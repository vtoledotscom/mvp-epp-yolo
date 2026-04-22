"""
scripts/06_infer_rtsp.py
====================================

Versión RTSP homologada con el flujo moderno del MVP EPP.

Qué homologa respecto a 06_infer_video.py
-----------------------------------------
- evaluación global multi-persona
- filtro por inspection_zone e ignore_zones
- histéresis temporal por persona
- evidencia estructurada
- serialización mysql-ready
- persistencia opcional en MySQL/MariaDB
- dibujo de solo la escena evaluada

Qué conserva del flujo RTSP
---------------------------
- apertura de stream RTSP
- reconexión robusta
- timeout configurable
- visualización en ventana OpenCV
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.config import (
    LOG_DIR,
    RTSP_URL,
    CONFIDENCE,
    IOU_THRESHOLD,
    SMOOTHING_ALPHA,
    MAX_MISSING_FRAMES,
    MIN_BOX_AREA,
    MAX_CENTER_DISTANCE,
    MYSQL_ENABLED,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    MYSQL_CONNECT_TIMEOUT,
)
from utils.detection_stabilizer import DetectionStabilizer
from utils.logger import get_logger, mask_rtsp_url
from utils.person_tracker import PersonTracker
from utils.ppe_associator import evaluate_all_persons
from utils.business_rules import RulesEngine
from utils.frame_buffer import FrameBuffer
from utils.compliance_state_manager import ComplianceStateManager
from utils.zone_utils import (
    is_person_in_ignore_zones,
    is_person_in_inspection_zone,
)
from utils.evidence_manager import EvidenceManager
from utils.event_serializer_mysql import EventSerializerMySQL
from utils.mysql_event_repository import MySQLEventRepository

# Configuraciones principales (se pueden ajustar vía variables de entorno o .env)
MODEL_PATH = os.getenv("INFERENCE_MODEL_PATH", "data/runs/epp_mvp/weights/best.pt")
CAMERA_ID = os.getenv("CAMERA_ID", "cam_rtsp_001")
SCENARIO_ID = os.getenv("SCENARIO_ID", "helmet_and_vest_required")

SHOW_WINDOW = os.getenv("SHOW_WINDOW", "true").lower() == "true"
DEBUG_VISUAL = os.getenv("DEBUG_VISUAL", "true").lower() == "true"
DEBUG_LOG_PER_PERSON = os.getenv("DEBUG_LOG_PER_PERSON", "true").lower() == "true"
DEBUG_DRAW_REGIONS = os.getenv("DEBUG_DRAW_REGIONS", "true").lower() == "true"
DEBUG_DRAW_EPP = os.getenv("DEBUG_DRAW_EPP", "true").lower() == "true"

FRAME_BUFFER_SECONDS = int(os.getenv("FRAME_BUFFER_SECONDS", "3"))

RTSP_RECONNECT_DELAY_SECONDS = float(os.getenv("RTSP_RECONNECT_DELAY_SECONDS", "3"))
RTSP_MAX_RECONNECT_ATTEMPTS = int(os.getenv("RTSP_MAX_RECONNECT_ATTEMPTS", "0"))
RTSP_OPEN_WARMUP_FRAMES = int(os.getenv("RTSP_OPEN_WARMUP_FRAMES", "3"))
RTSP_OPEN_WARMUP_TIMEOUT_SECONDS = float(os.getenv("RTSP_OPEN_WARMUP_TIMEOUT_SECONDS", "8"))
RTSP_NO_FRAME_TIMEOUT_SECONDS = float(os.getenv("RTSP_NO_FRAME_TIMEOUT_SECONDS", "10"))
RTSP_READ_FAIL_THRESHOLD = int(os.getenv("RTSP_READ_FAIL_THRESHOLD", "15"))
RTSP_FORCE_FPS = float(os.getenv("RTSP_FORCE_FPS", "15"))

DISPLAY_WINDOW_NAME = os.getenv("DISPLAY_WINDOW_NAME", "EPP RTSP Inference")

logger = get_logger("infer_rtsp_epp", str(LOG_DIR / "infer_rtsp_epp.log"))

CLASS_COLORS = {
    "helmet": (0, 255, 255),
    "person": (255, 0, 0),
    "vest": (0, 165, 255),
}


def build_stabilizer() -> DetectionStabilizer:
    return DetectionStabilizer(
        iou_threshold=IOU_THRESHOLD,
        smoothing_alpha=SMOOTHING_ALPHA,
        max_missing_frames=MAX_MISSING_FRAMES,
        min_box_area=MIN_BOX_AREA,
        max_center_distance=MAX_CENTER_DISTANCE,
        class_colors=CLASS_COLORS,
    )


def draw_valid_evaluated_scene(frame, persons_eval: list[dict], class_colors: dict):
    annotated = frame.copy()

    for item in persons_eval:
        person_box = item.get("person_box")
        helmet = item.get("helmet")
        vest = item.get("vest")

        if person_box:
            x1, y1, x2, y2 = person_box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                class_colors.get("person", (255, 0, 0)),
                2,
            )

        if helmet and helmet.get("box"):
            x1, y1, x2, y2 = helmet["box"]
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                class_colors.get("helmet", (0, 255, 255)),
                2,
            )

        if vest and vest.get("box"):
            x1, y1, x2, y2 = vest["box"]
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                class_colors.get("vest", (0, 165, 255)),
                2,
            )

    return annotated


def load_camera_config(camera_id: str, default_scenario_id: str) -> dict:
    cameras_path = ROOT / "config" / "cameras.json"

    if not cameras_path.exists():
        logger.warning("No existe config/cameras.json. Se usarán valores por defecto.")
        return {
            "camera_id": camera_id,
            "name": camera_id,
            "scenario_id": default_scenario_id,
            "inspection_zone": None,
            "ignore_zones": [],
            "min_zone_overlap": 0.25,
            "min_non_compliant_frames": 5,
            "min_compliant_frames": 3,
            "zone_name": "inspection_zone",
        }

    data = json.loads(cameras_path.read_text(encoding="utf-8"))
    cameras = data.get("cameras", {})

    camera_data = None
    if isinstance(cameras, dict):
        camera_data = cameras.get(camera_id)
    elif isinstance(cameras, list):
        for item in cameras:
            if item.get("camera_id") == camera_id:
                camera_data = item
                break

    if not camera_data:
        logger.warning(
            "camera_id=%s no encontrado en cameras.json. Se usarán valores por defecto.",
            camera_id,
        )
        return {
            "camera_id": camera_id,
            "name": camera_id,
            "scenario_id": default_scenario_id,
            "inspection_zone": None,
            "ignore_zones": [],
            "min_zone_overlap": 0.25,
            "min_non_compliant_frames": 5,
            "min_compliant_frames": 3,
            "zone_name": "inspection_zone",
        }

    return {
        "camera_id": camera_id,
        "name": camera_data.get("name", camera_id),
        "scenario_id": camera_data.get("scenario_id", default_scenario_id),
        "inspection_zone": camera_data.get("inspection_zone"),
        "ignore_zones": camera_data.get("ignore_zones", []),
        "min_zone_overlap": float(camera_data.get("min_zone_overlap", 0.25)),
        "min_non_compliant_frames": int(camera_data.get("min_non_compliant_frames", 5)),
        "min_compliant_frames": int(camera_data.get("min_compliant_frames", 3)),
        "zone_name": camera_data.get("zone_name", "inspection_zone"),
    }


def draw_rectangle_zone(frame, zone: dict | None, color=(255, 255, 0), label="ZONE"):
    if not zone:
        return
    if zone.get("type", "rectangle") != "rectangle":
        return

    x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_person_status(frame, persons_status):
    annotated = frame.copy()

    for item in persons_status:
        bbox = item.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        person_id = item["person_id"]
        status = item["status"]
        violations = item["violations"]
        temporal = item.get("temporal_label")

        if status == "compliant":
            text = f"ID {person_id} | OK"
            color = (0, 255, 0)
        else:
            violations_text = ",".join(violations) if violations else "review"
            text = f"ID {person_id} | {violations_text}"
            color = (0, 0, 255)

        if temporal:
            text = f"{text} | {temporal}"

        cv2.putText(
            annotated,
            text,
            (x1, min(y2 + 20, annotated.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def draw_debug_box(frame, box, color, label):
    if not box:
        return

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_debug_point(frame, box, color, label):
    if not box:
        return

    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.circle(frame, (cx, cy), 4, color, -1)
    cv2.putText(
        frame,
        label,
        (cx + 4, cy - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_person_debug(frame, person_eval, status=None, violations=None):
    person_box = person_eval.get("person_box")
    head_box = person_eval.get("head_box")
    torso_box = person_eval.get("torso_box")
    helmet = person_eval.get("helmet")
    vest = person_eval.get("vest")
    person_id = person_eval.get("person_id")
    helmet_score = float(person_eval.get("helmet_score", 0.0))
    vest_score = float(person_eval.get("vest_score", 0.0))

    if person_box:
        draw_debug_box(frame, person_box, (255, 0, 0), f"{person_id}")

        if status == "non_compliant":
            vx1, vy1, _, _ = person_box
            text = ",".join(violations or []) if violations else "review"
            cv2.putText(
                frame,
                text,
                (vx1, max(20, vy1 - 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    if DEBUG_DRAW_REGIONS:
        if head_box:
            draw_debug_box(frame, head_box, (0, 255, 255), "HEAD")

        if torso_box:
            draw_debug_box(frame, torso_box, (0, 165, 255), "TORSO")

    if DEBUG_DRAW_EPP:
        if helmet:
            draw_debug_box(frame, helmet["box"], (0, 255, 255), f"helmet {helmet_score:.2f}")
            draw_debug_point(frame, helmet["box"], (0, 255, 255), "H")

        if vest:
            draw_debug_box(frame, vest["box"], (0, 165, 255), f"vest {vest_score:.2f}")
            draw_debug_point(frame, vest["box"], (0, 165, 255), "V")


def log_person_evaluation(frame_count, person_eval, observed_status, confirmed_status, violations):
    logger.info(
        "EPP_EVAL | frame=%s | person_id=%s | observed_status=%s | confirmed_status=%s | violations=%s | "
        "person_box=%s | head_box=%s | torso_box=%s | "
        "helmet_ok=%s | helmet_score=%.3f | helmet_box=%s | "
        "vest_ok=%s | vest_score=%.3f | vest_box=%s",
        frame_count,
        person_eval.get("person_id"),
        observed_status,
        confirmed_status,
        violations,
        person_eval.get("person_box"),
        person_eval.get("head_box"),
        person_eval.get("torso_box"),
        person_eval.get("helmet_ok"),
        float(person_eval.get("helmet_score", 0.0)),
        person_eval.get("helmet", {}).get("box") if person_eval.get("helmet") else None,
        person_eval.get("vest_ok"),
        float(person_eval.get("vest_score", 0.0)),
        person_eval.get("vest", {}).get("box") if person_eval.get("vest") else None,
    )


def release_capture(cap):
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def try_read_warmup_frames(cap, warmup_frames: int, timeout_seconds: float) -> bool:
    start = time.time()
    ok_reads = 0

    while time.time() - start < timeout_seconds:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            ok_reads += 1
            if ok_reads >= warmup_frames:
                return True
        else:
            time.sleep(0.05)

    return False


def open_rtsp_capture(rtsp_url: str):
    return cv2.VideoCapture(rtsp_url)


def connect_rtsp(rtsp_url: str, safe_rtsp_url: str):
    attempts = 0

    while True:
        attempts += 1
        logger.info("Intentando abrir RTSP | attempt=%s | url=%s", attempts, safe_rtsp_url)
        cap = open_rtsp_capture(rtsp_url)

        if not cap.isOpened():
            logger.warning("RTSP no abrió correctamente | attempt=%s | url=%s", attempts, safe_rtsp_url)
            release_capture(cap)
        else:
            warmup_ok = try_read_warmup_frames(
                cap=cap,
                warmup_frames=max(1, RTSP_OPEN_WARMUP_FRAMES),
                timeout_seconds=RTSP_OPEN_WARMUP_TIMEOUT_SECONDS,
            )

            if warmup_ok:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 1 or fps > 120:
                    fps = RTSP_FORCE_FPS

                logger.info(
                    "RTSP conectado correctamente | attempt=%s | width=%s | height=%s | fps=%s",
                    attempts,
                    width,
                    height,
                    fps,
                )
                return cap, width, height, fps

            logger.warning("RTSP abrió pero no entregó frames válidos | attempt=%s", attempts)
            release_capture(cap)

        if RTSP_MAX_RECONNECT_ATTEMPTS > 0 and attempts >= RTSP_MAX_RECONNECT_ATTEMPTS:
            raise RuntimeError(f"No se pudo conectar al RTSP después de {attempts} intentos.")

        time.sleep(RTSP_RECONNECT_DELAY_SECONDS)


def main() -> None:
    safe_rtsp = mask_rtsp_url(RTSP_URL)

    if not RTSP_URL:
        logger.error("RTSP_URL no está definido en .env")
        return

    if not Path(MODEL_PATH).exists():
        logger.error("No existe el modelo: %s", MODEL_PATH)
        return

    scenarios_path = ROOT / "config" / "scenarios.json"
    if not scenarios_path.exists():
        logger.error("No existe el archivo de escenarios: %s", scenarios_path)
        return

    camera_cfg = load_camera_config(CAMERA_ID, SCENARIO_ID)
    scenario_id = camera_cfg["scenario_id"]
    inspection_zone = camera_cfg["inspection_zone"]
    ignore_zones = camera_cfg["ignore_zones"]
    min_zone_overlap = camera_cfg["min_zone_overlap"]
    min_non_compliant_frames = camera_cfg["min_non_compliant_frames"]
    min_compliant_frames = camera_cfg["min_compliant_frames"]
    zone_name = camera_cfg["zone_name"]

    logger.info("RTSP_URL=%s", safe_rtsp)
    logger.info("MODEL_PATH=%s", MODEL_PATH)
    logger.info("CAMERA_ID=%s", CAMERA_ID)
    logger.info("SCENARIO_ID=%s", SCENARIO_ID)
    logger.info("camera_cfg=%s", camera_cfg)

    model = YOLO(MODEL_PATH)

    stabilizer = build_stabilizer()
    tracker = PersonTracker(max_missing_seconds=1.0)
    rules = RulesEngine(str(scenarios_path))
    compliance_manager = ComplianceStateManager(
        default_min_non_compliant_frames=min_non_compliant_frames,
        default_min_compliant_frames=min_compliant_frames,
    )
    evidence = EvidenceManager(base_path="evidence")
    serializer = EventSerializerMySQL(base_path="evidence/json")

    mysql_repo = None
    if MYSQL_ENABLED:
        mysql_repo = MySQLEventRepository(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            connect_timeout=MYSQL_CONNECT_TIMEOUT,
        )
        logger.info(
            "MySQL habilitado | host=%s | port=%s | database=%s | user=%s",
            MYSQL_HOST,
            MYSQL_PORT,
            MYSQL_DATABASE,
            MYSQL_USER,
        )

    cap = None
    width = 1280
    height = 720
    fps = RTSP_FORCE_FPS

    try:
        cap, width, height, fps = connect_rtsp(RTSP_URL, safe_rtsp)
        frame_buffer = FrameBuffer(max_frames=max(1, int(fps * FRAME_BUFFER_SECONDS)))

        frame_count = 0
        consecutive_read_failures = 0
        last_valid_frame_ts = time.time()
        events_count = 0

        while True:
            ret, frame = cap.read()
            now = time.time()

            if ret and frame is not None and frame.size > 0:
                consecutive_read_failures = 0
                last_valid_frame_ts = now
            else:
                consecutive_read_failures += 1
                no_frame_elapsed = now - last_valid_frame_ts

                if (
                    consecutive_read_failures >= RTSP_READ_FAIL_THRESHOLD
                    or no_frame_elapsed >= RTSP_NO_FRAME_TIMEOUT_SECONDS
                ):
                    logger.warning(
                        "RTSP sin frames válidos | failures=%s | elapsed=%.2f | reconectando",
                        consecutive_read_failures,
                        no_frame_elapsed,
                    )
                    release_capture(cap)
                    cap, width, height, fps = connect_rtsp(RTSP_URL, safe_rtsp)
                    frame_buffer = FrameBuffer(max_frames=max(1, int(fps * FRAME_BUFFER_SECONDS)))
                    consecutive_read_failures = 0
                    last_valid_frame_ts = time.time()

                time.sleep(0.05)
                continue

            frame_count += 1
            frame_buffer.add_frame(frame)

            results = model.predict(
                frame,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                verbose=False,
            )

            stabilizer.update(results, model.names, width, height)
            stable_detections = stabilizer.get_visible_tracks_as_detections()
            persons = tracker.update(stable_detections, fps=fps)

            valid_persons = []
            for person in persons:
                person_box = person.get("box")
                if not person_box:
                    continue

                if is_person_in_ignore_zones(person_box, ignore_zones):
                    continue

                if not is_person_in_inspection_zone(
                    person_box,
                    inspection_zone=inspection_zone,
                    min_overlap=min_zone_overlap,
                ):
                    continue

                valid_persons.append(person)

            persons_eval = evaluate_all_persons(valid_persons, stable_detections)
            eval_by_person_id = {item["person_id"]: item for item in persons_eval}

            persons_status = []
            persons_debug = []
            active_person_ids = set()

            annotated_base = frame.copy()
            annotated_base = draw_valid_evaluated_scene(
                frame=annotated_base,
                persons_eval=persons_eval,
                class_colors=CLASS_COLORS,
            )

            if inspection_zone and inspection_zone.get("type", "rectangle") == "rectangle":
                draw_rectangle_zone(annotated_base, inspection_zone, color=(255, 255, 0), label="INSPECTION")

            for idx, ignore_zone in enumerate(ignore_zones or []):
                if ignore_zone.get("type", "rectangle") == "rectangle":
                    draw_rectangle_zone(annotated_base, ignore_zone, color=(128, 128, 128), label=f"IGNORE_{idx+1}")

            for person in valid_persons:
                person_id = person["person_id"]
                active_person_ids.add(person_id)

                person_eval = eval_by_person_id.get(person_id)
                if not person_eval:
                    continue

                observed_status, violations, rule = rules.evaluate(scenario_id, person_eval)

                changed, event_type, state_snapshot = compliance_manager.update(
                    person_id=person_id,
                    observed_status=observed_status,
                    observed_violations=violations,
                    frame_number=frame_count,
                    min_non_compliant_frames=min_non_compliant_frames,
                    min_compliant_frames=min_compliant_frames,
                )

                confirmed_status = state_snapshot.get("confirmed_status", observed_status)
                confirmed_violations = state_snapshot.get("confirmed_violations", violations)

                if DEBUG_LOG_PER_PERSON:
                    log_person_evaluation(
                        frame_count=frame_count,
                        person_eval=person_eval,
                        observed_status=observed_status,
                        confirmed_status=confirmed_status,
                        violations=confirmed_violations,
                    )

                temporal_label = None
                candidate_status = state_snapshot.get("candidate_status")
                candidate_count = state_snapshot.get("candidate_count", 0)
                if candidate_status:
                    temporal_label = f"cand={candidate_status}:{candidate_count}"

                person_state = {
                    "person_id": person_id,
                    "bbox": person.get("box"),
                    "status": confirmed_status,
                    "violations": confirmed_violations,
                    "helmet_ok": person_eval.get("helmet_ok", False),
                    "vest_ok": person_eval.get("vest_ok", False),
                    "temporal_label": temporal_label,
                }
                persons_status.append(person_state)
                persons_debug.append({
                    "person_id": person_id,
                    "person_eval": person_eval,
                    "status": confirmed_status,
                    "violations": confirmed_violations,
                })

                if not changed or event_type is None:
                    continue

                # NUEVO: Solo procesar violation_started (no violation_resolved)
                if event_type != "violation_started":
                    continue

                event_id = f"{CAMERA_ID}_{person_id}_{frame_count}"
                evidence_paths = {}

                # Ya no necesita el if event_type == "violation_started" porque ya se filtró arriba
                evidence_paths["image_full_path"] = evidence.save_full_image(frame, event_id)
                evidence_paths["image_annotated_path"] = evidence.save_annotated_image(annotated_base, event_id)
                evidence_paths["image_crop_path"] = evidence.save_person_crop(
                    frame=frame,
                    person_box=person_eval.get("person_box"),
                    event_id=event_id,
                )
                evidence_paths["video_path"] = evidence.save_video_clip(
                    frames=frame_buffer.get_frames(),
                    fps=fps,
                    width=width,
                    height=height,
                    event_id=event_id,
                )

                event = serializer.build_event(
                    camera_id=CAMERA_ID,
                    scenario_id=scenario_id,
                    zone_name=zone_name,
                    person_track_id=person_id,
                    frame_number=frame_count,
                    status=confirmed_status,
                    event_type=event_type,
                    violation_codes=confirmed_violations,
                    person_eval=person_eval,
                    observed_status=observed_status,
                    state_snapshot=state_snapshot,
                    evidence_paths=evidence_paths,
                    model_version=str(MODEL_PATH),
                )

                if mysql_repo is not None:
                    try:
                        inserted = mysql_repo.save_event(event)
                        logger.info(
                            "MySQL save_event | event_id=%s | inserted=%s",
                            event["event_id"],
                            inserted,
                        )

                        if event["event_type"] == "violation_resolved":
                            affected = mysql_repo.close_open_violation(
                                camera_id=event["camera_id"],
                                person_track_id=event["person_track_id"],
                                scenario_id=event["scenario_id"],
                                resolved_event_id=event["event_id"],
                                resolved_at=event["event_confirmed_at"],
                            )
                            logger.info(
                                "MySQL close_open_violation | event_id=%s | affected=%s",
                                event["event_id"],
                                affected,
                            )

                    except Exception as exc:
                        logger.error(
                            "Error guardando evento en MySQL | event_id=%s | error=%s",
                            event["event_id"],
                            exc,
                        )

                events_count += 1
                logger.warning(
                    "ALERTA EPP | frame=%s | person_id=%s | scenario=%s | status=%s | "
                    "event_type=%s | violations=%s | event=%s",
                    frame_count,
                    person_id,
                    scenario_id,
                    confirmed_status,
                    event_type,
                    confirmed_violations,
                    event,
                )

            annotated = draw_person_status(annotated_base, persons_status)

            if DEBUG_VISUAL:
                for item in persons_debug:
                    draw_person_debug(
                        annotated,
                        item["person_eval"],
                        status=item["status"],
                        violations=item["violations"],
                    )

            if SHOW_WINDOW:
                cv2.imshow(DISPLAY_WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            visible_tracks = stabilizer.get_visible_tracks()
            logger.info(
                "Frame=%s | detecciones_estables=%s | persons=%s | persons_valid=%s | eventos=%s",
                frame_count,
                len(visible_tracks),
                len(persons),
                len(valid_persons),
                events_count,
            )

            compliance_manager.cleanup_absent(active_person_ids=active_person_ids)

    finally:
        release_capture(cap)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()