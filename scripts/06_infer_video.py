"""
NOTA DE ESTA VERSIÓN
--------------------
Se ajustó el dibujado del video para mostrar solo la escena evaluada.
Ya no se usa el dibujo global del estabilizador como capa principal.
Eso reduce ruido visual y evita mostrar detecciones que no participan
realmente en la decisión del MVP.

scripts/06_infer_video_.py
======================================================

Versión extendida del MVP EPP con:
- detecciones estabilizadas
- evaluación global multi-persona
- filtro por zona de inspección e ignore zones
- histéresis temporal por persona
- evidencia full / annotated / crop / clip
- serialización lista para MySQL
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.config import (
    LOG_DIR,
    CONFIDENCE,
    PERSON_CONFIDENCE,
    HELMET_CONFIDENCE,
    VEST_CONFIDENCE,
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

from utils.logger import get_logger
from utils.detection_stabilizer import DetectionStabilizer
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

VIDEO_PATH = os.getenv("INFERENCE_VIDEO_PATH", "data/raw_videos/deteccion_test_720.mp4")
MODEL_PATH = os.getenv("INFERENCE_MODEL_PATH", "data/runs/epp_mvp/weights/best.pt")
OUTPUT_PATH = os.getenv(
    "INFERENCE_VIDEO_OUTPUT_PATH",
    "runs/detect/detection_epp_vest.mp4"
)

CAMERA_ID = os.getenv("CAMERA_ID", "cam_rtsp_003")
SCENARIO_ID = os.getenv("SCENARIO_ID", "vest_required")

DEBUG_VISUAL = os.getenv("DEBUG_VISUAL", "true").lower() == "true"
DEBUG_LOG_PER_PERSON = os.getenv("DEBUG_LOG_PER_PERSON", "true").lower() == "true"
DEBUG_DRAW_REGIONS = os.getenv("DEBUG_DRAW_REGIONS", "true").lower() == "true"
DEBUG_DRAW_EPP = os.getenv("DEBUG_DRAW_EPP", "true").lower() == "true"

logger = get_logger("infer_video_epp", str(LOG_DIR / "infer_video.log"))

CLASS_COLORS = {
    "helmet": (0, 255, 255), #amarillo
    "person": (255, 0, 0), #azul
    "vest": (0, 165, 255), #naranjo
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
    """
    Dibuja únicamente los objetos que realmente participan en la evaluación EPP.

    Qué dibuja:
    - persona válida dentro de la zona de inspección
    - casco asociado a esa persona
    - chaleco asociado a esa persona

    Qué NO dibuja:
    - personas fuera de la zona
    - cascos/chalecos sueltos no asociados
    - ruido visual del resto del frame

    Por qué existe:
    Antes se usaba stabilizer.draw(frame), que dibujaba TODAS las detecciones
    estabilizadas del frame. Eso generaba una visualización cargada y poco fiel
    a la lógica de decisión real del MVP.

    Con esta función, el video final representa solo la escena evaluada.
    """
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
        logger.warning(
            "No existe config/cameras.json. Se usarán valores por defecto."
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


def main() -> None:
    logger.info("Inicio inferencia offline EPP v4")
    logger.info("VIDEO_PATH=%s", VIDEO_PATH)
    logger.info("MODEL_PATH=%s", MODEL_PATH)
    logger.info("OUTPUT_PATH=%s", OUTPUT_PATH)
    logger.info("CAMERA_ID=%s", CAMERA_ID)
    logger.info("SCENARIO_ID=%s", SCENARIO_ID)

    if not Path(VIDEO_PATH).exists():
        logger.error("No existe el video: %s", VIDEO_PATH)
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
    logger.warning("DEBUG_CAMERA_ID=%s | DEBUG_SCENARIO_ID_FINAL=%s | camera_cfg=%s", CAMERA_ID, scenario_id, camera_cfg)
    logger.warning(
        "DEBUG_INPUT | env_CAMERA_ID=%s | env_SCENARIO_ID=%s | CAMERA_ID=%s | SCENARIO_ID_DEFAULT=%s",
        os.getenv("CAMERA_ID"),
        os.getenv("SCENARIO_ID"),
        CAMERA_ID,
        SCENARIO_ID,
    )

    logger.warning(
        "DEBUG_FINAL | camera_cfg=%s | scenario_id_final=%s",
        camera_cfg,
        scenario_id,
    )
    inspection_zone = camera_cfg["inspection_zone"]
    ignore_zones = camera_cfg["ignore_zones"]
    min_zone_overlap = camera_cfg["min_zone_overlap"]
    min_non_compliant_frames = camera_cfg["min_non_compliant_frames"]
    min_compliant_frames = camera_cfg["min_compliant_frames"]
    zone_name = camera_cfg["zone_name"]

    logger.info("camera_cfg=%s", camera_cfg)

    model = YOLO(MODEL_PATH)
    logger.info("Clases del modelo: %s", model.names)

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


    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error("No se pudo abrir el video: %s", VIDEO_PATH)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 15.0
        logger.warning("FPS inválido detectado. Se usará fallback=%s", fps)

    frame_buffer = FrameBuffer(max_frames=int(fps * 3))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    events_count = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not ret or frame is None:
            logger.info("Fin de video o frame inválido | frame=%s", frame_count)
            break

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

        # Dibuja solo la escena realmente evaluada:
        # personas válidas + EPP asociado.
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

        out.write(annotated)

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

    cap.release()
    out.release()

    logger.info("Fin inferencia offline EPP v4")
    logger.info("Video guardado en: %s", OUTPUT_PATH)
    logger.info("Total eventos generados: %s", events_count)
    print(f"Video guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()