"""
scripts/06_infer_rtsp.py
========================

Propósito
---------
Ejecutar inferencia EPP en tiempo real sobre un stream RTSP y mostrar:
- bounding boxes estabilizados
- tracking lógico de personas
- evaluación de EPP por persona
- alertas por transición de estado
- evidencia en imagen y video
- debug visual configurable solo para eventos reales
- visualización en ventana OpenCV
- reconexión RTSP robusta
- timeout configurable y watchdog de congelamiento

Comportamiento de alertas
-------------------------
- Si una persona aparece por primera vez en NO_OK:
  se emite violation_started y se guarda evidencia.
- Si una persona pasa de OK a NO_OK:
  se emite violation_started y se guarda evidencia.
- Si una persona pasa de NO_OK a OK:
  se emite violation_resolved y NO se guarda evidencia.
- Si una persona permanece en el mismo estado:
  no se genera evento nuevo.
"""

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
)
from utils.detection_stabilizer import DetectionStabilizer
from utils.logger import get_logger, mask_rtsp_url
from utils.person_tracker import PersonTracker
from utils.ppe_associator import evaluate_all_persons
from utils.business_rules import RulesEngine
from utils.alert_manager import AlertManager
from utils.evidence_manager import EvidenceManager
from utils.event_serializer import EventSerializer
from utils.frame_buffer import FrameBuffer
from utils.camera_registry import CameraRegistry

MODEL_PATH = os.getenv("INFERENCE_MODEL_PATH", "data/runs/epp_mvp/weights/best.pt")
CAMERA_ID = os.getenv("CAMERA_ID", "cam_rtsp_001")
SCENARIO_ID = os.getenv("SCENARIO_ID", "helmet_and_vest_required")

SHOW_WINDOW = os.getenv("SHOW_WINDOW", "true").lower() == "true"
DEBUG_VISUAL = os.getenv("DEBUG_VISUAL", "true").lower() == "true"
DEBUG_LOG_PER_PERSON = os.getenv("DEBUG_LOG_PER_PERSON", "true").lower() == "true"
DEBUG_DRAW_REGIONS = os.getenv("DEBUG_DRAW_REGIONS", "false").lower() == "true"
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

        if status == "compliant":
            text = f"ID {person_id} | OK"
            color = (0, 255, 0)
        else:
            violations_text = ",".join(violations) if violations else "review"
            text = f"ID {person_id} | {violations_text}"
            color = (0, 0, 255)

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


def log_person_evaluation(frame_count, person_eval, status, violations):
    logger.info(
        "EPP_EVAL | frame=%s | person_id=%s | status=%s | violations=%s | "
        "person_box=%s | head_box=%s | torso_box=%s | "
        "helmet_ok=%s | helmet_score=%.3f | helmet_box=%s | "
        "vest_ok=%s | vest_score=%.3f | vest_box=%s",
        frame_count,
        person_eval.get("person_id"),
        status,
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


def load_camera_scenario(camera_id: str, default_scenario_id: str) -> str:
    cameras_path = ROOT / "config" / "cameras.json"
    try:
        registry = CameraRegistry(str(cameras_path))
        camera_data = registry.get_camera(camera_id)
        return camera_data.get("scenario_id", default_scenario_id)
    except Exception as exc:
        logger.warning(
            f"No se pudo resolver camera_id en cameras.json. "
            f"Se usará SCENARIO_ID={default_scenario_id} | error={exc}"
        )
        return default_scenario_id


def build_event_evidence_frame(frame, person_id, person_box, violations, person_eval):
    evidence_frame = frame.copy()

    x1, y1, x2, y2 = person_box
    text = f"{person_id} | {','.join(violations)}" if violations else f"{person_id} | review"

    cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        evidence_frame,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    helmet = person_eval.get("helmet")
    vest = person_eval.get("vest")

    if helmet:
        hx1, hy1, hx2, hy2 = helmet["box"]
        cv2.rectangle(evidence_frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

    if vest:
        vx1, vy1, vx2, vy2 = vest["box"]
        cv2.rectangle(evidence_frame, (vx1, vy1), (vx2, vy2), (0, 165, 255), 2)

    return evidence_frame


def build_event_evidence_clip(frames, person_id, person_box, violations, person_eval):
    annotated_frames = []
    for frame in frames:
        annotated_frames.append(
            build_event_evidence_frame(
                frame=frame,
                person_id=person_id,
                person_box=person_box,
                violations=violations,
                person_eval=person_eval,
            )
        )
    return annotated_frames


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
        logger.info(f"Intentando abrir RTSP | attempt={attempts} | url={safe_rtsp_url}")
        cap = open_rtsp_capture(rtsp_url)

        if not cap.isOpened():
            logger.warning(f"RTSP no abrió correctamente | attempt={attempts} | url={safe_rtsp_url}")
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
                    f"RTSP conectado correctamente | attempt={attempts} | "
                    f"width={width} | height={height} | fps={fps}"
                )
                return cap, width, height, fps

            logger.warning(f"RTSP abrió pero no entregó frames válidos | attempt={attempts}")
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
        logger.error(f"No existe el modelo: {MODEL_PATH}")
        return

    scenarios_path = ROOT / "config" / "scenarios.json"
    if not scenarios_path.exists():
        logger.error(f"No existe el archivo de escenarios: {scenarios_path}")
        return

    scenario_id = load_camera_scenario(CAMERA_ID, SCENARIO_ID)

    model = YOLO(MODEL_PATH)

    stabilizer = build_stabilizer()
    tracker = PersonTracker(
        max_missing_seconds=1.2,
        max_center_distance=140.0,
        max_match_cost=0.85,
    )
    rules = RulesEngine(str(scenarios_path))
    alert_manager = AlertManager(default_cooldown=15)
    evidence = EvidenceManager(base_path="evidence")
    serializer = EventSerializer(base_path="evidence/json")

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
            detections = stabilizer.get_visible_tracks_as_detections()
            persons = tracker.update(detections)

            person_evals = evaluate_all_persons(persons, detections)
            eval_map = {item["person_id"]: item for item in person_evals}

            persons_status = []
            persons_debug = []
            alerted_person_ids = set()

            for person in persons:
                person_id = person["person_id"]
                person_eval = eval_map.get(person_id)
                if not person_eval:
                    continue

                status, violations, rule = rules.evaluate(scenario_id, person_eval)

                if DEBUG_LOG_PER_PERSON:
                    log_person_evaluation(frame_count, person_eval, status, violations)

                person_box = person.get("box")
                if not person_box:
                    continue

                persons_status.append({
                    "person_id": person_id,
                    "bbox": person_box,
                    "status": status,
                    "violations": violations,
                    "helmet_ok": person_eval.get("helmet_ok", False),
                    "vest_ok": person_eval.get("vest_ok", False),
                })

                persons_debug.append({
                    "person_id": person_id,
                    "person_eval": person_eval,
                    "status": status,
                    "violations": violations,
                })

                cooldown = int(rule.get("alert_cooldown_seconds", 15))
                should_emit, event_type = alert_manager.should_alert(
                    person_id=person_id,
                    status=status,
                    violations=violations,
                    cooldown=cooldown,
                )

                if should_emit:
                    event_id = f"{CAMERA_ID}_{person_id}_{frame_count}"
                    image_path = None
                    video_path = None

                    if event_type == "violation_started":
                        alerted_person_ids.add(person_id)

                        evidence_frame = build_event_evidence_frame(
                            frame=frame,
                            person_id=person_id,
                            person_box=person_box,
                            violations=violations,
                            person_eval=person_eval,
                        )

                        if rule.get("save_image", True):
                            image_path = evidence.save_image(evidence_frame, event_id)

                        if rule.get("save_video", True):
                            clip_frames = frame_buffer.get_frames()
                            annotated_clip_frames = build_event_evidence_clip(
                                frames=clip_frames,
                                person_id=person_id,
                                person_box=person_box,
                                violations=violations,
                                person_eval=person_eval,
                            )
                            video_path = evidence.save_video_clip(
                                frames=annotated_clip_frames,
                                fps=fps,
                                width=width,
                                height=height,
                                event_id=event_id,
                            )

                    serializer.build_event(
                        camera_id=CAMERA_ID,
                        scenario_id=scenario_id,
                        person_id=person_id,
                        status=status,
                        violations=violations,
                        image_path=image_path,
                        video_path=video_path,
                        event_type=event_type,
                    )

            annotated = stabilizer.draw(frame)
            annotated = draw_person_status(annotated, persons_status)

            if DEBUG_VISUAL:
                for item in persons_debug:
                    if item["person_id"] not in alerted_person_ids:
                        continue
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

    finally:
        release_capture(cap)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()