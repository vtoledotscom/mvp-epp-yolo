from itertools import count
from typing import Dict, List, Optional
from utils.config import PERSON_CONFIDENCE, HELMET_CONFIDENCE, VEST_CONFIDENCE

import cv2


class DetectionStabilizer:
    """
    Corrección importante:
    expone get_visible_tracks_as_detections(), de modo que tracking y
    evaluación EPP usen las cajas estabilizadas y no las detecciones crudas.
    """

    def __init__(
        self,
        iou_threshold: float,
        smoothing_alpha: float,
        max_missing_frames: int,
        min_box_area: int,
        max_center_distance: int,
        class_colors: Optional[Dict[str, tuple]] = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.smoothing_alpha = smoothing_alpha
        self.max_missing_frames = max_missing_frames
        self.min_box_area = min_box_area
        self.max_center_distance = max_center_distance
        self.class_colors = class_colors or {}

        self.tracked_objects: Dict[int, dict] = {}
        self._next_track_id = count(1)

    @staticmethod
    def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    @staticmethod
    def box_area(box: List[int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def box_center(box: List[int]):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def center_distance(self, box_a: List[int], box_b: List[int]) -> float:
        ax, ay = self.box_center(box_a)
        bx, by = self.box_center(box_b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def smooth_box(self, prev_box: List[int], curr_box: List[int]) -> List[int]:
        alpha = self.smoothing_alpha
        return [
            int((1 - alpha) * prev_box[0] + alpha * curr_box[0]),
            int((1 - alpha) * prev_box[1] + alpha * curr_box[1]),
            int((1 - alpha) * prev_box[2] + alpha * curr_box[2]),
            int((1 - alpha) * prev_box[3] + alpha * curr_box[3]),
        ]

    @staticmethod
    def clamp_box(box: List[int], width: int, height: int) -> List[int]:
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        return [x1, y1, x2, y2]

    def extract_detections(self, results, model_names, frame_width: int, frame_height: int) -> list[dict]:
        detections = []

        if not results or results[0].boxes is None:
            return detections

        class_thresholds = {
            "person": PERSON_CONFIDENCE,
            "helmet": HELMET_CONFIDENCE,
            "vest": VEST_CONFIDENCE,
        }

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = model_names[cls_id]

            min_conf = class_thresholds.get(class_name, PERSON_CONFIDENCE)
            if conf < min_conf:
                continue

            coords = list(map(int, box.xyxy[0].tolist()))
            coords = self.clamp_box(coords, frame_width, frame_height)

            if self.box_area(coords) < self.min_box_area:
                continue

            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class_name": class_name,
                "conf": conf,
                "box": coords,
            })

        return detections

    def match_detection_to_track(self, detection: dict, used_track_ids: set) -> Optional[int]:
        best_track_id = None
        best_score = -1.0

        det_box = detection["box"]
        det_class = detection["class_name"]

        for track_id, track in self.tracked_objects.items():
            if track_id in used_track_ids:
                continue

            if track["class_name"] != det_class:
                continue

            prev_box = track["box"]
            iou = self.calculate_iou(det_box, prev_box)
            distance = self.center_distance(det_box, prev_box)

            if iou >= self.iou_threshold:
                score = iou
            elif distance <= self.max_center_distance:
                score = 0.30 / (1.0 + distance)
            else:
                continue

            if score > best_score:
                best_score = score
                best_track_id = track_id

        return best_track_id

    def update(self, results, model_names, frame_width: int, frame_height: int) -> List[dict]:
        detections = self.extract_detections(results, model_names, frame_width, frame_height)

        for track in self.tracked_objects.values():
            track["updated"] = False

        used_track_ids = set()

        for detection in detections:
            matched_track_id = self.match_detection_to_track(detection, used_track_ids)

            if matched_track_id is not None:
                prev_track = self.tracked_objects[matched_track_id]
                smoothed = self.smooth_box(prev_track["box"], detection["box"])

                self.tracked_objects[matched_track_id]["box"] = self.clamp_box(
                    smoothed, frame_width, frame_height
                )
                self.tracked_objects[matched_track_id]["conf"] = detection["conf"]
                self.tracked_objects[matched_track_id]["missing_frames"] = 0
                self.tracked_objects[matched_track_id]["updated"] = True
                used_track_ids.add(matched_track_id)
            else:
                new_id = next(self._next_track_id)
                self.tracked_objects[new_id] = {
                    "track_id": new_id,
                    "class_name": detection["class_name"],
                    "box": detection["box"],
                    "conf": detection["conf"],
                    "missing_frames": 0,
                    "updated": True,
                }
                used_track_ids.add(new_id)

        to_delete = []
        for track_id, track in self.tracked_objects.items():
            if not track["updated"]:
                track["missing_frames"] += 1

            if track["missing_frames"] > self.max_missing_frames:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracked_objects[track_id]

        return detections

    def draw(self, frame):
        annotated = frame.copy()

        for track in self.tracked_objects.values():
            if track["missing_frames"] > self.max_missing_frames:
                continue

            x1, y1, x2, y2 = track["box"]
            class_name = track["class_name"]
            conf = track["conf"]

            color = self.class_colors.get(class_name, (255, 255, 255))
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def get_visible_tracks(self) -> List[dict]:
        return [
            {
                "track_id": track.get("track_id", track_id),
                "class_name": track["class_name"],
                "box": track["box"],
                "conf": track["conf"],
                "missing_frames": track["missing_frames"],
            }
            for track_id, track in self.tracked_objects.items()
            if track["missing_frames"] <= self.max_missing_frames
        ]

    def get_visible_tracks_as_detections(self) -> List[dict]:
        return [
            {
                "track_id": track["track_id"],
                "class_name": track["class_name"],
                "box": track["box"],
                "conf": track["conf"],
            }
            for track in self.get_visible_tracks()
        ]