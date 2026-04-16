"""
utils/person_tracker.py
=======================

Tracker de personas más robusto para cruces y oclusiones cortas.

Mejoras:
- usa IoU + distancia + área + aspect ratio
- predicción simple por velocidad
- asignación uno a uno por mejor costo
- menor fragmentación de IDs
"""

from itertools import count
import math
import time


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area(box):
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def _aspect_ratio(box):
    x1, y1, x2, y2 = box
    w = max(1.0, (x2 - x1))
    h = max(1.0, (y2 - y1))
    return w / h


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = _area(box_a)
    area_b = _area(box_b)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


class PersonTracker:
    def __init__(
        self,
        max_missing_seconds: float = 1.2,
        max_center_distance: float = 140.0,
        max_match_cost: float = 0.85,
    ):
        self.max_missing_seconds = max_missing_seconds
        self.max_center_distance = max_center_distance
        self.max_match_cost = max_match_cost

        self._next_id = count(1)
        self.tracks = {}

    def _predict_center(self, track, now_ts):
        cx, cy = track["center"]
        vx, vy = track.get("velocity", (0.0, 0.0))
        dt = max(0.0, now_ts - track["last_ts"])
        return (cx + vx * dt, cy + vy * dt)

    def _compute_match_cost(self, track, det_box, now_ts):
        prev_box = track["box"]
        prev_area = _area(prev_box)
        det_area = _area(det_box)

        prev_ar = _aspect_ratio(prev_box)
        det_ar = _aspect_ratio(det_box)

        det_center = _center(det_box)
        pred_center = self._predict_center(track, now_ts)

        center_distance = _dist(pred_center, det_center)
        if center_distance > self.max_center_distance:
            return None

        iou_value = _iou(prev_box, det_box)

        if max(prev_area, det_area) > 0:
            area_diff = abs(prev_area - det_area) / max(prev_area, det_area)
        else:
            area_diff = 1.0

        aspect_diff = abs(prev_ar - det_ar) / max(prev_ar, det_ar, 1e-6)

        normalized_distance = min(1.0, center_distance / max(1.0, self.max_center_distance))

        w_iou = 0.45
        w_dist = 0.30
        w_area = 0.15
        w_aspect = 0.10

        cost = (
            w_iou * (1.0 - iou_value) +
            w_dist * normalized_distance +
            w_area * min(1.0, area_diff) +
            w_aspect * min(1.0, aspect_diff)
        )

        return cost

    def _create_track(self, det_box, now_ts):
        tid = f"p_{next(self._next_id):05d}"
        c = _center(det_box)

        self.tracks[tid] = {
            "box": det_box,
            "center": c,
            "velocity": (0.0, 0.0),
            "last_ts": now_ts,
            "hits": 1,
            "age": 1,
            "missing_count": 0,
        }
        return tid

    def _update_track(self, tid, det_box, now_ts):
        track = self.tracks[tid]

        prev_center = track["center"]
        new_center = _center(det_box)
        dt = max(1e-6, now_ts - track["last_ts"])

        vx = (new_center[0] - prev_center[0]) / dt
        vy = (new_center[1] - prev_center[1]) / dt

        old_vx, old_vy = track.get("velocity", (0.0, 0.0))
        smoothed_vx = 0.7 * old_vx + 0.3 * vx
        smoothed_vy = 0.7 * old_vy + 0.3 * vy

        track["box"] = det_box
        track["center"] = new_center
        track["velocity"] = (smoothed_vx, smoothed_vy)
        track["last_ts"] = now_ts
        track["hits"] += 1
        track["age"] += 1
        track["missing_count"] = 0

    def update(self, detections, fps: float = 15.0):
        now = time.time()
        person_detections = [d for d in detections if d.get("class_name") == "person"]

        for track in self.tracks.values():
            track["age"] += 1

        candidate_pairs = []

        for tid, track in self.tracks.items():
            for det_idx, det in enumerate(person_detections):
                det_box = det["box"]
                cost = self._compute_match_cost(track, det_box, now)

                if cost is None:
                    continue

                if cost <= self.max_match_cost:
                    candidate_pairs.append((cost, tid, det_idx))

        candidate_pairs.sort(key=lambda x: x[0])

        assigned_tracks = set()
        assigned_dets = set()

        for cost, tid, det_idx in candidate_pairs:
            if tid in assigned_tracks or det_idx in assigned_dets:
                continue

            det_box = person_detections[det_idx]["box"]
            self._update_track(tid, det_box, now)

            assigned_tracks.add(tid)
            assigned_dets.add(det_idx)

        for det_idx, det in enumerate(person_detections):
            if det_idx in assigned_dets:
                continue

            tid = self._create_track(det["box"], now)
            assigned_tracks.add(tid)

        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in assigned_tracks:
                track["missing_count"] += 1

            if (now - track["last_ts"]) > self.max_missing_seconds:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        results = []
        for tid, track in self.tracks.items():
            results.append({
                "person_id": tid,
                "box": track["box"],
            })

        return results