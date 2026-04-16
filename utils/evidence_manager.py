from __future__ import annotations

from pathlib import Path
import cv2


class EvidenceManager:
    def __init__(self, base_path: str = "evidence") -> None:
        self.base_path = Path(base_path)
        self.images_full_path = self.base_path / "images_full"
        self.images_annotated_path = self.base_path / "images_annotated"
        self.images_crop_path = self.base_path / "images_crop"
        self.videos_path = self.base_path / "videos"

        for path in [
            self.images_full_path,
            self.images_annotated_path,
            self.images_crop_path,
            self.videos_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_crop(frame, box: list[int] | None):
        if frame is None or box is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box

        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def save_full_image(self, frame, event_id: str) -> str:
        image_path = self.images_full_path / f"{event_id}.jpg"
        cv2.imwrite(str(image_path), frame)
        return str(image_path)

    def save_annotated_image(self, frame, event_id: str) -> str:
        image_path = self.images_annotated_path / f"{event_id}.jpg"
        cv2.imwrite(str(image_path), frame)
        return str(image_path)

    def save_person_crop(self, frame, person_box: list[int] | None, event_id: str) -> str | None:
        crop = self._safe_crop(frame, person_box)
        if crop is None:
            return None

        image_path = self.images_crop_path / f"{event_id}.jpg"
        cv2.imwrite(str(image_path), crop)
        return str(image_path)

    def save_video_clip(
        self,
        frames,
        fps: float,
        width: int,
        height: int,
        event_id: str,
    ) -> str | None:
        if not frames:
            return None

        video_path = self.videos_path / f"{event_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)

        writer.release()
        return str(video_path)