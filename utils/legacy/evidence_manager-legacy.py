from pathlib import Path
import cv2


class EvidenceManager:
    def __init__(self, base_path: str = "evidence") -> None:
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.videos_path = self.base_path / "videos"

        self.images_path.mkdir(parents=True, exist_ok=True)
        self.videos_path.mkdir(parents=True, exist_ok=True)

    def save_image(self, frame, event_id: str) -> str:
        image_path = self.images_path / f"{event_id}.jpg"
        cv2.imwrite(str(image_path), frame)
        return str(image_path)

    def save_video_clip(self, frames, fps: float, width: int, height: int, event_id: str) -> str | None:
        if not frames:
            return None

        video_path = self.videos_path / f"{event_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)

        writer.release()
        return str(video_path)