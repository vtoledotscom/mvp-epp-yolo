import json
from datetime import datetime
from pathlib import Path


class EventSerializer:
    """
    Mantiene un único archivo JSON acumulando eventos agrupados por persona.

    Corrección importante:
    - Ya no agrupa solo por person_id.
    - Ahora usa una clave compuesta por camera_id + person_id.
    """

    def __init__(
        self,
        base_path: str = "evidence/json",
        filename: str = "events.json",
        duplicate_window_seconds: int = 10,
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.events_file = self.base_path / filename
        self.duplicate_window_seconds = duplicate_window_seconds

    def _load_document(self) -> dict:
        if not self.events_file.exists():
            return {
                "generated_at": None,
                "total_events": 0,
                "persons": {}
            }

        try:
            content = self.events_file.read_text(encoding="utf-8").strip()

            if not content:
                return {
                    "generated_at": None,
                    "total_events": 0,
                    "persons": {}
                }

            data = json.loads(content)

            if not isinstance(data, dict):
                raise ValueError("El documento principal no es un objeto JSON.")

            if "persons" not in data or not isinstance(data["persons"], dict):
                data["persons"] = {}

            if "total_events" not in data:
                data["total_events"] = 0

            if "generated_at" not in data:
                data["generated_at"] = None

            return data

        except Exception:
            return {
                "generated_at": None,
                "total_events": 0,
                "persons": {}
            }

    def _save_document(self, document: dict) -> None:
        self.events_file.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _parse_timestamp(self, timestamp_str: str) -> datetime | None:
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", ""))
        except Exception:
            return None

    def _build_event_key(self, event: dict) -> str:
        camera_id = event.get("camera_id", "")
        person_id = event.get("person_id", "")
        event_type = event.get("event_type", "")
        violations = "_".join(sorted(event.get("violations", [])))
        return f"{camera_id}_{person_id}_{event_type}_{violations}"

    def _is_duplicate(self, person_bucket: dict, event: dict) -> bool:
        event_key = self._build_event_key(event)
        event_time = self._parse_timestamp(event["timestamp"])

        if event_time is None:
            return False

        for existing in person_bucket.get("events", []):
            existing_key = self._build_event_key(existing)

            if existing_key != event_key:
                continue

            existing_time = self._parse_timestamp(existing.get("timestamp", ""))
            if existing_time is None:
                continue

            diff_seconds = abs((event_time - existing_time).total_seconds())

            if diff_seconds <= self.duplicate_window_seconds:
                return True

        return False

    def _build_person_bucket_key(self, camera_id: str, person_id: str | int) -> str:
        return f"{camera_id}::{person_id}"

    def _get_or_create_person_bucket(
        self,
        document: dict,
        person_id: str | int,
        camera_id: str,
        scenario_id: str,
    ) -> dict:
        person_key = self._build_person_bucket_key(camera_id, person_id)

        if person_key not in document["persons"]:
            document["persons"][person_key] = {
                "bucket_id": person_key,
                "person_id": str(person_id),
                "camera_id": camera_id,
                "scenario_id": scenario_id,
                "total_events": 0,
                "events": []
            }

        return document["persons"][person_key]

    def build_event(
        self,
        camera_id: str,
        scenario_id: str,
        person_id: str | int,
        status: str,
        violations: list[str],
        image_path: str | None,
        video_path: str | None,
        event_type: str | None = None,
    ) -> dict:
        now = datetime.utcnow()

        event = {
            "event_id": f"{camera_id}_{person_id}_{int(now.timestamp())}",
            "timestamp": now.isoformat() + "Z",
            "camera_id": camera_id,
            "scenario_id": scenario_id,
            "person_id": str(person_id),
            "status": status,
            "event_type": event_type,
            "violations": violations,
            "image_path": image_path,
            "video_path": video_path,
        }

        document = self._load_document()
        person_bucket = self._get_or_create_person_bucket(
            document=document,
            person_id=person_id,
            camera_id=camera_id,
            scenario_id=scenario_id,
        )

        if self._is_duplicate(person_bucket, event):
            return event

        person_bucket["events"].append(event)
        person_bucket["total_events"] = len(person_bucket["events"])

        total = 0
        for _, bucket in document["persons"].items():
            total += len(bucket.get("events", []))

        document["total_events"] = total
        document["generated_at"] = now.isoformat() + "Z"

        self._save_document(document)
        return event