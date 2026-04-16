from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class EventSerializerMySQL:
    """
    Genera dos salidas:
    - events_mysql_ready.json
    - mysql_events.jsonl

    El objetivo es dejar una estructura simple de consumir para
    persistencia posterior en MySQL.
    """

    def __init__(
        self,
        base_path: str = "evidence/json",
        json_filename: str = "events_mysql_ready.json",
        jsonl_filename: str = "mysql_events.jsonl",
        duplicate_window_seconds: int = 10,
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.json_file = self.base_path / json_filename
        self.jsonl_file = self.base_path / jsonl_filename
        self.duplicate_window_seconds = duplicate_window_seconds

    def _load_json(self) -> dict:
        if not self.json_file.exists():
            return {
                "generated_at": None,
                "total_events": 0,
                "events": [],
            }

        try:
            content = self.json_file.read_text(encoding="utf-8").strip()
            if not content:
                raise ValueError("Archivo JSON vacío.")
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("Formato JSON inválido.")
            data.setdefault("generated_at", None)
            data.setdefault("total_events", 0)
            data.setdefault("events", [])
            return data
        except Exception:
            return {
                "generated_at": None,
                "total_events": 0,
                "events": [],
            }

    def _save_json(self, data: dict) -> None:
        self.json_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _append_jsonl(self, event: dict) -> None:
        with self.jsonl_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_ts(ts: str) -> datetime | None:
        try:
            return datetime.fromisoformat(ts.replace("Z", ""))
        except Exception:
            return None

    def _event_signature(self, event: dict) -> tuple:
        return (
            event.get("camera_id"),
            event.get("person_track_id"),
            event.get("event_type"),
            tuple(sorted(event.get("violation_codes", []))),
        )

    def _is_duplicate(self, existing_events: list[dict], new_event: dict) -> bool:
        new_sig = self._event_signature(new_event)
        new_ts = self._parse_ts(new_event["event_confirmed_at"])
        if new_ts is None:
            return False

        for item in existing_events:
            if self._event_signature(item) != new_sig:
                continue

            item_ts = self._parse_ts(item.get("event_confirmed_at", ""))
            if item_ts is None:
                continue

            diff = abs((new_ts - item_ts).total_seconds())
            if diff <= self.duplicate_window_seconds:
                return True

        return False

    def build_event(
        self,
        camera_id: str,
        scenario_id: str,
        zone_name: str | None,
        person_track_id: str,
        frame_number: int,
        status: str,
        event_type: str | None,
        violation_codes: list[str],
        person_eval: dict,
        observed_status: str,
        state_snapshot: dict,
        evidence_paths: dict | None = None,
        model_version: str | None = None,
    ) -> dict:
        now = datetime.utcnow().isoformat() + "Z"

        helmet = person_eval.get("helmet")
        vest = person_eval.get("vest")

        event = {
            "event_id": f"{camera_id}_{person_track_id}_{frame_number}_{int(datetime.utcnow().timestamp())}",
            "camera_id": camera_id,
            "scenario_id": scenario_id,
            "zone_name": zone_name,
            "person_track_id": str(person_track_id),
            "frame_number": int(frame_number),
            "event_type": event_type,
            "status": status,
            "observed_status": observed_status,
            "violation_codes": violation_codes[:],
            "event_observed_at": now,
            "event_confirmed_at": now,
            "model_version": model_version,
            "person_box": person_eval.get("person_box"),
            "head_box": person_eval.get("head_box"),
            "torso_box": person_eval.get("torso_box"),
            "helmet_box": helmet.get("box") if helmet else None,
            "vest_box": vest.get("box") if vest else None,
            "helmet_score": float(person_eval.get("helmet_score", 0.0)),
            "vest_score": float(person_eval.get("vest_score", 0.0)),
            "helmet_ok": bool(person_eval.get("helmet_ok", False)),
            "vest_ok": bool(person_eval.get("vest_ok", False)),
            "confirmed_status_snapshot": {
                "confirmed_status": state_snapshot.get("confirmed_status"),
                "confirmed_violations": state_snapshot.get("confirmed_violations", []),
                "frames_seen": state_snapshot.get("frames_seen", 0),
            },
            "evidence": evidence_paths or {},
        }

        data = self._load_json()
        if not self._is_duplicate(data["events"], event):
            data["events"].append(event)
            data["total_events"] = len(data["events"])
            data["generated_at"] = now
            self._save_json(data)
            self._append_jsonl(event)

        return event