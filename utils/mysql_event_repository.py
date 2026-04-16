from __future__ import annotations

import json
from typing import Any

import mysql.connector
from mysql.connector import Error


class MySQLEventRepository:
    """
    Repositorio para persistir eventos EPP confirmados en MariaDB/MySQL.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        connect_timeout: int = 5,
    ) -> None:
        self.config = {
            "host": host,
            "port": int(port),
            "user": user,
            "password": password,
            "database": database,
            "autocommit": False,
            "connection_timeout": int(connect_timeout),
        }

    def _connect(self):
        return mysql.connector.connect(**self.config)

    @staticmethod
    def _to_json(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _normalize_ts(value: str | None) -> str | None:
        if not value:
            return None
        return value.replace("Z", "")

    def event_exists(self, event_id: str) -> bool:
        conn = None
        cursor = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM epp_events WHERE event_id = %s LIMIT 1",
                (event_id,),
            )
            return cursor.fetchone() is not None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def save_event(self, event: dict) -> bool:
        conn = None
        cursor = None

        try:
            conn = self._connect()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM epp_events WHERE event_id = %s LIMIT 1",
                (event["event_id"],),
            )
            if cursor.fetchone() is not None:
                conn.rollback()
                return False

            insert_event_sql = """
                INSERT INTO epp_events (
                    event_id,
                    camera_id,
                    scenario_id,
                    zone_name,
                    person_track_id,
                    event_type,
                    status,
                    observed_status,
                    frame_number,
                    event_observed_at,
                    event_confirmed_at,
                    helmet_ok,
                    vest_ok,
                    helmet_score,
                    vest_score,
                    person_box_json,
                    head_box_json,
                    torso_box_json,
                    helmet_box_json,
                    vest_box_json,
                    violation_codes_json,
                    confirmed_status_snapshot_json,
                    model_version
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
            """

            cursor.execute(
                insert_event_sql,
                (
                    event["event_id"],
                    event["camera_id"],
                    event["scenario_id"],
                    event.get("zone_name"),
                    event["person_track_id"],
                    event["event_type"],
                    event["status"],
                    event["observed_status"],
                    int(event["frame_number"]),
                    self._normalize_ts(event.get("event_observed_at")),
                    self._normalize_ts(event.get("event_confirmed_at")),
                    int(bool(event.get("helmet_ok", False))),
                    int(bool(event.get("vest_ok", False))),
                    float(event["helmet_score"]) if event.get("helmet_score") is not None else None,
                    float(event["vest_score"]) if event.get("vest_score") is not None else None,
                    self._to_json(event.get("person_box")),
                    self._to_json(event.get("head_box")),
                    self._to_json(event.get("torso_box")),
                    self._to_json(event.get("helmet_box")),
                    self._to_json(event.get("vest_box")),
                    self._to_json(event.get("violation_codes", [])),
                    self._to_json(event.get("confirmed_status_snapshot", {})),
                    event.get("model_version"),
                ),
            )

            evidence = event.get("evidence", {}) or {}

            insert_evidence_sql = """
                INSERT INTO epp_event_evidence (
                    event_id,
                    image_full_path,
                    image_annotated_path,
                    image_crop_path,
                    video_path
                ) VALUES (%s, %s, %s, %s, %s)
            """

            cursor.execute(
                insert_evidence_sql,
                (
                    event["event_id"],
                    evidence.get("image_full_path"),
                    evidence.get("image_annotated_path"),
                    evidence.get("image_crop_path"),
                    evidence.get("video_path"),
                ),
            )

            conn.commit()
            return True

        except Error:
            if conn:
                conn.rollback()
            raise

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def close_open_violation(
        self,
        camera_id: str,
        person_track_id: str,
        scenario_id: str,
        resolved_event_id: str,
        resolved_at: str,
    ) -> int:
        conn = None
        cursor = None

        try:
            conn = self._connect()
            cursor = conn.cursor()

            sql = """
                UPDATE epp_events
                SET resolved_by_event_id = %s,
                    resolved_at = %s
                WHERE camera_id = %s
                  AND person_track_id = %s
                  AND scenario_id = %s
                  AND event_type = 'violation_started'
                  AND resolved_by_event_id IS NULL
                ORDER BY event_confirmed_at DESC
                LIMIT 1
            """

            cursor.execute(
                sql,
                (
                    resolved_event_id,
                    self._normalize_ts(resolved_at),
                    camera_id,
                    person_track_id,
                    scenario_id,
                ),
            )
            affected = cursor.rowcount
            conn.commit()
            return affected

        except Error:
            if conn:
                conn.rollback()
            raise

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()