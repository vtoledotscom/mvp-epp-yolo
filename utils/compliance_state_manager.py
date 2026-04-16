from __future__ import annotations


class ComplianceStateManager:
    """
    Maneja histéresis temporal por persona.

    Conceptos:
    - observed_status: estado observado en el frame actual
    - confirmed_status: estado confirmado después de N frames consecutivos
    - candidate_status: posible nuevo estado en evaluación

    Reglas:
    - Un cambio a non_compliant solo se confirma si se observa
      min_non_compliant_frames veces consecutivas.
    - Un cambio a compliant solo se confirma si se observa
      min_compliant_frames veces consecutivas.
    - Cuando se confirma el cambio:
        * compliant -> non_compliant => violation_started
        * non_compliant -> compliant => violation_resolved
    """

    def __init__(
        self,
        default_min_non_compliant_frames: int = 5,
        default_min_compliant_frames: int = 3,
    ) -> None:
        self.default_min_non_compliant_frames = max(1, int(default_min_non_compliant_frames))
        self.default_min_compliant_frames = max(1, int(default_min_compliant_frames))
        self.states: dict[str, dict] = {}

    def _build_default_state(self, observed_status: str, violations: list[str]) -> dict:
        return {
            "confirmed_status": observed_status,
            "confirmed_violations": violations[:],
            "candidate_status": None,
            "candidate_violations": [],
            "candidate_count": 0,
            "frames_seen": 1,
            "last_frame_number": None,
        }

    def get_confirmed_state(self, person_id: str) -> dict | None:
        return self.states.get(person_id)

    def update(
        self,
        person_id: str,
        observed_status: str,
        observed_violations: list[str],
        frame_number: int,
        min_non_compliant_frames: int | None = None,
        min_compliant_frames: int | None = None,
    ) -> tuple[bool, str | None, dict]:
        """
        Retorna:
        - state_changed: bool
        - event_type: violation_started | violation_resolved | None
        - state_snapshot: dict
        """
        if observed_status not in {"compliant", "non_compliant"}:
            raise ValueError(f"Estado observado inválido: {observed_status}")

        min_non_compliant = max(
            1,
            int(min_non_compliant_frames or self.default_min_non_compliant_frames),
        )
        min_compliant = max(
            1,
            int(min_compliant_frames or self.default_min_compliant_frames),
        )

        if person_id not in self.states:
            self.states[person_id] = self._build_default_state(
                observed_status=observed_status,
                violations=observed_violations,
            )
            self.states[person_id]["last_frame_number"] = frame_number

            event_type = "violation_started" if observed_status == "non_compliant" else None
            return (event_type is not None), event_type, self.states[person_id].copy()

        state = self.states[person_id]
        state["frames_seen"] += 1
        state["last_frame_number"] = frame_number

        confirmed_status = state["confirmed_status"]

        if observed_status == confirmed_status:
            state["confirmed_violations"] = observed_violations[:]
            state["candidate_status"] = None
            state["candidate_violations"] = []
            state["candidate_count"] = 0
            return False, None, state.copy()

        # Hay un estado candidato distinto al confirmado
        if state["candidate_status"] == observed_status:
            state["candidate_count"] += 1
            state["candidate_violations"] = observed_violations[:]
        else:
            state["candidate_status"] = observed_status
            state["candidate_violations"] = observed_violations[:]
            state["candidate_count"] = 1

        required_frames = (
            min_non_compliant
            if observed_status == "non_compliant"
            else min_compliant
        )

        if state["candidate_count"] < required_frames:
            return False, None, state.copy()

        previous_confirmed = state["confirmed_status"]
        state["confirmed_status"] = observed_status
        state["confirmed_violations"] = observed_violations[:]
        state["candidate_status"] = None
        state["candidate_violations"] = []
        state["candidate_count"] = 0

        event_type = None
        if previous_confirmed == "compliant" and observed_status == "non_compliant":
            event_type = "violation_started"
        elif previous_confirmed == "non_compliant" and observed_status == "compliant":
            event_type = "violation_resolved"

        return True, event_type, state.copy()

    def mark_missing(self, person_id: str) -> None:
        """
        Extensible para manejo futuro de personas desaparecidas.
        Por ahora no resuelve automáticamente el estado al perderse el track.
        """
        if person_id not in self.states:
            return

    def cleanup_absent(
        self,
        active_person_ids: set[str],
        max_absent_frames: int = 60,
    ) -> None:
        """
        Limpia estados temporales antiguos de personas que ya no están.
        """
        to_delete = []
        for person_id, state in self.states.items():
            if person_id in active_person_ids:
                continue

            last_frame = state.get("last_frame_number")
            if last_frame is None:
                to_delete.append(person_id)

        for person_id in to_delete:
            del self.states[person_id]