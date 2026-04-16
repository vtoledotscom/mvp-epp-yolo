import time


class AlertManager:
    """
    Gestiona transiciones de estado por persona para evitar alertas repetidas.

    Estados manejados:
    - OK
    - NO_OK

    Reglas:
    - Si la persona aparece por primera vez en NO_OK => emite violation_started
    - Si cambia de OK a NO_OK => emite violation_started
    - Si cambia de NO_OK a OK => emite violation_resolved
    - Si permanece en el mismo estado => no emite evento nuevo
    """

    def __init__(self, default_cooldown: int = 15) -> None:
        self.default_cooldown = default_cooldown
        self.last_alert = {}
        self.person_states = {}
        # {
        #   person_id: {
        #       "last_status": "OK" | "NO_OK",
        #       "last_violations": [...],
        #       "last_change_ts": float
        #   }
        # }

    def _normalize_status(self, status: str) -> str:
        return "NO_OK" if status == "non_compliant" else "OK"

    def _build_key(self, person_id: int, state: str, violations: list[str]) -> tuple:
        return (person_id, state, tuple(sorted(violations)))

    def should_alert(
        self,
        person_id: int,
        status: str,
        violations: list[str],
        cooldown: int | None = None,
    ) -> tuple[bool, str | None]:
        """
        Retorna:
        - should_emit
        - event_type:
            - violation_started
            - violation_resolved
            - None
        """
        now = time.time()
        current_state = self._normalize_status(status)

        previous = self.person_states.get(person_id)
        previous_state = previous["last_status"] if previous else None

        # Primera vez que vemos a la persona
        if previous_state is None:
            self.person_states[person_id] = {
                "last_status": current_state,
                "last_violations": violations[:],
                "last_change_ts": now,
            }

            # Si aparece ya incumpliendo, se debe alertar.
            if current_state == "NO_OK":
                key = self._build_key(person_id, current_state, violations)
                self.last_alert[key] = now
                return True, "violation_started"

            return False, None

        # Si no hubo cambio de estado, no se emite un nuevo evento.
        if previous_state == current_state:
            self.person_states[person_id]["last_violations"] = violations[:]
            return False, None

        event_type = None

        if previous_state == "OK" and current_state == "NO_OK":
            event_type = "violation_started"
        elif previous_state == "NO_OK" and current_state == "OK":
            event_type = "violation_resolved"

        self.person_states[person_id] = {
            "last_status": current_state,
            "last_violations": violations[:],
            "last_change_ts": now,
        }

        if event_type is None:
            return False, None

        wait_seconds = cooldown if cooldown is not None else self.default_cooldown
        key = self._build_key(person_id, current_state, violations)
        last_ts = self.last_alert.get(key, 0.0)

        if now - last_ts < wait_seconds:
            return False, None

        self.last_alert[key] = now
        return True, event_type