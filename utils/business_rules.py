import json
from pathlib import Path


class RulesEngine:
    """
    Carga escenarios desde JSON y evalúa el cumplimiento EPP por persona.
    """

    def __init__(self, scenarios_path: str) -> None:
        path = Path(scenarios_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.scenarios = data["scenarios"]

    def get_rule(self, scenario_id: str) -> dict:
        if scenario_id not in self.scenarios:
            raise KeyError(f"Escenario no definido: {scenario_id}")
        return self.scenarios[scenario_id]

    def evaluate(self, scenario_id: str, person_eval: dict) -> tuple[str, list[str], dict]:
        """
        Retorna:
        - status
        - violations
        - rule
        """
        rule = self.get_rule(scenario_id)

        helmet_ok = bool(person_eval.get("helmet_ok", False))
        vest_ok = bool(person_eval.get("vest_ok", False))

        violations = []

        if rule.get("helmet_required", False) and not helmet_ok:
            violations.append("missing_helmet")

        if rule.get("vest_required", False) and not vest_ok:
            violations.append("missing_vest")

        status = "compliant" if not violations else "non_compliant"
        return status, violations, rule