from __future__ import annotations

import json
from typing import Any, Dict, List


def build_saturation_prompt(jhat_trajectory: List[float], patience: int, min_delta: float, trial_index: int) -> list:
    system = {
        "role": "system",
        "content": (
            "You are a tuning monitor for a distribution-level ratio objective R_hat.\n"
            "Given the trajectory of R_hat across trials, estimate whether additional trials\n"
            "are likely to reduce R_hat by at least min_delta over the next 'patience' trials.\n"
            "Return strict JSON: {continue: bool, expected_improvement: float, reason: string}."
        ),
    }
    user_payload: Dict[str, Any] = {
        "task": "saturation_check",
        "trial_index": int(trial_index),
        "jhat_trajectory": [float(x) for x in jhat_trajectory],
        "min_delta": float(min_delta),
        "patience": int(patience),
        "instructions": "If expected_improvement < min_delta, set continue=false; otherwise true.",
    }
    user = {"role": "user", "content": json.dumps(user_payload)}
    return [system, user]
