
from typing import Dict, Any, Tuple

def judge_score(trial: Dict[str, Any]) -> Tuple:
    """Compute an *orderable* score tuple according to the user's criterion.
    Higher is better when sorted with Python's default tuple ordering, so we
    structure it as (tier, primary, secondary):
      - Tier 2: solved optimal within time limit (prefer *less* time)
      - Tier 1: infeasible/unbounded proven within time limit (prefer *less* time)
      - Tier 0: hit time limit (prefer *better* primal, then *better* dual)
    For minimization, smaller primal/dual are better; for maximization, larger are better.
    """
    t = float(trial.get("solve_time", 0.0))
    T = float(trial.get("time_limit", 0.0))
    status = (trial.get("status") or "").lower()
    primal = trial.get("primal")
    dual = trial.get("dual")
    sense = (trial.get("obj_sense") or "minimize").lower()

    # Normalize comparison direction
    def _score_value(val):
        if val is None: 
            return float("-inf") if sense == "maximize" else float("inf")
        return float(val)

    if t >= T - 1e-2:
        # Tier 0: time limit reached
        primary = -_score_value(primal) if sense == "maximize" else -(-_score_value(primal))  # noop, keep explicit
        # For tuple ordering: higher is better -> we invert by multiplying by -1 for minimization
        if sense == "minimize":
            primary = -_score_value(primal)
            secondary = -_score_value(dual)
        else:
            primary = _score_value(primal)
            secondary = _score_value(dual)
        return (0, primary, secondary)
    else:
        # Solved or infeasible within time
        if "optimal" in status:
            # Tier 2: prefer shorter time
            return (2, -t, 0.0)
        else:
            # Tier 1: infeasible/unbounded/other, prefer shorter time
            return (1, -t, 0.0)
