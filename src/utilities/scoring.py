from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple

EPS = 1e-9
CAP_LIMIT = 1000


def status_to_tier(s: str) -> int:
    s = (s or "").strip().lower()
    if any(x in s for x in ("optimal", "infeasible", "unbounded", "opt", "inf", "unb")):
        return 0
    return 1


def gap_from_bounds(primal: Any, dual: Any) -> float:
    try:
        p = float(primal)
        d = float(dual)
        if math.isfinite(p) and math.isfinite(d):
            return abs(p - d) / (1.0 + abs(p))
    except Exception:
        pass
    return 1.0


def sigma_for_instance(metrics: Dict[str, Any], tau: float, n_max: float) -> float:
    """Placeholder: scoring is to be defined by the new method.

    This function intentionally raises until the new scoring definition is provided.
    """
    raise NotImplementedError("sigma_for_instance is undefined. Please provide the new scoring definition.")


def geomean(values: Iterable[float]) -> float:
    vals = [max(float(v), EPS) for v in values]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def j_hat_sigma(per_instance_metrics: Dict[str, Dict[str, Any]], tau: float) -> Tuple[float, Dict[str, float]]:
    sigmas: Dict[str, float] = {}
    # This computes J-hat from per-instance sigmas once sigma_for_instance is defined
    for key, m in per_instance_metrics.items():
        sigmas[key] = sigma_for_instance(m, tau=tau, n_max=1.0)
    return geomean(sigmas.values()), sigmas


def gm(values: Iterable[float]) -> float:
    vals = [max(float(v), EPS) for v in values]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def sigma_config(score_parts: Dict[str, float], f_cfg: Dict[str, Dict[str, Any]]) -> float:
    """Placeholder for baseline-configurable scoring.

    Intentionally unimplemented until the new scoring method is provided.
    """
    raise NotImplementedError("sigma_config is undefined. Please provide the new scoring definition.")


def r_hat_ratio(tinf_p: Dict[str, float], tinf_q: Dict[str, float], cap: float = 1e3) -> float:
    """Compute R_hat(p,q) = exp(mean_i log(min(Tp_i/Tq_i, cap)))."""
    ratios: list[float] = []
    for k, tp in tinf_p.items():
        tq = tinf_q.get(k)
        if tq is None or not (math.isfinite(tp) and math.isfinite(tq)) or tq <= 0:
            continue
        ratios.append(min(tp / max(tq, EPS), float(cap)))
    if not ratios:
        return float('inf')
    return geomean(ratios)
