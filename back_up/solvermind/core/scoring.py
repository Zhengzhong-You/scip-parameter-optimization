from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple


# Canonical scoring constants and helpers (single source of truth)
EPS = 1e-9
CAP_LIMIT = 1000  # Matches README caps at 1e3


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


def _f1(x: float) -> float:
    return min(CAP_LIMIT * float(x), CAP_LIMIT)


def _f2(x: float) -> float:
    return min(float(x), CAP_LIMIT)


def _f3(x: float) -> float:
    return min(math.exp(float(x)) - 1.0, CAP_LIMIT)


def _f4(x: float) -> float:
    return min(1e-6 * float(x), CAP_LIMIT)


def sigma_for_instance(metrics: Dict[str, Any], tau: float, n_max: float) -> float:
    status = metrics.get("status", "")
    t = float(metrics.get("solve_time") or metrics.get("time") or 0.0)
    primal = metrics.get("primal")
    dual = metrics.get("dual")
    nodes = metrics.get("n_nodes", metrics.get("nodes"))
    try:
        nodes_val = float(nodes) if nodes is not None else 0.0
    except Exception:
        nodes_val = 0.0

    tier = status_to_tier(status)
    tf = (t / float(max(tau, EPS))) if tau else 0.0
    gap_val = metrics.get("gap")
    if gap_val is None:
        gap_val = gap_from_bounds(primal, dual)
    nrm_nodes = nodes_val / float(max(n_max, EPS))

    val = _f1(float(tier)) + _f2(tf) + _f3(float(gap_val)) + _f4(nrm_nodes)
    return max(val, EPS)


def geomean(values: Iterable[float]) -> float:
    vals = [max(float(v), EPS) for v in values]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def j_hat_sigma(per_instance_metrics: Dict[str, Dict[str, Any]], tau: float) -> Tuple[float, Dict[str, float]]:
    nodes = []
    for m in per_instance_metrics.values():
        try:
            v = m.get("n_nodes", m.get("nodes"))
            nodes.append(float(v))
        except Exception:
            pass
    n_max = max([n for n in nodes if math.isfinite(n)], default=1.0)
    if n_max <= 0:
        n_max = 1.0

    sigmas: Dict[str, float] = {}
    for key, m in per_instance_metrics.items():
        sigmas[key] = sigma_for_instance(m, tau=tau, n_max=n_max)

    return geomean(sigmas.values()), sigmas


def gm(values: Iterable[float]) -> float:
    vals = [max(float(v), EPS) for v in values]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# Configurable sigma (for baselines that define f1..f4 in YAML)
def _cap_linear(x: float, a: float, cap: float) -> float:
    return min(a * max(0.0, float(x)), float(cap))


def _cap_exp(x: float, base: float, cap: float) -> float:
    return min((float(base) ** max(0.0, float(x))) - 1.0, float(cap))


def _make_f(cfg: Dict[str, Any]):
    kind = str(cfg.get("kind"))
    if kind == "cap_linear":
        a = float(cfg.get("a", 1.0)); cap = float(cfg.get("cap", 1e3))
        return lambda x: _cap_linear(x, a=a, cap=cap)
    if kind == "cap_exp":
        base = float(cfg.get("base", math.e)); cap = float(cfg.get("cap", 1e3))
        return lambda x: _cap_exp(x, base=base, cap=cap)
    raise ValueError(f"Unknown f kind: {kind}")


def sigma_config(score_parts: Dict[str, float], f_cfg: Dict[str, Dict[str, Any]]) -> float:
    f1 = _make_f(f_cfg["f1"])  # type: ignore[index]
    f2 = _make_f(f_cfg["f2"])  # type: ignore[index]
    f3 = _make_f(f_cfg["f3"])  # type: ignore[index]
    f4 = _make_f(f_cfg["f4"])  # type: ignore[index]
    val = f1(score_parts.get("tier", 1.0)) + \
          f2(score_parts.get("t_over_tau", 1.0)) + \
          f3(score_parts.get("gap", 1.0)) + \
          f4(score_parts.get("nodes_over_Nmax", 0.0))
    return float(max(1e-12, val))
