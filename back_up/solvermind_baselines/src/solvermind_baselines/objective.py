import math
from typing import Dict, Any, List, Tuple
from .scoring import sigma as sigma_config, status_to_tier


def compute_gap(z_pr, z_du, eps: float = 1e-9) -> float:
    if z_pr is None or z_du is None:
        return float("inf")
    if (z_pr >= 0 and z_du <= 0) or (z_pr <= 0 and z_du >= 0):
        return float("inf")
    denom = max(eps, min(abs(z_pr), abs(z_du)))
    return abs(z_pr - z_du) / denom


def aggregate_sigma_for_instance(run_out: Dict[str, Any], tau: float, N_max: float, f_cfg: Dict[str, Any]) -> float:
    t = float(run_out.get("time", tau))
    s = str(run_out.get("status", "other"))
    z_pr = run_out.get("primal", None)
    z_du = run_out.get("dual", None)
    nodes = float(run_out.get("nodes", 0.0))

    tier = status_to_tier(s)
    gap = compute_gap(z_pr, z_du)
    parts = {
        "tier": tier,
        "t_over_tau": min(1.0, t / max(1e-9, tau)),
        "gap": gap if math.isfinite(gap) else 1e9,
        "nodes_over_Nmax": min(100.0, nodes / max(1.0, N_max)),
    }
    return sigma_config(parts, f_cfg)


def evaluate_config_on_set(
    cfg: Dict[str, Any],
    instances: List[str],
    runner_fn,
    tau: float,
    f_cfg: Dict[str, Any],
    eps: float,
    N_max: float,
    extra_runner_kwargs: Dict[str, Any] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    extra_runner_kwargs = extra_runner_kwargs or {}
    sigmas = []
    per_logs = []
    for inst in instances:
        out = runner_fn(cfg, inst, tau, **extra_runner_kwargs)
        per_logs.append({"instance": inst, **out})
        s = aggregate_sigma_for_instance(out, tau=tau, N_max=N_max, f_cfg=f_cfg)
        sigmas.append(s + eps)
    L = float(sum(math.log(x) for x in sigmas) / max(1, len(sigmas)))
    return L, per_logs
