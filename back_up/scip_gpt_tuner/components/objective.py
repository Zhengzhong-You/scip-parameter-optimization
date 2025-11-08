from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple

from solvermind.core.scoring import (
    sigma_for_instance,
    j_hat_sigma,
    geomean as _geomean,
    gm as _gm,
)


def compute_sigma(metrics: Dict[str, Any], tau: float, n_max: float) -> float:
    return sigma_for_instance(metrics, tau=tau, n_max=n_max)


def compute_j_hat(per_instance_metrics: Dict[str, Dict[str, Any]], tau: float) -> Tuple[float, Dict[str, float]]:
    return j_hat_sigma(per_instance_metrics, tau=tau)


def geomean(values: Iterable[float]) -> float:
    return _geomean(values)


def gm(values: Iterable[float]) -> float:
    return _gm(values)
