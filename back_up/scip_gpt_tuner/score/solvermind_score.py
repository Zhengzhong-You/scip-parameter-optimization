from __future__ import annotations

# Thin wrapper to keep backward-compatible import paths while using
# the canonical scoring implementation in solvermind.core.scoring
from typing import Dict, Any, Iterable, Tuple

from solvermind.core.scoring import (  # noqa: F401
    EPS,
    CAP_LIMIT,
    sigma_for_instance,
    j_hat_sigma,
    geomean,
    gm,
    gap_from_bounds as _gap_from_bounds,
)
