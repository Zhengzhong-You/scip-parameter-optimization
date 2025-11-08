from __future__ import annotations

from typing import List, Dict, Any
from .pipeline import run_tuning as _run_tuning


def tune_batch(
    instances: List[str],
    time_limit: float = 600.0,
    max_trials: int = 16,
    max_edits: int = 3,
    outdir: str = "runs",
    gpt_model: str = "gpt-5",
    seed: int = 0,
    early_stop_patience: int = 0,
    early_stop_delta: float = 0.0,
    whitelist_regime: str = "curated",
) -> Dict[str, Any]:
    return _run_tuning(
        instances=instances,
        time_limit=time_limit,
        max_trials=max_trials,
        max_edits=max_edits,
        outdir=outdir,
        gpt_model=gpt_model,
        seed=seed,
        early_stop_patience=early_stop_patience,
        early_stop_delta=early_stop_delta,
        whitelist_regime=whitelist_regime,
    )

