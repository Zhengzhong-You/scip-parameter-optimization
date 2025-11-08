from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
from ..run.run_instance_cli import run_instance_cli
from .features_batch import instance_name


def run_batch_cli(instances: List[str], settings_file: str, time_limit: float, outdir: str, seed: int, trial_id: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    per_m: Dict[str, Dict[str, Any]] = {}
    for p in instances:
        name = instance_name(p)
        # Place logs in a dedicated subfolder per instance when outdir is a batch root
        outdir_i = outdir if os.path.basename(os.path.normpath(outdir)) == name else os.path.join(outdir, name)
        m = run_instance_cli(p, settings_file=settings_file, meta_applied={}, time_limit=time_limit, outdir=outdir_i, seed=seed, trial_id=trial_id)
        per_m[name] = m
    return per_m
