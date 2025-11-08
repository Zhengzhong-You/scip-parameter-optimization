from __future__ import annotations

import os
from typing import Dict, Any, List

from ..utils.scip_cli import extract_basic_features


def instance_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def collect_batch_features(instances: List[str], dry_run: bool = False) -> Dict[str, Any]:
    """Collect minimal, robust features via SCIP CLI for each instance.

    When dry_run=True, returns empty dicts per instance to avoid CLI calls.
    """
    out: Dict[str, Any] = {}
    for p in instances:
        name = instance_name(p)
        if dry_run:
            out[name] = {}
            continue
        try:
            out[name] = extract_basic_features(p, presolve=True)
        except Exception:
            out[name] = {"error": "feature_extraction_failed"}
    return out
