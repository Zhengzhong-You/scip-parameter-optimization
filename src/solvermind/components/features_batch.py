from __future__ import annotations

import os
from typing import Dict, Any, List

from utilities.scip_cli import run_scip_script


def instance_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def collect_batch_features(instances: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in instances:
        name = instance_name(p)
        try:
            # Light feature extraction via CLI
            cmds = [f"read {p}", "set limits nodes 0", "optimize", "quit"]
            rc, txt = run_scip_script(cmds)
            feats: Dict[str, Any] = {}
            # Parse minimal counts
            import re
            m = re.search(r"original problem has\s+(\d+)\s+variables.*?and\s+(\d+)\s+constraints", txt, re.I)
            if m:
                feats["nvars"] = int(m.group(1)); feats["nconss"] = int(m.group(2))
            out[name] = feats
        except Exception:
            out[name] = {"error": "feature_extraction_failed"}
    return out

