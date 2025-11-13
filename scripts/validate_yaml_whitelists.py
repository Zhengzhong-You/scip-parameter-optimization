#!/usr/bin/env python3
"""
Validate that typed YAML whitelists contain only parameters and value domains
that SCIP 9.2.4 accepts. For each YAML file, we create a .set with a safe test
value per parameter and run SCIP with it to detect any ERRORs during loading.

Usage:
  export SCIP_BIN=/opt/homebrew/bin/scip  # ensure 9.2.4
  source scip_env/bin/activate
  PYTHONPATH=./src python3 scripts/validate_yaml_whitelists.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict, List

from utilities.whitelist import load_yaml_whitelist
from utilities.scip_cli import run_scip_script, scip_version


def pick_test_value(item: Dict[str, Any]) -> str:
    t = item.get("type")
    if t == "int":
        lo = int(item["lower"]); hi = int(item["upper"])
        v = lo if lo == hi else int((lo + hi) // 2)
        return str(v)
    if t == "float":
        lo = float(item["lower"]); hi = float(item["upper"])
        v = (lo + hi) / 2.0
        # format conservatively to avoid sci-notation
        return f"{v:.6g}"
    if t == "bool":
        # choose false
        return "false"
    if t == "cat":
        # choose first choice
        choices = item.get("choices") or []
        if not choices:
            return ""
        return str(choices[0])
    raise ValueError(f"Unknown type in YAML: {t}")


def validate_yaml(path: str) -> tuple[bool, str]:
    items = load_yaml_whitelist(path)
    with tempfile.NamedTemporaryFile("w", suffix=".set", delete=False) as tf:
        for it in items:
            name = it.get("name")
            if not name:
                continue
            val = pick_test_value(it)
            tf.write(f"{name} = {val}\n")
        tf.flush()
        set_path = tf.name
    # Run SCIP with the settings file, no instance (just load and quit)
    rc, out = run_scip_script([f"set load {set_path}", "quit"])
    os.unlink(set_path)
    ok = (rc == 0) and ("ERROR" not in (out or ""))
    return ok, out


def main() -> int:
    sv = scip_version()
    print(f"SCIP version: {sv}")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    curated = os.path.join(root, "src", "utilities", "whitelists", "curated.yaml")
    minimal = os.path.join(root, "src", "utilities", "whitelists", "minimal.yaml")
    for label, p in ("curated", curated), ("minimal", minimal):
        ok, out = validate_yaml(p)
        print(f"[{label}] valid={ok}")
        if not ok:
            print(out[:2000])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

