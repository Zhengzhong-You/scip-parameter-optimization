#!/usr/bin/env python3
"""
Check that whitelist parameter names exist in the current SCIP CLI.

It validates:
  - Typed YAML whitelists under src/utilities/whitelists/{curated,minimal}.yaml
  - Programmatic lists from utilities.whitelist.get_whitelist(regime)

It prints unknown/missing parameters per source and basic stats.

Usage:
  source scip_env/bin/activate
  PYTHONPATH=./src python3 scripts/check_whitelist_params.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

from utilities.scip_cli import get_default_params, scip_version
from utilities.whitelist import get_whitelist, load_yaml_whitelist


def check_names(names: List[str], known: set[str]) -> List[str]:
    return [n for n in names if n not in known]


def main() -> int:
    sv = scip_version() or "unknown"
    params = get_default_params()
    known = set(params.keys())

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    curated_yaml = os.path.join(root, "src", "utilities", "whitelists", "curated.yaml")
    minimal_yaml = os.path.join(root, "src", "utilities", "whitelists", "minimal.yaml")

    print(f"SCIP version: {sv}")
    print(f"Known parameter count: {len(known)}\n")

    # YAML typed whitelists
    results: Dict[str, List[str]] = {}
    for label, pth in ("yaml_curated", curated_yaml), ("yaml_minimal", minimal_yaml):
        items = load_yaml_whitelist(pth)
        names = [it.get("name") for it in items if it.get("name")]
        unknown = check_names(names, known)
        results[label] = unknown
        print(f"[{label}] total={len(names)}, unknown={len(unknown)}")
        for u in unknown:
            print(f"  - {u}")
        print()

    # Programmatic whitelists used by LLM flow
    for regime in ("curated", "minimal"):
        lst = get_whitelist(regime=regime).get("params", [])
        unknown = check_names(lst, known)
        results[f"prog_{regime}"] = unknown
        print(f"[prog_{regime}] total={len(lst)}, unknown={len(unknown)}")
        for u in unknown:
            print(f"  - {u}")
        print()

    any_unknown = any(results.values())
    if any_unknown:
        print("There are unknown parameters. Consider updating the whitelists to match this SCIP version.")
        return 1
    print("All whitelist parameters are known.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

