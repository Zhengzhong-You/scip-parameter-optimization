"""
Log Utility Functions
====================

Common utility functions for parsing SCIP log files that are shared
between logs.py and est_time.py modules.
"""

from __future__ import annotations
import re
import math
from typing import Dict, Any, List, Tuple
from collections import deque


# Progress table parsing utilities
_HDR_RE = re.compile(r"^\s*time\s*\|\s*node\s*\|\s*left\s*\|", re.I)
_ROW_RE = re.compile(r"^\s*(?:\*|d)?\d+(?:\.\d+)?s\|")
# Restart markers (SCIP prints lines signaling restarts)
_RESTART_MARK = re.compile(r"(Restart triggered.*|performing user restart|\(restart\).*)", re.I)


def _slice_after_last_restart(log_text: str) -> str:
    """Return the portion of the log after the final restart marker.

    If no restart markers are found, return the original text.
    """
    if not log_text:
        return log_text
    lines = log_text.splitlines(True)  # keepends
    last_idx = -1
    for idx, ln in enumerate(lines):
        if _RESTART_MARK.search(ln):
            last_idx = idx
    if last_idx >= 0 and last_idx + 1 < len(lines):
        return "".join(lines[last_idx + 1 :])
    return log_text


def _normalize_col(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().lower())
    n = n.replace("dual bound", "dual").replace("primal bound", "primal")
    n = n.replace("dualbound", "dual").replace("primalbound", "primal")
    return n


def _parse_float_cell(s: str) -> float | None:
    t = s.strip()
    if t.endswith("s") and t[:-1].replace(".", "", 1).replace("-", "", 1).isdigit():
        t = t[:-1]
    t = t.replace("%", "")
    try:
        return float(t)
    except Exception:
        tl = t.lower()
        if tl in ("inf", "+inf", "infinite", "infinity"):
            return float("inf")
        if tl in ("-inf", "-infinity"):
            return float("-inf")
    return None


def parse_progress_series(log_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Parse the SCIP progress table into a series of rows with named columns.

    Returns (columns, rows), where each row has possible keys among:
      time, node, left, dual, primal, gap
    """
    lines = log_text.splitlines() if isinstance(log_text, str) else list(log_text)
    cols: List[str] = []
    rows: List[Dict[str, Any]] = []
    parsing = False
    for ln in lines:
        if _HDR_RE.match(ln):
            # Parse header columns by splitting on '|'
            parts = [p.strip() for p in ln.split("|")]
            cols = [_normalize_col(p) for p in parts if p.strip()]
            parsing = True
            continue
        if parsing and _ROW_RE.match(ln):
            parts = [p.strip() for p in ln.split("|")]
            # align to header count (skip empty fragments at ends)
            vals = [p for p in parts if p != ""]
            row: Dict[str, Any] = {}
            for idx, name in enumerate(cols[: len(vals)]):
                key = None
                if name.startswith("time"):
                    key = "time"
                elif name.startswith("node"):
                    key = "node"
                elif name.startswith("left"):
                    key = "left"
                elif name.startswith("dual"):
                    key = "dual"
                elif name.startswith("primal"):
                    key = "primal"
                elif name.startswith("gap"):
                    key = "gap"
                if key:
                    row[key] = _parse_float_cell(vals[idx])
            rows.append(row)
        # Stop parsing on a blank line after table or summary begins
        if parsing and (ln.strip().startswith("SCIP Status") or ln.strip().startswith("Presolving Time") or ln.strip() == ""):
            parsing = False
    return cols, rows