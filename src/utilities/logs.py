from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple
from collections import deque
import math
import re

# Import common log parsing utilities
from .log_utils import parse_progress_series, _slice_after_last_restart, _normalize_col, _parse_float_cell

# Import estimation functions from the separate module
from .est_time import (
    estimate_svb_from_log,
    estimate_total_nodes_svb,
    estimate_remaining_time,
    compute_T_infty,
    diagnose_t_infty,
    format_t_infty_diagnostic,
    per_instance_T_infty
)


def parse_scip_log_lines(lines: List[str]) -> Dict[str, Any]:
    HEADER_RE = re.compile(r'^\s*time\s*\|\s*node\s*\|\s*left\s*\|', re.I)
    TIMEROW_RE = re.compile(r'^\s*(?:\*|d)?\d+(?:\.\d+)?s\|')
    ORIGINAL_RE = re.compile(r'original problem has .*? and \d+\s+constraints', re.I)
    PRESOLVE_SUMMARY_RE = re.compile(r'^\s*presolving\s*\(', re.I)
    PRESOLVE_TIME_RE = re.compile(r'^\s*Presolving Time\s*:', re.I)
    RESTART_RE = re.compile(r'(Restart triggered.*|performing user restart|\(restart\).*)', re.I)

    SUMMARY_KEYS = [
        re.compile(r'^\s*SCIP Status\s*:', re.I),
        re.compile(r'^\s*Solving Time\s*\(sec\)\s*:', re.I),
        re.compile(r'^\s*Solving Nodes\s*:', re.I),
        re.compile(r'^\s*Primal Bound\s*:', re.I),
        re.compile(r'^\s*Dual Bound\s*:', re.I),
        re.compile(r'^\s*Gap\s*:', re.I),
    ]

    res = {
        "original_problem": None,
        "presolve_blocks": [],
        "progress_blocks": [],
        "restarts": [],
        "summary_lines": []
    }
    in_presolve = False
    curr_presolve = None
    curr_prog = None

    def end_presolve():
        nonlocal in_presolve, curr_presolve
        if in_presolve and curr_presolve:
            res["presolve_blocks"].append(curr_presolve)
        in_presolve = False
        curr_presolve = None

    def end_progress():
        nonlocal curr_prog
        if curr_prog and curr_prog["rows"]:
            res["progress_blocks"].append({
                "header": curr_prog["header"],
                "rows": list(curr_prog["rows"])
            })
        curr_prog = None

    for line in lines:
        line = line.rstrip()
        if res["original_problem"] is None and ORIGINAL_RE.search(line):
            res["original_problem"] = line.strip()
        if RESTART_RE.search(line):
            res["restarts"].append(line.strip())
        if PRESOLVE_SUMMARY_RE.match(line):
            end_presolve(); in_presolve = True; curr_presolve = [line]; continue
        if in_presolve and curr_presolve is not None:
            curr_presolve.append(line)
            if PRESOLVE_TIME_RE.match(line):
                end_presolve(); continue
        if HEADER_RE.match(line):
            end_progress();
            curr_prog = {"header": line, "rows": deque(maxlen=5)}
            continue
        if curr_prog is not None and TIMEROW_RE.match(line):
            curr_prog["rows"].append(line)
        if any(pat.match(line) for pat in SUMMARY_KEYS):
            res["summary_lines"].append(line.strip())

    end_presolve(); end_progress()
    return res


def shrink_scip_log_for_gpt(log_text: str, max_length: int = 1500) -> str:
    if not log_text:
        return ""
    parsed = parse_scip_log_lines(log_text.splitlines())
    out = []
    if parsed["original_problem"]:
        out.append("PROBLEM SIZE:")
        out.append(parsed["original_problem"]); out.append("")
    if parsed["presolve_blocks"]:
        # If there are restarts, use the last presolve block (after restart), otherwise use the first one
        restart_count = len(parsed["restarts"])
        if restart_count > 0 and len(parsed["presolve_blocks"]) > 1:
            # Use the last presolve block (after restart)
            presolve_block = parsed["presolve_blocks"][-1]
        else:
            # Use the first presolve block
            presolve_block = parsed["presolve_blocks"][0]

        out.append("PRESOLVING:")
        out.extend(presolve_block[:8]); out.append("")
    if parsed["progress_blocks"]:
        # Find root node performance after the last restart
        # Look for the last progress block that contains node 1 rows after restart indicators
        root_block = None
        restart_count = len(parsed["restarts"])

        if restart_count > 0:
            # Look for progress blocks after restarts - find the last one with node 1
            for block in reversed(parsed["progress_blocks"]):
                rows = list(block["rows"])
                for row in rows:
                    # Check if this row shows node 1 (root after restart)
                    if "|     1 |" in row and "unknown" in row:
                        root_block = block
                        break
                if root_block:
                    break

        # If no restart root found, use first block
        if not root_block:
            root_block = parsed["progress_blocks"][0]

        # Show root node performance after last restart
        out.append("ROOT NODE PERFORMANCE:")
        out.append(root_block["header"])
        root_rows = [row for row in list(root_block["rows"]) if "|     1 |" in row][:5]
        if not root_rows:  # fallback to first 5 rows if no node 1 found
            root_rows = list(root_block["rows"])[:5]
        out.extend(root_rows)
        out.append("")

        # Also include the last few rows as before
        last_block = parsed["progress_blocks"][-1]
        out.append("SOLVING PROGRESS:")
        out.append(last_block["header"])
        out.extend(list(last_block["rows"])[-3:]); out.append("")
    if parsed["restarts"]:
        out.append(f"RESTARTS: {len(parsed['restarts'])} restart(s)"); out.append("")
    if parsed["summary_lines"]:
        out.append("FINAL RESULTS:")
        out.extend(parsed["summary_lines"])
    result = "\n".join(out)
    if len(result) > max_length:
        result = result[:max_length-15] + "\n...[truncated]"
    return result



