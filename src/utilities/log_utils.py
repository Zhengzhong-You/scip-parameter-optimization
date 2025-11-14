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


def per_instance_T_infty(per_instance_results: Dict[str, Dict[str, Any]], tau: float) -> Dict[str, float]:
    """
    Compute T_infinity values for each instance from optimization results.

    Args:
        per_instance_results: Dict mapping instance names to result dictionaries
        tau: Time limit in seconds

    Returns:
        Dict mapping instance names to T_infinity values
    """
    from utilities.est_time import compute_t_infinity_surrogate
    from pathlib import Path

    tinf_dict = {}

    for instance_name, result in per_instance_results.items():
        try:
            # Get log content if available
            log_path = result.get("log_path")
            log_text = ""
            if log_path and Path(log_path).exists():
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        log_text = f.read()
                except:
                    pass

            # Create summary from results
            summary = {
                "solve_time": result.get("solve_time", result.get("time", tau)),
                "n_nodes": result.get("n_nodes", result.get("nodes", 0)),
                "primal": result.get("primal"),
                "dual": result.get("dual")
            }

            # Compute T_infinity using est_time module
            est_result = compute_t_infinity_surrogate(log_text, tau, summary)
            tinf_dict[instance_name] = float(est_result.get("T_infinity", tau))

        except Exception:
            # Fallback to tau if computation fails
            tinf_dict[instance_name] = float(tau)

    return tinf_dict


def shrink_scip_log_for_gpt(log_content: str, max_chars: int = 4000) -> str:
    """
    Shrink SCIP log content to fit within GPT context limits.

    Keeps the most important parts: header, final statistics, and some middle content.

    Args:
        log_content: Full SCIP log content
        max_chars: Maximum characters to retain

    Returns:
        Shortened log content
    """
    if len(log_content) <= max_chars:
        return log_content

    lines = log_content.split('\n')
    if len(lines) <= 20:
        return log_content[:max_chars]

    # Keep first 10 lines (header info)
    header_lines = lines[:10]

    # Keep last 20 lines (final statistics)
    footer_lines = lines[-20:]

    # Calculate remaining space for middle content
    header_text = '\n'.join(header_lines)
    footer_text = '\n'.join(footer_lines)
    separator = '\n[... log truncated ...]\n'

    remaining_chars = max_chars - len(header_text) - len(footer_text) - len(separator)

    if remaining_chars > 200:
        # Take some middle content
        middle_start = len(lines) // 3
        middle_end = 2 * len(lines) // 3
        middle_lines = lines[middle_start:middle_end]

        middle_text = '\n'.join(middle_lines)
        if len(middle_text) > remaining_chars:
            middle_text = middle_text[:remaining_chars - 3] + "..."

        return header_text + separator + middle_text + separator + footer_text
    else:
        # Just header and footer
        return header_text + separator + footer_text


def diagnose_t_infty(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose T_infinity computation and provide detailed information.

    Args:
        log_text: SCIP log content
        tau: Time limit in seconds
        summary: Summary dictionary with solve_time, n_nodes, etc.

    Returns:
        Diagnostic information dictionary
    """
    from utilities.est_time import compute_t_infinity_surrogate

    try:
        result = compute_t_infinity_surrogate(log_text, tau, summary)

        diagnostic = {
            "status": "success",
            "T_infinity": result.get("T_infinity", tau),
            "method": result.get("method", "unknown"),
            "solved": result.get("solved", False),
            "error": result.get("error"),
            "details": {
                "tau": tau,
                "actual_time": summary.get("solve_time", 0),
                "actual_nodes": summary.get("n_nodes", 0),
                "b_hat": result.get("b_hat"),
                "theta_hat": result.get("theta_hat"),
                "samples_used": result.get("samples_used")
            }
        }

        return diagnostic

    except Exception as e:
        return {
            "status": "error",
            "T_infinity": float(tau),
            "method": "fallback",
            "solved": False,
            "error": str(e),
            "details": {
                "tau": tau,
                "actual_time": summary.get("solve_time", 0),
                "actual_nodes": summary.get("n_nodes", 0)
            }
        }


def format_t_infty_diagnostic(diagnostic: Dict[str, Any]) -> str:
    """
    Format T_infinity diagnostic information as readable text.

    Args:
        diagnostic: Diagnostic dictionary from diagnose_t_infty

    Returns:
        Formatted diagnostic text
    """
    status = diagnostic.get("status", "unknown")
    t_inf = diagnostic.get("T_infinity", 0)
    method = diagnostic.get("method", "unknown")
    solved = diagnostic.get("solved", False)
    error = diagnostic.get("error")
    details = diagnostic.get("details", {})

    lines = [
        f"T_infinity Diagnostic",
        f"Status: {status}",
        f"T_infinity: {t_inf:.2f}s",
        f"Method: {method}",
        f"Solved: {solved}",
    ]

    if error:
        lines.append(f"Error: {error}")

    lines.append("\nDetails:")
    lines.append(f"  Time limit (τ): {details.get('tau', 0):.2f}s")
    lines.append(f"  Actual time: {details.get('actual_time', 0):.2f}s")
    lines.append(f"  Actual nodes: {details.get('actual_nodes', 0):,}")

    if details.get('b_hat') is not None:
        lines.append(f"  Predicted nodes (b̂): {details['b_hat']:.1f}")
    if details.get('theta_hat') is not None:
        lines.append(f"  Time per node (θ̂): {details['theta_hat']:.6f}s/node")
    if details.get('samples_used') is not None:
        lines.append(f"  Samples used: {details['samples_used']}")

    return "\n".join(lines)