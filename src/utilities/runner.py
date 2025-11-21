from __future__ import annotations

import os
import re
import time
import subprocess
from typing import Dict, Any, Optional

from .version_check import ensure_scip_version
from .logs import parse_scip_log_lines
from .scip_cli import _scip_bin

_VERSION_CHECKED = False


def _ensure_version_once():
    global _VERSION_CHECKED
    if not _VERSION_CHECKED:
        ensure_scip_version()
        _VERSION_CHECKED = True


def _build_batch_script(instance_path: str) -> str:
    lines = []
    lines.append("set display freq 100")
    lines.append(f"read {instance_path}")
    lines.append("optimize")
    lines.append("quit")
    return "\n".join(lines) + "\n"


def _parse_summary(log_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    VER_RE = re.compile(r"^\s*SCIP version\s+([0-9]+\.[0-9]+\.[0-9]+)\b", re.I)
    STATUS_RE = re.compile(r"^\s*SCIP Status\s*:\s*(.+?)\s*$", re.I)
    TIME_RE = re.compile(r"^\s*(?:Solving|Total)\s+Time\s*\(sec\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    NODES_RE = re.compile(r"^\s*Solving Nodes\s*:\s*([0-9]+)", re.I)
    PR_RE = re.compile(r"^\s*Primal Bound\s*:\s*([+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)", re.I)
    DU_RE = re.compile(r"^\s*Dual Bound\s*:\s*([+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)", re.I)
    GAP_RE = re.compile(r"^\s*Gap\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I)
    LPIT_RE = re.compile(r"^\s*LP Iterations\s*:\s*([0-9]+)", re.I)

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = VER_RE.match(line)
                if m and "version" not in out:
                    out["version"] = m.group(1); continue
                if "SCIP Status" in line and "Solution Status" not in line:
                    m = STATUS_RE.match(line)
                    if m:
                        out["status"] = m.group(1).strip(); continue
                m = TIME_RE.match(line)
                if m:
                    try: out["solve_time"] = float(m.group(1))
                    except Exception: pass
                    continue
                m = NODES_RE.match(line)
                if m:
                    try: out["n_nodes"] = int(m.group(1))
                    except Exception: pass
                    continue
                m = PR_RE.match(line)
                if m:
                    try: out["primal"] = float(m.group(1))
                    except Exception: pass
                    continue
                m = DU_RE.match(line)
                if m:
                    try: out["dual"] = float(m.group(1))
                    except Exception: pass
                    continue
                m = GAP_RE.match(line)
                if m:
                    try: out["gap"] = float(m.group(1)) / 100.0
                    except Exception: pass
                    continue
                m = LPIT_RE.match(line)
                if m:
                    try: out["lp_iterations"] = int(m.group(1))
                    except Exception: pass
                    continue
    except Exception:
        pass
    return out


def _print_trial_summary(metrics: Dict[str, Any], instance_path: str, trial_id: Optional[int] = None) -> None:
    """Print trial summary in SCIP format to terminal"""
    print(f"\n{'='*60}")
    print(f"TRIAL SUMMARY - {os.path.basename(instance_path)}" + (f" (Trial {trial_id})" if trial_id is not None else ""))
    print(f"{'='*60}")

    # Format status
    status = metrics.get("status", "unknown")
    print(f"SCIP Status        : {status}")

    # Format solving time
    solve_time = metrics.get("solve_time", 0.0)
    print(f"Solving Time (sec) : {solve_time:.2f}")

    # Format nodes if available
    n_nodes = metrics.get("n_nodes")
    if n_nodes is not None:
        print(f"Solving Nodes      : {n_nodes}")

    # Format primal bound
    primal = metrics.get("primal")
    if primal is not None and primal != float("inf"):
        print(f"Primal Bound       : {primal:+.14e}")

    # Format dual bound
    dual = metrics.get("dual")
    if dual is not None and dual != float("inf"):
        print(f"Dual Bound         : {dual:+.14e}")

    # Format gap
    gap = metrics.get("gap")
    if gap is not None:
        print(f"Gap                : {gap*100:.2f} %")

    print(f"{'='*60}\n")


def run_instance(instance_path: str, params: Dict[str, Any], time_limit: float, outdir: str, seed: Optional[int] = None, trial_id: Optional[int] = None) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    log_dir = os.path.join(outdir, "log"); os.makedirs(log_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    inst_name = os.path.splitext(os.path.basename(instance_path))[0]
    log_path = os.path.join(log_dir, f"{inst_name}_scip_trial_{trial_id if trial_id is not None else ts}.log")

    # Build settings .set file
    set_path = os.path.join(outdir, f"params_{trial_id if trial_id is not None else ts}.set")
    with open(set_path, "w", encoding="utf-8") as f:
        for k, v in (params or {}).items():
            f.write(f"{k} = {v}\n")
        f.write(f"\nlimits/time = {float(time_limit)}\n")
        if seed is not None:
            try: f.write(f"randomization/randomseedshift = {int(seed)}\n")
            except Exception: pass

    _ensure_version_once()
    script = _build_batch_script(instance_path)

    start = time.time()
    try:
        with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
            proc = subprocess.Popen([
                _scip_bin(), "-s", set_path
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
               text=True, bufsize=1, universal_newlines=True)

            # Send script input
            proc.stdin.write(script)
            proc.stdin.close()

            # Stream output line by line with real-time flushing
            for line in proc.stdout:
                lf.write(line)
                lf.flush()  # Force flush to disk immediately

            proc.wait()
    except FileNotFoundError:
        raise RuntimeError("SCIP CLI not found. Ensure 'scip' is in PATH or set SCIP_BIN to the SCIP binary path.")
    end = time.time()

    # Validate log file exists and is not empty
    if not os.path.exists(log_path):
        raise RuntimeError(f"FATAL ERROR: SCIP log file not found: {log_path}")

    if os.path.getsize(log_path) == 0:
        raise RuntimeError(f"FATAL ERROR: SCIP log file is empty: {log_path}")

    summary = _parse_summary(log_path)
    metrics: Dict[str, Any] = {
        "timestamp": ts,
        "log_path": log_path,
        "status": summary.get("status", "unknown"),
        "solve_time": summary.get("solve_time", end - start),
        "time_limit": time_limit,
        "primal": summary.get("primal", float("inf")),
        "dual": summary.get("dual", float("inf")),
        "gap": summary.get("gap", None),
        "n_nodes": summary.get("n_nodes", None),
        "lp_iterations": summary.get("lp_iterations", None),
        "n_solutions": None,
        "obj_sense": None,
        "applied_params": params or {},
    }

    # Print trial summary to terminal
    _print_trial_summary(metrics, instance_path, trial_id)

    return metrics

