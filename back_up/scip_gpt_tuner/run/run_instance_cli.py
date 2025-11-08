from __future__ import annotations

import os
import re
import time
import shlex
import subprocess
from typing import Dict, Any, Optional


def _which_scip() -> str:
    return os.environ.get("SCIP_BIN", "scip")


def _build_batch_script(instance_path: str) -> str:
    lines = []
    # Increase display verbosity somewhat for richer logs
    lines.append("set display freq 100")

    # Read problem and optimize (no meta toggles; params come from .set file)
    lines.append(f"read {instance_path}")
    lines.append("optimize")
    lines.append("quit")

    return "\n".join(lines) + "\n"


_VER_RE = re.compile(r"^\s*SCIP version\s+([0-9]+\.[0-9]+\.[0-9]+)\b", re.I)
_STATUS_RE = re.compile(r"^\s*SCIP Status\s*:\s*(.+?)\s*$", re.I)
_TIME_RE = re.compile(r"^\s*(?:Solving|Total)\s+Time\s*\(sec\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_NODES_RE = re.compile(r"^\s*Solving Nodes\s*:\s*([0-9]+)", re.I)
_PRIMAL_RE = re.compile(r"^\s*Primal Bound\s*:\s*([+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)", re.I)
_DUAL_RE = re.compile(r"^\s*Dual Bound\s*:\s*([+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)", re.I)
_GAP_RE = re.compile(r"^\s*Gap\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I)
_LPIT_RE = re.compile(r"^\s*LP Iterations\s*:\s*([0-9]+)", re.I)


def _parse_summary(log_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _VER_RE.match(line)
                if m and "version" not in out:
                    out["version"] = m.group(1)
                    continue
                if "SCIP Status" in line and "Solution Status" not in line:
                    m = _STATUS_RE.match(line)
                    if m:
                        out["status"] = m.group(1).strip()
                        continue
                m = _TIME_RE.match(line)
                if m:
                    try:
                        out["solve_time"] = float(m.group(1))
                    except Exception:
                        pass
                    continue
                m = _NODES_RE.match(line)
                if m:
                    try:
                        out["n_nodes"] = int(m.group(1))
                    except Exception:
                        pass
                    continue
                m = _PRIMAL_RE.match(line)
                if m:
                    try:
                        out["primal"] = float(m.group(1))
                    except Exception:
                        pass
                    continue
                m = _DUAL_RE.match(line)
                if m:
                    try:
                        out["dual"] = float(m.group(1))
                    except Exception:
                        pass
                    continue
                m = _GAP_RE.match(line)
                if m:
                    try:
                        out["gap"] = float(m.group(1)) / 100.0
                    except Exception:
                        pass
                    continue
                m = _LPIT_RE.match(line)
                if m:
                    try:
                        out["lp_iterations"] = int(m.group(1))
                    except Exception:
                        pass
                    continue
    except Exception:
        pass
    return out


def run_instance_cli(instance_path: str, settings_file: str, meta_applied: Dict[str, Any], time_limit: float, outdir: str, seed: Optional[int] = None, trial_id: Optional[int] = None) -> Dict[str, Any]:
    """Run SCIP via CLI with a settings file and a minimal command script (no meta toggles).

    Returns a metrics dict summarizing status, time, bounds, nodes, etc.
    """
    os.makedirs(outdir, exist_ok=True)
    log_dir = os.path.join(outdir, "log")
    os.makedirs(log_dir, exist_ok=True)

    # Use trial-based naming if trial_id is provided, otherwise fall back to timestamp
    ts = int(time.time() * 1000)  # Always generate timestamp for use in metrics
    inst_name = os.path.splitext(os.path.basename(instance_path))[0]
    if trial_id is not None:
        log_path = os.path.join(log_dir, f"{inst_name}_scip_trial_{trial_id}.log")
    else:
        log_path = os.path.join(log_dir, f"{inst_name}_scip_{ts}.log")

    scip_bin = _which_scip()
    script = _build_batch_script(instance_path)

    start = time.time()
    try:
        proc = subprocess.run(
            [scip_bin, "-s", settings_file],
            input=script.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        out_text = proc.stdout.decode("utf-8", errors="replace")
        with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
            lf.write(out_text)
    except FileNotFoundError:
        raise RuntimeError("SCIP CLI not found. Ensure 'scip' is in PATH or set SCIP_BIN to the SCIP binary path.")
    end = time.time()

    # Parse summary from logfile
    summary = _parse_summary(log_path)
    # Version is enforced centrally; do not raise here to avoid duplication

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
        "applied_params": {},
        "applied_meta": {},
    }

    return metrics
