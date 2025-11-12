"""
Unified Optimizer Runner Interface
=================================

Provides a consistent interface for running different optimization methods
(SMAC, RBFOpt, etc.) with the same command-line interface and behavior.

This avoids code duplication and ensures all optimizers work consistently
with the L1 MINLP paper implementation.
"""

import argparse
import json
import os
import time as _time
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List, Callable

from .datasets import discover_instances, train_test_split_by_fraction
from .whitelist import get_typed_whitelist
from .runner import run_instance as scip_run
from .logs import per_instance_T_infty
from .est_time import diagnose_t_infty, format_t_infty_diagnostic


def create_runner_function(out_dir: str, tau: float):
    """Create a unified runner function for optimization methods."""
    counter = {"k": 0}

    def runner_fn(pcfg, instance, tau_):
        name = os.path.splitext(os.path.basename(instance))[0]
        outdir_i = os.path.join(out_dir, name)
        # Bump trial id and name files accordingly
        counter["k"] += 1
        res = scip_run(instance, params=pcfg, time_limit=tau_, outdir=outdir_i, trial_id=counter["k"])
        return res

    return runner_fn


def compute_baseline_t_infinity(instances: List[str], tau: float, out_dir: str) -> Dict[str, float]:
    """Compute baseline T_infinity for default parameters."""
    per_base = {}
    for pth in instances:
        nm = os.path.splitext(os.path.basename(pth))[0]
        outdir_i = os.path.join(out_dir, nm)
        # Baseline is trial 0 so naming matches LLM rules
        res = scip_run(pth, params={}, time_limit=tau, outdir=outdir_i, trial_id=0)
        per_base[nm] = res
    return per_instance_T_infty(per_base, tau=tau)


def save_results(best_cfg: Dict[str, Any], best_rhat: float, tinf_best: Dict[str, float],
                 tinf_base: Dict[str, float], instances: List[str], tau: float,
                 out_dir: str, method: str, total_time: float = 0.0):
    """Save optimization results in unified format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _to_jsonable(o):
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        try:
            import numpy as _np
            if isinstance(o, _np.generic):
                return o.item()
        except Exception:
            pass
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(x) for x in o]
        return str(o)

    # Save best configuration and results
    (out / "best_config.json").write_text(json.dumps(_to_jsonable(best_cfg), indent=2))
    (out / "best_R_hat.txt").write_text(str(best_rhat))

    # Re-run with best config to get detailed logs
    runner_fn = create_runner_function(out_dir, tau)
    per_m = {}
    per_logs = []
    for inst in instances:
        result = runner_fn(best_cfg, inst, tau)
        name = Path(inst).stem
        per_m[name] = result
        per_logs.append({"instance": inst, **result})

    (out / "per_instance.json").write_text(json.dumps(_to_jsonable(per_logs), indent=2))
    (out / "tinf_candidate.json").write_text(json.dumps(tinf_best, indent=2))
    (out / "tinf_baseline.json").write_text(json.dumps(tinf_base, indent=2))

    # Unified CSV schema
    import pandas as pd
    trials_df = pd.DataFrame([{
        "trial": -1,
        "r_hat": float(best_rhat),
        "config": json.dumps(best_cfg),
        "total_time": total_time
    }])
    trials_df.to_csv(out / "trials.csv", index=False)

    # Per-instance CSV
    import csv as _csv
    with open(out / "per_instance.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "trial", "instance", "T_infty", "status", "solve_time",
            "primal", "dual", "gap", "n_nodes", "log_path"
        ])
        w.writeheader()
        for row in per_logs:
            name = Path(row["instance"]).stem
            w.writerow({
                "trial": -1,
                "instance": name,
                "T_infty": float(tinf_best.get(name, float("nan"))),
                "status": row.get("status"),
                "solve_time": row.get("solve_time", row.get("time")),
                "primal": row.get("primal"),
                "dual": row.get("dual"),
                "gap": row.get("gap"),
                "n_nodes": row.get("n_nodes", row.get("nodes")),
                "log_path": row.get("log_path"),
            })

    # Per-instance diagnostics
    try:
        with open(out / "t_infty_diagnostics.txt", "w", encoding="utf-8") as df:
            for row in per_logs:
                lp = row.get("log_path")
                name = Path(row["instance"]).stem
                try:
                    log_text = open(lp, "r", encoding="utf-8", errors="ignore").read() if lp else ""
                except Exception:
                    log_text = ""
                diag = diagnose_t_infty(log_text, tau=tau, summary=row)
                df.write(f"=== {name} ===\n")
                df.write(format_t_infty_diagnostic(diag) + "\n\n")
    except Exception:
        pass

    # Summary JSON
    try:
        summary = {
            "method": method,
            "r_hat": float(best_rhat),
            "best_config": best_cfg,
            "trials_csv": str(out / "trials.csv"),
            "per_instance_csv": str(out / "per_instance.csv"),
            "tinf_candidate": str(out / "tinf_candidate.json"),
            "tinf_baseline": str(out / "tinf_baseline.json"),
            "train_count": len(instances),
            "tau": float(tau),
            "total_time": float(total_time),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception:
        pass


def run_single_instance_optimization(method: str, optimizer_func: Callable,
                                   config_path: str, whitelist_regime: str,
                                   instance_path: str):
    """
    Unified interface for single-instance optimization.

    Args:
        method: "smac" or "rbfopt"
        optimizer_func: The actual optimization function
        config_path: Path to YAML config file
        whitelist_regime: "curated", "minimal", or "full"
        instance_path: Path to the single instance to optimize on
    """
    cfg = yaml.safe_load(open(config_path, "r"))

    # Use unified SCIP-driven typed whitelist
    if whitelist_regime not in ("curated", "minimal", "full"):
        raise SystemExit(f"Unsupported whitelist regime: {whitelist_regime}")
    wl = get_typed_whitelist(regime=whitelist_regime)

    # Always use a single provided instance (no train/test semantics)
    if not os.path.exists(instance_path):
        raise SystemExit(f"Instance not found: {instance_path}")
    instances = [os.path.abspath(instance_path)]

    tau = float(cfg["runner"]["tau"])

    # Place outputs directly under runs/, per request
    os.makedirs("runs", exist_ok=True)

    ts = int(_time.time())
    inst_name = os.path.splitext(os.path.basename(instances[0]))[0]
    out_run = os.path.join("runs", f"{method}_{inst_name}")

    # Clean existing run dir to ensure fresh runs
    shutil.rmtree(out_run, ignore_errors=True)
    os.makedirs(out_run, exist_ok=True)

    # Create runner function
    runner_fn = create_runner_function(out_run, tau)

    # Compute baseline T_infinity
    tinf_base = compute_baseline_t_infinity(instances, tau, out_run)

    # Run optimization
    if method == "smac":
        best_cfg, best_rhat, _, tinf_best = optimizer_func(
            wl, instances, runner_fn, tau, tinf_base,
            n_trials=cfg.get("smac", {}).get("n_trials", 10),
            seed=cfg.get("smac", {}).get("seed", 0),
            out_dir=out_run
        )
    elif method == "rbfopt":
        best_cfg, best_rhat, _, tinf_best = optimizer_func(
            wl, runner_fn, instances, tau, tinf_base,
            max_evaluations=cfg.get("rbfopt", {}).get("max_evaluations", 10),
            seed=cfg.get("rbfopt", {}).get("seed", 0),
            out_dir=out_run,
            tag=method
        )
    elif method == "solvermind":
        # SolverMind uses similar parameters to SMAC
        best_cfg, best_rhat, _, tinf_best = optimizer_func(
            wl, instances, runner_fn, tau, tinf_base,
            n_trials=cfg.get("solvermind", {}).get("n_trials", 10),
            seed=cfg.get("solvermind", {}).get("seed", 0),
            out_dir=out_run
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    print("Best R_hat(p,q):", best_rhat)
    print("Best config:", json.dumps(best_cfg, indent=2))

    return best_cfg, best_rhat


def create_unified_parser(method: str) -> argparse.ArgumentParser:
    """Create unified argument parser for optimization methods."""
    ap = argparse.ArgumentParser(description=f"Run {method.upper()} optimization with unified interface")
    ap.add_argument("--config", required=True, help="Configuration YAML file")
    ap.add_argument("--whitelist", required=True, help="Whitelist regime: curated|minimal|full")
    ap.add_argument("--instance", required=True, help="Path to a single instance file to tune on")
    return ap