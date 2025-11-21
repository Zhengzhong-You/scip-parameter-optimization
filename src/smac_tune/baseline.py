from typing import Dict, Any, List, Tuple
import json, time
import pandas as pd
from pathlib import Path

import ConfigSpace as CS
import importlib, sys, os, site

def _import_third_party_smac():
    """Import the pip-installed 'smac' by temporarily removing this project's 'src/smac' shadow package."""
    repo_pkg_dir = os.path.abspath(os.path.dirname(__file__))  # .../src/smac
    repo_src = os.path.abspath(os.path.join(repo_pkg_dir, '..'))
    orig_path = list(sys.path)
    try:
        # Remove src from path to avoid shadowing
        try:
            sys.path.remove(repo_src)
        except ValueError:
            pass
        # Remove local package binding if present
        if 'smac' in sys.modules:
            del sys.modules['smac']
        return importlib.import_module('smac')
    finally:
        sys.path[:] = orig_path

_smac = _import_third_party_smac()
HyperparameterOptimizationFacade = _smac.HyperparameterOptimizationFacade
Scenario = _smac.Scenario

from utilities.log_utils import per_instance_T_infty, diagnose_t_infty, format_t_infty_diagnostic
from utilities.scoring import r_hat_ratio
from .space import build_configspace


def run_smac(whitelist: List[Dict[str, Any]], instance: str, runner_fn, tau: float,
             tinf_base: Dict[str, float],
             n_trials: int = 100, seed: int = 0, out_dir: str = "./runs", tag: str = "smac"
             ) -> Tuple[Dict[str, Any], float, pd.DataFrame, Dict[str, float]]:
    cs, _ = build_configspace(whitelist)

    def objective(cfg: CS.Configuration, seed: int = 0) -> float:
        d = {k: cfg[k] for k in cfg}
        out = runner_fn(d, instance, tau)
        name = Path(instance).stem
        per_m = {name: out}
        tinf_cand = per_instance_T_infty(per_m, tau=tau)
        rhat = r_hat_ratio(tinf_cand, tinf_base, cap=1e3)
        return float(rhat)

    # Check if SMAC has verbose logging options
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create output directory structure first to put SMAC internal files there too
    out = Path(out_dir) / f"{tag}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    smac_internal_dir = out / "smac_internal"

    scenario = Scenario(cs, n_trials=int(n_trials), seed=int(seed), deterministic=True,
                       output_directory=str(smac_internal_dir))
    smac = HyperparameterOptimizationFacade(scenario, objective)

    start = time.time()
    incumbent = smac.optimize()
    total_time = time.time() - start

    # Convert ConfigSpace Configuration to JSON-serializable dict
    best_dict = {}
    for k in incumbent:
        val = incumbent[k]
        if hasattr(val, 'item'):  # numpy types
            best_dict[k] = val.item()
        elif isinstance(val, (bool, int, float, str)):  # native types
            best_dict[k] = val
        else:  # fallback for other types
            best_dict[k] = str(val)

    # Evaluate best config
    best_result = runner_fn(best_dict, instance, tau)
    name = Path(instance).stem
    per_m = {name: best_result}
    per_logs = [{"instance": instance, **best_result}]
    tinf_best = per_instance_T_infty(per_m, tau=tau)
    best_L = r_hat_ratio(tinf_best, tinf_base, cap=1e3)

    trials_df = pd.DataFrame([{"trial": -1, "r_hat": float(best_L), "config": json.dumps(best_dict), "total_time": total_time}])

    # Print optimal configuration to terminal
    print(f"\n=== SMAC OPTIMIZATION SUMMARY ===")
    print(f"Best R_hat: {best_L:.6f}")
    print(f"Best configuration: {best_dict}")
    print(f"Total optimization time: {total_time:.2f}s")

    # Output directory already created above - wrapped in try-catch for robustness
    try:
        (out / "best_config.json").write_text(json.dumps(best_dict, indent=2))
        (out / "best_R_hat.txt").write_text(str(best_L))
        (out / "per_instance.json").write_text(json.dumps(per_logs, indent=2))
        (out / "tinf_candidate.json").write_text(json.dumps(tinf_best, indent=2))
    except Exception as e:
        print(f"WARNING: Failed to write output files: {e}")
        print(f"Optimal configuration still available above!")

    # Unified CSV schema - wrapped in try-catch for robustness
    try:
        trials_df.to_csv(out / "trials.csv", index=False)
    except Exception as e:
        print(f"WARNING: Failed to write trials.csv: {e}")
    # Build per-instance CSV: instance, T_infty, and a few summary fields if available
    import csv as _csv
    with open(out / "per_instance.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["trial", "instance", "T_infty", "status", "solve_time", "primal", "dual", "gap", "n_nodes", "log_path"])
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
                lp = row.get("log_path"); name = Path(row["instance"]).stem
                try:
                    log_text = open(lp, "r", encoding="utf-8", errors="ignore").read() if lp else ""
                except Exception:
                    log_text = ""
                diag = diagnose_t_infty(log_text, tau=tau, summary=row)
                df.write(f"=== {name} ===\n")
                df.write(format_t_infty_diagnostic(diag) + "\n\n")
    except Exception:
        pass

    # Write baseline T_infty and a summary JSON
    (out / "tinf_baseline.json").write_text(json.dumps(tinf_base, indent=2))
    try:
        summary = {
            "method": "smac",
            "r_hat": float(best_L),
            "best_config": best_dict,
            "trials_csv": str(out / "trials.csv"),
            "per_instance_csv": str(out / "per_instance.csv"),
            "tinf_candidate": str(out / "tinf_candidate.json"),
            "tinf_baseline": str(out / "tinf_baseline.json"),
            "tau": float(tau),
            "total_time": float(total_time),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception:
        pass
    return best_dict, float(best_L), trials_df, tinf_best
