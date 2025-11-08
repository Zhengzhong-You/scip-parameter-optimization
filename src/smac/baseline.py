from typing import Dict, Any, List, Tuple
import json, time
import pandas as pd
from pathlib import Path

import ConfigSpace as CS
from smac import HyperparameterOptimizationFacade, Scenario

from utilities.logs import per_instance_T_infty, diagnose_t_infty, format_t_infty_diagnostic
from utilities.scoring import r_hat_ratio
from .space import build_configspace


def evaluate_config_rhat(cfg: Dict[str, Any], instances: List[str], runner_fn, tau: float,
                         tinf_base: Dict[str, float]) -> Tuple[float, List[Dict[str, Any]], Dict[str, float]]:
    per_m: Dict[str, Dict[str, Any]] = {}
    per_logs: List[Dict[str, Any]] = []
    for inst in instances:
        out = runner_fn(cfg, inst, tau)
        name = Path(inst).stem
        per_m[name] = out
        per_logs.append({"instance": inst, **out})
    tinf_cand = per_instance_T_infty(per_m, tau=tau)
    rhat = r_hat_ratio(tinf_cand, tinf_base, cap=1e3)
    return float(rhat), per_logs, tinf_cand


def run_smac(whitelist: List[Dict[str, Any]], runner_fn, instances: List[str], tau: float,
             tinf_base: Dict[str, float],
             n_trials: int = 100, seed: int = 0, out_dir: str = "./runs", tag: str = "smac"
             ) -> Tuple[Dict[str, Any], float, pd.DataFrame, Dict[str, float]]:
    cs, _ = build_configspace(whitelist)

    def objective(cfg: CS.Configuration) -> float:
        d = {k: cfg[k] for k in cfg}
        rhat, _ = evaluate_config_rhat(d, instances, runner_fn, tau, tinf_base)[:2]
        return float(rhat)

    scenario = Scenario(cs, n_trials=int(n_trials), seed=int(seed), deterministic=True)
    smac = HyperparameterOptimizationFacade(scenario, objective)

    start = time.time()
    incumbent = smac.optimize()
    total_time = time.time() - start

    best_dict = {k: incumbent[k] for k in incumbent}
    best_L, per_logs, tinf_best = evaluate_config_rhat(best_dict, instances, runner_fn, tau, tinf_base)

    trials_df = pd.DataFrame([{"trial": -1, "r_hat": float(best_L), "config": json.dumps(best_dict), "total_time": total_time}])

    out = Path(out_dir) / f"{tag}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_config.json").write_text(json.dumps(best_dict, indent=2))
    (out / "best_R_hat.txt").write_text(str(best_L))
    (out / "per_instance.json").write_text(json.dumps(per_logs, indent=2))
    (out / "tinf_candidate.json").write_text(json.dumps(tinf_best, indent=2))

    # Unified CSV schema
    trials_df.to_csv(out / "trials.csv", index=False)
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
            "train_count": len(instances),
            "tau": float(tau),
            "total_time": float(total_time),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception:
        pass
    return best_dict, float(best_L), trials_df, tinf_best
