from typing import Dict, Any, List, Tuple
import json, time
import pandas as pd
from pathlib import Path
import sys, os

# Import the third-party rbfopt package (no naming conflict anymore)
import rbfopt

from utilities.log_utils import per_instance_T_infty, diagnose_t_infty, format_t_infty_diagnostic
from utilities.scoring import r_hat_ratio


def run_rbfopt(whitelist: List[Dict[str, Any]], runner_fn, instance: str, tau: float,
               tinf_base: Dict[str, float],
               max_evaluations: int = 100, seed: int = 0, out_dir: str = "./runs", tag: str = "rbfopt"
               ) -> Tuple[Dict[str, Any], float, pd.DataFrame, Dict[str, float]]:
    print(f"# RBFOpt Optimization Log")
    print(f"# Instance: {Path(instance).stem}")
    print(f"# Max evaluations: {max_evaluations}")
    print(f"# Seed: {seed}")
    print(f"# Tau: {tau}\n")

    # Build RBFOpt variable space
    dim = len(whitelist)
    var_lower = []; var_upper = []; var_type = []; types = []; name_list = []; cat_maps = []
    for item in whitelist:
        name_list.append(item["name"]); t = item["type"]; types.append(t)
        if t == "float": var_lower.append(float(item["lower"])); var_upper.append(float(item["upper"])); var_type.append('R'); cat_maps.append(None)
        elif t == "int": var_lower.append(int(item["lower"])); var_upper.append(int(item["upper"])); var_type.append('I'); cat_maps.append(None)
        elif t == "bool": var_lower.append(0); var_upper.append(1); var_type.append('I'); cat_maps.append(("__bool__", [False, True]))
        elif t == "cat": choices = list(item["choices"]); var_lower.append(0); var_upper.append(len(choices)-1); var_type.append('I'); cat_maps.append((name_list[-1], choices))
        else: raise ValueError(f"Unknown type {t}")

    class BB(rbfopt.RbfoptUserBlackBox):
        def __init__(self):
            super().__init__(dim, var_lower, var_upper, var_type, self.evaluate)
            self.eval_count = 0
            self.best_result = None
            self.best_config = None
        def _decode(self, x):
            d = {}
            for idx, v in enumerate(x):
                t = types[idx]; name = name_list[idx]
                if t == "float": d[name] = float(v)
                elif t == "int": d[name] = int(round(v))
                elif t == "bool": d[name] = bool(int(round(v)))
                elif t == "cat": _, choices = cat_maps[idx]; d[name] = choices[int(round(v))]
            return d
        def evaluate(self, x, is_integer=None):
            self.eval_count += 1
            if self.eval_count > max_evaluations:
                return float('inf')  # Stop if exceeded max evaluations

            cfg = self._decode(x)

            print(f"\n### EVALUATION {self.eval_count}/{max_evaluations} ###")
            print(f"Running SCIP with tau={tau}s...")

            out = runner_fn(cfg, instance, tau)
            name = Path(instance).stem
            per_m = {name: out}

            # Get detailed T_infinity computation info
            from utilities.est_time import compute_t_infinity_surrogate, extract_log_samples
            log_path = out.get("log_path", "")
            est_result = {}
            terminal_times = []
            if log_path and os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()
                est_result = compute_t_infinity_surrogate(log_text, tau, out, silent=True)

                # Extract terminal sample times
                samples = extract_log_samples(log_text)
                if samples:
                    # Get last 10% of samples as terminal times
                    terminal_count = max(1, len(samples) // 10)
                    terminal_samples = samples[-terminal_count:]
                    terminal_times = [s.get('t_i', 0) for s in terminal_samples]

            tinf_cand = per_instance_T_infty(per_m, tau=tau)
            baseline_tinf = tinf_base.get(name, tau)
            candidate_tinf = tinf_cand.get(name, tau)
            rhat = r_hat_ratio(tinf_cand, tinf_base, cap=1e3)

            print(f"Final R_hat: {rhat:.4f}")

            # Track best result for final output
            if self.best_result is None or rhat < float(self.best_result.get('r_hat', float('inf'))):
                self.best_result = {
                    'r_hat': rhat,
                    'scip_result': out,
                    'instance': instance,
                    'cfg': cfg
                }
                self.best_config = cfg

            return float(rhat)

    # rbfopt 4.x uses 'rand_seed' instead of 'random_seed'
    print(f"Starting RBFOpt optimization: {dim}D parameter space")
    print("RBFOpt optimization starting...")

    # Suppress RBFOpt's verbose output
    import sys
    import contextlib
    from io import StringIO

    settings = rbfopt.RbfoptSettings(max_evaluations=int(max_evaluations), rand_seed=int(seed))
    bb = BB(); algo = rbfopt.RbfoptAlgorithm(settings, bb)

    start = time.time()
    # Allow all output including T_infinity debug prints to show
    best_val, best_x, iters, evals, _ = algo.optimize()
    total_time = time.time() - start
    print(f"\nRBFOpt completed: {min(evals, max_evaluations)} evaluations in {iters} iterations ({total_time:.1f}s total)")

    best_cfg = bb._decode(best_x); best_L = float(best_val)
    print(f"\n=== OPTIMIZATION SUMMARY ===")
    print(f"Best R_hat: {best_L:.4f}")
    print(f"Best configuration: {best_cfg}")
    print(f"Total optimization time: {total_time:.1f}s")

    trials_df = pd.DataFrame([{"trial": -1, "r_hat": float(best_L), "config": json.dumps(best_cfg), "total_time": total_time}])

    out = Path(out_dir) / f"{tag}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    (out / "best_R_hat.txt").write_text(str(best_L))

    # Use the actual best result from the optimization evaluations
    name = Path(instance).stem
    if bb.best_result:
        final_result = bb.best_result['scip_result']
        print(f"\nUsing best evaluation result:")
        print(f"  Best R_hat: {bb.best_result['r_hat']:.4f}")
        print(f"  From config: {bb.best_result['cfg']}")
    else:
        # Fallback if no best result tracked
        final_result = {"status": "no_evaluations", "solve_time": 0, "log_path": ""}

    per_m = {name: final_result}
    per_logs = [{"instance": instance, **final_result}]
    (out / "per_instance.json").write_text(json.dumps(per_logs, indent=2))
    tinf_best = per_instance_T_infty(per_m, tau=tau)
    (out / "tinf_candidate.json").write_text(json.dumps(tinf_best, indent=2))
    # Unified CSV schema
    trials_df.to_csv(out / "trials.csv", index=False)
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
            "method": "rbfopt",
            "r_hat": float(best_L),
            "best_config": best_cfg,
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

    return best_cfg, float(best_L), trials_df, tinf_best
