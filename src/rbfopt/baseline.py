from typing import Dict, Any, List, Tuple
import json, time
import pandas as pd
from pathlib import Path
import sys, os, importlib, site

# Avoid name collision with this package's own module 'rbfopt'.
# Dynamically import the third-party 'rbfopt' from site-packages.
def _import_third_party_rbfopt():
    """Import the pip-installed 'rbfopt' by temporarily removing this project's 'src' and the local package from sys.modules.

    This allows 'import rbfopt' to resolve to site-packages even though we have a local package named 'rbfopt'.
    """
    # Compute this repo's src path
    repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Save and modify sys.path and sys.modules
    orig_path = list(sys.path)
    orig_mod = sys.modules.get('rbfopt')
    try:
        # Remove our src from path if present
        try:
            sys.path.remove(repo_src)
        except ValueError:
            pass
        # Drop the local package binding so import finds site-packages
        if 'rbfopt' in sys.modules:
            del sys.modules['rbfopt']
        return importlib.import_module('rbfopt')
    finally:
        # Restore sys.path (do not restore sys.modules['rbfopt'] to avoid re-shadowing during this module's lifetime)
        sys.path[:] = orig_path

rbfopt = _import_third_party_rbfopt()

from utilities.logs import per_instance_T_infty
from utilities.scoring import r_hat_ratio


def run_rbfopt(whitelist: List[Dict[str, Any]], runner_fn, instances: List[str], tau: float,
               tinf_base: Dict[str, float],
               max_evaluations: int = 100, seed: int = 0, out_dir: str = "./runs", tag: str = "rbfopt"
               ) -> Tuple[Dict[str, Any], float, pd.DataFrame, Dict[str, float]]:
    # Build RBFOpt variable space
    dim = len(whitelist)
    var_lower = []; var_upper = []; var_type = ""; types = []; name_list = []; cat_maps = []
    for item in whitelist:
        name_list.append(item["name"]); t = item["type"]; types.append(t)
        if t == "float": var_lower.append(float(item["lower"])); var_upper.append(float(item["upper"])); var_type += "R"; cat_maps.append(None)
        elif t == "int": var_lower.append(int(item["lower"])); var_upper.append(int(item["upper"])); var_type += "I"; cat_maps.append(None)
        elif t == "bool": var_lower.append(0); var_upper.append(1); var_type += "I"; cat_maps.append(("__bool__", [False, True]))
        elif t == "cat": choices = list(item["choices"]); var_lower.append(0); var_upper.append(len(choices)-1); var_type += "I"; cat_maps.append((name_list[-1], choices))
        else: raise ValueError(f"Unknown type {t}")

    class BB(rbfopt.RbfoptUserBlackBox):
        def __init__(self): super().__init__(dim, var_lower, var_upper, var_type)
        def _decode(self, x):
            d = {}
            for idx, v in enumerate(x):
                t = types[idx]; name = name_list[idx]
                if t == "float": d[name] = float(v)
                elif t == "int": d[name] = int(round(v))
                elif t == "bool": d[name] = bool(int(round(v)))
                elif t == "cat": _, choices = cat_maps[idx]; d[name] = choices[int(round(v))]
            return d
        def evaluate(self, x, is_integer):
            cfg = self._decode(x)
            per_m: Dict[str, Dict[str, Any]] = {}
            for inst in instances:
                out = runner_fn(cfg, inst, tau)
                name = Path(inst).stem
                per_m[name] = out
            tinf_cand = per_instance_T_infty(per_m, tau=tau)
            rhat = r_hat_ratio(tinf_cand, tinf_base, cap=1e3)
            return float(rhat)

    # rbfopt 4.x uses 'rand_seed' instead of 'random_seed'
    settings = rbfopt.RbfoptSettings(max_evaluations=int(max_evaluations), rand_seed=int(seed))
    bb = BB(); algo = rbfopt.RbfoptAlgorithm(settings, bb)

    start = time.time()
    best_val, best_x, iters, evals, _ = algo.optimize()
    total_time = time.time() - start

    best_cfg = bb._decode(best_x); best_L = float(best_val)
    trials_df = pd.DataFrame([{"trial": -1, "r_hat": float(best_L), "config": json.dumps(best_cfg), "total_time": total_time}])

    out = Path(out_dir) / f"{tag}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    (out / "best_R_hat.txt").write_text(str(best_L))

    per_m = {}
    per_logs = []
    for inst in instances:
        out = runner_fn(best_cfg, inst, tau)
        name = Path(inst).stem
        per_m[name] = out
        per_logs.append({"instance": inst, **out})
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
            "train_count": len(instances),
            "tau": float(tau),
            "total_time": float(total_time),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception:
        pass

    return best_cfg, float(best_L), trials_df, tinf_best
