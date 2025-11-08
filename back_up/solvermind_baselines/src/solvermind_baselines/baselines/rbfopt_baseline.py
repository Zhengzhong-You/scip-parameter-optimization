from typing import Dict, Any, List, Tuple
import json, time
import pandas as pd
from pathlib import Path

import rbfopt

from ..objective import evaluate_config_on_set
from ..space import rbfopt_variables

def run_rbfopt(whitelist: List[Dict, Any], runner_fn, instances: List[str], tau: float,
               f_cfg: Dict[str, Any], epsilon: float, N_max: float,
               max_evaluations: int = 100, seed: int = 0, out_dir: str = "./runs", tag: str = "rbfopt"
               ) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
    dim, var_lower, var_upper, var_type, cat_maps = rbfopt_variables(whitelist)
    name_list = [item["name"] for item in whitelist]
    types = [item["type"] for item in whitelist]

    class BB(rbfopt.RbfoptUserBlackBox):
        def __init__(self):
            super().__init__(dim, var_lower, var_upper, var_type)
        def _decode(self, x):
            d = {}
            for idx, v in enumerate(x):
                t = types[idx]; name = name_list[idx]
                if t == "float":
                    d[name] = float(v)
                elif t == "int":
                    d[name] = int(round(v))
                elif t == "bool":
                    d[name] = bool(int(round(v)))
                elif t == "cat":
                    _, choices = cat_maps[idx]; d[name] = choices[int(round(v))]
                else:
                    raise ValueError(f"Unknown type {t}")
            return d
        def evaluate(self, x, is_integer):
            cfg = self._decode(x)
            L, _ = evaluate_config_on_set(cfg, instances, runner_fn, tau, f_cfg, epsilon, N_max)
            return float(L)

    settings = rbfopt.RbfoptSettings(max_evaluations=int(max_evaluations), random_seed=int(seed))
    bb = BB(); algo = rbfopt.RbfoptAlgorithm(settings, bb)

    start = time.time()
    best_val, best_x, iters, evals, _ = algo.optimize()
    total_time = time.time() - start

    best_cfg = bb._decode(best_x); best_L = float(best_val)
    trials_df = pd.DataFrame([{"trial": -1, "L": best_L, "config": json.dumps(best_cfg), "total_time": total_time}])

    out = Path(out_dir) / f"{tag}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    (out / "best_L.txt").write_text(str(best_L))
    _, per_logs = evaluate_config_on_set(best_cfg, instances, runner_fn, tau, f_cfg, epsilon, N_max)
    (out / "per_instance.json").write_text(json.dumps(per_logs, indent=2))

    return best_cfg, best_L, trials_df
