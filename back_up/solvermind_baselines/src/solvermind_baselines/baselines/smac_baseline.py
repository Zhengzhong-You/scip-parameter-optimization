from typing import Dict, Any, List, Tuple
import json, time
import pandas as pd
from pathlib import Path

import ConfigSpace as CS
from smac import HyperparameterOptimizationFacade, Scenario

from ..objective import evaluate_config_on_set
from ..space import build_configspace

def run_smac(whitelist: List[Dict[str, Any]], runner_fn, instances: List[str], tau: float,
             f_cfg: Dict[str, Any], epsilon: float, N_max: float,
             n_trials: int = 100, seed: int = 0, out_dir: str = "./runs", tag: str = "smac"
             ) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
    cs, _ = build_configspace(whitelist)
    def objective(cfg: CS.Configuration) -> float:
        d = {k: cfg[k] for k in cfg}
        L, _ = evaluate_config_on_set(d, instances, runner_fn, tau, f_cfg, epsilon, N_max)
        return L

    scenario = Scenario(cs, n_trials=int(n_trials), seed=int(seed), deterministic=True)
    smac = HyperparameterOptimizationFacade(scenario, objective)

    start = time.time()
    incumbent = smac.optimize()
    total_time = time.time() - start

    best_dict = {k: incumbent[k] for k in incumbent}
    best_L, per_logs = evaluate_config_on_set(best_dict, instances, runner_fn, tau, f_cfg, epsilon, N_max)

    trials_df = pd.DataFrame([{"trial": -1, "L": best_L, "config": json.dumps(best_dict), "total_time": total_time}])

    out = Path(out_dir) / f"{tag}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_config.json").write_text(json.dumps(best_dict, indent=2))
    (out / "best_L.txt").write_text(str(best_L))
    (out / "per_instance.json").write_text(json.dumps(per_logs, indent=2))

    return best_dict, best_L, trials_df
