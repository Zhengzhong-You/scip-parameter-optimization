import argparse, json, os, yaml
from solvermind_baselines.datasets import discover_instances, split_instances, explicit_split
from solvermind_baselines.space import load_whitelist
from solvermind_baselines.baselines.smac_baseline import run_smac
from solvermind_baselines.runner_dummy import run_instance as dummy_run
from solvermind_baselines.runner_scip import run_instance as scip_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--whitelist", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    wl  = load_whitelist(args.whitelist)

    data = cfg["data"]
    if data.get("use_explicit_split", False):
        train, test = explicit_split(data.get("train", []), data.get("test", []))
    else:
        files = discover_instances(data["instances_dir"], data["pattern"])
        train, test = split_instances(files, data.get("train_fraction", 0.3), data.get("train_cap", 20))

    runner_kind = cfg["runner"]["kind"]
    tau = float(cfg["runner"]["tau"])

    if runner_kind == "dummy":
        runner_fn = dummy_run; extra = {}
    elif runner_kind == "scip":
        runner_fn = scip_run
        extra = dict(
            bin = cfg["runner"]["scip"].get("bin"),
            workdir = cfg["runner"]["scip"].get("workdir"),
            preset_params = cfg["runner"]["scip"].get("preset_params", []),
            extra_flags = cfg["runner"]["scip"].get("extra_flags", []),
        )
    else:
        raise ValueError(f"Unknown runner kind: {runner_kind}")

    obj_cfg = cfg["objective"]
    L_cfg   = dict(f1=obj_cfg["f1"], f2=obj_cfg["f2"], f3=obj_cfg["f3"], f4=obj_cfg["f4"])
    epsilon = float(obj_cfg.get("epsilon", 1e-9))
    N_max   = float(obj_cfg.get("N_max", 1e6))

    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    best_cfg, best_L, trials_df = run_smac(
        whitelist=wl,
        runner_fn=lambda p,i,tau_: runner_fn(p,i,tau_, **extra),
        instances=train,
        tau=tau,
        f_cfg=L_cfg,
        epsilon=epsilon,
        N_max=N_max,
        n_trials=cfg["smac"]["n_trials"],
        seed=cfg["smac"]["seed"],
        out_dir=out_dir,
        tag="smac"
    )

    print("Best L(p):", best_L)
    print("Best config:", json.dumps(best_cfg, indent=2))

if __name__ == "__main__":
    main()
