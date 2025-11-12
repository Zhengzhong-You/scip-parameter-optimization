import argparse, json, os, yaml
from utilities.datasets import discover_instances
from utilities.whitelist import get_typed_whitelist
from utilities.runner import run_instance as scip_run
from utilities.logs import per_instance_T_infty
from .baseline import run_smac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--whitelist", required=True, help="Whitelist regime: curated|minimal|full")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    # Build typed whitelist directly from the programmatic curated list to keep all methods in sync
    regime = (args.whitelist or "curated").strip().lower()
    if regime not in ("curated", "minimal", "full"):
        raise SystemExit(f"Unsupported whitelist regime: {regime}")
    wl = get_typed_whitelist(regime=regime)

    data = cfg["data"]
    # Use the full set of discovered instances for tuning (no train/test split)
    if data.get("use_explicit_split", False):
        instances = data.get("train", [])
    else:
        instances = discover_instances(data["instances_dir"], data["pattern"])

    tau = float(cfg["runner"]["tau"])
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    def runner_fn(pcfg, instance, tau_):
        # Run via shared runner; return full metrics with log_path
        name = os.path.splitext(os.path.basename(instance))[0]
        outdir_i = os.path.join(out_dir, name)
        return scip_run(instance, params=pcfg, time_limit=tau_, outdir=outdir_i)

    # Precompute baseline T_infty for defaults on the target instances
    per_base = {}
    for pth in instances:
        nm = os.path.splitext(os.path.basename(pth))[0]
        outdir_i = os.path.join(out_dir, nm)
        per_base[nm] = scip_run(pth, params={}, time_limit=tau, outdir=outdir_i, trial_id=0)
    tinf_base = per_instance_T_infty(per_base, tau=tau)

    best_cfg, best_Rhat, _trials, _tinf = run_smac(
        whitelist=wl,
        runner_fn=runner_fn,
        instances=instances,
        tau=tau,
        tinf_base=tinf_base,
        n_trials=cfg["smac"]["n_trials"],
        seed=cfg["smac"]["seed"],
        out_dir=out_dir,
        tag="smac"
    )

    print("Best R_hat(p,q):", best_Rhat)
    print("Best config:", json.dumps(best_cfg, indent=2))


if __name__ == "__main__":
    main()
