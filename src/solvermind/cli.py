from __future__ import annotations

import argparse
import os
import json
import csv
import shutil
from typing import List

from .loop import tune_batch
from utilities.datasets import collect_instances, train_test_split


def main():
    parser = argparse.ArgumentParser(description="SolverMind tuner (SCIP + LLM)")
    parser.add_argument("--instance", required=False, help="Single instance file (.mps/.lp/.cip/...) for single-instance tuning")
    parser.add_argument("--instances-dir", required=False, help="Directory containing instance files for AC/ISAC experiments")
    parser.add_argument("--instances", nargs="*", default=None, help="Explicit list of instance file paths (overrides --instances-dir if provided)")
    parser.add_argument("--time-limit", type=float, default=600.0, help="Time budget per run in seconds (tau)")
    parser.add_argument("--trials", type=int, default=16, help="Trial budget K (default 16)")
    parser.add_argument("--max-edits", type=int, default=3, help="Edit cap m (max per-trial changes to apply)")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model to use (default gpt-5)")
    parser.add_argument("--outdir", default="runs", help="Output directory root")
    parser.add_argument("--seed", type=int, default=0, help="Seed shift for SCIP randomization")
    parser.add_argument("--mode", choices=["single", "ac", "ac-train", "ac-test", "isac"], default=None, help="Experiment mode: distributional tuning (ac), train-only, or instance-specific (isac)")
    parser.add_argument("--train-count", type=int, default=None, help="Override number of training instances L (default min(floor(0.3*N), 20))")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop if no improvement for this many trials (0=disabled)")
    parser.add_argument("--early-stop-delta", type=float, default=0.0, help="Minimum improvement in J-hat to reset patience")
    parser.add_argument("--whitelist-regime", choices=["minimal", "curated", "full"], default="curated", help="Whitelist regime for allowed parameters (default curated)")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        if args.instance:
            mode = "single"
        elif args.instances or args.instances_dir:
            mode = "ac"
        else:
            mode = "single"

    os.makedirs(args.outdir, exist_ok=True)

    if mode == "single":
        if not args.instance:
            parser.error("--instance is required for single mode")
        name = os.path.splitext(os.path.basename(args.instance))[0]
        run_dir = os.path.join(args.outdir, name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir, exist_ok=True)
        tb = tune_batch(
            instances=[args.instance],
            time_limit=args.time_limit,
            max_trials=args.trials,
            max_edits=args.max_edits,
            outdir=run_dir,
            gpt_model=args.model,
            seed=args.seed,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
            whitelist_regime=args.whitelist_regime,
        )
        print(f"CLI tuning completed. Trials CSV: {tb['trials_csv']}")
        return

    all_instances = collect_instances(args.instances, args.instances_dir)
    if not all_instances:
        parser.error("No instances found; provide --instances or --instances-dir")

    train, test = train_test_split(all_instances, L=(args.train_count if args.train_count is not None else None))

    if mode in ("ac", "ac-train"):
        run_dir = os.path.join(args.outdir, "ac_train")
        os.makedirs(run_dir, exist_ok=True)
        tb = tune_batch(
            instances=train,
            time_limit=args.time_limit,
            max_trials=args.trials,
            max_edits=args.max_edits,
            outdir=run_dir,
            gpt_model=args.model,
            seed=args.seed,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
            whitelist_regime=args.whitelist_regime,
        )
        print(f"Training completed. Trials CSV: {tb['trials_csv']}")
        print(f"Incumbent R_hat={tb['incumbent']['r_hat']:.6f} at trial {tb['incumbent']['trial']}")

        if mode == "ac" and test:
            from utilities.scoring import r_hat_ratio
            from utilities.runner import run_instance
            from utilities.logs import per_instance_T_infty
            inc = tb["incumbent"]; settings_file = inc.get("param_file")
            # Build params dict from saved .set
            params: dict = {}
            if settings_file and os.path.exists(settings_file):
                with open(settings_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = [x.strip() for x in line.split("=", 1)]
                        if k.startswith("limits/") or k.startswith("randomization/"):
                            continue
                        params[k] = v
            # Compute baseline T_infty on test with default params (q)
            per_base = {}
            out_root = os.path.join(args.outdir, "ac_test")
            os.makedirs(out_root, exist_ok=True)
            for pth in test:
                name = os.path.splitext(os.path.basename(pth))[0]
                outdir_i = os.path.join(out_root, name)
                per_base[name] = run_instance(pth, params={}, time_limit=args.time_limit, outdir=outdir_i, seed=args.seed, trial_id=0)
            tinf_base = per_instance_T_infty(per_base, tau=args.time_limit)
            # Compute candidate T_infty on test
            per_cand = {}
            for pth in test:
                name = os.path.splitext(os.path.basename(pth))[0]
                outdir_i = os.path.join(out_root, name)
                per_cand[name] = run_instance(pth, params=params, time_limit=args.time_limit, outdir=outdir_i, seed=args.seed)
            tinf_cand = per_instance_T_infty(per_cand, tau=args.time_limit)
            rtest = r_hat_ratio(tinf_cand, tinf_base)
            summ = {
                "train_trials_csv": tb["trials_csv"],
                "train_completed_trials": tb.get("completed_trials", 0),
                "train_incumbent": inc,
                "test_count": len(test),
                "test_r_hat": rtest,
            }
            with open(os.path.join(args.outdir, "ac_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2)
            print(f"Testing completed. R_hat(test)={rtest:.6f}.")
        return

    if mode == "isac":
        root = os.path.join(args.outdir, "isac")
        os.makedirs(root, exist_ok=True)
        rows = []
        for p in train:
            name = os.path.splitext(os.path.basename(p))[0]
            run_dir = os.path.join(root, name)
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
            os.makedirs(run_dir, exist_ok=True)
            tb = tune_batch(
                instances=[p],
                time_limit=args.time_limit,
                max_trials=args.trials,
                max_edits=args.max_edits,
                outdir=run_dir,
                gpt_model=args.model,
                seed=args.seed,
                early_stop_patience=args.early_stop_patience,
                early_stop_delta=args.early_stop_delta,
                whitelist_regime=args.whitelist_regime,
            )
            rows.append({
                "instance": name,
                "completed_trials": tb.get("completed_trials", 0),
                "incumbent_r_hat": tb["incumbent"]["r_hat"],
                "param_file": tb["incumbent"].get("param_file"),
            })
            print(f"ISAC {name}: R_hat={tb['incumbent']['r_hat']:.6f}")
        isac_csv = os.path.join(root, "isac_summary.csv")
        with open(isac_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["instance", "completed_trials", "incumbent_r_hat", "param_file"])
            writer.writeheader(); writer.writerows(rows)
        print(f"Wrote ISAC summary -> {isac_csv}")


if __name__ == "__main__":
    main()
