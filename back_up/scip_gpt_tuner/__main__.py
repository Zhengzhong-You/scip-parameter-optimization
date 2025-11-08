
import argparse, os, json, csv, time, math, shutil
from typing import List
from .loop.tune_batch import tune_batch
from .params.get_whitelist import get_whitelist
from .score.solvermind_score import j_hat_sigma

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
    # Removed: --show-reinforced, --dry-run, --debug (simplified CLI)
    parser.add_argument("--mode", choices=["single", "ac", "ac-train", "ac-test", "isac"], default=None, help="Experiment mode: distributional tuning (ac), train-only, test-only, or instance-specific (isac)")
    parser.add_argument("--train-count", type=int, default=None, help="Override number of training instances L (default min(floor(0.3*N), 20))")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop if no improvement for this many trials (0=disabled)")
    parser.add_argument("--early-stop-delta", type=float, default=0.0, help="Minimum improvement in J-hat to reset patience")
    parser.add_argument("--whitelist-regime", choices=["minimal", "curated", "full"], default="curated", help="Whitelist regime for allowed parameters (default curated)")
    # CLI is the only supported backend now (no flag needed)
    args = parser.parse_args()

    # No show-reinforced: always proceed to run

    # Resolve mode
    mode = args.mode
    if mode is None:
        if args.instance:
            mode = "single"
        elif args.instances or args.instances_dir:
            mode = "ac"
        else:
            mode = "single"

    # Use dataset utilities
    from .experiments.dataset import collect_instances as _collect_instances, train_test_split as _train_test_split
    def collect_instances() -> List[str]:
        return _collect_instances(args.instances, args.instances_dir)

    os.makedirs(args.outdir, exist_ok=True)

    if mode == "single":
        if not args.instance:
            parser.error("--instance is required for single mode")
        # Use the batch pipeline for a single instance with CLI backend
        name = os.path.splitext(os.path.basename(args.instance)) [0] if args.instance else "dry-run"
        run_dir = os.path.join(args.outdir, name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir, exist_ok=True)
        tb = tune_batch(
            instances=([args.instance] if args.instance else []),
            time_limit=args.time_limit,
            max_trials=args.trials,
            max_edits=args.max_edits,
            outdir=run_dir,
            gpt_model=args.model,
            seed=args.seed,
            dry_run=False,
            debug=False,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
            whitelist_regime=args.whitelist_regime,
        )
        print(f"CLI tuning completed. Trials CSV: {tb['trials_csv']}")
        return

    # Modes using batches of instances
    all_instances = collect_instances()
    if not all_instances:
        parser.error("No instances found; provide --instances or --instances-dir")

    N = len(all_instances)
    L = args.train_count if args.train_count is not None else None
    train, test = _train_test_split(all_instances, L=L)

    # Debug logging removed

    # Train-only (AC) or full AC (train+test)
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
            dry_run=False,
            debug=False,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
            whitelist_regime=args.whitelist_regime,
        )
        print(f"Training completed. Trials CSV: {tb['trials_csv']}")
        print(f"Incumbent J_hat={tb['incumbent']['j_hat']:.6f} at trial {tb['incumbent']['trial']}")

        # AC testing if requested or in full AC mode
        if mode == "ac" and test:
            # Evaluate incumbent on test set
            inc = tb["incumbent"]
            params = inc.get("params", {})
            per_m = {}
            # Evaluate incumbent on test set using CLI runner
            from .components.runner_cli import run_batch_cli
            per_m = run_batch_cli(
                instances=test,
                settings_file=inc.get("param_file"),
                time_limit=args.time_limit,
                outdir=os.path.join(args.outdir, "ac_test"),
                seed=args.seed,
                trial_id=None,  # Testing phase, no specific trial number
            )
            jtest, _ = j_hat_sigma(per_m, tau=args.time_limit)

            # Persist a simple summary JSON
            summ = {
                "train_trials_csv": tb["trials_csv"],
                "train_completed_trials": tb.get("completed_trials", 0),
                "train_incumbent": inc,
                "test_count": len(per_m),
                "test_j_hat": jtest,
            }
            with open(os.path.join(args.outdir, "ac_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2)
            print(f"Testing completed. J_hat(test)={jtest:.6f}. Summary: {os.path.join(args.outdir, 'ac_summary.json')}")
        return

    if mode == "ac-test":
        parser.error("ac-test mode requires a previously trained configuration; not implemented as a standalone CLI. Use mode=ac.")

    if mode == "isac":
        # Per-instance tuning with M=1
        root = os.path.join(args.outdir, "isac")
        os.makedirs(root, exist_ok=True)
        isac_rows = []
        from datetime import datetime
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
                dry_run=False,
                debug=False,
                early_stop_patience=args.early_stop_patience,
                early_stop_delta=args.early_stop_delta,
                whitelist_regime=args.whitelist_regime,
            )
            isac_rows.append({
                "instance": name,
                "completed_trials": tb.get("completed_trials", 0),
                "incumbent_j_hat": tb["incumbent"]["j_hat"],
                "param_file": tb["incumbent"].get("param_file"),
            })
            print(f"ISAC {name}: J_hat={tb['incumbent']['j_hat']:.6f}")
        # Write ISAC summary CSV
        isac_csv = os.path.join(root, "isac_summary.csv")
        with open(isac_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["instance", "completed_trials", "incumbent_j_hat", "param_file"])
            writer.writeheader()
            writer.writerows(isac_rows)
        print(f"Wrote ISAC summary -> {isac_csv}")
        return

    if __name__ == "__main__":
        main()
