#!/usr/bin/env python3
"""
Unified Optimizer CLI
====================

Single command-line interface for SMAC, RBFOpt, and SolverMind optimization.
Provides identical command syntax for all three methods.

Usage:
    python -m utilities.optimizer_cli smac --config ... --whitelist ... --instance ...
    python -m utilities.optimizer_cli rbfopt --config ... --whitelist ... --instance ...
    python -m utilities.optimizer_cli solvermind --config ... --whitelist ... --instance ...
"""

import argparse
import sys
import os

def main():
    # Create main parser
    parser = argparse.ArgumentParser(description="Unified optimizer CLI for SCIP parameter tuning")
    subparsers = parser.add_subparsers(dest='method', help='Optimization method')

    # Common arguments for all methods
    def add_common_args(subparser):
        subparser.add_argument("--config", required=True, help="Configuration YAML file")
        subparser.add_argument("--whitelist", required=True, help="Whitelist regime: curated|minimal|full")
        subparser.add_argument("--instance", required=True, help="Path to a single instance file to tune on")

    # SMAC subcommand
    smac_parser = subparsers.add_parser('smac', help='Run SMAC optimization')
    add_common_args(smac_parser)

    # RBFOpt subcommand
    rbfopt_parser = subparsers.add_parser('rbfopt', help='Run RBFOpt optimization')
    add_common_args(rbfopt_parser)

    # SolverMind subcommand
    solvermind_parser = subparsers.add_parser('solvermind', help='Run SolverMind LLM-based optimization')
    add_common_args(solvermind_parser)

    args = parser.parse_args()

    if not args.method:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate optimizer
    if args.method == 'smac':
        # Import SMAC dependencies with path manipulation to avoid conflicts
        SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        orig_path = list(sys.path)
        try:
            if SRC_DIR in sys.path:
                sys.path.remove(SRC_DIR)
            import ConfigSpace as CS
            from smac import HyperparameterOptimizationFacade as HPOF
            from smac import Scenario
        finally:
            sys.path[:] = orig_path
            if SRC_DIR not in sys.path:
                sys.path.append(SRC_DIR)

        from .optimizer_runner import run_single_instance_optimization

        # Inline SMAC optimization function
        def run_smac_unified(whitelist, instances, runner_fn, tau, tinf_base,
                            n_trials=10, seed=0, out_dir="./runs"):
            cs = CS.ConfigurationSpace()
            for item in whitelist:
                name = item["name"]
                typ = item["type"]
                if typ == "float":
                    hp = CS.hyperparameters.UniformFloatHyperparameter(
                        name, lower=float(item["lower"]), upper=float(item["upper"]),
                        log=bool(item.get("log", False))
                    )
                elif typ == "int":
                    hp = CS.hyperparameters.UniformIntegerHyperparameter(
                        name, lower=int(item["lower"]), upper=int(item["upper"]))
                elif typ == "bool":
                    choices = item.get("choices", [False, True])
                    hp = CS.hyperparameters.CategoricalHyperparameter(
                        name, choices=[bool(x) for x in choices])
                elif typ == "cat":
                    choices = item["choices"]
                    hp = CS.hyperparameters.CategoricalHyperparameter(name, choices=choices)
                else:
                    raise ValueError(f"Unknown type in whitelist: {typ}")
                cs.add_hyperparameter(hp)

            def objective(cfg_cs: CS.Configuration, seed: int) -> float:
                d = {k: cfg_cs[k] for k in cfg_cs}
                per_m = {}
                for inst in instances:
                    out = runner_fn(d, inst, tau)
                    name = os.path.splitext(os.path.basename(inst))[0]
                    per_m[name] = out

                from .logs import per_instance_T_infty
                from .scoring import r_hat_ratio
                tinf_cand = per_instance_T_infty(per_m, tau=tau)
                rhat = r_hat_ratio(tinf_cand, tinf_base, cap=1e3)
                return float(rhat)

            scenario = Scenario(
                cs,
                n_trials=int(n_trials),
                seed=int(seed),
                deterministic=True,
                output_directory=os.path.join(out_dir, "smac_internal"),
            )

            smac = HPOF(scenario, objective)
            incumbent = smac.optimize()
            best_dict = {k: incumbent[k] for k in incumbent}

            # Evaluate best config on all instances to get final results
            per_m = {}
            for inst in instances:
                out = runner_fn(best_dict, inst, tau)
                name = os.path.splitext(os.path.basename(inst))[0]
                per_m[name] = out

            from .logs import per_instance_T_infty
            from .scoring import r_hat_ratio
            tinf_best = per_instance_T_infty(per_m, tau=tau)
            best_rhat = r_hat_ratio(tinf_best, tinf_base, cap=1e3)

            # Clean up SMAC internal output
            try:
                import shutil
                shutil.rmtree(os.path.join(out_dir, "smac_internal"), ignore_errors=True)
                for d in os.listdir("."):
                    if d.startswith("smac3-output") or d.startswith("smac-output"):
                        shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

            return best_dict, float(best_rhat), None, tinf_best

        run_single_instance_optimization(
            method="smac",
            optimizer_func=run_smac_unified,
            config_path=args.config,
            whitelist_regime=args.whitelist,
            instance_path=args.instance
        )

    elif args.method == 'rbfopt':
        from .optimizer_runner import run_single_instance_optimization
        from rbfopt.baseline import run_rbfopt

        run_single_instance_optimization(
            method="rbfopt",
            optimizer_func=run_rbfopt,
            config_path=args.config,
            whitelist_regime=args.whitelist,
            instance_path=args.instance
        )

    elif args.method == 'solvermind':
        # Import SolverMind optimization function
        try:
            from solvermind.baseline import run_solvermind
            from .optimizer_runner import run_single_instance_optimization

            run_single_instance_optimization(
                method="solvermind",
                optimizer_func=run_solvermind,
                config_path=args.config,
                whitelist_regime=args.whitelist,
                instance_path=args.instance
            )
        except ImportError:
            print("❌ SolverMind optimization not available. Please ensure solvermind module is properly installed.")
            sys.exit(1)


if __name__ == "__main__":
    main()