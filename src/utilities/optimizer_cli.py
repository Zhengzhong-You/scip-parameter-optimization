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
        from .optimizer_runner import run_single_instance_optimization
        from smac_tune.baseline import run_smac

        run_single_instance_optimization(
            method="smac",
            optimizer_func=run_smac,
            config_path=args.config,
            whitelist_regime=args.whitelist,
            instance_path=args.instance
        )

    elif args.method == 'rbfopt':
        from .optimizer_runner import run_single_instance_optimization
        from rbfopt_tune.baseline import run_rbfopt

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
            print("SolverMind optimization not available. Please ensure solvermind module is properly installed.")
            sys.exit(1)


if __name__ == "__main__":
    main()