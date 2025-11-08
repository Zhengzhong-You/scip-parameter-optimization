#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# Activate scip_env environment
source scip_env/bin/activate

# Add src/ to Python path
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "Running SolverMind (LLM) on a single instance (quick smoke test)..."
python3 -m solvermind.cli \
  --instance ./instance/cvrp/mps/XML100_3346_25_50.mps.gz \
  --time-limit 10 \
  --trials 3 \
  --max-edits 3 \
  --outdir runs/single_quick \
  --whitelist-regime curated

# Optional: distribution-level (AC) quick example — uncomment to run
# echo "Running SolverMind (LLM) AC mode on a small batch (quick) ..."
# python3 -m solvermind.cli \
#   --instances-dir ./instance/cvrp/mps \
#   --mode ac \
#   --time-limit 60 \
#   --trials 3 \
#   --max-edits 3 \
#   --outdir runs/ac_quick \
#   --whitelist-regime curated

# Optional: SMAC baseline — uses configs/experiment.yaml. For a fast run,
# edit tau/n_trials in the config before uncommenting.
# echo "Running SMAC baseline (see runs/baselines/ for outputs) ..."
# python3 -m smac.cli \
#   --config configs/experiment.yaml \
#   --whitelist src/utilities/whitelists/curated.yaml

# Optional: RBFOpt baseline — uses configs/experiment.yaml. For a fast run,
# edit tau/max_evaluations in the config before uncommenting.
# echo "Running RBFOpt baseline (see runs/baselines/ for outputs) ..."
# python3 -m rbfopt.cli \
#   --config configs/experiment.yaml \
#   --whitelist src/utilities/whitelists/curated.yaml
