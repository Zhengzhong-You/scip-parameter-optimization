# SolverMind: LLM Tuner + SMAC + RBFOpt (Unified)

This repo provides a unified codebase to run three methods for SCIP parameter tuning:
- SolverMind (LLM‑guided closed‑loop tuning)
- SMAC baseline
- RBFOpt baseline

All three share a single set of utilities for scoring (R_hat via T_infty), running SCIP via CLI, dataset discovery/splits, whitelist management, and log parsing. If you change scoring or runner behavior, you change it once under `src/utilities` and all methods pick it up.

## Layout

- `src/`
  - `utilities/` — shared modules used by all methods
    - `scoring.py` — σ(p,i) and Ĵ_S definitions (single source of truth)
    - `runner.py` — runs SCIP via CLI, captures logs/metrics
    - `datasets.py` — instance discovery and train/test splits
    - `whitelist.py` — whitelist regimes (minimal/curated/full) and YAML loader
    - `scip_cli.py` — direct helpers for invoking SCIP
    - `logs.py` — SCIP log parsing and compact summaries
    - `whitelists/` — example YAML whitelists (curated/minimal)
  - `solvermind/` — LLM tuner (pipeline, components, GPT prompts/callers, CLI)
  - `smac/` — SMAC baseline (ConfigSpace mapping, optimizer wrapper, CLI)
  - `rbfopt/` — RBFOpt baseline (variable mapping, optimizer wrapper, CLI)
- `configs/experiment.yaml` — example baseline config (paths, objective, budgets)
- `instance/` — example instances folder (adjust to your data)

## Requirements

- Python 3.9+
- SCIP CLI installed and available on PATH (or set `SCIP_BIN`)
- LLM tuner: `openai` (Responses API)
- Baselines: `smac`, `rbfopt`, `ConfigSpace`, `pandas`

Install the extras you plan to use:
- `pip install openai pandas smac rbfopt ConfigSpace`

## Environment

- Set `PYTHONPATH` so Python can import from `src/`:
  - `export PYTHONPATH=$(pwd)/src:$PYTHONPATH`
- Optional (LLM): `export OPENAI_API_KEY=...`

## Quick Start

SolverMind (LLM), single instance
- `python -m solvermind.cli --instance path/to/instance.mps --time-limit 600 --trials 10 --outdir runs/single`
- Outputs under `runs/single/<instance>/` (`batch_trials.csv`, `per_instance_metrics.csv`, params, logs).

SolverMind (LLM), AC (train and test)
- `python -m solvermind.cli --instances-dir data/cvrp --mode ac --time-limit 900 --trials 16 --max-edits 3 --outdir runs/cvrp`
- Train artifacts under `runs/cvrp/ac_train/` and a test summary at `runs/cvrp/ac_summary.json`.

ISAC (LLM, per-instance tuning)
- `python -m solvermind.cli --instances-dir data/cvrp --mode isac --time-limit 900 --trials 16 --max-edits 3 --outdir runs/isac_cvrp`

SMAC baseline
- `python -m smac.cli --config configs/experiment.yaml --whitelist src/utilities/whitelists/curated.yaml`
  - Outputs under `runs/baselines/smac_*`: `trials.csv` (r_hat, config, total_time), `per_instance.csv` (T_infty, status, etc.), and JSONs.
  - Summary: `summary.json` (method, r_hat, best_config, artifact paths, train_count, tau, total_time).

RBFOpt baseline
- `python -m rbfopt.cli --config configs/experiment.yaml --whitelist src/utilities/whitelists/curated.yaml`
  - Outputs under `runs/baselines/rbfopt_*`: `trials.csv`, `per_instance.csv`, and JSONs as above.
  - Summary: `summary.json` (same schema as SMAC).

## Objective (R_hat)

We use an SVB‑based extrapolated time surrogate T_infty derived from logs and define
the batch objective as the capped geomean time ratio relative to a fixed baseline q:

R_hat(p,q) = exp(mean_i log(min(T_infty(p,i)/T_infty(q,i), 1e3))).

- T_infty is computed from solver logs by fitting b(G) ≈ C·φ^G via a linear fit on
  log(Δb/ΔG) vs Ḡ, estimating (C, φ), then extrapolating remaining nodes/time at timeout.
- Code: `src/utilities/logs.py` (estimate_svb_from_log, estimate_remaining_time, compute_T_infty).
- Aggregation: `src/utilities/scoring.py` (r_hat_ratio).

## Whitelist

- Canonical regimes: `src/utilities/whitelist.py` (choose via `--whitelist-regime` in LLM).
- YAML examples for baselines: `src/utilities/whitelists/*.yaml`.

## Notes

- Determinism: pin SCIP version and keep thread counts at defaults; use `--seed` where available.
- Logs: solver logs saved under each instance subfolder; LLM logs at `<outdir>/gpt_reasoning.log`.
- R_hat trajectory: each trial appends `R_HAT: <value>` to the reasoning log; `trials.csv` captures the full trajectory.
- Version check: enforced once per process when running SCIP (required version 9.2.4 by default).
