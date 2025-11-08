# SolverMind (LLM Tuning + Baselines)

LLM-guided parameter tuning for the SCIP solver, implementing the SolverMind closed-loop design described in main.tex. The repository now includes unified, shared utilities and three experiment entrypoints: SolverMind (LLM), SMAC, and RBFOpt. All share a common runner, scoring, dataset utilities, and whitelist helpers to avoid duplication.

Highlights
- Distribution-level tuning: minimizes the empirical geometric-mean objective Ĵ_S across a training batch S.
- Safe parameter space: only allows parameters from a curated whitelist and meta-toggles (emphasis, heuristics, presolve, separating).
- Schema and caps: LLM responses must be strict JSON; we enforce a max edit cap m per trial and reject default-equal proposals.
- Reproducible outputs: detailed logs, per-trial CSVs, and per-instance metrics suitable for analysis and paper figures.


## Installation

Prereqs
- Python 3.9+
- SCIP CLI installed and available on PATH (or set `SCIP_BIN` to the `scip` binary)
- OpenAI Python SDK (for the LLM; may be replaced by your own caller)

Install (editable)
- `pip install -e .`
- Set your API key: `export OPENAI_API_KEY=...`

Packages (declared in `solvermind_tuner.egg-info/requires.txt`)
- `openai>=1.40.0`
- `pandas>=2.0.0`


## Quick Start

Single instance (SolverMind LLM)
- `python scripts/run_solvermind.py --instance path/to/instance.mps --time-limit 600 --trials 10 --outdir runs/single`.
- Outputs: per-trial files under `runs/single/<instance>/` (trials.csv, params, logs).

AC (distribution-level training + testing)
- `python scripts/run_solvermind.py --instances-dir data/cvrp --mode ac --time-limit 900 --trials 16 --max-edits 3 --outdir runs/cvrp`.
- Train–test split: L = min(floor(0.3*N), 20). Override with `--train-count`.
- Outputs:
  - Train: `runs/cvrp/ac_train/batch_trials.csv`, `per_instance_metrics.csv`, `gpt_reasoning.log`.
  - Test: `runs/cvrp/ac_summary.json` with incumbent and test Ĵ_S.

ISAC (per-instance tuning, M=1)
- `python scripts/run_solvermind.py --instances-dir data/cvrp --mode isac --time-limit 900 --trials 16 --max-edits 3 --outdir runs/isac_cvrp`.
- Outputs per instance under `runs/isac_cvrp/isac/<name>/` and a summary at `runs/isac_cvrp/isac/isac_summary.csv`.

Dry-run mode has been removed. For quick smoke tests, use small `--time-limit` and `--trials`.

Baselines
- SMAC: `python scripts/run_smac.py --config solvermind_baselines/configs/experiment.yaml --whitelist solvermind_baselines/whitelists/curated.yaml`
- RBFOpt: `python scripts/run_rbf.py --config solvermind_baselines/configs/experiment.yaml --whitelist solvermind_baselines/whitelists/curated.yaml`

SCIP CLI notes
- All experiments use the SCIP CLI under the hood (no PySCIPOpt dependency).
- Optional: set `SCIP_BIN=/path/to/scip` if `scip` is not on PATH.


## CLI Reference

Base
- `python scripts/run_solvermind.py` (CLI wrapper calling the SolverMind tuner).

Core flags
- `--mode {single,ac,ac-train,ac-test,isac}`
  - single: single-instance tuning loop
  - ac-train: batch tuning on train set S only
  - ac: train as above, then evaluate incumbent on test set T
  - ac-test: reserved (expected to test a saved config)
  - isac: instance-specific tuning (M=1) per training instance
- `--instance` path to a single model file (single mode)
- `--instances-dir` directory containing instances
- `--instances` explicit list of instance paths (overrides `--instances-dir`)
- `--train-count` L override for train–test split; default L = min(floor(0.3*N), 20)
- `--time-limit` per-run budget τ (seconds)
- `--trials` trial budget K (default 16)
- `--max-edits` m (max per-trial parameter edits; default 3)
- `--model` LLM model id (default `gpt-5`)
- `--seed` shift for SCIP randomization
- `--outdir` output directory root
- `--early-stop-patience`, `--early-stop-delta` numeric early stopping on Ĵ_S
- `--whitelist-regime {minimal,curated,full}` choose the allowed parameter set (default `curated`)

Outputs
- Single mode: instance subfolder under `--outdir` with `trials.csv`, `gpt_reasoning.log`, and per‑trial param files.
- AC train: `ac_train/batch_trials.csv`, `per_instance_metrics.csv`, `gpt_reasoning.log`.
- AC test: `ac_summary.json` with Ĵ_S(test) and incumbent metadata.
- ISAC: `isac/<instance>/...` and `isac/isac_summary.csv`.


## Objective (Production-Ready Scoring)

Per-instance scalarization σ(p,i)
- `σ(p,i) = f1(tier) + f2(t/τ) + f3(gap) + f4(nodes/Nmax)` with caps at 1e3.
- `tier=0` for conclusive statuses (`opt`, `inf`, `unb`), otherwise `tier=1`.
- `gap = |primal-dual| / (1 + |primal|)` if both defined; else 1.
- `f1(x)=min(1e3*x, 1e3)`, `f2(x)=min(x, 1e3)`, `f3(x)=min(exp(x)-1, 1e3)`, `f4(x)=min(1e-6*x, 1e3)`.

Distribution-level objective Ĵ_S(p)
- `Ĵ_S(p) = exp(mean_i log σ(p,i))` (geometric mean) computed across the training set S.
- The tuning agent minimizes Ĵ_S over trials and tracks the incumbent.

Code
- Core: `solvermind/core/scoring.py` (σ, Ĵ_S); thin wrappers in tuner `components/objective.py`.


## Architecture

Top-level
- Pipeline: `solvermind/tuner/pipeline/solvermind.py` (core batch tuning per Algorithm 1)
- Components (plug-and-replace): `solvermind/tuner/components/`
- Shared core for all experiments: `solvermind/core/` (runner, scoring, datasets, whitelist)
  - `features_batch.py`: placeholder (static feature extraction removed)
  - `prompt_builder.py`: constructs LLM input with batch features, history, and max_changes
  - `validator.py`: validates params/meta against whitelist and defaults; enforces edit cap m
  - `objective.py`: σ and Ĵ_S computation glue
  - `stopping.py`: early stopping policy
  - `history.py`: lightweight prompt history store
- GPT: `gpt/build_tuning_prompt.py` and `gpt/call_gpt.py` (OpenAI Responses API)
- Whitelist: `params/get_whitelist.py` (validation in `components/validator.py`)
- Solver execution: `run/run_instance_cli.py`
- Logging and metrics: `log/parse_scip_log.py`, `log/summarize_run_metrics.py`
- Legacy single-instance loop: removed (CLI batch pipeline used for single-instance mode)
- Batch wrapper for pipeline: `loop/tune_batch.py`
- Dataset & splits: `experiments/dataset.py`

LLM constraints and schema
- The prompt asks for strict JSON with keys: `params` (map), `meta` (map with emphasis/heuristics/presolve/separating), and `reasons` (string).
- The module `gpt/build_tuning_prompt.py` includes a `constraints.max_changes` matching `--max-edits`.

Validation and safety
- Whitelist restricts editable parameters to a curated set and meta toggles.
- Unknown names are rejected; default-equal values are rejected (nudging non-default exploration).
- Edit cap: at most `m` individual parameter changes are applied per trial (meta toggles are tracked separately).


## Whitelist
File: `solvermind/tuner/params/get_whitelist.py` (LLM tuner) and YAML loaders in `solvermind/core/whitelist.py` (baselines)
- Regimes (select via `--whitelist-regime`):
  - Full: every tunable parameter discovered via `set default; set save …` (large; for exhaustive tuning)
    - Excludes `limits/*` and thread-related parameters (e.g., names containing `/threads` or starting with `parallel/`).
  - Curated (default): a mid-size list covering primary control knobs:
    - Branching: `branching/scorefunc`, `branching/preferbinary`, and `branching/relpscost/{minreliable,maxreliable,sbiterquot,sbiterofs}`
    - Node selection: `nodeselection/{bestestimate,dfs,bfs}/stdpriority`, `nodeselection/childsel`
    - Cutting planes (global + per-family): `separating/{maxrounds,maxroundsroot,maxcuts,maxcutsroot}`, and for families `{gomory,mir,cmir,flowcover,clique,knapsackcover,oddcycle}/{freq,maxrounds,maxroundsroot}`
    - Presolve: `presolving/{maxrounds,maxrestarts,abortfac}` and presolver `/{probing,aggregation,boundshift,dualfix,implications,trivial}/maxrounds`
    - Primal heuristics: `heuristics/feaspump/{freq,freqofs,maxdepth,maxlpiterquot,maxlpiterofs,beforecuts}`, `heuristics/rins/{nodesofs,nodesquot,minnodes,maxnodes,nwaitingnodes,minfixingrate}`, `heuristics/localbranching/{neighborhoodsize,nodesofs,nodesquot,lplimfac}`, `heuristics/rens/{nodesofs,nodesquot,minnodes,maxnodes,minfixingrate,startsol}`
    - Tolerances: `numerics/{feastol,epsilon,dualfeastol}`
  - Minimal: focused, high-impact subset:
    - Branching: `branching/scorefunc`
    - Node selection: `nodeselection/{bestestimate,dfs}/stdpriority`
    - Cutting planes: `separating/{maxroundsroot,maxcutsroot}`
    - Presolve: `presolving/{maxrounds,abortfac}`
    - Heuristics: `heuristics/feaspump/{freq,maxlpiterquot}`

How defaults are obtained
- We call SCIP CLI with `set default; set save <tmp>` and parse the saved `.set` to get default values.
- The defaults shown to the LLM (and in `gpt_reasoning.log`) are filtered to the current whitelist regime so that the LLM only sees defaults for allowed parameters.


## Outputs and File Semantics

AC training (`--mode ac` or `ac-train`)
- `batch_trials.csv` — per-trial rows with `trial`, `j_hat`, `solved_pct`, `gm_time`, `gm_gap`, `gm_nodes`, `applied_param_count`, `param_file`.
- `per_instance_metrics.csv` — per-instance rows per trial with status, solve time, bounds, gap, nodes, and a `sigma` value.
- `gpt_reasoning.log` — full prompt and responses per trial for auditability.

AC testing (`--mode ac`)
- `ac_summary.json` — final summary with train artifacts, incumbent, and test Ĵ_S.

ISAC (`--mode isac`)
- One folder per instance plus `isac_summary.csv` for quick aggregation.

Single instance (`--mode single`)
- Trial artifacts in an instance-specific subdirectory (no top-level summary.csv).

Parameter set files (`params_trial_<k>.set`)
- Can be used to reproduce runs with `scip -s params_trial_<k>.set -f instance.mps`.
- Contain meta toggles (as comments) and explicit name=value pairs for set parameters plus `limits/time`.


## Production Tips

- Determinism and stability
  - Pin SCIP version, Python version, and thread counts across runs.
  - Use `--seed` to shift solver randomization; record seeds with outputs.
- LLM reliability
  - Set sensible `--max-edits` (m) and `--trials` (K) budgets for your SLA.
  - Consider rate limits and error retries in `gpt/call_gpt.py` if you scale up.
- Logging and storage
  - Logs can be large; rotate or compress if needed. `display/freq` is set to 100 for richer logs; reduce if storage is tight.
- Security
  - Keep `OPENAI_API_KEY` secure; avoid committing logs that include sensitive instance details.


## Extending and Customizing

- Swap objective
  - Edit `score/solvermind_score.py` (σ and Ĵ_S), or add your own module and change `components/objective.py` to route to it.
- Change the whitelist or validation
  - Edit `params/get_whitelist.py` and/or `components/validator.py`.
- Modify prompts and reasoning
  - Update `components/prompt_builder.py` and `gpt/build_tuning_prompt.py`.
- Parallelize or batch execution
  - Replace `components/runner_cli.py` with a parallel CLI runner; keep the same function signature.
- Early stopping policy
  - Tweak `components/stopping.py` or add an LLM-guided saturation prompt (easy to integrate next to numeric patience).


## Troubleshooting

- SCIP not found on PATH
  - Set `SCIP_BIN=/path/to/scip` or ensure `scip` is on PATH.
- No instances discovered
  - Check `--instances-dir` and file extensions: `.mps`, `.lp`, `.cip`, or their `.gz` variants.
- LLM returns non-JSON or schema violations
  - The code raises a clear error in `gpt/call_gpt.py`. Consider adding retries or using `response_format` if supported by your model version.
- Meta toggles not applying
  - Applied via CLI command script in `run/run_instance_cli.py`.


## Repository Layout

- `solvermind/tuner/__main__.py` — CLI
- `solvermind/tuner/pipeline/solvermind.py` — Orchestrator for batch tuning
- `solvermind/tuner/components/` — Modular building blocks
- `solvermind/tuner/loop/` — Batch wrapper
- `solvermind/tuner/gpt/` — Prompt builder and OpenAI call site
- `solvermind/tuner/params/` — Whitelist and validator
- `solvermind/tuner/run/` — CLI runner per instance
- `solvermind/tuner/log/` — Log parsing and summarization
- `solvermind/tuner/score/` — Scoring functions (σ and Ĵ_S)
- `solvermind/tuner/experiments/` — Dataset discovery and splitting
- `main.tex` — Paper draft that this code follows (Problem Formulation and Algorithm sections)


## Notes

- This repository currently defaults to `gpt-5` as the model id. Adjust via `--model` to match your deployment (e.g., `gpt-4.1` or a compatible Responses API model).
- The solver is SCIP-specific via the SCIP CLI. Adapting to other solvers would involve modifying `run/*`, whitelist parameters, and (if desired) feature inputs.
