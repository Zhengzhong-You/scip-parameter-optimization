
## What you get

```
solvermind_baselines/
├── README.md                         # quick start
├── requirements.txt                  # smac, ConfigSpace, rbfopt, etc.
├── configs/
│   └── experiment.yaml               # τ, seeds, budgets, paths
├── whitelists/
│   ├── curated.yaml                  # mid-size knob set (edit as needed)
│   └── minimal.yaml                  # ultra-compact numeric-first list
├── scripts/
│   ├── run_smac.py                   # CLI for SMAC baseline
│   └── run_rbfopt.py                 # CLI for RBFOpt baseline
└── src/solvermind_baselines/
    ├── __init__.py
    ├── datasets.py                   # instance discovery & splitting
    ├── scoring.py                    # f1..f4, σ(p,i), status tier
    ├── space.py                      # build SMAC ConfigSpace, RBFOpt vars
    ├── io_utils.py                   # dump params -> SCIP .set commands
    ├── objective.py                  # evaluate L(p) on full S (your Eq. 10–13)
    ├── runner_dummy.py               # fast synthetic runner to test the loop
    ├── runner_scip.py                # real SCIP runner (subprocess + log parse)
    └── baselines/
        ├── smac_baseline.py          # SMAC training loop
        └── rbfopt_baseline.py        # RBFOpt training loop
```

### Why this matches your paper

* **Full‑set evaluation:** Each objective call evaluates a candidate **on the entire training set (S)** under the same (\tau). This keeps the comparison fair and sidesteps SMAC’s intensification/capping: even if SMAC internally wants to stage budgets, our objective is already an **aggregate scalar** over all instances.
* **σ(p,i) exactly as specified:** `scoring.py` builds (\sigma) with your (f_1)–(f_4) (cap-linear / cap-exp options). You can change caps and slopes in `configs/experiment.yaml`.
* **Mixed discrete–continuous parameters:**

  * **SMAC:** native continuous / integer / categorical support (via `ConfigSpace`).
  * **RBFOpt:** integers are native; categoricals use **label encoding** with a reversible mapping.
* **Reproducible:** Seeds are fixed; one call = one **complete** evaluation.

---

## How to run

1. **Install**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip setuptools wheel
   pip install -r requirements.txt
   ```

2. **Configure** (edit `configs/experiment.yaml`)

   * `runner.kind`: `"dummy"` (smoke test) or `"scip"` (real runs)
   * `runner.tau`: per‑instance time limit (\tau) (sec)
   * `data.instances_dir`: folder with your `.lp/.mps/.cip` files
   * Objective caps/weights: `objective.f1..f4` (matches your Table on (f_k))
   * Budgets: `smac.n_trials`, `rbfopt.max_evaluations`

3. **Choose parameter whitelist**

   * Start with `whitelists/minimal.yaml` (numeric‑heavy for portability).
   * For real experiments, switch to `whitelists/curated.yaml` and **update categorical choices** to match your SCIP build (enum names can vary slightly). You can add/remove parameters freely.

4. **Run baselines**

   ```bash
   # SMAC
   python -m scripts.run_smac   --config configs/experiment.yaml --whitelist whitelists/minimal.yaml

   # RBFOpt
   python -m scripts.run_rbfopt --config configs/experiment.yaml --whitelist whitelists/minimal.yaml
   ```

Artifacts (best config, per‑instance logs, etc.) are saved under `runs/`.

---

## File-by-file overview (so you know where to debug)

### `src/solvermind_baselines/objective.py`

* `evaluate_config_on_set(...)`
  → **Core of Eq. (12)**. Runs the solver on each (i\in S), computes (\sigma(p,i)), then returns (L(p)=\frac{1}{M}\sum \log \sigma(p,i)).
* `aggregate_sigma_for_instance(...)`
  → Builds status tier, normalized time (t/\tau), gap (g(p,i)) (your piecewise definition, with `+∞` when signs differ), normalized nodes, then calls `scoring.sigma(...)`.

### `src/solvermind_baselines/scoring.py`

* Implements the (f_1…f_4) families you described (cap-linear, cap-exp), plus the status tier mapping (opt/inf/unb → tier 0; others → tier 1). Change slopes/caps in `configs/experiment.yaml`.

### `src/solvermind_baselines/runner_scip.py`

* **Real SCIP** subprocess runner:

  * Writes a temp `.set` file from the proposed config.
  * Runs: `set limits time τ` → `read <instance>` → `set load tuned.set` → `optimize` → `display statistics`.
  * Parses: status (opt/inf/unb/tl/other), solving time, primal/dual bounds, node count.
    The regexes are robust across 8.x–9.x, but if your build logs differ, tweak the patterns here.

> For quick end‑to‑end sanity checks, use `runner_dummy.py` first—fast and deterministic‑ish.

### `src/solvermind_baselines/space.py`

* **SMAC**: builds a `ConfigSpace.ConfigurationSpace` from a YAML whitelist.
* **RBFOpt**: creates variable arrays (`R`/`I` types) and label maps for categoricals.

### `src/solvermind_baselines/baselines/smac_baseline.py`

* Minimal SMAC3 loop using `HyperparameterOptimizationFacade`.
* Objective call **always** does full‑set evaluation (so intensification/capping won’t bias budgets).
* Saves best config + per‑instance logs.

### `src/solvermind_baselines/baselines/rbfopt_baseline.py`

* RBFOpt black‑box with label‑decoded categoricals.
* Returns best config, best (L), and per‑instance logs.

---

## Notes you’ll likely care about

* **Budget parity**: Set `smac.n_trials` and `rbfopt.max_evaluations` to the same number. In both cases, **one evaluation = full (S) with τ per instance**.
* **Categoricals**: `branching/scorefunc` etc. have build‑dependent string choices. Update the `choices` arrays in `whitelists/*.yaml` to match your SCIP (run `scip -c "set help"` or check the manual).
* **Scaling**: For floats with wide ranges, you can set `log: true` in YAML (SMAC uses log‑scaled domain).
* **Early stopping**: Kept **out** of the baselines to match your “full evaluation” fairness rule. You can add Ask/Tell callbacks later if you want richer logs per trial.
* **Generalization**: The launcher scripts currently optimize on `train`. After you’re happy with a baseline config, just reuse `best_config.json` to evaluate on your `test` split with your reporting harness.

---

## Where to plug your paper choices

* **(f_1)–(f_4) settings** → `configs/experiment.yaml` (`objective` section).
* **Whitelist regimes** (Full / Curated / Minimal) → `whitelists/*.yaml`.
  The shipped `curated.yaml` mirrors the categories you listed (branching, node selection, cuts, presolve, heuristics) with numeric knobs by default.
* **(\tau) sweep** (`300/900/3600s`) → change `runner.tau` and rerun both baselines.
  The outputs under `runs/` can be aggregated into your Tables 6–8.

---

### Quick smoke test (no SCIP required)

1. Keep runner kind = `dummy` in `configs/experiment.yaml`.
2. Run SMAC and RBFOpt commands above.
3. You should see best configs and (L(p)) printed, with artifacts written to `runs/`.

---

