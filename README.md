# SCIP Parameter Optimization Framework

A comprehensive framework for optimizing SCIP solver parameters using multiple optimization methods including SolverMind (GPT-based), RBFOpt, and SMAC.

## Features

- **Multi-Method Optimization**: Support for SolverMind (LLM-based), RBFOpt, and SMAC optimizers
- **Advanced T_infinity Estimation**: Enhanced SVB (Single Variable Branching) model with robust fitting
- **Comprehensive Parameter Space**: Curated SCIP parameter whitelist with automatic type inference
- **Organized Output**: Structured results with diagnostics and performance metrics
- **Range Constraints**: Prevents unrealistic parameter predictions with configurable bounds

## Directory Structure

```
├── configs/          # Optimization configuration files
├── src/              # Source code
│   ├── utilities/    # Core utilities and estimation models
│   ├── solvermind/   # GPT-based optimization
│   ├── rbfopt_tune/ # RBFOpt optimization
│   └── smac_tune/   # SMAC optimization
└── instance/         # Test problem instances
```

## Key Components

### T_infinity Estimation (`src/utilities/est_time.py`)
- Enhanced SVB model fitting with L1 MINLP optimization
- Range constraints preventing b̂_i estimates from differing by >2x factor
- Robust sample filtering using unique gap values
- Debug output for detailed analysis

### Optimization Methods
- **SolverMind**: LLM-guided parameter tuning with iterative refinement
- **RBFOpt**: Radial basis function optimization for black-box problems
- **SMAC**: Sequential Model-based Algorithm Configuration

### Parameter Management (`src/utilities/whitelist.py`)
- Automatic SCIP parameter discovery and type inference
- Curated parameter whitelist for focused optimization
- Support for minimal, curated, and full parameter sets

## Recent Improvements

- ✅ Fixed SMAC JSON serialization for ConfigSpace boolean types
- ✅ Enhanced SVB fitting with range constraints (2x factor bounds)
- ✅ Organized SMAC internal files under structured output directories
- ✅ Improved debug output with sample analysis and calculation breakdown
- ✅ Robust sample filtering using unique gap values only

## Usage

Run optimization using the unified CLI:

```bash
# SolverMind optimization
PYTHONPATH=./src python -m utilities.optimizer_cli solvermind --config configs/experiment.yaml --whitelist curated --instance instance/cvrp/mps/example.mps.gz

# RBFOpt optimization
PYTHONPATH=./src python -m utilities.optimizer_cli rbfopt --config configs/experiment.yaml --whitelist curated --instance instance/cvrp/mps/example.mps.gz

# SMAC optimization
PYTHONPATH=./src python -m utilities.optimizer_cli smac --config configs/experiment.yaml --whitelist curated --instance instance/cvrp/mps/example.mps.gz
```

## Configuration

Optimization settings are controlled via YAML configuration files in `configs/`. Key parameters:

- `tau`: Time limit for SCIP solver runs
- `n_trials`: Number of optimization trials
- `whitelist`: Parameter set ("minimal", "curated", "full")

## Output Structure

Results are organized in timestamped directories under `runs/`:
```
runs/
└── [method]_[instance]/
    └── [method]_[timestamp]/
        ├── best_config.json      # Optimal parameters found
        ├── per_instance.json     # Detailed run results
        ├── summary.json          # Optimization summary
        └── [method]_internal/    # Method-specific internal files
```