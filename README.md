# SCIP Parameter Optimization Framework

A comprehensive framework for optimizing SCIP solver parameters using multiple optimization methods including SolverMind (GPT-based), RBFOpt, and SMAC.

## Installation

### Quick Installation (macOS)

The easiest way to install all dependencies is using the automated installation script:

```bash
python3 install.py
```

This will:
1. Install Python 3.11+ via Homebrew
2. Install SCIP 9.2.4 via Homebrew
3. Install SWIG (required for pyrfr compilation)
4. Create a virtual environment named `scip_env`
5. Install all Python dependencies from `requirements.txt`
6. Run verification tests

#### Installation Options

```bash
# Use custom virtual environment name
python3 install.py --venv-name my_env

# Skip Homebrew installation (if already installed)
python3 install.py --skip-brew

# Test existing installation
python3 install.py --test-only
```

### Manual Installation

#### System Requirements

- **Operating System**: macOS (for automated script), Linux/Windows (manual installation)
- **Python**: 3.11 or later
- **SCIP Optimizer**: Version 9.2.4 (fixed version for reproducibility)
- **SWIG**: Required for compiling pyrfr (SMAC dependency)

#### Step-by-Step Manual Installation

1. **Install Homebrew** (macOS only):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install system dependencies**:
   ```bash
   brew install python@3.11
   brew install scip      # This installs SCIP 9.2.4
   brew install swig      # Required for pyrfr compilation
   ```

3. **Create virtual environment**:
   ```bash
   /usr/local/bin/python3.11 -m venv scip_env
   source scip_env/bin/activate
   ```

4. **Install Python packages**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   python -c "import pyscipopt; print(f'PySCIPOpt: {pyscipopt.__version__}')"
   scip --version
   ```

### Verification

To verify your installation is working correctly:

```bash
# Activate environment
source scip_env/bin/activate

# Run test
python3 install.py --test-only
```

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