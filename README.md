# SCIP Parameter Optimization Framework

A comprehensive framework for optimizing SCIP solver parameters using multiple optimization methods including SolverMind (GPT-based), RBFOpt, and SMAC.

## Installation

### Quick Installation (macOS and Linux)

The easiest way to install all dependencies is using the automated installation script:

```bash
python3 install.py
```

**What it does:**
- Detects your platform (macOS or Linux)
- Installs system dependencies (Python 3.11+, SWIG, build tools)
- For macOS: Installs SCIP 9.2.4 via Homebrew
- For Linux: Installs build dependencies (SCIP needs manual installation, see below)
- Creates a virtual environment named `scip_env`
- Installs all Python dependencies from `requirements.txt`
- Runs verification tests

#### Installation Options

```bash
# Use custom virtual environment name
python3 install.py --venv-name my_env

# Skip system dependency installation (if already installed)
python3 install.py --skip-system-deps

# Test existing installation
python3 install.py --test-only
```

#### Linux SCIP Installation

On Linux, SCIP needs to be installed separately. Choose one option:

**Option 1: Using Conda (Recommended)**
```bash
conda install -c conda-forge scip=9.2.4
```

**Option 2: Build from Source**
```bash
# Download SCIP 9.2.4
wget https://scipopt.org/download/release/scip-9.2.4.tgz
tar xzf scip-9.2.4.tgz
cd scip-9.2.4
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### Manual Installation

#### System Requirements

- **Operating System**: macOS or Linux (automated script), Windows (manual installation required)
- **Python**: 3.11 or later (required for numpy 2.3.4+, scipy 1.16.3+, and other dependencies)
- **SCIP Optimizer**: Version 9.2.4 (fixed version for reproducibility)
- **SWIG**: Required for compiling pyrfr (SMAC dependency)
- **Build Tools**: gcc, g++, make, gfortran (for compiling scientific packages)

#### Step-by-Step Manual Installation

**On macOS:**

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install system dependencies**:
   ```bash
   brew install python@3.11
   brew install scip      # This installs SCIP 9.2.4
   brew install swig      # Required for pyrfr compilation
   ```

**On Linux (Debian/Ubuntu):**

1. **Update package lists**:
   ```bash
   sudo apt-get update
   ```

2. **Install system dependencies**:
   ```bash
   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
       build-essential swig libgmp-dev libreadline-dev zlib1g-dev \
       libbz2-dev liblapack-dev libblas-dev gfortran
   ```

3. **Install SCIP** (choose one option):
   ```bash
   # Option 1: Using Conda
   conda install -c conda-forge scip=9.2.4

   # Option 2: Build from source (see Linux SCIP Installation section above)
   ```

**On Linux (RedHat/CentOS/Fedora):**

1. **Install system dependencies**:
   ```bash
   sudo yum install -y python311 python311-devel gcc gcc-c++ make swig \
       gmp-devel readline-devel zlib-devel bzip2-devel \
       lapack-devel blas-devel gcc-gfortran
   ```

2. **Install SCIP** (same as Debian/Ubuntu, see above)

**All Platforms:**

3. **Create virtual environment**:
   ```bash
   python3.11 -m venv scip_env
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