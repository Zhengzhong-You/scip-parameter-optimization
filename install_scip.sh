#!/bin/bash
# SCIP Installation Script for Linux
# This script installs SCIP 9.2.4 using conda

set -e

echo "============================================================"
echo "Installing SCIP 9.2.4 via conda"
echo "============================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo ""
    echo "Please install conda first:"
    echo "  Option 1: Miniconda (recommended)"
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "  Option 2: Use your system's conda if already installed"
    echo "    Make sure conda is in your PATH"
    exit 1
fi

echo "✓ Found conda: $(conda --version)"
echo ""

# Install SCIP 9.2.4
echo "Installing SCIP 9.2.4 from conda-forge..."
conda install -c conda-forge scip=9.2.4 -y

echo ""
echo "============================================================"
echo "Verifying SCIP installation"
echo "============================================================"
echo ""

# Verify SCIP is installed
if command -v scip &> /dev/null; then
    echo "✓ SCIP installed successfully!"
    scip --version | head -n 1
else
    echo "✗ SCIP installation may have failed"
    echo "Try running manually: conda install -c conda-forge scip=9.2.4"
    exit 1
fi

echo ""
echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo ""
echo "SCIP 9.2.4 is now installed and ready to use."
echo ""
echo "Next steps:"
echo "  1. Activate your virtual environment: source scip_env/bin/activate"
echo "  2. Run the optimizer:"
echo "     PYTHONPATH=./src python -m utilities.optimizer_cli smac \\"
echo "       --config configs/experiment.yaml \\"
echo "       --whitelist curated \\"
echo "       --instance instance/cvrp/mps/XML100_1131_01_40.mps.gz"
