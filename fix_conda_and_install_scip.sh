#!/bin/bash
# Fix conda configuration and install SCIP 9.2.4

set -e

echo "=================================================="
echo "Fixing conda configuration"
echo "=================================================="

# Show current conda config
echo "Current conda channels:"
conda config --show channels

# Remove all conda-forge mirrors
echo ""
echo "Removing broken mirrors..."
conda config --remove channels conda-forge 2>/dev/null || true
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 2>/dev/null || true
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/conda-forge 2>/dev/null || true

# Add official conda-forge channel
echo "Adding official conda-forge channel..."
conda config --add channels conda-forge
conda config --set channel_priority strict

echo ""
echo "New conda channels:"
conda config --show channels

echo ""
echo "=================================================="
echo "Installing SCIP 9.2.4 from conda-forge"
echo "=================================================="

# Install SCIP
conda install -c conda-forge scip=9.2.4 -y

echo ""
echo "=================================================="
echo "Verifying SCIP installation"
echo "=================================================="

# Check if SCIP is available
if command -v scip &> /dev/null; then
    echo "✓ SCIP installed successfully!"
    scip --version | head -n 1
    echo ""
    echo "SCIP location: $(which scip)"
else
    echo "✗ SCIP not found in PATH"
    echo ""
    echo "Checking conda environment..."
    conda list scip
    exit 1
fi

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Now you can run:"
echo "  source scip_env/bin/activate"
echo "  PYTHONPATH=./src python -m utilities.optimizer_cli smac \\"
echo "    --config configs/experiment.yaml \\"
echo "    --whitelist curated \\"
echo "    --instance instance/cvrp/mps/XML100_1131_01_40.mps.gz"
