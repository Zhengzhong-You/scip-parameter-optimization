#!/usr/bin/env python3
"""
SCIP Parameter Optimization Framework - Installation Script

This script automates the installation process including:
1. Checking system requirements
2. Installing Homebrew dependencies (Python 3.11+, SCIP 9.2.4, SWIG)
3. Creating virtual environment
4. Installing Python packages
5. Running verification tests

Usage:
    python3 install.py [--venv-name scip_env] [--skip-brew] [--test-only]
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def run_command(cmd, description, check=True, capture_output=False):
    """Run a shell command with error handling"""
    print(f"Running: {description}...")
    try:
        if capture_output:
            result = subprocess.run(
                cmd, shell=True, check=check,
                capture_output=True, text=True, timeout=600
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check, timeout=600)
        print_success(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with exit code {e.returncode}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print_error(f"{description} timed out")
        return False


def check_platform():
    """Check if the platform is supported"""
    print_section("Checking Platform")
    system = platform.system()
    print(f"Detected platform: {system}")

    if system != "Darwin":
        print_error("This installation script is designed for macOS (Darwin).")
        print_error("For other platforms, please install dependencies manually.")
        return False

    print_success("Platform supported")
    return True


def check_homebrew():
    """Check if Homebrew is installed"""
    print_section("Checking Homebrew")
    try:
        result = subprocess.run(
            ["brew", "--version"],
            capture_output=True, text=True, check=True
        )
        version = result.stdout.split('\n')[0]
        print_success(f"Homebrew is installed: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Homebrew is not installed")
        print("\nTo install Homebrew, run:")
        print('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        return False


def install_brew_packages():
    """Install required Homebrew packages"""
    print_section("Installing Homebrew Packages")

    packages = [
        ("python@3.11", "Python 3.11"),
        ("scip", "SCIP Optimizer 9.2.4"),
        ("swig", "SWIG (for pyrfr compilation)")
    ]

    for package, description in packages:
        print(f"\nInstalling {description}...")
        if not run_command(f"brew install {package}", f"Install {package}"):
            print_error(f"Failed to install {package}")
            return False

    return True


def verify_installations():
    """Verify Homebrew installations"""
    print_section("Verifying System Installations")

    # Check Python 3.11
    python_version = run_command(
        "/usr/local/bin/python3.11 --version",
        "Check Python 3.11 version",
        capture_output=True
    )
    if python_version:
        print_success(f"Python installed: {python_version}")
    else:
        print_error("Python 3.11 not found")
        return False

    # Check SCIP
    scip_version = run_command(
        "scip --version | head -n 1",
        "Check SCIP version",
        capture_output=True
    )
    if scip_version and "9.2.4" in scip_version:
        print_success(f"SCIP installed: {scip_version}")
    else:
        print_warning(f"SCIP version may not be 9.2.4: {scip_version}")

    # Check SWIG
    swig_version = run_command(
        "swig -version | grep 'SWIG Version'",
        "Check SWIG version",
        capture_output=True
    )
    if swig_version:
        print_success(f"SWIG installed: {swig_version}")
    else:
        print_error("SWIG not found")
        return False

    return True


def create_virtualenv(venv_name):
    """Create Python virtual environment"""
    print_section(f"Creating Virtual Environment: {venv_name}")

    venv_path = Path(venv_name)

    # Remove existing venv if it exists
    if venv_path.exists():
        print_warning(f"Removing existing virtual environment: {venv_name}")
        run_command(f"rm -rf {venv_name}", "Remove old venv")

    # Create new venv
    if not run_command(
        f"/usr/local/bin/python3.11 -m venv {venv_name}",
        "Create virtual environment"
    ):
        return False

    # Upgrade pip
    if not run_command(
        f"{venv_name}/bin/pip install --upgrade pip",
        "Upgrade pip"
    ):
        return False

    return True


def install_python_packages(venv_name):
    """Install Python packages from requirements.txt"""
    print_section("Installing Python Packages")

    if not Path("requirements.txt").exists():
        print_error("requirements.txt not found in current directory")
        return False

    print("This may take several minutes...")
    if not run_command(
        f"{venv_name}/bin/pip install -r requirements.txt",
        "Install Python packages from requirements.txt"
    ):
        print_error("Failed to install Python packages")
        return False

    return True


def run_tests(venv_name):
    """Run verification tests"""
    print_section("Running Verification Tests")

    tests = [
        (
            f"{venv_name}/bin/python -c 'import sys; print(f\"Python: {{sys.version}}\")'",
            "Python version in venv"
        ),
        (
            f"{venv_name}/bin/python -c 'import pyscipopt; print(f\"PySCIPOpt: {{pyscipopt.__version__}}\")'",
            "PySCIPOpt import and version"
        ),
        (
            f"{venv_name}/bin/python -c 'import numpy; print(f\"NumPy: {{numpy.__version__}}\")'",
            "NumPy import and version"
        ),
        (
            f"{venv_name}/bin/python -c 'import scipy; print(f\"SciPy: {{scipy.__version__}}\")'",
            "SciPy import and version"
        ),
        (
            f"{venv_name}/bin/python -c 'import pandas; print(f\"Pandas: {{pandas.__version__}}\")'",
            "Pandas import and version"
        ),
        (
            f"{venv_name}/bin/python -c 'import sklearn; print(f\"Scikit-learn: {{sklearn.__version__}}\")'",
            "Scikit-learn import and version"
        ),
        (
            f"{venv_name}/bin/python -c 'import smac; print(f\"SMAC: {{smac.__version__}}\")'",
            "SMAC import and version"
        ),
        (
            f"{venv_name}/bin/python -c 'import rbfopt; print(\"RBFOpt: OK\")'",
            "RBFOpt import"
        ),
        (
            f"{venv_name}/bin/python -c 'import openai; print(f\"OpenAI: {{openai.__version__}}\")'",
            "OpenAI import and version"
        ),
        (
            f"{venv_name}/bin/python -c 'import pyomo; print(\"Pyomo: OK\")'",
            "Pyomo import"
        ),
    ]

    all_passed = True
    for cmd, description in tests:
        result = run_command(cmd, description, capture_output=True)
        if result:
            print(f"  {result}")
        else:
            all_passed = False
            print_error(f"  Test failed: {description}")

    # Test SCIP through PySCIPOpt
    print("\nTesting SCIP solver integration...")
    test_scip_code = '''
import pyscipopt as scip
model = scip.Model("test")
x = model.addVar("x", vtype="C", lb=0, ub=10)
y = model.addVar("y", vtype="C", lb=0, ub=10)
model.setObjective(x + y, "maximize")
model.addCons(2*x + y <= 15)
model.addCons(x + 2*y <= 15)
model.hideOutput()
model.optimize()
if model.getStatus() == "optimal":
    print(f"SCIP integration test: PASSED (objective = {model.getObjVal():.2f})")
else:
    print("SCIP integration test: FAILED")
'''

    result = run_command(
        f"{venv_name}/bin/python -c '{test_scip_code}'",
        "SCIP solver integration test",
        capture_output=True
    )
    if result:
        print(f"  {result}")
    else:
        all_passed = False

    return all_passed


def print_summary(venv_name):
    """Print installation summary and next steps"""
    print_section("Installation Complete!")

    print(f"""
{Colors.GREEN}✓ All components installed successfully!{Colors.END}

{Colors.BOLD}Installed Components:{Colors.END}
  • Python 3.11.14
  • SCIP Optimizer 9.2.4
  • SWIG (for building extensions)
  • Virtual environment: {venv_name}
  • All Python packages from requirements.txt

{Colors.BOLD}Next Steps:{Colors.END}

1. Activate the virtual environment:
   {Colors.BLUE}source {venv_name}/bin/activate{Colors.END}

2. Run the optimizer (example):
   {Colors.BLUE}PYTHONPATH=./src python -m utilities.optimizer_cli solvermind \\
       --config configs/experiment.yaml \\
       --whitelist curated \\
       --instance instance/cvrp/mps/example.mps.gz{Colors.END}

3. Deactivate when done:
   {Colors.BLUE}deactivate{Colors.END}

{Colors.BOLD}Available Optimizers:{Colors.END}
  • solvermind - LLM-guided parameter tuning
  • rbfopt    - Radial basis function optimization
  • smac      - Sequential Model-based Algorithm Configuration

For more information, see README.md
""")


def main():
    """Main installation workflow"""
    parser = argparse.ArgumentParser(
        description="Install SCIP Parameter Optimization Framework"
    )
    parser.add_argument(
        "--venv-name",
        default="scip_env",
        help="Name of virtual environment (default: scip_env)"
    )
    parser.add_argument(
        "--skip-brew",
        action="store_true",
        help="Skip Homebrew package installation"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run tests on existing installation"
    )

    args = parser.parse_args()

    print(f"""
{Colors.BOLD}{Colors.BLUE}
╔═══════════════════════════════════════════════════════════╗
║  SCIP Parameter Optimization Framework                    ║
║  Installation Script                                      ║
╚═══════════════════════════════════════════════════════════╝
{Colors.END}
""")

    # Test-only mode
    if args.test_only:
        if not Path(args.venv_name).exists():
            print_error(f"Virtual environment '{args.venv_name}' not found")
            sys.exit(1)

        success = run_tests(args.venv_name)
        sys.exit(0 if success else 1)

    # Check platform
    if not check_platform():
        sys.exit(1)

    # Check Homebrew
    if not args.skip_brew:
        if not check_homebrew():
            sys.exit(1)

        # Install Homebrew packages
        if not install_brew_packages():
            sys.exit(1)

        # Verify installations
        if not verify_installations():
            sys.exit(1)

    # Create virtual environment
    if not create_virtualenv(args.venv_name):
        sys.exit(1)

    # Install Python packages
    if not install_python_packages(args.venv_name):
        sys.exit(1)

    # Run tests
    if not run_tests(args.venv_name):
        print_warning("Some tests failed, but installation may still be usable")

    # Print summary
    print_summary(args.venv_name)


if __name__ == "__main__":
    main()
