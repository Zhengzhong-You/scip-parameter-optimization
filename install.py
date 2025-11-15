#!/usr/bin/env python3
"""
SCIP Parameter Optimization Framework - Installation Script

This script automates the installation process including:
1. Checking system requirements
2. Installing dependencies (Python 3.11+, SCIP 9.2.4, SWIG)
3. Creating virtual environment
4. Installing Python packages
5. Running verification tests

Supported Platforms:
- macOS (via Homebrew)
- Linux (via apt/yum package managers)

Usage:
    python3 install.py [--venv-name scip_env] [--skip-system-deps] [--test-only]
"""

import argparse
import os
import platform
import shutil
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


def print_info(message):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")


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


def get_platform_info():
    """Detect platform and return relevant information"""
    system = platform.system()

    if system == "Darwin":
        return {
            "os": "macOS",
            "package_manager": "brew",
            "python_cmd": "/usr/local/bin/python3.11",
            "supported": True
        }
    elif system == "Linux":
        # Detect Linux distribution with multiple methods
        pkg_mgr = "unknown"

        # Method 1: Check for distribution-specific files
        if Path("/etc/debian_version").exists():
            pkg_mgr = "apt"
        elif Path("/etc/redhat-release").exists():
            pkg_mgr = "yum"
        elif Path("/etc/fedora-release").exists():
            pkg_mgr = "yum"
        elif Path("/etc/SuSE-release").exists() or Path("/etc/SUSE-release").exists():
            pkg_mgr = "zypper"
        elif Path("/etc/arch-release").exists():
            pkg_mgr = "pacman"

        # Method 2: Try using os-release (standard on most modern Linux distros)
        if pkg_mgr == "unknown" and Path("/etc/os-release").exists():
            try:
                with open("/etc/os-release") as f:
                    os_release = f.read().lower()
                    if any(distro in os_release for distro in ["ubuntu", "debian", "mint"]):
                        pkg_mgr = "apt"
                    elif any(distro in os_release for distro in ["rhel", "centos", "fedora", "red hat"]):
                        pkg_mgr = "yum"
                    elif "suse" in os_release:
                        pkg_mgr = "zypper"
                    elif "arch" in os_release:
                        pkg_mgr = "pacman"
            except:
                pass

        # Method 3: Check which package manager commands exist
        if pkg_mgr == "unknown":
            if shutil.which("apt-get"):
                pkg_mgr = "apt"
            elif shutil.which("yum"):
                pkg_mgr = "yum"
            elif shutil.which("dnf"):
                pkg_mgr = "dnf"
            elif shutil.which("zypper"):
                pkg_mgr = "zypper"
            elif shutil.which("pacman"):
                pkg_mgr = "pacman"

        return {
            "os": "Linux",
            "package_manager": pkg_mgr,
            "python_cmd": "python3.11",
            "supported": pkg_mgr in ["apt", "yum", "dnf"]
        }
    else:
        return {
            "os": system,
            "package_manager": "unknown",
            "python_cmd": "python3",
            "supported": False
        }


def check_platform():
    """Check if the platform is supported"""
    print_section("Checking Platform")

    platform_info = get_platform_info()
    print(f"Detected platform: {platform_info['os']}")
    print(f"Package manager: {platform_info['package_manager']}")

    if not platform_info['supported']:
        print_error(f"Platform {platform_info['os']} is not automatically supported")
        print_info("You can still use --skip-system-deps and install dependencies manually")
        return False, platform_info

    print_success("Platform supported")
    return True, platform_info


def find_python():
    """Find suitable Python 3.11+ installation"""
    python_candidates = [
        "/usr/local/bin/python3.11",
        "/usr/bin/python3.11",
        "python3.11",
        "python3",
    ]

    for cmd in python_candidates:
        try:
            result = subprocess.run(
                f"{cmd} --version",
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            version_str = result.stdout.strip()
            # Extract version number
            version_parts = version_str.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])

            if major == 3 and minor >= 11:
                print_success(f"Found suitable Python: {cmd} ({version_str})")
                return cmd
            else:
                print_warning(f"Python version too old: {version_str}")
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
            continue

    return None


def install_macos_dependencies():
    """Install dependencies on macOS via Homebrew"""
    print_section("Installing macOS Dependencies")

    # Check Homebrew
    if not shutil.which("brew"):
        print_error("Homebrew is not installed")
        print_info("Install Homebrew first:")
        print('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        return False

    packages = [
        ("python@3.11", "Python 3.11"),
        ("scip", "SCIP Optimizer"),
        ("swig", "SWIG (for pyrfr)")
    ]

    for package, description in packages:
        print(f"\nInstalling {description}...")
        if not run_command(f"brew install {package}", f"Install {package}"):
            print_warning(f"Failed to install {package}, it may already be installed")

    return True


def install_linux_dependencies(pkg_manager):
    """Install dependencies on Linux"""
    print_section("Installing Linux Dependencies")

    if pkg_manager == "apt":
        print_info("Using apt package manager (Debian/Ubuntu)")

        # Update package lists
        run_command("sudo apt-get update", "Update apt package lists", check=False)

        packages = [
            "python3.11",
            "python3.11-venv",
            "python3.11-dev",
            "build-essential",
            "swig",
            "libgmp-dev",
            "libreadline-dev",
            "zlib1g-dev",
            "libbz2-dev",
            "liblapack-dev",
            "libblas-dev",
            "gfortran"
        ]

        print("\nInstalling required packages...")
        cmd = f"sudo apt-get install -y {' '.join(packages)}"
        if not run_command(cmd, "Install apt packages"):
            print_warning("Some packages may not have installed correctly")

        # SCIP typically needs to be built from source or installed from COIN-OR
        print_warning("\nSCIP 9.2.4 needs to be installed manually on Linux")
        print_info("Option 1: Build from source: https://scipopt.org/")
        print_info("Option 2: Use conda: conda install -c conda-forge scip=9.2.4")
        print_info("Continuing with Python environment setup...")

    elif pkg_manager in ["yum", "dnf"]:
        mgr_cmd = "dnf" if pkg_manager == "dnf" else "yum"
        print_info(f"Using {mgr_cmd} package manager (RedHat/CentOS/Fedora)")

        packages = [
            "python311",
            "python311-devel",
            "gcc",
            "gcc-c++",
            "make",
            "swig",
            "gmp-devel",
            "readline-devel",
            "zlib-devel",
            "bzip2-devel",
            "lapack-devel",
            "blas-devel",
            "gcc-gfortran"
        ]

        print("\nInstalling required packages...")
        cmd = f"sudo {mgr_cmd} install -y {' '.join(packages)}"
        if not run_command(cmd, f"Install {mgr_cmd} packages"):
            print_warning("Some packages may not have installed correctly")

        print_warning("\nSCIP 9.2.4 needs to be installed manually on Linux")
        print_info("Option 1: Build from source: https://scipopt.org/")
        print_info("Option 2: Use conda: conda install -c conda-forge scip=9.2.4")
        print_info("Continuing with Python environment setup...")

    else:
        print_warning(f"Package manager '{pkg_manager}' is not fully supported")
        print_info("You'll need to manually install:")
        print_info("  - Python 3.11+")
        print_info("  - Python development headers (python3.11-dev)")
        print_info("  - Build tools (gcc, g++, make, gfortran)")
        print_info("  - SWIG")
        print_info("  - Development libraries (GMP, readline, zlib, bzip2, LAPACK, BLAS)")
        print_info("  - SCIP 9.2.4")

    return True


def verify_system_dependencies(platform_info):
    """Verify that system dependencies are installed"""
    print_section("Verifying System Dependencies")

    all_ok = True

    # Check Python
    python_cmd = find_python()
    if python_cmd:
        version = run_command(f"{python_cmd} --version", "Python version", capture_output=True)
        print_success(f"Python: {version}")
    else:
        print_error("Python 3.11+ not found")
        all_ok = False

    # Check SCIP
    if shutil.which("scip"):
        scip_version = run_command("scip --version | head -n 1", "SCIP version", capture_output=True)
        if scip_version:
            print_success(f"SCIP: {scip_version[:80]}")
            if "9.2.4" not in scip_version:
                print_warning("SCIP version may not be 9.2.4 (recommended)")
    else:
        print_warning("SCIP not found in PATH - you may need to install it manually")
        print_info("See: https://scipopt.org/ or use conda: conda install -c conda-forge scip=9.2.4")

    # Check SWIG
    if shutil.which("swig"):
        swig_version = run_command("swig -version | grep 'SWIG Version'", "SWIG version", capture_output=True)
        if swig_version:
            print_success(f"SWIG: {swig_version}")
    else:
        print_error("SWIG not found - required for building pyrfr (SMAC dependency)")
        all_ok = False

    return all_ok


def create_virtualenv(venv_name):
    """Create Python virtual environment"""
    print_section(f"Creating Virtual Environment: {venv_name}")

    venv_path = Path(venv_name)

    # Remove existing venv if it exists
    if venv_path.exists():
        print_warning(f"Removing existing virtual environment: {venv_name}")
        run_command(f"rm -rf {venv_name}", "Remove old venv")

    # Find Python
    python_cmd = find_python()
    if not python_cmd:
        print_error("Could not find Python 3.11+")
        return False

    # Create new venv
    if not run_command(
        f"{python_cmd} -m venv {venv_name}",
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

    print("This may take several minutes (especially compiling scipy, scikit-learn, pyrfr)...")
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
        print_warning("SCIP test failed - ensure SCIP is installed correctly")

    return all_passed


def print_summary(venv_name, platform_info):
    """Print installation summary and next steps"""
    print_section("Installation Complete!")

    print(f"""
{Colors.GREEN}✓ Virtual environment and Python packages installed!{Colors.END}

{Colors.BOLD}Installed Components:{Colors.END}
  • Python 3.11+ virtual environment
  • Virtual environment: {venv_name}
  • All Python packages from requirements.txt

{Colors.BOLD}System Dependencies (verify these are installed):{Colors.END}
  • Python 3.11 or later
  • SCIP Optimizer 9.2.4 (recommended)
  • SWIG (for building extensions)

{Colors.BOLD}Next Steps:{Colors.END}

1. Activate the virtual environment:
   {Colors.BLUE}source {venv_name}/bin/activate{Colors.END}

2. Verify SCIP installation (if not done already):
   {Colors.BLUE}scip --version{Colors.END}

   If SCIP is not installed, see:
   - macOS: brew install scip
   - Linux: Build from source (https://scipopt.org/) or use conda
   - Conda: conda install -c conda-forge scip=9.2.4

3. Run the optimizer (example):
   {Colors.BLUE}PYTHONPATH=./src python -m utilities.optimizer_cli solvermind \\
       --config configs/experiment.yaml \\
       --whitelist curated \\
       --instance instance/cvrp/mps/example.mps.gz{Colors.END}

4. Deactivate when done:
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
        description="Install SCIP Parameter Optimization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 install.py                              # Full installation
  python3 install.py --skip-system-deps           # Only setup Python environment
  python3 install.py --venv-name my_env           # Custom venv name
  python3 install.py --test-only                  # Test existing installation
        """
    )
    parser.add_argument(
        "--venv-name",
        default="scip_env",
        help="Name of virtual environment (default: scip_env)"
    )
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Skip system dependency installation (Python, SCIP, SWIG)"
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
    supported, platform_info = check_platform()

    # Install system dependencies if not skipped
    if not args.skip_system_deps:
        if not supported:
            print_error("Platform not supported for automatic dependency installation")
            print_info("Use --skip-system-deps to skip this step and install manually")
            sys.exit(1)

        # Install dependencies based on platform
        if platform_info['os'] == "macOS":
            if not install_macos_dependencies():
                print_error("Failed to install macOS dependencies")
                sys.exit(1)
        elif platform_info['os'] == "Linux":
            if not install_linux_dependencies(platform_info['package_manager']):
                print_error("Failed to install Linux dependencies")
                sys.exit(1)

        # Verify installations
        if not verify_system_dependencies(platform_info):
            print_warning("Some system dependencies may be missing")
            print_info("You can continue, but some features may not work")
    else:
        print_info("Skipping system dependency installation")
        # Still verify what's available
        verify_system_dependencies(platform_info)

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
    print_summary(args.venv_name, platform_info)


if __name__ == "__main__":
    main()
