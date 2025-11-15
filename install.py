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
            print_info("You may need to run manually: sudo apt-get install -y " + " ".join(packages))

        # SCIP typically needs to be built from source or installed from COIN-OR
        # Install SCIP 9.2.4 - try conda first, then build from source
        print_info("\nInstalling SCIP 9.2.4...")
        scip_installed = False

        # Try conda first if available
        if shutil.which("conda"):
            print_info("Found conda - trying to install SCIP 9.2.4 from conda-forge...")
            scip_cmd = "conda install -c conda-forge scip=9.2.4 -y"
            if run_command(scip_cmd, "Install SCIP 9.2.4 via conda", check=False):
                # Verify SCIP is actually in PATH
                if shutil.which("scip"):
                    print_success("SCIP 9.2.4 installed successfully via conda")
                    scip_installed = True
                else:
                    print_warning("Conda install succeeded but SCIP not in PATH")
            else:
                print_warning("Failed to install SCIP via conda")

        # If conda failed or not available, build from source
        if not scip_installed:
            print_info("\nBuilding SCIP 9.2.4 from source...")
            print_info("This will take 5-10 minutes...")

            # Create build directory
            build_dir = Path.home() / "scip_build"
            build_dir.mkdir(exist_ok=True)

            # Download SCIP 9.2.4
            scip_url = "https://github.com/scipopt/scip/archive/refs/tags/v924.tar.gz"
            scip_tar = build_dir / "scip-9.2.4.tar.gz"
            scip_src = build_dir / "scip-924"

            print_info(f"Downloading SCIP 9.2.4 to {build_dir}...")
            # Try wget first, then curl as fallback
            if shutil.which("wget"):
                download_cmd = f"cd {build_dir} && wget -O scip-9.2.4.tar.gz {scip_url}"
            elif shutil.which("curl"):
                download_cmd = f"cd {build_dir} && curl -L -o scip-9.2.4.tar.gz {scip_url}"
            else:
                print_error("Neither wget nor curl found - cannot download SCIP")
                download_cmd = None

            if not download_cmd or not run_command(download_cmd, "Download SCIP 9.2.4", check=False):
                print_error("Failed to download SCIP")
                print_info("You may need to install SCIP manually:")
                print_info("  Download from: https://scipopt.org/")
            else:
                # Extract
                print_info("Extracting SCIP source...")
                extract_cmd = f"cd {build_dir} && tar xzf scip-9.2.4.tar.gz"
                if run_command(extract_cmd, "Extract SCIP", check=False):
                    # Build and install
                    print_info("Building SCIP (this takes 5-10 minutes)...")
                    build_commands = [
                        f"cd {scip_src} && mkdir -p build",
                        f"cd {scip_src}/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local",
                        f"cd {scip_src}/build && make -j$(nproc)",
                        f"cd {scip_src}/build && sudo make install"
                    ]

                    build_success = True
                    for cmd in build_commands:
                        if not run_command(cmd, f"Build step: {cmd.split('&&')[-1].strip()}", check=False):
                            build_success = False
                            break

                    if build_success:
                        print_success("SCIP 9.2.4 built and installed successfully!")

                        # Update library cache so SCIP libraries are found
                        print_info("Updating library cache...")
                        run_command("sudo ldconfig", "Update ldconfig", check=False)

                        # Add /usr/local/bin to PATH if SCIP is there but not in PATH
                        if not shutil.which("scip") and Path("/usr/local/bin/scip").exists():
                            print_info("Adding /usr/local/bin to PATH...")
                            os.environ["PATH"] = f"/usr/local/bin:{os.environ.get('PATH', '')}"

                            # Also add to user's .bashrc for future sessions
                            bashrc = Path.home() / ".bashrc"
                            path_export = 'export PATH=/usr/local/bin:$PATH'
                            try:
                                with open(bashrc, 'r') as f:
                                    content = f.read()
                                if path_export not in content:
                                    with open(bashrc, 'a') as f:
                                        f.write(f'\n# Added by SCIP installer\n{path_export}\n')
                                    print_info("Added /usr/local/bin to ~/.bashrc")
                            except:
                                pass

                        # Verify SCIP is now accessible and report its location
                        scip_locations = [
                            "/usr/local/bin/scip",
                            "/usr/bin/scip",
                            str(Path.home() / "miniconda3/bin/scip"),
                            str(Path.home() / "anaconda3/bin/scip"),
                        ]

                        found_at = None
                        for loc in scip_locations:
                            if Path(loc).exists():
                                found_at = loc
                                break

                        if shutil.which("scip"):
                            scip_in_path = shutil.which("scip")
                            print_success(f"SCIP is in PATH at: {scip_in_path}")
                            scip_installed = True
                        elif found_at:
                            print_success(f"SCIP installed at: {found_at}")
                            print_info(f"Not in PATH, but will be found by patched scip_cli.py")
                            scip_installed = True
                        else:
                            print_warning("SCIP built but not found at expected locations")
                            print_info("Checked locations:")
                            for loc in scip_locations:
                                print_info(f"  - {loc}")
                    else:
                        print_error("Failed to build SCIP from source")
                        print_info("You may need to install dependencies or try manually:")
                        print_info("  https://scipopt.org/doc/html/INSTALL.php")

        if not scip_installed:
            print_warning("\nSCIP installation incomplete")
            print_info("The framework will still work but you need SCIP to run optimizations")
            print_info("You can install SCIP later with:")
            print_info("  conda install -c conda-forge scip=9.2.4")
            print_info("Or build from source: https://scipopt.org/")

        print_info("Continuing with Python environment setup...")

    elif pkg_manager in ["yum", "dnf"]:
        mgr_cmd = "dnf" if pkg_manager == "dnf" else "yum"
        print_info(f"Using {mgr_cmd} package manager (RedHat/CentOS/Fedora)")

        # Check if EPEL repository is needed (for SWIG on older RHEL/CentOS)
        if mgr_cmd == "yum":
            print_info("Enabling EPEL repository (required for SWIG and other packages)...")
            epel_cmd = "sudo yum install -y epel-release"
            run_command(epel_cmd, "Install EPEL repository", check=False)

        # Define packages with alternatives for different RHEL/CentOS versions
        package_groups = [
            (["python3.11", "python311", "python3"], "Python 3.11"),
            (["python3.11-devel", "python311-devel", "python3-devel"], "Python development headers"),
            (["gcc"], "GCC compiler"),
            (["gcc-c++"], "G++ compiler"),
            (["make"], "Make"),
            (["swig"], "SWIG"),
            (["gmp-devel"], "GMP development"),
            (["readline-devel"], "Readline development"),
            (["zlib-devel"], "Zlib development"),
            (["bzip2-devel"], "Bzip2 development"),
            (["lapack-devel"], "LAPACK development"),
            (["blas-devel"], "BLAS development"),
            (["gcc-gfortran"], "Fortran compiler"),
        ]

        print("\nInstalling required packages (trying alternatives if needed)...")

        installed_count = 0
        failed_packages = []

        for pkg_alternatives, description in package_groups:
            installed = False
            for pkg in pkg_alternatives:
                # First check if already installed
                check_cmd = f"{mgr_cmd} list installed {pkg} 2>/dev/null"
                check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

                if check_result.returncode == 0:
                    print_success(f"{description}: {pkg} (already installed)")
                    installed = True
                    installed_count += 1
                    break

                # Try to install
                install_cmd = f"sudo {mgr_cmd} install -y {pkg}"
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print_success(f"{description}: {pkg} installed")
                    installed = True
                    installed_count += 1
                    break
                else:
                    # Try next alternative
                    continue

            if not installed:
                print_warning(f"{description}: Not found (tried: {', '.join(pkg_alternatives)})")
                failed_packages.append(description)

        print(f"\nInstalled {installed_count}/{len(package_groups)} package groups")

        if failed_packages:
            print_warning(f"\nFailed to install: {', '.join(failed_packages)}")
            print_info("These packages may not be available in your repositories")

        # Check critical packages
        critical_ok = True
        if not shutil.which("gcc"):
            print_error("GCC compiler not found - required for building Python packages")
            critical_ok = False
        if not shutil.which("swig"):
            print_error("SWIG not found - required for pyrfr (SMAC dependency)")
            critical_ok = False

        if not critical_ok:
            print_error("\nCritical packages missing! Installation cannot continue.")
            print_info("Please ensure you have sudo privileges and try again.")
            return False

        # Install SCIP 9.2.4 - try conda first, then build from source
        print_info("\nInstalling SCIP 9.2.4...")
        scip_installed = False

        # Try conda first if available
        if shutil.which("conda"):
            print_info("Found conda - trying to install SCIP 9.2.4 from conda-forge...")
            scip_cmd = "conda install -c conda-forge scip=9.2.4 -y"
            if run_command(scip_cmd, "Install SCIP 9.2.4 via conda", check=False):
                # Verify SCIP is actually in PATH
                if shutil.which("scip"):
                    print_success("SCIP 9.2.4 installed successfully via conda")
                    scip_installed = True
                else:
                    print_warning("Conda install succeeded but SCIP not in PATH")
            else:
                print_warning("Failed to install SCIP via conda")

        # If conda failed or not available, build from source
        if not scip_installed:
            print_info("\nBuilding SCIP 9.2.4 from source...")
            print_info("This will take 5-10 minutes...")

            # Create build directory
            build_dir = Path.home() / "scip_build"
            build_dir.mkdir(exist_ok=True)

            # Download SCIP 9.2.4
            scip_url = "https://github.com/scipopt/scip/archive/refs/tags/v924.tar.gz"
            scip_tar = build_dir / "scip-9.2.4.tar.gz"
            scip_src = build_dir / "scip-924"

            print_info(f"Downloading SCIP 9.2.4 to {build_dir}...")
            # Try wget first, then curl as fallback
            if shutil.which("wget"):
                download_cmd = f"cd {build_dir} && wget -O scip-9.2.4.tar.gz {scip_url}"
            elif shutil.which("curl"):
                download_cmd = f"cd {build_dir} && curl -L -o scip-9.2.4.tar.gz {scip_url}"
            else:
                print_error("Neither wget nor curl found - cannot download SCIP")
                download_cmd = None

            if not download_cmd or not run_command(download_cmd, "Download SCIP 9.2.4", check=False):
                print_error("Failed to download SCIP")
                print_info("You may need to install SCIP manually:")
                print_info("  Download from: https://scipopt.org/")
            else:
                # Extract
                print_info("Extracting SCIP source...")
                extract_cmd = f"cd {build_dir} && tar xzf scip-9.2.4.tar.gz"
                if run_command(extract_cmd, "Extract SCIP", check=False):
                    # Build and install
                    print_info("Building SCIP (this takes 5-10 minutes)...")
                    build_commands = [
                        f"cd {scip_src} && mkdir -p build",
                        f"cd {scip_src}/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local",
                        f"cd {scip_src}/build && make -j$(nproc)",
                        f"cd {scip_src}/build && sudo make install"
                    ]

                    build_success = True
                    for cmd in build_commands:
                        if not run_command(cmd, f"Build step: {cmd.split('&&')[-1].strip()}", check=False):
                            build_success = False
                            break

                    if build_success:
                        print_success("SCIP 9.2.4 built and installed successfully!")

                        # Update library cache so SCIP libraries are found
                        print_info("Updating library cache...")
                        run_command("sudo ldconfig", "Update ldconfig", check=False)

                        # Add /usr/local/bin to PATH if SCIP is there but not in PATH
                        if not shutil.which("scip") and Path("/usr/local/bin/scip").exists():
                            print_info("Adding /usr/local/bin to PATH...")
                            os.environ["PATH"] = f"/usr/local/bin:{os.environ.get('PATH', '')}"

                            # Also add to user's .bashrc for future sessions
                            bashrc = Path.home() / ".bashrc"
                            path_export = 'export PATH=/usr/local/bin:$PATH'
                            try:
                                with open(bashrc, 'r') as f:
                                    content = f.read()
                                if path_export not in content:
                                    with open(bashrc, 'a') as f:
                                        f.write(f'\n# Added by SCIP installer\n{path_export}\n')
                                    print_info("Added /usr/local/bin to ~/.bashrc")
                            except:
                                pass

                        # Verify SCIP is now accessible and report its location
                        scip_locations = [
                            "/usr/local/bin/scip",
                            "/usr/bin/scip",
                            str(Path.home() / "miniconda3/bin/scip"),
                            str(Path.home() / "anaconda3/bin/scip"),
                        ]

                        found_at = None
                        for loc in scip_locations:
                            if Path(loc).exists():
                                found_at = loc
                                break

                        if shutil.which("scip"):
                            scip_in_path = shutil.which("scip")
                            print_success(f"SCIP is in PATH at: {scip_in_path}")
                            scip_installed = True
                        elif found_at:
                            print_success(f"SCIP installed at: {found_at}")
                            print_info(f"Not in PATH, but will be found by patched scip_cli.py")
                            scip_installed = True
                        else:
                            print_warning("SCIP built but not found at expected locations")
                            print_info("Checked locations:")
                            for loc in scip_locations:
                                print_info(f"  - {loc}")
                    else:
                        print_error("Failed to build SCIP from source")
                        print_info("You may need to install dependencies or try manually:")
                        print_info("  https://scipopt.org/doc/html/INSTALL.php")

        if not scip_installed:
            print_warning("\nSCIP installation incomplete")
            print_info("The framework will still work but you need SCIP to run optimizations")
            print_info("You can install SCIP later with:")
            print_info("  conda install -c conda-forge scip=9.2.4")
            print_info("Or build from source: https://scipopt.org/")

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

    # Check for SWIG before attempting pyrfr installation
    if not shutil.which("swig"):
        print_error("SWIG is not installed - required for pyrfr (SMAC dependency)")
        print_info("Please install SWIG first:")
        print_info("  RHEL/CentOS/Fedora: sudo yum install -y swig")
        print_info("  Debian/Ubuntu: sudo apt-get install -y swig")
        print_info("  Conda: conda install -c conda-forge swig")
        print_info("\nAttempting to install packages anyway (some may fail)...")

    print("This may take several minutes (especially compiling scipy, scikit-learn, pyrfr)...")
    result = run_command(
        f"{venv_name}/bin/pip install -r requirements.txt",
        "Install Python packages from requirements.txt",
        check=False
    )

    if not result:
        print_error("Failed to install Python packages")
        print_info("\nCommon causes and solutions:")
        print_info("  1. Missing SWIG: Install swig package (see above)")
        print_info("  2. Missing compilers: Install gcc, g++, gfortran")
        print_info("  3. Missing development headers: Install python3.11-dev(el)")
        print_info("  4. Missing libraries: Install gmp-devel, blas-devel, lapack-devel")
        print_info("\nTo retry after fixing dependencies:")
        print_info(f"  source {venv_name}/bin/activate")
        print_info(f"  pip install -r requirements.txt")
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


def patch_scip_cli():
    """Patch src/utilities/scip_cli.py to find SCIP in common locations"""
    print_section("Patching SCIP CLI")

    # Debug: Print current working directory
    cwd = os.getcwd()
    print_info(f"Current working directory: {cwd}")

    # Debug: Check if SCIP exists in various locations
    print_info("Checking SCIP locations before patching:")
    scip_check_locations = [
        "/usr/local/bin/scip",
        "/usr/bin/scip",
        str(Path.home() / "miniconda3/bin/scip"),
        str(Path.home() / "anaconda3/bin/scip"),
        str(Path.home() / ".local/bin/scip"),
        "./scip",
        "../scip",
    ]

    for loc in scip_check_locations:
        if Path(loc).exists():
            abs_path = Path(loc).resolve()
            print_success(f"  Found SCIP at: {loc} -> {abs_path}")
        else:
            print_info(f"  Not found: {loc}")

    scip_in_path = shutil.which("scip")
    if scip_in_path:
        print_success(f"  SCIP in PATH: {scip_in_path}")
    else:
        print_warning("  SCIP NOT in PATH")

    scip_cli_path = Path("src/utilities/scip_cli.py")

    if not scip_cli_path.exists():
        print_warning(f"src/utilities/scip_cli.py not found at {scip_cli_path.resolve()}")
        print_info(f"Directory contents: {list(Path('src/utilities').iterdir()) if Path('src/utilities').exists() else 'src/utilities not found'}")
        return False

    print_info(f"Patching file: {scip_cli_path.resolve()}")

    # Read the current file
    try:
        with open(scip_cli_path, 'r') as f:
            content = f.read()

        # Check if already patched
        if "common_locations" in content:
            print_success("scip_cli.py already has the fix - skipping")
            return True

        # Find and replace the old _scip_bin function
        old_function = '''def _scip_bin() -> str:
    return os.environ.get("SCIP_BIN", "scip")'''

        new_function = '''def _scip_bin() -> str:
    """Get SCIP binary path, checking common installation locations."""
    # First check SCIP_BIN environment variable
    if "SCIP_BIN" in os.environ:
        return os.environ["SCIP_BIN"]

    # Check if 'scip' is in PATH
    scip_path = shutil.which("scip")
    if scip_path:
        return scip_path

    # Check common installation locations
    common_locations = [
        "/usr/local/bin/scip",
        "/usr/bin/scip",
        os.path.expanduser("~/miniconda3/bin/scip"),
        os.path.expanduser("~/anaconda3/bin/scip"),
        os.path.expanduser("~/.local/bin/scip"),
        "./scip",  # Current directory
        "../scip", # Parent directory
    ]

    for location in common_locations:
        if os.path.isfile(location) and os.access(location, os.X_OK):
            return location

    # Fall back to 'scip' and let it fail with a clear error
    return "scip"'''

        if old_function in content:
            content = content.replace(old_function, new_function)

            # Write back
            with open(scip_cli_path, 'w') as f:
                f.write(content)

            print_success("Successfully patched scip_cli.py")
            print_info("SCIP will now be automatically detected in the following order:")
            print_info("  1. $SCIP_BIN environment variable")
            print_info("  2. 'scip' in PATH")
            print_info("  3. /usr/local/bin/scip")
            print_info("  4. /usr/bin/scip")
            print_info("  5. ~/miniconda3/bin/scip")
            print_info("  6. ~/anaconda3/bin/scip")
            print_info("  7. ~/.local/bin/scip")
            print_info("  8. ./scip (current directory)")
            print_info("  9. ../scip (parent directory)")

            # Verify the patch worked by reading back
            with open(scip_cli_path, 'r') as f:
                patched_content = f.read()

            if "common_locations" in patched_content:
                print_success("Patch verification: File successfully updated")
            else:
                print_error("Patch verification: WARNING - Patch may not have applied correctly")

            return True
        else:
            print_warning("Could not find expected function signature - file may have been modified")
            return False

    except Exception as e:
        print_error(f"Failed to patch scip_cli.py: {e}")
        return False


def print_summary(venv_name, platform_info, patch_applied=False):
    """Print installation summary and next steps"""
    print_section("Installation Complete!")

    print(f"""
{Colors.GREEN}✓ Virtual environment and Python packages installed!{Colors.END}

{Colors.BOLD}Installed Components:{Colors.END}
  • Python 3.11+ virtual environment
  • Virtual environment: {venv_name}
  • All Python packages from requirements.txt
  • scip_cli.py patch: {"✓ APPLIED" if patch_applied else "✗ NOT APPLIED"}

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

    # Patch scip_cli.py to find SCIP automatically
    patch_result = patch_scip_cli()

    # Final verification before summary
    print_section("Final Verification")
    print_info(f"Installation directory: {os.getcwd()}")
    print_info(f"Virtual environment: {Path(args.venv_name).resolve()}")

    # Show what files exist
    if Path("src/utilities/scip_cli.py").exists():
        print_success("src/utilities/scip_cli.py exists")
    else:
        print_error("src/utilities/scip_cli.py NOT FOUND")

    if Path(args.venv_name).exists():
        print_success(f"{args.venv_name}/ exists")
    else:
        print_error(f"{args.venv_name}/ NOT FOUND")

    # Print summary
    print_summary(args.venv_name, platform_info, patch_result)


if __name__ == "__main__":
    main()
