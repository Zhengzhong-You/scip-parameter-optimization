#!/usr/bin/env python3
"""
SCIP Parameter Optimization Framework - Installation Script
STRICT MODE: Source-only build, exit on ANY failure, NO FALLBACKS
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def fatal_error(message):
    """Print error and exit with -1"""
    print(f"\033[91m✗ {message}\033[0m")
    print(f"\033[91m✗ Installation failed - exiting with error code -1\033[0m")
    sys.exit(-1)


def run_cmd(cmd, desc):
    """Run command, exit on failure"""
    print(f"\033[94mRunning: {desc}\033[0m")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\033[91mFAILED: {desc}\033[0m")
        print(f"Command: {cmd}")
        print(f"Exit code: {result.returncode}")
        if result.stdout: print(f"STDOUT:\n{result.stdout}")
        if result.stderr: print(f"STDERR:\n{result.stderr}")
        fatal_error(f"Build step failed: {desc}")
    print(f"\033[92m✓ {desc}\033[0m")


def install_system_deps():
    """Install system dependencies"""
    print("\033[94mInstalling system dependencies...\033[0m")

    # Detect package manager
    if shutil.which("yum"):
        # RHEL/CentOS/Fedora/AnolisOS
        packages = [
            "python3.11", "python3.11-devel", "gcc", "gcc-c++", "make",
            "wget", "tar", "cmake", "swig", "gmp-devel", "readline-devel",
            "zlib-devel", "bzip2-devel", "lapack-devel", "blas-devel", "gcc-gfortran"
        ]
        run_cmd(f"sudo yum install -y {' '.join(packages)}", "Install YUM packages")
    elif shutil.which("apt-get"):
        # Debian/Ubuntu
        run_cmd("sudo apt-get update", "Update package lists")
        packages = [
            "python3.11", "python3.11-dev", "gcc", "g++", "make",
            "wget", "tar", "cmake", "swig", "libgmp-dev", "libreadline-dev",
            "zlib1g-dev", "libbz2-dev", "liblapack-dev", "libblas-dev", "gfortran"
        ]
        run_cmd(f"sudo apt-get install -y {' '.join(packages)}", "Install APT packages")
    else:
        fatal_error("Unsupported package manager - need yum or apt-get")


def main():
    """Main installation - source only, no fallbacks"""
    print("\033[1m\033[94m" + "="*60 + "\033[0m")
    print("\033[1m\033[94mSCIP Parameter Optimization - Source Build Only\033[0m")
    print("\033[1m\033[94m" + "="*60 + "\033[0m")

    # Install system dependencies first
    install_system_deps()

    # Check dependencies are now available
    deps = ["cmake", "make", "gcc", "g++", "wget", "tar", "python3"]
    for dep in deps:
        if not shutil.which(dep):
            fatal_error(f"Required dependency still missing after install: {dep}")
    print("\033[92m✓ All dependencies found\033[0m")

    # Setup paths
    build_dir = Path("scip_build")
    install_dir = Path("scip_install")

    # Clean build
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if install_dir.exists():
        shutil.rmtree(install_dir)

    # Build SCIP from source
    run_cmd(f"mkdir -p {build_dir}", "Create build directory")

    # Download
    run_cmd(f"cd {build_dir} && wget -O scip-9.2.4.tar.gz 'https://scipopt.org/download.php?fname=scip-9.2.4.tar.gz'",
            "Download SCIP 9.2.4")

    # Extract
    run_cmd(f"cd {build_dir} && tar xzf scip-9.2.4.tar.gz", "Extract SCIP")

    # Build
    scip_src = build_dir / "scip-9.2.4"
    run_cmd(f"cd {scip_src} && mkdir -p build", "Create build directory")

    run_cmd(f"cd {scip_src}/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX={Path.cwd()/install_dir} -DPAPILO=off -DZIMPL=off -DIPOPT=off",
            "Configure SCIP")

    run_cmd(f"cd {scip_src}/build && make -j$(nproc || echo 4)", "Compile SCIP")

    run_cmd(f"cd {scip_src}/build && make install", "Install SCIP")

    # Verify
    scip_bin = install_dir / "bin" / "scip"
    if not scip_bin.exists() or not scip_bin.is_file():
        fatal_error(f"SCIP binary not found: {scip_bin}")

    # Test SCIP
    result = subprocess.run([str(scip_bin), "--version"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0 or "9.2.4" not in result.stdout:
        fatal_error("SCIP version test failed")
    print(f"\033[92m✓ SCIP 9.2.4 installed: {scip_bin}\033[0m")

    # Python environment
    venv = "scip_env"
    if Path(venv).exists():
        shutil.rmtree(venv)

    run_cmd(f"python3 -m venv {venv}", "Create Python environment")

    pip_exe = f"{venv}/bin/pip" if platform.system() != "Windows" else f"{venv}\\Scripts\\pip"
    run_cmd(f"{pip_exe} install --upgrade pip", "Upgrade pip")

    if not Path("requirements.txt").exists():
        fatal_error("requirements.txt not found")

    run_cmd(f"{pip_exe} install -r requirements.txt", "Install Python packages")

    print("\033[1m\033[92m" + "="*60 + "\033[0m")
    print("\033[1m\033[92mInstallation Complete!\033[0m")
    print("\033[1m\033[92m" + "="*60 + "\033[0m")
    print(f"SCIP: {install_dir}/bin/scip")
    print(f"Python: {venv}")
    print("\nActivate: source scip_env/bin/activate")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fatal_error("Installation interrupted")
    except Exception as e:
        fatal_error(f"Unexpected error: {e}")