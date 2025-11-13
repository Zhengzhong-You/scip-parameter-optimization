#!/usr/bin/env python3
"""
Simple SCIP CLI runner with instance and parameter file support.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_scip(instance_path, param_file=None, time_limit=None, output_log=None):
    """
    Run SCIP with given instance and parameters.

    Args:
        instance_path: Path to the MPS instance file
        param_file: Optional path to SCIP parameter file
        time_limit: Optional time limit in seconds
        output_log: Optional path to save the log output
    """

    # Build SCIP command
    cmd = ["scip"]

    # Add parameter file if provided
    if param_file:
        if not os.path.exists(param_file):
            print(f"❌ Parameter file not found: {param_file}")
            return None
        cmd.extend(["-s", param_file])

    # Build command sequence (back to simple approach)
    scip_commands = []

    if time_limit:
        scip_commands.append(f"set limits time {time_limit}")

    scip_commands.append("set display freq 100")
    scip_commands.append(f"read {os.path.abspath(instance_path)}")  # Use absolute path
    scip_commands.append("optimize")
    scip_commands.append("quit")

    # Execute commands one by one instead of joining with semicolons
    for scip_cmd in scip_commands:
        cmd.extend(["-c", scip_cmd])

    print(f"🚀 Running SCIP command:")
    print(f"   Instance: {instance_path}")
    print(f"   Parameters: {param_file or 'default'}")
    print(f"   Time limit: {time_limit or 'default'}")
    print(f"   Commands: {scip_commands}")
    print(f"   Full command: {' '.join(cmd)}")

    # Check if instance file exists
    if not os.path.exists(instance_path):
        print(f"❌ Instance file not found: {instance_path}")
        return None

    try:
        # Open log file for real-time writing if requested
        log_file = None
        if output_log:
            log_file = open(output_log, 'w')
            print(f"📄 Logging to: {output_log}")

        print("🚀 Starting SCIP... (Press Ctrl+C to interrupt)")
        print("=" * 60)

        # Run SCIP with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        output_lines = []

        # Read output line by line in real-time
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                # Print to console
                print(line.rstrip())

                # Save to log file if requested
                if log_file:
                    log_file.write(line)
                    log_file.flush()  # Force write to disk

                # Store for return value
                output_lines.append(line)

        # Wait for process to complete
        return_code = process.wait()

        if log_file:
            log_file.close()
            print(f"\n📄 Log saved to: {output_log}")

        print("=" * 60)

        # Print result summary
        if return_code == 0:
            print("✅ SCIP completed successfully")
        else:
            print(f"❌ SCIP failed with return code: {return_code}")

        output = ''.join(output_lines)
        return {
            'returncode': return_code,
            'output': output,
            'stdout': output,
            'stderr': ''
        }

    except subprocess.TimeoutExpired:
        print("⏰ SCIP process timed out")
        return None
    except Exception as e:
        print(f"❌ Error running SCIP: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_scip.py <instance_path> [param_file] [time_limit] [output_log]")
        print("Example: python run_scip.py instance.mps.gz params.set 1600 output.log")
        sys.exit(1)

    instance_path = sys.argv[1]
    param_file = sys.argv[2] if len(sys.argv) > 2 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    output_log = sys.argv[4] if len(sys.argv) > 4 else None

    result = run_scip(instance_path, param_file, time_limit, output_log)

    if result:
        # Print key statistics if available
        lines = result['output'].split('\n')
        for line in lines:
            if 'Solving Time' in line or 'Primal Bound' in line or 'Dual Bound' in line or 'Gap' in line:
                print(f"📊 {line.strip()}")


if __name__ == "__main__":
    main()