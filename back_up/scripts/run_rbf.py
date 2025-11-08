#!/usr/bin/env python3
"""Run RBFOpt baseline using shared SolverMind core utilities.

This wrapper sets up PYTHONPATH to include the baselines' src/ layout
and then delegates to the baseline's run logic.
"""

import os
import sys

THIS_DIR = os.path.dirname(__file__)
BASELINES_SRC = os.path.abspath(os.path.join(THIS_DIR, "..", "solvermind_baselines", "src"))
if BASELINES_SRC not in sys.path:
    sys.path.insert(0, BASELINES_SRC)

from solvermind_baselines.scripts.run_rbfopt import main  # type: ignore  # noqa: E402

if __name__ == "__main__":
    main()
