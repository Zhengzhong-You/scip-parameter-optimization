"""SolverMind tuner public surface (official name).

Thin wrappers that delegate to the legacy implementation to avoid code
duplication while exposing the official solvermind.* package path.
"""

from scip_gpt_tuner.loop.tune_batch import tune_batch  # noqa: F401

