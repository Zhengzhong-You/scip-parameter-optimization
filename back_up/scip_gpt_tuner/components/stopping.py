from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 0
    min_delta: float = 0.0

    def __post_init__(self):
        self._best = float("inf")
        self._streak = 0

    def update(self, current: float) -> bool:
        """Update with current objective; return True if should stop."""
        if current + self.min_delta < self._best:
            self._best = current
            self._streak = 0
            return False
        self._streak += 1
        return self.patience > 0 and self._streak >= self.patience

