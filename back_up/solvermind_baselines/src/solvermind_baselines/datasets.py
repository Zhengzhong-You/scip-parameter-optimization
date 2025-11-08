from typing import List, Tuple

from solvermind.core.datasets import (
    discover_instances as discover_instances,  # noqa: F401
    train_test_split_by_fraction,
)


def split_instances(files: List[str], train_fraction: float, train_cap: int) -> Tuple[List[str], List[str]]:
    return train_test_split_by_fraction(files, train_fraction, train_cap)


def explicit_split(train: List[str], test: List[str]):
    return list(train), list(test)
