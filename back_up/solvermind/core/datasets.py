from __future__ import annotations

import glob
import math
import os
from typing import List, Tuple


def collect_instances(instances: List[str] | None = None, instances_dir: str | None = None) -> List[str]:
    if instances:
        return [os.path.abspath(p) for p in instances]
    if instances_dir:
        exts = {".mps", ".lp", ".cip"}
        paths: List[str] = []
        for root, _, files in os.walk(instances_dir):
            for fn in files:
                base, ext = os.path.splitext(fn)
                if ext.lower() in exts or fn.lower().endswith(".mps.gz") or fn.lower().endswith(".lp.gz"):
                    paths.append(os.path.join(root, fn))
        paths.sort()
        return paths
    return []


def train_test_split(paths: List[str], L: int | None = None) -> Tuple[List[str], List[str]]:
    N = len(paths)
    if L is None:
        L = min(int(math.floor(0.3 * N)), 20)
        if L == 0 and N > 0:
            L = 1
    train = paths[:L]
    test = paths[L:]
    return train, test


def train_test_split_by_fraction(paths: List[str], train_fraction: float, train_cap: int) -> Tuple[List[str], List[str]]:
    n = len(paths)
    n_train = min(int(n * float(train_fraction) + 1e-9), int(train_cap))
    return paths[:n_train], paths[n_train:]


def discover_instances(instances_dir: str, pattern: str) -> list:
    files = sorted(glob.glob(os.path.join(instances_dir, pattern)))
    return files
