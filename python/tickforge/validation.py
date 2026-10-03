"""Time-series cross-validation with purging and embargo.

Labels built from overlapping forward windows make neighbouring samples share
information, so a plain K-fold or a train/test split with no gap leaks the
test set into training. Following Lopez de Prado (2018), training samples
whose label window overlaps the test block are purged, and an additional
embargo is dropped after the test block to cover serial correlation in
features.

All splitters work on positional indices of time-ordered samples and take the
label ``horizon`` in samples.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class PurgedKFold:
    n_splits: int = 5
    horizon: int = 0
    embargo: int = 0

    def split(self, n_samples: int) -> Iterator[Split]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        idx = np.arange(n_samples)
        for test in np.array_split(idx, self.n_splits):
            if test.size == 0:
                continue
            lo, hi = test[0], test[-1]
            # Before the test block: a training label at i spans (i, i+horizon].
            # After it: test labels reach hi+horizon, then the embargo.
            train = idx[(idx < lo - self.horizon) | (idx > hi + self.horizon + self.embargo)]
            yield train, test


@dataclass(frozen=True)
class WalkForward:
    """Rolling (or expanding) origin evaluation: train on the past, test on the next block."""

    train_size: int
    test_size: int
    horizon: int = 0
    step: int | None = None
    expanding: bool = False

    def split(self, n_samples: int) -> Iterator[Split]:
        step = self.step or self.test_size
        if min(self.train_size, self.test_size, step) <= 0:
            raise ValueError("sizes must be positive")
        start = self.train_size + self.horizon
        while start + self.test_size <= n_samples:
            train_end = start - self.horizon  # exclusive; last label ends before the test block starts
            train_start = 0 if self.expanding else train_end - self.train_size
            yield np.arange(train_start, train_end), np.arange(start, start + self.test_size)
            start += step


def assert_no_overlap(train: np.ndarray, test: np.ndarray, horizon: int) -> None:
    """Raise if any training label window intersects the test label window."""
    lo, hi = test.min(), test.max() + horizon
    bad = (train + horizon >= lo) & (train <= hi)
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} training samples overlap the test window")
