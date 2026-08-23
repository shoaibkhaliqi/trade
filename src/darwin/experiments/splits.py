"""Dataset splitting: chronological plans and leak-safe walk-forward folds.

Rules encoded here (tested, not aspirational):
- Time flows forward: validation comes entirely after train; test after both.
- Walk-forward folds use a ROLLING ORIGIN: each fold trains on ``train_bars``
  ending at an embargo gap, then evaluates on the following ``test_bars``.
  The embargo exists because adjacent candles share information - training on
  data up to bar k and testing at k+1 is soft leakage, not validation.
- With ``step == test_bars`` the test segments tile the tail contiguously,
  so aggregate statistics cover the out-of-sample period without gaps.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """Half-open [start, end) index range into a dataset."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def slice(self) -> slice:
        return slice(self.start, self.end)


@dataclass(frozen=True)
class SplitPlan:
    train: Segment
    validation: Segment
    test: Segment


def chronological_split(
    n_rows: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitPlan:
    """Split [0, n) chronologically into train/validation/test."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        msg = "fractions must be within (0, 1)"
        raise ValueError(msg)
    if train_frac + val_frac >= 1:
        msg = "train+validation fractions must leave room for test"
        raise ValueError(msg)

    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return SplitPlan(
        train=Segment(0, train_end),
        validation=Segment(train_end, val_end),
        test=Segment(val_end, n_rows),
    )


def walk_forward_splits(
    n_rows: int,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
    step_bars: int | None = None,
) -> list[tuple[Segment, Segment]]:
    """Rolling-origin folds: (train_segment, test_segment) pairs.

    Fold k: train = [k*step, k*step + train_bars),
            embargo gap of ``embargo_bars``,
            test  = [k*step + train_bars + embargo, ... + test_bars).
    Folds are produced while the test segment fits inside [0, n).
    """
    if train_bars <= 0 or test_bars <= 0:
        msg = "train_bars and test_bars must be positive"
        raise ValueError(msg)
    if embargo_bars < 0:
        msg = "embargo_bars must be >= 0"
        raise ValueError(msg)
    step = step_bars if step_bars is not None else test_bars
    if step <= 0:
        msg = "step_bars must be positive"
        raise ValueError(msg)

    folds: list[tuple[Segment, Segment]] = []
    origin = 0
    while True:
        test_start = origin + train_bars + embargo_bars
        test_end = test_start + test_bars
        if test_end > n_rows:
            break
        folds.append((Segment(origin, test_start - embargo_bars),
                      Segment(test_start, test_end)))
        origin += step
    if not folds:
        msg = (
            f"dataset too short ({n_rows} rows) for "
            f"train={train_bars} embargo={embargo_bars} test={test_bars}"
        )
        raise ValueError(msg)
    return folds


def iter_folds(
    folds: list[tuple[Segment, Segment]],
) -> Iterator[tuple[int, Segment, Segment]]:
    """Numbered fold iterator - sugar for runners."""
    for i, (train_seg, test_seg) in enumerate(folds):
        yield i, train_seg, test_seg
