"""Split-engine tests: ordering, embargo, coverage - the geometry of honesty."""

from __future__ import annotations

import pytest

from darwin.experiments.splits import (
    Segment,
    chronological_split,
    walk_forward_splits,
)


class TestChronological:
    def test_exact_boundaries_and_ordering(self) -> None:
        plan = chronological_split(1000, train_frac=0.7, val_frac=0.15)

        assert (plan.train.start, plan.train.end) == (0, 700)
        assert (plan.validation.start, plan.validation.end) == (700, 850)
        assert (plan.test.start, plan.test.end) == (850, 1000)
        # strict forward ordering, zero overlap
        assert plan.train.end <= plan.validation.start
        assert plan.validation.end <= plan.test.start

    def test_test_gets_remainder(self) -> None:
        plan = chronological_split(999)
        assert len(plan.train) + len(plan.validation) + len(plan.test) == 999

    def test_invalid_fractions_rejected(self) -> None:
        with pytest.raises(ValueError):
            chronological_split(100, train_frac=0.0)
        with pytest.raises(ValueError):
            chronological_split(100, train_frac=0.9, val_frac=0.2)


class TestWalkForward:
    def test_hand_computed_geometry(self) -> None:
        # n=100, train=50, test=10, embargo=5, step=10
        folds = walk_forward_splits(
            100, train_bars=50, test_bars=10, embargo_bars=5, step_bars=10
        )

        assert len(folds) == 4  # origins 0,10,20,30 keep test_end<=100
        t0, s0 = folds[0]
        assert (t0.start, t0.end) == (0, 50)
        assert (s0.start, s0.end) == (55, 65)
        t3, s3 = folds[3]
        assert (t3.start, t3.end) == (30, 80)
        assert (s3.start, s3.end) == (85, 95)

    def test_embargo_gaps_are_enforced(self) -> None:
        for train_seg, test_seg in walk_forward_splits(
            300, train_bars=120, test_bars=30, embargo_bars=8
        ):
            gap = test_seg.start - train_seg.end
            assert gap == 8

    def test_contiguous_coverage_when_step_equals_test(self) -> None:
        folds = walk_forward_splits(200, train_bars=60, test_bars=20)
        tests = [test for _, test in folds]

        assert tests[0].start == 60 + 0 * 20
        # pairwise comparison of intentionally offset sequences (lengths differ)
        for prev, nxt in zip(tests, tests[1:], strict=False):
            assert nxt.start == prev.end          # no gaps...
            assert nxt.start >= prev.start        # ...and always forward
        assert tests[-1].end == 200               # tail covered exactly

    def test_train_never_sees_future_of_its_own_test(self) -> None:
        for train_seg, test_seg in walk_forward_splits(500, train_bars=200,
                                                       test_bars=50, embargo_bars=10):
            assert train_seg.end < test_seg.start

    def test_too_short_dataset_raises_with_clear_error(self) -> None:
        with pytest.raises(ValueError, match="dataset too short"):
            walk_forward_splits(40, train_bars=100, test_bars=10)


def test_segment_is_half_open() -> None:
    seg = Segment(5, 9)
    assert len(seg) == 4
    assert seg.slice() == slice(5, 9)
