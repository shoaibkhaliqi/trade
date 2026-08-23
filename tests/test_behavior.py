"""Behavioral diversity tests - fingerprints, clone detection, distances."""

from __future__ import annotations

import pytest

from darwin.evolution.behavior import (
    behavior_distance,
    behavior_diversity,
    format_behavior_report,
    summarize_behavior,
)


def _actions(positions: list[float]) -> list[int]:
    """Any action stream; positions are what the fingerprint uses."""
    return [1 if p > 0 else 0 for p in positions]


class TestSummarizeBehavior:
    def test_always_long_signature(self) -> None:
        positions = [1.0] * 100
        b = summarize_behavior(_actions(positions), positions)
        assert b["pos_long_frac"] == 1.0
        assert b["pos_flat_frac"] == 0.0
        assert b["actions"]["long"] == 100

    def test_flat_agent(self) -> None:
        positions = [0.0] * 50
        b = summarize_behavior([0] * 50, positions)
        assert b["pos_flat_frac"] == 1.0
        assert b["behavior_hash"] == summarize_behavior([0] * 50, positions)["behavior_hash"]

    def test_resubmitted_long_equals_quiet_long(self) -> None:
        """LONG-every-bar and HOLD-while-long are the SAME behavior."""
        loud = summarize_behavior([1] * 10, [1.0] * 10)
        quiet = summarize_behavior([0] * 10, [1.0] * 10)
        assert loud["behavior_hash"] == quiet["behavior_hash"]
        assert behavior_distance(loud, quiet) == 0.0

    def test_misaligned_rejected(self) -> None:
        with pytest.raises(ValueError):
            summarize_behavior([0, 1], [0.0])


class TestDistances:
    def test_identical_hash_is_zero(self) -> None:
        positions = [0.0, 1.0, 1.0, 0.0]
        a = summarize_behavior(_actions(positions), positions)
        b = summarize_behavior(_actions(positions), positions)
        assert behavior_distance(a, b) == 0.0

    def test_opposite_behaviors_are_far(self) -> None:
        long_agent = summarize_behavior([1] * 10, [1.0] * 10)
        short_agent = summarize_behavior([2] * 10, [-1.0] * 10)
        assert behavior_distance(long_agent, short_agent) == pytest.approx(1.0)

    def test_half_overlap_is_half(self) -> None:
        a = summarize_behavior([1] * 10, [1.0] * 10)
        b = summarize_behavior([1] * 10, [1.0] * 5 + [0.0] * 5)
        assert behavior_distance(a, b) == pytest.approx(0.5)


class TestPopulation:
    def test_monoculture_detected(self) -> None:
        positions = [1.0] * 20
        clones = [summarize_behavior(_actions(positions), positions)
                  for _ in range(4)]
        report = behavior_diversity(clones)
        assert report.n_unique_behaviors == 1
        assert report.monoculture
        assert "monoculture" in format_behavior_report(report)

    def test_diverse_population_passes(self) -> None:
        behaviors = [
            summarize_behavior([1] * 10, [1.0] * 10),           # always long
            summarize_behavior([0] * 10, [0.0] * 10),           # flat
            summarize_behavior([2] * 10, [-1.0] * 10),          # always short
            summarize_behavior([1] * 5 + [0] * 5, [1.0] * 5 + [0.0] * 5),  # half-long
        ]
        report = behavior_diversity(behaviors)
        assert report.n_unique_behaviors == 4
        assert not report.monoculture
        assert report.mean_pairwise > 0.3

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            behavior_diversity([])
