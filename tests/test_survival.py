"""Survival verdict tests - graduated states, ordered checks, honest reasons."""

from __future__ import annotations

import pytest

from darwin.evolution.survival import (
    STATUS_ALIVE,
    STATUS_DEAD,
    STATUS_WEAK,
    SurvivalConfig,
    evaluate_survival,
)


def _metrics(ret=0.01, dd=0.03, trades=12):
    return {"total_return": ret, "max_drawdown": dd, "n_trades": trades}


class TestConfigValidation:
    def test_threshold_ordering_enforced(self) -> None:
        with pytest.raises(ValueError, match="dead_fitness"):
            SurvivalConfig(dead_fitness=-0.1, weak_fitness=-0.5)  # inverted
        with pytest.raises(ValueError, match="dead_drawdown"):
            SurvivalConfig(dead_drawdown=0.05, weak_drawdown=0.10)


class TestAlive:
    def test_healthy_agent_is_alive(self) -> None:
        v = evaluate_survival(_metrics(ret=0.05, dd=0.04, trades=20),
                              fitness_total=0.8, cfg=SurvivalConfig())
        assert v.status == STATUS_ALIVE
        assert v.reasons == ()


class TestWeak:
    def test_weak_by_fitness(self) -> None:
        v = evaluate_survival(_metrics(ret=-0.01, dd=0.02, trades=8),
                              fitness_total=-1.0, cfg=SurvivalConfig())
        assert v.status == STATUS_WEAK
        assert any("weak floor" in r for r in v.reasons)

    def test_weak_by_drawdown(self) -> None:
        v = evaluate_survival(_metrics(ret=0.02, dd=0.12, trades=30),
                              fitness_total=0.4, cfg=SurvivalConfig())
        assert v.status == STATUS_WEAK
        assert any("drawdown" in r for r in v.reasons)

    def test_paralysis_is_weak_by_default(self) -> None:
        # flat agent vs a rising baseline scores ~ -0.72 under spec compass
        v = evaluate_survival(_metrics(ret=0.0, dd=0.0, trades=0),
                              fitness_total=-0.72, cfg=SurvivalConfig())
        assert v.status == STATUS_WEAK
        assert any("weak floor" in r for r in v.reasons)


class TestDead:
    def test_dead_by_drawdown(self) -> None:
        v = evaluate_survival(_metrics(ret=-0.05, dd=0.25, trades=40),
                              fitness_total=-1.0, cfg=SurvivalConfig())
        assert v.status == STATUS_DEAD
        assert any("drawdown" in r for r in v.reasons)

    def test_dead_by_fitness_floor(self) -> None:
        v = evaluate_survival(_metrics(ret=-0.03, dd=0.05, trades=15),
                              fitness_total=-2.5, cfg=SurvivalConfig())
        assert v.status == STATUS_DEAD
        assert any("fitness" in r for r in v.reasons)

    def test_drawdown_dominates_weak_zone(self) -> None:
        """A dd beyond the DEATH line must be dead even if fitness looks weak-ish."""
        v = evaluate_survival(_metrics(ret=0.01, dd=0.30, trades=50),
                              fitness_total=-0.7, cfg=SurvivalConfig())
        assert v.status == STATUS_DEAD

    def test_paralysis_death_is_opt_in(self) -> None:
        cfg = SurvivalConfig(paralysis_is_death=True)
        v = evaluate_survival(_metrics(ret=0.0, dd=0.0, trades=0),
                              fitness_total=-0.3, cfg=cfg)
        assert v.status == STATUS_DEAD
        assert any("paralysis" in r for r in v.reasons)

    def test_boundary_inclusive(self) -> None:
        v = evaluate_survival(_metrics(dd=0.20), fitness_total=0.0,
                              cfg=SurvivalConfig())
        assert v.status == STATUS_DEAD  # >= threshold counts


class TestReasons:
    def test_multiple_reasons_accumulate(self) -> None:
        v = evaluate_survival(_metrics(ret=-0.05, dd=0.25, trades=40),
                              fitness_total=-3.0, cfg=SurvivalConfig())
        assert v.status == STATUS_DEAD
        assert len(v.reasons) >= 2
        assert any("drawdown" in r for r in v.reasons)
        assert any("fitness" in r for r in v.reasons)
