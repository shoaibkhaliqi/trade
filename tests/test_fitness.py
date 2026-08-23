"""Fitness tests - including the paralysis trap each compass must survive."""

from __future__ import annotations

import pytest

from darwin.evolution.fitness import FitnessConfig, compute_fitness, preset


def _metrics(ret=0.0, sharpe=0.0, sortino=0.0, dd=0.0, fees=0.0, trades=0):
    return {
        "total_return": ret, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": dd, "fees_paid": fees, "n_trades": trades,
        "initial_capital_proxy": 1000.0,
    }


BASELINE = 0.0092  # M9's real buy&hold on the scoring window


def _cfg(**kw):
    kw.setdefault("baseline_return", BASELINE)
    return FitnessConfig(**kw)


class TestParalysisTrap:
    """The M9 discovery: 5-of-8 agents scored 0.00% by never trading."""

    def test_flat_agent_scores_negative_under_spec(self) -> None:
        flat = _metrics(ret=0.0, trades=0)
        b = compute_fitness(flat, _cfg())
        # gold rose 0.92% while the agent did nothing: negative value added,
        # scaled by the 5% target band => -0.0092/0.05
        assert b.total < 0
        assert b.components["return"] == pytest.approx(-0.0092 / 0.05)

    def test_risk_parity_strawman_crowns_paralysis(self) -> None:
        """Documents WHY baseline-relative return is non-negotiable."""
        flat = _metrics(ret=0.0, trades=0)
        loser = _metrics(ret=-0.013, sharpe=-1.3, dd=0.04, trades=11)

        cfg = preset("risk_parity", baseline_return=BASELINE)
        flat_score = compute_fitness(flat, cfg).total
        loser_score = compute_fitness(loser, cfg).total

        assert flat_score > loser_score  # the failure mode, made explicit

    def test_spec_ranking_prefers_active_winner_over_flat(self) -> None:
        winner = _metrics(ret=0.0093, sharpe=1.36, sortino=1.78,
                          dd=0.034, fees=1.24, trades=9)
        flat = _metrics(ret=0.0, trades=0)

        cfg = _cfg()
        assert compute_fitness(winner, cfg).total > compute_fitness(flat, cfg).total


class TestComponents:
    def test_excess_return_scaling_and_clipping(self) -> None:
        cfg = _cfg(target_return=0.05)
        at_target = compute_fitness(
            _metrics(ret=BASELINE + 0.05), cfg
        )
        beyond = compute_fitness(_metrics(ret=BASELINE + 0.20), cfg)
        assert at_target.components["return"] == pytest.approx(1.0)
        assert beyond.components["return"] == pytest.approx(1.0)  # clipped

    def test_underperforming_baseline_is_negative(self) -> None:
        cfg = _cfg()
        b = compute_fitness(_metrics(ret=BASELINE - 0.05), cfg)
        assert b.components["return"] == pytest.approx(-1.0)

    def test_drawdown_penalty_grows_and_clips(self) -> None:
        cfg = _cfg(dd_tolerance=0.10)
        half = compute_fitness(_metrics(ret=BASELINE, dd=0.05), cfg)
        capped = compute_fitness(_metrics(ret=BASELINE, dd=0.25), cfg)
        assert half.components["drawdown_pen"] == pytest.approx(0.5)
        assert capped.components["drawdown_pen"] == pytest.approx(1.0)

    def test_overtrade_penalty_only_above_cap(self) -> None:
        cfg = _cfg(trade_cap=200)
        ok = compute_fitness(_metrics(ret=BASELINE, trades=200), cfg)
        over = compute_fitness(_metrics(ret=BASELINE, trades=400), cfg)
        assert ok.components["overtrade_pen"] == pytest.approx(0.0)
        assert over.components["overtrade_pen"] == pytest.approx(1.0)

    def test_no_component_can_exceed_its_weight(self) -> None:
        cfg = _cfg()
        extreme = _metrics(ret=1.0, sharpe=99.0, sortino=99.0,
                           dd=1.0, fees=1e6, trades=10**6)
        b = compute_fitness(extreme, cfg)
        assert b.total <= cfg.w_return + cfg.w_sharpe + cfg.w_sortino + 1e-12
        assert b.total >= -(cfg.w_drawdown + cfg.w_fees + cfg.w_overtrade) - 1e-12


class TestPresets:
    def test_unknown_preset_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown fitness preset"):
            preset("sharpe_ultra")

    def test_presets_rank_same_population_differently(self) -> None:
        flat = _metrics(ret=0.0, trades=0)
        cautious_winner = _metrics(ret=0.02, sharpe=1.2, sortino=1.5,
                                   dd=0.02, trades=30)
        wild_winner = _metrics(ret=0.06, sharpe=0.8, sortino=0.9,
                               dd=0.18, trades=400)

        for preset_name in ("spec", "conservative"):
            cfg = preset(preset_name, baseline_return=BASELINE)
            scores = {
                "cautious": compute_fitness(cautious_winner, cfg).total,
                "wild": compute_fitness(wild_winner, cfg).total,
                "flat": compute_fitness(flat, cfg).total,
            }
            assert scores["cautious"] > scores["flat"]
        # conservative compass specifically demotes the wild winner below cautious
        cons = preset("conservative", baseline_return=BASELINE)
        assert (
            compute_fitness(cautious_winner, cons).total
            > compute_fitness(wild_winner, cons).total
        )

    def test_negative_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FitnessConfig(w_return=-1.0)
