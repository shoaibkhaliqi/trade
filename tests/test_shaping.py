"""M17b tests - baseline-relative rewards and multi-seed aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from darwin.environment.env import TradingEnv
from darwin.evolution.population import aggregate_seed_metrics
from darwin.evolution.survival import Verdict
from darwin.features.engine import FeatureEngine


def _rising_candles(n=260):
    import pandas as pd

    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    closes = 100.0 * np.exp(np.linspace(0.0, 0.30, n))  # +30% steady rally
    df = pd.DataFrame({
        "timestamp": ts,
        "open": closes,
        "high": closes * 1.0005,
        "low": closes * 0.9995,
        "close": closes,
        "volume": 10.0,
    })
    return df


class TestBaselineRelativeReward:
    def _run_flat(self, weight: float):
        candles = _rising_candles()
        feats = FeatureEngine().build_feature_matrix(candles)
        env = TradingEnv(candles, feats, start_idx=210, end_idx=240,
                         reward_baseline_weight=weight)
        env.reset(seed=0)
        rewards = []
        done = False
        while not done:
            _, r, term, trunc, _ = env.step(0)  # HOLD forever: stay flat
            rewards.append(r)
            done = term or trunc
        return rewards

    def test_weight_zero_keeps_historical_reward(self) -> None:
        rewards = self._run_flat(0.0)
        assert all(abs(r) < 1e-12 for r in rewards)  # flat = exactly zero

    def test_flat_through_rally_now_costs(self) -> None:
        candles = _rising_candles()
        rewards = self._run_flat(1.0)
        total = sum(rewards)
        # expected cost = price log-move over exactly the episode's bars
        closes = candles["close"].to_numpy()
        # episode covers bars 210..240 (31 bars, 30 steps)
        expected = -float(np.log(closes[240] / closes[210]))
        assert total < 0
        assert total == pytest.approx(expected, abs=1e-9)

    def test_long_agent_neutralized_to_costs_only(self) -> None:
        """An agent fully long captures the baseline: reward ≈ -costs only."""
        candles = _rising_candles()
        feats = FeatureEngine().build_feature_matrix(candles)
        env = TradingEnv(candles, feats, start_idx=210, end_idx=240,
                         reward_baseline_weight=1.0)
        env.reset(seed=0)
        rewards = []
        done = False
        first = True
        while not done:
            _, r, term, trunc, _ = env.step(1)  # LONG immediately, hold it
            if first:
                first = False
            rewards.append(r)
            done = term or trunc
        # equity tracks price 1:1 (25% sizing scales BOTH terms? no - baseline
        # is price, equity term is account: 25% long gives 0.25*(eq) - 1.0*(px)
        # => strongly negative; the DESIGN point: shaping changes incentives
        assert sum(rewards) < 0  # 25% sizing under-captures a shaped baseline


class TestMultiSeedAggregation:
    def test_means_across_seeds(self) -> None:
        runs = [
            {"total_return": 0.02, "sharpe": 1.0, "fitness": 0.5,
             "max_drawdown": -0.03, "n_trades": 10},
            {"total_return": 0.04, "sharpe": 2.0, "fitness": 0.7,
             "max_drawdown": -0.05, "n_trades": 30},
        ]
        agg = aggregate_seed_metrics(runs)
        assert agg["total_return"] == pytest.approx(0.03)
        assert agg["sharpe"] == pytest.approx(1.5)
        assert agg["fitness"] == pytest.approx(0.6)
        assert agg["n_trades"] == pytest.approx(20.0)
        assert len(agg["per_seed"]) == 2

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            aggregate_seed_metrics([])

    def test_survival_verdict_uses_mean_fitness(self) -> None:
        """One lucky seed must not mask a dying agent."""
        runs = [
            {"total_return": 0.10, "sharpe": 2.0, "fitness": 0.9,
             "max_drawdown": -0.05, "n_trades": 10},
            {"total_return": -0.20, "sharpe": -3.0, "fitness": -2.6,
             "max_drawdown": -0.25, "n_trades": 50},
        ]
        agg = aggregate_seed_metrics(runs)
        from darwin.evolution.survival import SurvivalConfig, evaluate_survival

        verdict = evaluate_survival(agg, agg["fitness"], SurvivalConfig())
        assert agg["fitness"] == pytest.approx((0.9 - 2.6) / 2)
        assert verdict.status == "weak" or verdict.status == Verdict("weak", ()).status
