"""Benchmark strategy tests: determinism, contracts, and known behaviors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.agents.strategies import (
    BuyAndHoldStrategy,
    MovingAverageCrossStrategy,
    RandomTraderStrategy,
    RSIMeanReversionStrategy,
    VWAPMeanReversionStrategy,
    default_benchmarks,
)
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.features.engine import FeatureEngine


def _candles(opens, closes, start: str = "2024-01-01") -> pd.DataFrame:
    o = np.asarray(opens, dtype=float)
    c = np.asarray(closes, dtype=float)
    ts = pd.date_range(start, periods=len(o), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": o,
            "high": np.maximum(o, c) + 1.0,
            "low": np.minimum(o, c) - 1.0,
            "close": c,
            "volume": 10.0,
        }
    )


class TestContract:
    def test_all_benchmarks_produce_full_action_vectors(self, make_ohlcv) -> None:
        candles = make_ohlcv(n=300)
        for strategy in default_benchmarks():
            actions = strategy.generate_actions(candles)
            assert len(actions) == 300
            assert all(isinstance(a, Action) for a in actions)


class TestBuyAndHold:
    def test_persistent_target_state(self) -> None:
        actions = BuyAndHoldStrategy().generate_actions(_candles([100] * 5, [100] * 5))
        assert all(a == Action.LONG for a in actions)

    def test_simulator_output_matches_analytical_value(self) -> None:
        # flat entry at open=100, final close 123, no costs, full position
        closes = [100.0, 102.0, 104.0, 106.0, 123.0]
        candles = _candles([100.0] * 5, closes)
        cfg = SimulatorConfig(
            initial_capital=1000.0, taker_fee_pct=0.0, slippage_pct=0.0,
            position_size_pct=100.0, close_at_end=False,
        )
        result = TradingSimulator(cfg).run(
            candles, BuyAndHoldStrategy().generate_actions(candles)
        )
        # qty = round(1000/100, 8) = 10 exactly
        assert result.final_equity == pytest.approx(10.0 * 123.0, rel=1e-12)


class TestRandomTrader:
    def test_same_seed_is_bit_identical(self, make_ohlcv) -> None:
        candles = make_ohlcv(n=300)
        a = RandomTraderStrategy(seed=7).generate_actions(candles)
        b = RandomTraderStrategy(seed=7).generate_actions(candles)
        assert a == b

    def test_different_seeds_diverge(self, make_ohlcv) -> None:
        candles = make_ohlcv(n=300)
        a = RandomTraderStrategy(seed=1).generate_actions(candles)
        b = RandomTraderStrategy(seed=2).generate_actions(candles)
        assert a != b

    def test_action_domain(self, make_ohlcv) -> None:
        candles = make_ohlcv(n=300)
        actions = set(RandomTraderStrategy().generate_actions(candles))
        assert actions <= {Action.HOLD, Action.LONG, Action.SHORT}

    def test_probability_validation(self) -> None:
        with pytest.raises(ValueError):
            RandomTraderStrategy(p_hold=1.0, p_long=1.0)
        with pytest.raises(ValueError):
            RandomTraderStrategy(p_hold=-0.1)


class TestMovingAverageCross:
    def test_warmup_rows_are_hold_then_pure_long_on_uptrend(self, make_ohlcv) -> None:
        candles = make_ohlcv(n=120)  # persistent uptrend
        actions = MovingAverageCrossStrategy(fast=5, slow=20).generate_actions(candles)

        assert all(a == Action.HOLD for a in actions[:19])  # ema_20 warmup
        assert all(a == Action.LONG for a in actions[19:])

    def test_fast_must_be_smaller_than_slow(self) -> None:
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(fast=50, slow=20)


class TestRSIReversion:
    def test_extremes_map_to_expected_directions(self) -> None:
        up = _candles([100.0 + i for i in range(40)], [100.5 + i for i in range(40)])
        down = _candles([200.0 - i for i in range(40)], [199.5 - i for i in range(40)])

        actions_up = RSIMeanReversionStrategy().generate_actions(up)
        actions_down = RSIMeanReversionStrategy().generate_actions(down)

        assert all(a == Action.SHORT for a in actions_up[14:])
        assert all(a == Action.LONG for a in actions_down[14:])
        assert all(a == Action.HOLD for a in actions_up[:14])

    def test_band_validation(self) -> None:
        with pytest.raises(ValueError):
            RSIMeanReversionStrategy(lower=70, upper=30)


class TestVWAPReversion:
    def test_action_matches_sign_of_vwap_distance(self, make_ohlcv) -> None:
        candles = make_ohlcv(n=240)
        threshold = 0.004
        actions = VWAPMeanReversionStrategy(threshold=threshold).generate_actions(candles)

        dist = FeatureEngine().compute(candles)["vwap_dist"].to_numpy()
        for action, d in zip(actions, dist, strict=True):
            if np.isnan(d):
                expected = Action.HOLD
            elif d > threshold:
                expected = Action.SHORT
            elif d < -threshold:
                expected = Action.LONG
            else:
                expected = Action.HOLD
            assert action == expected

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            VWAPMeanReversionStrategy(threshold=0.0)
