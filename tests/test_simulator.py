"""Simulator tests: every PnL number is independently recomputed by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from darwin.config import load_config
from darwin.environment.simulator import (
    MIN_QTY,
    Action,
    SimulatorConfig,
    TradingSimulator,
    Wallet,
)


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


def _cfg(**kw) -> SimulatorConfig:
    base = dict(
        initial_capital=1000.0,
        taker_fee_pct=0.1,
        slippage_pct=0.0,
        position_size_pct=25.0,
        close_at_end=False,
    )
    base.update(kw)
    return SimulatorConfig(**base)


class TestWallet:
    def test_open_long_charges_exact_fee(self) -> None:
        w = Wallet(1000.0)

        fee = w.open(direction=+1, qty=2.0, price=100.0, fee_rate=0.001)

        assert fee == pytest.approx(0.2)
        assert w.cash == pytest.approx(999.8)
        assert w.qty == 2.0
        assert w.entry == 100.0
        assert w.fees_paid == pytest.approx(0.2)

    def test_close_long_realizes_pnl_and_updates_cash(self) -> None:
        w = Wallet(1000.0)
        w.open(direction=+1, qty=2.0, price=100.0, fee_rate=0.001)

        gross, fee = w.close(price=110.0, fee_rate=0.001)

        assert gross == pytest.approx(20.0)
        assert fee == pytest.approx(0.22)
        assert w.cash == pytest.approx(1019.58)  # 999.8 + 20 - 0.22
        assert not w.has_position

    def test_short_profits_on_decline(self) -> None:
        w = Wallet(1000.0)
        w.open(direction=-1, qty=2.0, price=100.0, fee_rate=0.001)

        gross, fee = w.close(price=90.0, fee_rate=0.001)

        assert gross == pytest.approx(20.0)   # sold 100, bought back 90
        assert fee == pytest.approx(0.18)
        assert w.cash == pytest.approx(1019.62)

    def test_unrealized_and_equity_identity(self) -> None:
        w = Wallet(500.0)
        assert w.unrealized(mark=123.0) == 0.0
        assert w.equity(mark=123.0) == 500.0

        w.open(direction=+1, qty=3.0, price=100.0, fee_rate=0.0)
        assert w.unrealized(mark=110.0) == pytest.approx(30.0)
        assert w.equity(mark=110.0) == pytest.approx(530.0)

    def test_state_guards_raise(self) -> None:
        w = Wallet(100.0)
        with pytest.raises(RuntimeError):
            w.close(price=100.0, fee_rate=0.0)
        w.open(direction=+1, qty=1.0, price=100.0, fee_rate=0.0)
        with pytest.raises(RuntimeError):
            w.open(direction=+1, qty=1.0, price=100.0, fee_rate=0.0)


class TestSimulatorConfig:
    def test_config_validation_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError, match="initial_capital"):
            _cfg(initial_capital=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            _cfg(taker_fee_pct=-0.01)
        with pytest.raises(ValueError, match="position_size_pct"):
            _cfg(position_size_pct=0.0)
        with pytest.raises(ValueError, match="position_size_pct"):
            _cfg(position_size_pct=101.0)


class TestTradingSimulator:
    def test_next_open_execution_hand_math(self) -> None:
        candles = _candles([100, 110, 120], [105, 115, 125])
        sim = TradingSimulator(_cfg())

        result = sim.run(candles, [Action.LONG, Action.HOLD, Action.HOLD])

        # decided after candle 0 -> filled at candle 1's OPEN, not candle 0
        qty = round(250.0 / 110.0, 8)
        fee_in = qty * 110.0 * 0.001
        expected_final = 1000.0 - fee_in + qty * (125.0 - 110.0)
        curve = result.equity_curve

        assert curve.loc[1, "entry_price"] == 110.0
        assert curve["position_qty"].iloc[-1] == qty
        assert curve["equity"].iloc[-1] == pytest.approx(expected_final, rel=1e-12)
        assert result.trades == ()
        assert not result.closed_at_end

    def test_round_trip_records_both_legs_of_fees(self) -> None:
        candles = _candles([100, 110, 120], [105, 115, 125])
        sim = TradingSimulator(_cfg())

        result = sim.run(candles, [Action.LONG, Action.CLOSE, Action.HOLD])

        qty = round(250.0 / 110.0, 8)
        gross = qty * (120.0 - 110.0)
        fee_in = qty * 110.0 * 0.001
        fee_out = qty * 120.0 * 0.001
        trade = result.trades[0]

        assert trade.direction == "long"
        assert trade.entry_price == 110.0
        assert trade.exit_price == 120.0
        assert trade.entry_ts == candles["timestamp"].iloc[1]
        assert trade.exit_ts == candles["timestamp"].iloc[2]
        assert trade.bars_held == 1
        assert trade.fees_paid == pytest.approx(fee_in + fee_out, rel=1e-12)
        assert trade.net_pnl == pytest.approx(gross - fee_in - fee_out, rel=1e-12)
        assert result.final_equity == pytest.approx(
            1000.0 - fee_in + gross - fee_out, rel=1e-12
        )

    def test_pending_on_last_candle_never_fills(self) -> None:
        candles = _candles([100, 110, 120], [105, 115, 125])

        result = TradingSimulator(_cfg()).run(
            candles, [Action.HOLD, Action.HOLD, Action.LONG]
        )

        assert result.n_unfilled_actions == 1
        assert result.trades == ()
        assert result.final_equity == 1000.0

    def test_hold_only_constant_equity(self) -> None:
        candles = _candles([100, 101, 102, 103, 104], [100, 101, 102, 103, 104])

        result = TradingSimulator(_cfg()).run(
            candles, [Action.HOLD] * 5
        )

        curve = result.equity_curve
        assert (curve["equity"] == 1000.0).all()
        assert (curve["drawdown"] == 0.0).all()
        assert (curve["fees_cum"] == 0.0).all()

    def test_long_is_idempotent_while_long(self) -> None:
        candles = _candles([100, 110, 120, 130], [105, 115, 125, 135])

        a = TradingSimulator(_cfg()).run(
            candles, [Action.LONG, Action.HOLD, Action.LONG, Action.HOLD]
        )
        b = TradingSimulator(_cfg()).run(
            candles, [Action.LONG, Action.HOLD, Action.HOLD, Action.HOLD]
        )

        assert_frame_equal(a.equity_curve, b.equity_curve)

    def test_flip_short_to_long_single_candle_event(self) -> None:
        candles = _candles([100, 90, 95], [95, 85, 100])
        sim = TradingSimulator(_cfg())

        result = sim.run(candles, [Action.SHORT, Action.LONG, Action.HOLD])

        qty1 = round(250.0 / 90.0, 8)
        fee_in = qty1 * 90.0 * 0.001
        gross_short = -qty1 * (95.0 - 90.0)          # short lost as price rose
        fee_out = qty1 * 95.0 * 0.001
        trade = result.trades[0]

        assert len(result.trades) >= 1
        assert trade.direction == "short"
        assert trade.gross_pnl == pytest.approx(gross_short, rel=1e-12)
        assert trade.fees_paid == pytest.approx(fee_in + fee_out, rel=1e-12)
        # sizing for the new long used equity marked at close of candle 1
        decision_eq = 1000.0 - fee_in + qty1 * (90.0 - 85.0)
        qty2 = round(decision_eq * 0.25 / 95.0, 8)
        assert result.equity_curve["position_qty"].iloc[-1] == qty2
        assert trade.bars_held == 1

    def test_short_round_trip_profit_hand_math(self) -> None:
        candles = _candles([100, 90, 85], [95, 85, 80])

        result = TradingSimulator(_cfg()).run(
            candles, [Action.SHORT, Action.CLOSE, Action.HOLD]
        )

        qty = round(250.0 / 90.0, 8)
        fee_in = qty * 90.0 * 0.001
        gross = qty * (90.0 - 85.0)
        fee_out = qty * 85.0 * 0.001
        trade = result.trades[0]

        assert trade.direction == "short"
        assert trade.net_pnl == pytest.approx(gross - fee_in - fee_out, rel=1e-12)
        assert result.final_equity == pytest.approx(
            1000.0 - fee_in + gross - fee_out, rel=1e-12
        )
        assert result.final_equity > 1000.0

    def test_forced_liquidation_uses_final_close_and_adverse_slippage(self) -> None:
        candles = _candles([100, 110, 120], [105, 115, 125])
        sim = TradingSimulator(
            _cfg(slippage_pct=0.1, taker_fee_pct=0.0, close_at_end=True)
        )

        result = sim.run(candles, [Action.LONG, Action.HOLD, Action.HOLD])

        assert result.closed_at_end
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(110.0 * 1.001, abs=1e-12)
        assert trade.exit_price == pytest.approx(125.0 * 0.999, abs=1e-12)
        assert trade.exit_ts == candles["timestamp"].iloc[-1]
        assert result.equity_curve["position_qty"].iloc[-1] == 0.0

    def test_equity_curve_rowwise_invariants(self) -> None:
        rng = np.random.default_rng(3)
        n = 60
        rets = rng.normal(0.0, 0.01, n)
        closes = 100.0 * np.exp(np.cumsum(rets))
        opens = np.concatenate([[100.0], closes[:-1]])
        candles = _candles(opens, closes)
        pattern = [Action.LONG, Action.HOLD, Action.SHORT, Action.CLOSE] * (n // 4)

        result = TradingSimulator().run(candles, pattern)
        curve = result.equity_curve

        resid = (curve["cash"] + curve["unrealized"] - curve["equity"]).abs()
        assert (resid < 1e-9).all()
        assert (curve["drawdown"] <= 1e-12).all()
        recomputed_dd = curve["equity"] / curve["equity"].cummax() - 1.0
        assert np.allclose(curve["drawdown"], recomputed_dd, atol=1e-15)

    def test_deterministic_repeat_runs(self) -> None:
        rng = np.random.default_rng(3)
        n = 60
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        candles = _candles(np.r_[100.0, closes[:-1]], closes)
        pattern = [Action.LONG, Action.HOLD, Action.SHORT, Action.CLOSE] * (n // 4)

        a = TradingSimulator().run(candles, pattern)
        b = TradingSimulator().run(candles, pattern)

        assert_frame_equal(a.equity_curve, b.equity_curve)
        assert list(a.trades) == list(b.trades)

    def test_tiny_size_fills_are_skipped(self) -> None:
        candles = _candles([100, 110, 120], [105, 115, 125])
        # 1e-8 % of 1000 -> notional 1e-7 -> qty rounds to 0.0 (< MIN_QTY)
        tiny = _cfg(position_size_pct=1e-8)

        result = TradingSimulator(tiny).run(candles, [Action.LONG, Action.HOLD, Action.HOLD])

        assert result.n_skipped_fills == 1
        assert result.n_unfilled_actions == 0
        assert (result.equity_curve["position_qty"] == 0.0).all()
        assert MIN_QTY > 0  # sanity: guard constant meaningful


def test_development_yaml_simulator_block_maps_to_config() -> None:
    """Documents the YAML->config naming contract (fixed_position_size_pct rename)."""
    sim = load_config("development")["simulator"]
    cfg = SimulatorConfig(
        initial_capital=sim["initial_capital"],
        taker_fee_pct=sim["taker_fee_pct"],
        slippage_pct=sim["slippage_pct"],
        position_size_pct=sim["fixed_position_size_pct"],
    )
    assert cfg.initial_capital == 1000.0
    assert cfg.taker_fee_pct == 0.055
    assert cfg.slippage_pct == 0.02
    assert cfg.position_size_pct == 25.0
