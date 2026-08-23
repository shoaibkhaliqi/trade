"""Risk engine tests - written as adversarial attacks on the gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.environment.env import TradingEnv
from darwin.environment.simulator import (
    Action,
    SimulatorConfig,
)
from darwin.execution.risk import RiskConfig, RiskContext, RiskManager
from darwin.features.engine import FeatureEngine

DAY1 = pd.Timestamp("2024-01-01", tz="UTC")
DAY2 = pd.Timestamp("2024-01-02", tz="UTC")


def make_ctx(**kw) -> RiskContext:
    defaults: dict = dict(
        timestamp=DAY1,
        bar_index=0,
        mark=100.0,
        equity=1000.0,
        peak_equity=1000.0,
        position_qty=0.0,
        entry_price=float("nan"),
        total_trades=0,
        bars_since_exit=None,
    )
    defaults.update(kw)
    return RiskContext(**defaults)


class TestConfigValidation:
    def test_rejects_nonsensical_limits(self) -> None:
        with pytest.raises(ValueError, match="position_size"):
            RiskConfig(max_position_size_pct=0.0)
        with pytest.raises(ValueError, match="position_size"):
            RiskConfig(max_position_size_pct=101.0)
        with pytest.raises(ValueError, match="leverage"):
            RiskConfig(max_leverage=0.0)
        with pytest.raises(ValueError, match="stop_loss_pct"):
            RiskConfig(stop_loss_pct=-1.0)
        with pytest.raises(ValueError, match="trades_per_day"):
            RiskConfig(max_trades_per_day=0)


class TestStopLossTakeProfit:
    def test_stop_loss_fires_long_inclusive_boundary(self) -> None:
        rm = RiskManager(RiskConfig(stop_loss_pct=2.0))
        ctx = make_ctx(position_qty=1.0, entry_price=100.0, mark=98.0)

        action, size = rm.apply(Action.HOLD, ctx)

        assert action == Action.CLOSE
        assert size is None
        assert rm.stats.auto_exits["stop_loss"] == 1

    def test_stop_loss_fires_short_mirrored(self) -> None:
        rm = RiskManager(RiskConfig(stop_loss_pct=2.0))
        ctx = make_ctx(position_qty=-1.0, entry_price=100.0, mark=102.1)

        action, _ = rm.apply(Action.HOLD, ctx)

        assert action == Action.CLOSE
        assert rm.stats.auto_exits["stop_loss"] == 1

    def test_take_profit_fires(self) -> None:
        rm = RiskManager(RiskConfig(take_profit_pct=3.0))
        ctx = make_ctx(position_qty=1.0, entry_price=100.0, mark=103.0)

        action, _ = rm.apply(Action.HOLD, ctx)

        assert action == Action.CLOSE
        assert rm.stats.auto_exits["take_profit"] == 1

    def test_no_trigger_leaves_proposal_untouched(self) -> None:
        rm = RiskManager(RiskConfig(stop_loss_pct=2.0, take_profit_pct=3.0))
        ctx = make_ctx(position_qty=1.0, entry_price=100.0, mark=100.5)

        action, size = rm.apply(Action.HOLD, ctx)

        assert action == Action.HOLD
        assert size is None


class TestKillSwitch:
    def _latched_manager(self) -> RiskManager:
        rm = RiskManager(RiskConfig(max_drawdown_pct=20.0))
        ctx = make_ctx(
            position_qty=1.0,
            entry_price=125.0,
            mark=100.0,
            equity=790.0,
            peak_equity=1000.0,
        )
        rm.apply(Action.HOLD, ctx)
        return rm

    def test_trips_and_flattens(self) -> None:
        rm = self._latched_manager()
        assert rm.killed
        assert rm.stats.kill_switch_tripped_at == DAY1
        assert rm.stats.auto_exits["max_drawdown"] == 1

    def test_latch_vetoes_all_subsequent_entries_across_days(self) -> None:
        rm = self._latched_manager()

        action, _ = rm.apply(Action.LONG, make_ctx(timestamp=DAY2))

        assert action == Action.HOLD
        assert rm.stats.entries_vetoed == 1

    def test_close_is_never_blocked_even_when_latched(self) -> None:
        rm = self._latched_manager()
        action, _ = rm.apply(Action.CLOSE, make_ctx())
        assert action == Action.CLOSE


class TestDailyLossAndTradeCaps:
    def test_daily_loss_blocks_entries_then_resets_at_midnight(self) -> None:
        rm = RiskManager(RiskConfig(max_daily_loss_pct=5.0))
        rm.apply(Action.HOLD, make_ctx(equity=1000.0))  # opens the day bucket

        blocked, _ = rm.apply(Action.LONG, make_ctx(equity=940.0))
        assert blocked == Action.HOLD  # -6% <= -5%

        allowed, _ = rm.apply(Action.LONG, make_ctx(timestamp=DAY2, equity=900.0))
        assert allowed == Action.LONG  # fresh day

    def test_max_trades_per_day_blocks_then_resets(self) -> None:
        rm = RiskManager(RiskConfig(max_trades_per_day=1))
        ok, _ = rm.apply(Action.LONG, make_ctx(total_trades=0))
        assert ok == Action.LONG

        blocked, _ = rm.apply(Action.LONG, make_ctx(total_trades=1))
        assert blocked == Action.HOLD

        ok2, _ = rm.apply(Action.LONG, make_ctx(timestamp=DAY2, total_trades=1))
        assert ok2 == Action.LONG


class TestCooldown:
    def test_entries_blocked_within_cooldown_window(self) -> None:
        rm = RiskManager(RiskConfig(cooldown_bars=2))

        early, _ = rm.apply(Action.LONG, make_ctx(bars_since_exit=1))
        assert early == Action.HOLD

        ready, _ = rm.apply(Action.LONG, make_ctx(bar_index=2, bars_since_exit=2))
        assert ready == Action.LONG


class TestEntrySizingClamps:
    def test_size_and_leverage_caps_shrink_entries(self) -> None:
        rm = RiskManager(RiskConfig(max_position_size_pct=30.0, max_leverage=0.4))
        action, size = rm.apply(Action.LONG, make_ctx(), base_size_pct=100.0)

        assert action == Action.LONG
        assert size == pytest.approx(30.0)  # min(100, 30, 40)
        assert rm.stats.entries_shrunk == 1

    def test_risk_per_trade_cap_uses_stop_distance(self) -> None:
        # explicit headroom caps so ONLY the risk-per-trade rule can bind
        rm = RiskManager(
            RiskConfig(max_position_size_pct=100.0, max_leverage=1.0,
                       max_risk_per_trade_pct=1.0)
        )
        # stop 2% away => may risk 1% => size <= 50%
        _, size2 = rm.apply(
            Action.LONG, make_ctx(), base_size_pct=100.0, stop_distance_pct=0.02
        )
        # stop 4% away => size <= 25%
        _, size4 = rm.apply(
            Action.LONG, make_ctx(), base_size_pct=100.0, stop_distance_pct=0.04
        )

        assert size2 == pytest.approx(50.0)
        assert size4 == pytest.approx(25.0)

    def test_without_sl_the_risk_cap_cannot_apply(self) -> None:
        rm = RiskManager(
            RiskConfig(max_position_size_pct=100.0, max_leverage=1.0,
                       max_risk_per_trade_pct=1.0)
        )
        _, size = rm.apply(
            Action.LONG, make_ctx(), base_size_pct=100.0, stop_distance_pct=None
        )
        assert size == pytest.approx(100.0)


def _falling_frames():
    """Long-enough falling-market frame (>200 bars clears warmup) + features."""
    n = 260
    drop = 0.005
    closes = [100.0 * (1 - drop) ** i for i in range(n)]
    opens = [c / (1 - drop) for c in closes]
    o = np.asarray(opens)
    c = np.asarray(closes)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    candles = pd.DataFrame(
        {
            "timestamp": ts,
            "open": o,
            "high": np.maximum(o, c) * 1.0005,
            "low": np.minimum(o, c) * 0.9995,
            "close": c,
            "volume": 10.0,
        }
    )
    return candles, FeatureEngine().build_feature_matrix(candles)


class TestEnvIntegration:
    def test_hostile_agent_cannot_bypass_limits(self) -> None:
        """An agent that spams LONG forever into a crash stays inside limits."""
        candles, feats = _falling_frames()
        cfg = SimulatorConfig(
            initial_capital=1000.0, taker_fee_pct=0.055,
            slippage_pct=0.02, position_size_pct=100.0,
        )
        risk = RiskManager(RiskConfig(
            stop_loss_pct=2.0,
            max_drawdown_pct=10.0,
            max_trades_per_day=1,
            cooldown_bars=0,
        ))
        env = TradingEnv(candles, feats, config=cfg, risk=risk)
        env.reset(seed=0)

        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(1)  # spam LONG forever
            done = terminated or truncated

        assert env.risk is not None
        assert env.risk.stats.auto_exits.get("stop_loss", 0) >= 1
        assert env.risk.stats.entries_vetoed > 0
        assert float(env.last_result.equity_curve["position_qty"].iloc[-1]) == 0.0
        # losses contained: a 3%/bar crash with unlimited longs would be ruin
        assert env.last_result.final_equity > 700.0

    def test_risky_env_is_deterministic(self) -> None:
        candles, feats = _falling_frames()
        cfg = SimulatorConfig(taker_fee_pct=0.055, slippage_pct=0.02)

        def rollout() -> list[float]:
            risk = RiskManager(RiskConfig(stop_loss_pct=2.0, cooldown_bars=2))
            env = TradingEnv(candles, feats, config=cfg, risk=risk)
            env.reset(seed=5)
            rewards = []
            done = False
            rng = np.random.default_rng(11)
            while not done:
                _, r, term, trunc, _ = env.step(int(rng.integers(0, 4)))
                rewards.append(r)
                done = term or trunc
            return rewards

        assert rollout() == rollout()
