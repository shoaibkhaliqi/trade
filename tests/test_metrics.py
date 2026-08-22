"""Metrics tests: every statistic is recomputed independently by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.environment.simulator import TradeRecord
from darwin.evaluation.metrics import MetricsReport, periods_per_year


def _curve(equity: list[float]) -> pd.DataFrame:
    n = len(equity)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "equity": equity,
            "position_qty": 0.0,
            "fees_cum": 0.0,
        }
    )


def _trade(net_pnl: float) -> TradeRecord:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return TradeRecord(
        trade_id=0, direction="long", qty=1.0, entry_ts=ts, entry_price=100.0,
        exit_ts=ts, exit_price=100.0 + net_pnl, gross_pnl=net_pnl,
        fees_paid=0.0, net_pnl=net_pnl, bars_held=1,
    )


def test_periods_per_year_matches_timeframe() -> None:
    assert periods_per_year("1h") == pytest.approx(8_760.0)
    assert periods_per_year("15m") == pytest.approx(35_040.0)
    assert periods_per_year("5m") == pytest.approx(105_120.0)


class TestRiskAdjustedRatios:
    def test_sharpe_and_sortino_hand_computed(self) -> None:
        # per-bar returns: +10%, -5%, +2%
        curve = _curve([100.0, 110.0, 104.5, 106.59])
        m = MetricsReport.from_parts(curve, (), "1h")

        r = np.array([0.10, -0.05, 0.02])
        mean_r = r.mean()
        std_r = r.std(ddof=1)
        ppy = 8_760.0
        expected_sharpe = mean_r / std_r * np.sqrt(ppy)
        ddev = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
        expected_sortino = mean_r / ddev * np.sqrt(ppy)

        assert m.sharpe == pytest.approx(expected_sharpe, rel=1e-12)
        assert m.sortino == pytest.approx(expected_sortino, rel=1e-12)

    def test_max_drawdown_exact_value(self) -> None:
        m = MetricsReport.from_parts(_curve([100.0, 120.0, 90.0, 110.0]), (), "1h")
        assert m.max_drawdown == pytest.approx(-0.25)

    def test_flat_equity_reports_zero_not_nan(self) -> None:
        m = MetricsReport.from_parts(_curve([100.0] * 50), (), "15m")
        assert m.total_return == 0.0
        assert m.sharpe == 0.0
        assert m.sortino == 0.0
        assert m.max_drawdown == 0.0


class TestTradeStatistics:
    def test_profit_factor_win_rate_avg(self) -> None:
        trades = (_trade(+10.0), _trade(+20.0), _trade(-5.0))
        m = MetricsReport.from_parts(_curve([100.0] * 4), trades, "1h")

        assert m.profit_factor == pytest.approx(30.0 / 5.0)
        assert m.win_rate == pytest.approx(2.0 / 3.0)
        assert m.avg_trade_net == pytest.approx(25.0 / 3.0)
        assert m.n_trades == 3

    def test_no_losses_means_infinite_profit_factor(self) -> None:
        trades = (_trade(+3.0), _trade(+4.0))
        m = MetricsReport.from_parts(_curve([100.0] * 3), trades, "1h")
        assert m.profit_factor == float("inf")

    def test_only_losses_means_zero_profit_factor(self) -> None:
        trades = (_trade(-3.0), _trade(-4.0))
        m = MetricsReport.from_parts(_curve([100.0] * 3), trades, "1h")
        assert m.profit_factor == 0.0

    def test_no_trades_yields_nan_trade_stats_without_crash(self) -> None:
        m = MetricsReport.from_parts(_curve([100.0, 101.0]), (), "1h")
        assert np.isnan(m.profit_factor)
        assert np.isnan(m.win_rate)
        assert np.isnan(m.avg_trade_net)
        assert m.n_trades == 0
        assert m.exposure == 0.0
