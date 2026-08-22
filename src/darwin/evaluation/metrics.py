"""Performance metrics for simulation results.

Definitions (written down precisely because they are routinely abused):

- Per-bar returns ``r``: pct change of the equity mark series.
- Sharpe   = mean(r) / std(r) * sqrt(periods_per_year); std uses ddof=1;
  risk-free rate assumed 0 (USDT cash baseline).
- Sortino  = mean(r) / downside_dev * sqrt(periods_per_year),
  downside_dev = sqrt(mean(min(r, 0)^2)) - only downside dispersion counts.
- Both ratios report 0.0 when the denominator is zero: mathematically
  undefined ("no dispersion" / "no losing bars"), reported as honest zero.
- Profit factor = sum of winning net PnL / |sum of losing net PnL|;
  inf when there are wins but no losses, nan when there are no trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from darwin.data.schema import expected_interval
from darwin.environment.simulator import SimResult, TradeRecord


def periods_per_year(timeframe: str) -> float:
    seconds_per_year = 365.0 * 24 * 3600
    return seconds_per_year / expected_interval(timeframe).total_seconds()


@dataclass(frozen=True)
class MetricsReport:
    total_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    profit_factor: float
    win_rate: float
    n_trades: int
    fees_paid: float
    avg_trade_net: float
    exposure: float          # fraction of bars holding a position
    periods_per_year: float

    @classmethod
    def from_parts(
        cls,
        curve: pd.DataFrame,
        trades: tuple[TradeRecord, ...],
        timeframe: str,
    ) -> MetricsReport:
        equity = curve["equity"]
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        # recomputed here so the metric never trusts an externally supplied column
        max_dd = float((equity / equity.cummax() - 1.0).min())

        r = equity.pct_change().dropna()
        ppy = periods_per_year(timeframe)
        sharpe = 0.0
        sortino = 0.0
        if len(r) > 1:
            mean_r = float(r.mean())
            std_r = float(r.std(ddof=1))
            if std_r > 0:
                sharpe = mean_r / std_r * np.sqrt(ppy)
            downside = np.minimum(r.to_numpy(), 0.0)
            ddev = float(np.sqrt((downside**2).mean()))
            if ddev > 0:
                sortino = mean_r / ddev * np.sqrt(ppy)

        nets = np.array([t.net_pnl for t in trades])
        n_trades = len(nets)
        profit_factor = float("nan")
        win_rate = float("nan")
        avg_trade = float("nan")
        if n_trades:
            wins = nets[nets > 0]
            losses = nets[nets < 0]
            loss_sum = abs(float(losses.sum()))
            profit_factor = (
                float(wins.sum()) / loss_sum if loss_sum > 1e-15 else float("inf")
            )
            win_rate = float(len(wins)) / n_trades
            avg_trade = float(nets.mean())

        exposure = float((curve["position_qty"] != 0.0).mean())
        fees_paid = float(curve["fees_cum"].iloc[-1])

        return cls(
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe=sharpe,
            sortino=sortino,
            profit_factor=profit_factor,
            win_rate=win_rate,
            n_trades=n_trades,
            fees_paid=fees_paid,
            avg_trade_net=avg_trade,
            exposure=exposure,
            periods_per_year=ppy,
        )

    @classmethod
    def from_result(cls, result: SimResult, timeframe: str) -> MetricsReport:
        return cls.from_parts(result.equity_curve, result.trades, timeframe)


TABLE_COLUMNS = [
    "total_return", "max_drawdown", "sharpe", "sortino",
    "profit_factor", "win_rate", "n_trades", "fees_paid",
    "avg_trade_net", "exposure",
]


def format_value(col: str, value: float) -> str:
    if col in {"n_trades"}:
        return f"{int(value):>7d}"
    if value != value or value in (float("inf"), float("-inf")):  # nan / +-inf
        return f"{str(value):>9}"
    if col in {"total_return", "max_drawdown", "win_rate", "exposure"}:
        return f"{value:>9.2%}"
    return f"{value:>9.3f}"


def format_header() -> str:
    labels = ["strategy", *TABLE_COLUMNS]
    widths = ["<16", ">9", ">9", ">9", ">9", ">9", ">9", ">7", ">9", ">9", ">9"]
    return " ".join(f"{lab:{w}}" for lab, w in zip(labels, widths, strict=True))


def format_row(name: str, m: MetricsReport) -> str:
    cells = [f"{name:<16}"]
    for col in TABLE_COLUMNS:
        cells.append(format_value(col, getattr(m, col)))
    return " ".join(cells)
