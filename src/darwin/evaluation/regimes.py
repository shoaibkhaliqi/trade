"""Market regime labeling and per-regime agent evaluation.

Causality contract (same religion as M2):
- a regime label at bar t uses only data with timestamp <= t
- trend: trailing return over ``trend_window`` bars
- volatility: trailing std of log returns vs the EXPANDING MEDIAN OF PAST
  volatility (shifted one bar) - the classifier may not see today's chaos
  when deciding whether today is chaotic

Per-regime performance sums the agent's equity LOG-returns over each
combined label's bars - additive across disjoint time sets, so bucketing
never assumes contiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from darwin.data.schema import TIMESTAMP_COL

TREND_LABELS = (
    "strong_bear", "weak_bear", "sideways", "weak_bull", "strong_bull",
)


@dataclass(frozen=True)
class RegimeConfig:
    trend_window: int = 96
    vol_window: int = 96
    strong_threshold: float = 0.02   # |window return| bounding the strong bands
    weak_threshold: float = 0.005    # |window return| bounding the weak bands
    vol_quantile: float = 0.5        # median split for high/low volatility
    min_warmup: int | None = None    # defaults to max(trend, vol window)


def regime_timeline(candles: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    """Per-bar regime labels: trend, volatility, and their combination."""
    close = candles["close"].astype("float64")
    log_ret = np.log(close / close.shift(1))

    trend_ret = close / close.shift(cfg.trend_window) - 1.0
    trend = pd.Series("sideways", index=candles.index, dtype="object")
    trend[trend_ret <= -cfg.strong_threshold] = "strong_bear"
    trend[(trend_ret <= -cfg.weak_threshold)
          & (trend_ret > -cfg.strong_threshold)] = "weak_bear"
    trend[(trend_ret > -cfg.weak_threshold)
          & (trend_ret < cfg.weak_threshold)] = "sideways"
    trend[(trend_ret >= cfg.weak_threshold)
          & (trend_ret < cfg.strong_threshold)] = "weak_bull"
    trend[trend_ret >= cfg.strong_threshold] = "strong_bull"

    vol = log_ret.rolling(cfg.vol_window).std()
    # causal threshold: median of PAST vol only (shift 1 excludes today)
    vol_ref = vol.expanding(min_periods=cfg.vol_window).median().shift(1)
    vol_regime = pd.Series("low", index=candles.index, dtype="object")
    vol_regime[vol > vol_ref] = "high"
    vol_regime[vol.isna() | vol_ref.isna()] = "warmup"

    warmup = cfg.min_warmup or max(cfg.trend_window, cfg.vol_window)
    trend[:warmup] = "warmup"
    vol_regime[:warmup] = "warmup"

    out = pd.DataFrame({
        TIMESTAMP_COL: candles[TIMESTAMP_COL].to_numpy(),
        "trend_ret": trend_ret.to_numpy(),
        "vol": vol.to_numpy(),
        "trend": trend.to_numpy(),
        "vol_regime": vol_regime.to_numpy(),
    })
    out["combined"] = np.where(
        out["trend"].to_numpy() == "warmup",
        "warmup",
        out["trend"].astype(str) + "/" + out["vol_regime"].astype(str),
    )
    return out


def regime_performance(
    equity: pd.Series,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Agent return per combined-regime label over the aligned window.

    ``equity`` must be row-aligned with ``timeline`` (one mark per bar).
    """
    if len(equity) != len(timeline):
        msg = f"equity ({len(equity)}) and timeline ({len(timeline)}) misaligned"
        raise ValueError(msg)

    eq = equity.astype("float64").to_numpy()
    log_step = np.diff(np.log(eq), prepend=np.nan)  # per-bar log return of equity
    labels = timeline["combined"].to_numpy()

    rows: list[dict[str, Any]] = []
    for label in pd.unique(labels):
        mask = labels == label
        steps = log_step[mask & ~np.isnan(log_step)]
        total_log = float(steps.sum())
        rows.append({
            "regime": label,
            "bars": int(mask.sum()),
            "time_share": float(mask.mean()),
            "log_return": total_log,
            "return": float(np.exp(total_log) - 1.0),
        })
    # warmup included so per-label returns ALWAYS reconstruct the total;
    # consumers may ignore that row, but no bar goes missing silently
    frame = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
    return frame


def format_regime_table(frame: pd.DataFrame) -> str:
    header = f"{'regime':<22}{'bars':>7}{'share':>8}{'return':>10}"
    lines = [header]
    for _, row in frame.iterrows():
        lines.append(
            f"{row['regime']:<22}{int(row['bars']):>7d}"
            f"{row['time_share']:>8.1%}{row['return']:>+10.2%}"
        )
    return "\n".join(lines)
