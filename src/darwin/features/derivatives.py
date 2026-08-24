"""Derivatives features: funding rates + open interest merged causally onto bars.

Causality contract (same religion as multi_timeframe):
- a funding settlement stamped T is PUBLIC at T (it settles then) -> a bar at
  time t sees the latest settlement with timestamp <= t
- an OI snapshot stamped T reports the book as of T -> visible to bars t >= T
- trailing stats (ma/z) are computed on the EVENT series using only past
  events, then merged - never on the merged bar frame looking forward.

NaN policy: bars before the first event carry NaN (warmup honesty); the
downstream fillna(0) convention is a documented modeling choice made by the
consumer (supervised_hunt), not hidden here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from darwin.data.schema import TIMESTAMP_COL


def _merge_backward(
    base: pd.DataFrame, events: pd.DataFrame, value_col: str, out_col: str
) -> pd.DataFrame:
    events = events[[TIMESTAMP_COL, value_col]].sort_values(TIMESTAMP_COL)
    merged = pd.merge_asof(
        base[[TIMESTAMP_COL]].sort_values(TIMESTAMP_COL),
        events,
        left_on=TIMESTAMP_COL,
        right_on=TIMESTAMP_COL,
        direction="backward",
    )
    out = base.copy()
    out[out_col] = merged[value_col].to_numpy()
    return out


def merge_derivatives_features(
    base_feats: pd.DataFrame,
    funding: pd.DataFrame,
    oi: pd.DataFrame | None = None,
    *,
    z_window: int = 90,          # trailing funding events for z-score
    oi_change_bars: int = 24,    # OI change lookback in OI-snapshot units
) -> pd.DataFrame:
    """Add fnd_rate, fnd_ma3, fnd_z, oi, oi_chg columns to a feature frame."""
    out = base_feats.copy()
    have_funding = funding is not None and len(funding) > 0
    have_oi = oi is not None and len(oi) > 0

    if have_funding:
        fnd = funding.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        fnd["fnd_ma3"] = fnd["funding_rate"].rolling(3, min_periods=3).mean()
        mean = fnd["funding_rate"].expanding(min_periods=z_window // 3).mean()
        std = fnd["funding_rate"].expanding(min_periods=z_window // 3).std()
        fnd["fnd_z"] = (fnd["funding_rate"] - mean) / std.replace(0.0, np.nan)
        out = _merge_backward(out, fnd, "funding_rate", "fnd_rate")
        out = _merge_backward(out, fnd, "fnd_ma3", "fnd_ma3")
        out = _merge_backward(out, fnd, "fnd_z", "fnd_z")
    else:
        out["fnd_rate"] = np.nan
        out["fnd_ma3"] = np.nan
        out["fnd_z"] = np.nan

    if have_oi:
        oi_s = oi.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        oi_s["oi_ref"] = oi_s["open_interest"].shift(oi_change_bars)
        oi_s["oi_chg"] = oi_s["open_interest"] / oi_s["oi_ref"] - 1.0
        out = _merge_backward(out, oi_s, "open_interest", "oi")
        out = _merge_backward(out, oi_s, "oi_chg", "oi_chg")
    else:
        out["oi"] = np.nan
        out["oi_chg"] = np.nan

    return out
