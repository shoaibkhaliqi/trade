"""Multi-timeframe features: higher-TF context merged causally onto a base TF.

THE causality rule (the whole point of this module):
- a higher-timeframe candle stamped with open time T closes at T + interval.
  Its features describe [T, T+interval) and are only KNOWABLE at T + interval.
- therefore a base-TF bar at time t may only see htf features whose
  ``open_time + interval <= t`` - implemented by merging on the htf
  AVAILABILITY time (open + interval), never on the open time.

Merging on open time would leak one full htf bar of the future - the exact
bug this module's tests are designed to catch.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from darwin.data.schema import TIMESTAMP_COL
from darwin.features.schema import ALL_FEATURES

DEFAULT_HTF_COLUMNS = [
    "ema_dist_20", "ema_dist_50", "rsi_14", "rv_20", "atr_pct", "ret_1",
]


def merge_timeframe_features(
    base: pd.DataFrame,
    htf: pd.DataFrame,
    *,
    htf_interval: str = "1h",
    columns: list[str] | None = None,
    prefix: str = "h1_",
) -> pd.DataFrame:
    """Attach causal htf feature columns onto the base feature frame.

    Both frames must carry TIMESTAMP_COL + the requested feature columns,
    sorted ascending. Returns base with additional ``prefix + column`` cols.
    """
    cols = columns or DEFAULT_HTF_COLUMNS
    missing = [c for c in cols if c not in htf.columns]
    if missing:
        msg = f"htf frame missing columns: {missing}"
        raise ValueError(msg)

    interval = pd.Timedelta(htf_interval)
    htf_avail = htf[[TIMESTAMP_COL, *cols]].copy()
    htf_avail["available_at"] = htf_avail[TIMESTAMP_COL] + interval
    htf_avail = htf_avail.sort_values("available_at")

    base_sorted = base[[TIMESTAMP_COL]].copy()
    base_sorted["_row"] = range(len(base_sorted))
    merged = pd.merge_asof(
        base_sorted.sort_values(TIMESTAMP_COL),
        htf_avail.sort_values("available_at"),
        left_on=TIMESTAMP_COL,
        right_on="available_at",
        direction="backward",
    ).sort_values("_row")

    out = base.copy()
    for col in cols:
        out[prefix + col] = merged[col].to_numpy()
    return out


def build_multi_timeframe_matrix(
    base_ohlcv: Any,
    base_feats: pd.DataFrame,
    htf_ohlcv: Any,
    htf_feats: pd.DataFrame,
    *,
    htf_interval: str = "1h",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Full multi-TF artifact: timestamp + base features + prefixed htf features."""
    enriched = merge_timeframe_features(
        base_feats, htf_feats, htf_interval=htf_interval, columns=columns
    )
    return enriched[["timestamp", *ALL_FEATURES,
                     *(c for c in enriched.columns if c.startswith("h1_"))]]
