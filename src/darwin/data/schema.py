"""Canonical OHLCV schema shared by every later layer.

Contract:
- exactly these columns, in this order
- `timestamp` = candle OPEN time, tz-aware datetime64[ns, UTC]
- prices/volume = float64
- one row per (symbol, timeframe) timestamp, sorted ascending

Later additions (funding rate, open interest) extend this table with new
columns; they never change existing semantics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

TIMESTAMP_COL = "timestamp"
PRICE_COLUMNS = ["open", "high", "low", "close"]
OHLCV_COLUMNS = [TIMESTAMP_COL, *PRICE_COLUMNS, "volume"]

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}


def expected_interval(timeframe: str) -> pd.Timedelta:
    """Return the canonical bar interval for a timeframe label like ``'5m'``."""
    try:
        return pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    except KeyError:
        supported = ", ".join(sorted(TIMEFRAME_SECONDS))
        msg = f"unsupported timeframe '{timeframe}'. Supported: {supported}"
        raise ValueError(msg) from None


def to_canonical_timestamps(values: Any, *, unit: str | None = None) -> pd.Series:
    """Cast any datetime-like Series to the canonical dtype: datetime64[ns, UTC].

    Single source of truth for timestamp dtype: pandas 3 may produce
    microsecond-resolution stamps depending on constructor, Parquet may round-
    trip at 'us' - every producer/loader funnels through here so comparisons
    and merges never silently mismatch units. ``unit`` is forwarded to
    ``pd.to_datetime`` for raw integer input (e.g. unit='ms' for epoch millis).
    """
    return pd.to_datetime(values, utc=True, errors="coerce", unit=unit).astype(
        "datetime64[ns, UTC]"
    )


def empty_ohlcv() -> pd.DataFrame:
    """An empty frame with the canonical dtypes - useful as a stable return value."""
    data: dict[str, pd.Series] = {c: pd.Series(dtype="float64") for c in PRICE_COLUMNS}
    data["volume"] = pd.Series(dtype="float64")
    data[TIMESTAMP_COL] = pd.Series(dtype="datetime64[ns, UTC]")
    return pd.DataFrame(data)[OHLCV_COLUMNS]
