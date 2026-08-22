"""Shared test fixtures for darwin.data tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.data.schema import OHLCV_COLUMNS, expected_interval, to_canonical_timestamps


@pytest.fixture
def make_ohlcv():
    """Factory producing small deterministic OHLCV frames in canonical dtypes."""

    def _make(
        n: int = 8,
        timeframe: str = "1h",
        start: str = "2024-01-01",
        base_price: float = 100.0,
    ) -> pd.DataFrame:
        idx = np.arange(n, dtype=float)
        ts = pd.date_range(start, periods=n, freq=expected_interval(timeframe), tz="UTC")
        open_ = base_price + idx * 0.5
        close = open_ + 0.25
        high = np.maximum(open_, close) + 0.5
        low = np.minimum(open_, close) - 0.5
        volume = 1000.0 + idx * 7.0
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        df["timestamp"] = to_canonical_timestamps(df["timestamp"])
        return df[OHLCV_COLUMNS]

    return _make
