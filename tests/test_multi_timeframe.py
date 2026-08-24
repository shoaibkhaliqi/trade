"""Multi-timeframe causality tests - the boundary is the whole game."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.data.schema import to_canonical_timestamps
from darwin.features.engine import FeatureEngine
from darwin.features.multi_timeframe import merge_timeframe_features
from darwin.features.schema import ALL_FEATURES


def _ohlcv(n: int, freq: str, start: str, base: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    idx = np.arange(n, dtype=float)
    close = base + idx * 0.1
    df = pd.DataFrame({
        "timestamp": ts, "open": close - 0.05, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 10.0,
    })
    df["timestamp"] = to_canonical_timestamps(df["timestamp"])
    return df


@pytest.fixture
def frames():
    base_ohlcv = _ohlcv(600, "15min", "2024-01-01")
    htf_ohlcv = _ohlcv(150, "1h", "2024-01-01", base=100.0)
    base_feats = FeatureEngine().build_feature_matrix(base_ohlcv)
    htf_feats = FeatureEngine().build_feature_matrix(htf_ohlcv)
    return base_ohlcv, htf_ohlcv, base_feats, htf_feats


class TestCausality:
    def test_boundary_bar_sees_only_closed_htf_bar(self, frames) -> None:
        """The 15m bar at exactly 20:00 must see the 19:00 htf bar, NOT 20:00's.

        20:00 chosen past ema_20's warmup so both candidates are non-NaN.
        """
        base_ohlcv, htf_ohlcv, base_feats, htf_feats = frames
        merged = merge_timeframe_features(base_feats, htf_feats)

        boundary = pd.Timestamp("2024-01-01 20:00", tz="UTC")
        row = base_feats.index[base_feats["timestamp"] == boundary][0]

        htf_19 = htf_feats.loc[htf_feats["timestamp"] == boundary
                               - pd.Timedelta("1h"), "ema_dist_20"].iloc[0]
        htf_20 = htf_feats.loc[htf_feats["timestamp"] == boundary,
                               "ema_dist_20"].iloc[0]
        assert not np.isnan(htf_19) and not np.isnan(htf_20)
        assert htf_19 != pytest.approx(htf_20)  # otherwise the test is vacuous

        got = merged.loc[row, "h1_ema_dist_20"]
        assert got == pytest.approx(htf_19)

    def test_future_htf_perturbation_cannot_leak(self, frames) -> None:
        base_ohlcv, htf_ohlcv, base_feats, htf_feats = frames
        k = 100  # within the 150-bar htf frame, past its own warmup
        perturbed = htf_feats.copy()
        perturbed.loc[k:, "rsi_14"] = 99.0

        a = merge_timeframe_features(base_feats, htf_feats)
        b = merge_timeframe_features(base_feats, perturbed)

        cutoff = htf_feats["timestamp"].iloc[k] + pd.Timedelta("1h")
        cols = [c for c in b.columns if c.startswith("h1_")]
        before = base_feats["timestamp"] < cutoff
        after = base_feats["timestamp"] >= cutoff
        # guard against vacuous pass: the perturbation MUST change late bars
        assert (a.loc[after, "h1_rsi_14"].fillna(-1)
                != b.loc[after, "h1_rsi_14"].fillna(-1)).any()
        equal = (a.loc[before, cols].fillna(-999).to_numpy()
                 == b.loc[before, cols].fillna(-999).to_numpy())
        assert equal.all(), "htf future data leaked into base features"

    def test_pre_warmup_htf_rows_yield_nan(self, frames) -> None:
        _b, _h, base_feats, htf_feats = frames
        merged = merge_timeframe_features(base_feats, htf_feats)
        # before the first htf bar closes (+1h), nothing is available
        early = merged[merged["timestamp"] < pd.Timestamp("2024-01-01 02:00",
                                                          tz="UTC")]
        assert early["h1_rsi_14"].isna().all()


class TestContract:
    def test_all_base_features_preserved(self, frames) -> None:
        _b, _h, base_feats, htf_feats = frames
        merged = merge_timeframe_features(base_feats, htf_feats)
        for col in ALL_FEATURES:
            assert col in merged.columns

    def test_missing_htf_column_rejected(self, frames) -> None:
        _b, _h, base_feats, htf_feats = frames
        with pytest.raises(ValueError, match="missing columns"):
            merge_timeframe_features(base_feats, htf_feats, columns=["nope"])
