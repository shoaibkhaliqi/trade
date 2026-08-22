"""Tests for DataValidator: every defect class must be detected."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.data.validator import DataValidator


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator()


class TestCleanData:
    def test_clean_frame_passes(self, validator, make_ohlcv) -> None:
        rep = validator.validate(make_ohlcv(10), "1h")

        assert rep.ok
        assert rep.n_rows == 10
        assert rep.n_duplicates == 0
        assert rep.monotonic_index
        assert rep.timezone_utc
        assert rep.n_missing_candles == 0
        assert rep.n_invalid_ohlc == 0
        assert rep.n_negative_volume == 0
        assert rep.n_nan_rows == 0


class TestTimestampDefects:
    def test_duplicate_timestamp_detected(self, validator, make_ohlcv) -> None:
        df = pd.concat([make_ohlcv(5), make_ohlcv(5).tail(1)], ignore_index=True)
        rep = validator.validate(df, "1h")

        assert not rep.ok
        assert rep.n_duplicates == 1
        assert any("duplicate" in e for e in rep.errors)

    def test_unsorted_timestamps_detected(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(6).sample(frac=1.0, random_state=0).reset_index(drop=True)
        rep = validator.validate(df, "1h")

        assert not rep.monotonic_index
        assert any("not sorted" in e for e in rep.errors)

    def test_tz_naive_timestamps_rejected(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(5)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        rep = validator.validate(df, "1h")

        assert not rep.timezone_utc
        assert not rep.ok
        assert any("tz-aware" in e for e in rep.errors)


class TestGapDetection:
    def test_missing_candle_counted_with_range(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(6)
        expected_gap_after = df.loc[2, "timestamp"]   # last candle before the hole
        expected_next = df.loc[4, "timestamp"]        # first candle after the hole
        df = df.drop(index=3).reset_index(drop=True)

        rep = validator.validate(df, "1h")

        # gaps are warnings (often legitimate), never silent
        assert rep.n_missing_candles == 1
        assert len(rep.gap_ranges) == 1
        prev_ts, next_ts = rep.gap_ranges[0]
        assert prev_ts == expected_gap_after
        assert next_ts == expected_next
        assert any("missing candle" in w for w in rep.warnings)

    def test_multi_candle_gap_counts_each(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(10).drop(index=[3, 4]).reset_index(drop=True)  # 2h hole

        rep = validator.validate(df, "1h")

        assert rep.n_missing_candles == 2


class TestValueDefects:
    def test_invalid_ohlc_relationship_detected(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(5)
        df.loc[1, "high"] = df.loc[1, "close"] - 1.0  # high below close: impossible
        rep = validator.validate(df, "1h")

        assert rep.n_invalid_ohlc == 1
        assert not rep.ok

    def test_nonpositive_price_detected(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(5)
        df.loc[2, "open"] = 0.0
        rep = validator.validate(df, "1h")

        assert rep.n_invalid_ohlc == 1

    def test_negative_volume_detected(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(5)
        df.loc[3, "volume"] = -12.0
        rep = validator.validate(df, "1h")

        assert rep.n_negative_volume == 1
        assert not rep.ok

    def test_nan_values_detected(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(5)
        df.loc[4, "close"] = np.nan
        rep = validator.validate(df, "1h")

        assert rep.n_nan_rows == 1
        assert not rep.ok


class TestSchemaDefects:
    def test_missing_columns_is_fatal(self, validator, make_ohlcv) -> None:
        df = make_ohlcv(5).drop(columns=["volume"])
        rep = validator.validate(df, "1h")

        assert not rep.ok
        assert any("missing columns" in e for e in rep.errors)

    def test_unsupported_timeframe_raises(self, validator, make_ohlcv) -> None:
        with pytest.raises(ValueError, match="unsupported timeframe"):
            validator.validate(make_ohlcv(5), "7m")
