"""Tests for DataCleaner: deterministic repair of each defect class."""

from __future__ import annotations

import numpy as np
import pandas as pd

from darwin.data.cleaner import DataCleaner
from darwin.data.validator import DataValidator


def _dirty_frame(make_ohlcv):
    df = make_ohlcv(8)
    df = pd.concat([df, df.tail(1)], ignore_index=True)          # duplicate row
    df.loc[1, "high"], df.loc[1, "low"] = df.loc[1, "low"], df.loc[1, "high"]  # impossible
    df.loc[2, "volume"] = -50.0                                   # negative volume
    df.loc[3, "close"] = np.nan                                   # missing value
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffled


def test_clean_repairs_all_defects(make_ohlcv) -> None:
    dirty = _dirty_frame(make_ohlcv)
    cleaner = DataCleaner()
    cleaned, rep = cleaner.clean(dirty)

    # 9 rows in: dup removed (1), impossible OHLC + neg volume (2), nan (1) => 5 survive
    assert rep.rows_in == 9
    assert rep.rows_out == 5
    assert rep.duplicates_removed == 1
    assert rep.invalid_rows_removed == 2
    assert rep.nan_rows_removed == 1
    assert rep.unsorted_fixed
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["timestamp"].is_unique


def test_cleaned_frame_passes_validation(make_ohlcv) -> None:
    cleaned, _ = DataCleaner().clean(_dirty_frame(make_ohlcv))
    rep = DataValidator().validate(cleaned, "1h")

    assert rep.ok, rep.errors


def test_preserves_canonical_column_order_and_dtypes(make_ohlcv) -> None:
    cleaned, _ = DataCleaner().clean(make_ohlcv(4))
    assert list(cleaned.columns)[:2] == ["timestamp", "open"]
    assert str(cleaned["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert all(cleaned[c].dtype == "float64" for c in ("open", "high", "low", "close", "volume"))


def test_clean_does_not_fill_gaps(make_ohlcv) -> None:
    """Gaps must survive cleaning untouched - filling is an explicit later decision."""
    df = make_ohlcv(6).drop(index=[2]).reset_index(drop=True)
    cleaned, rep = DataCleaner().clean(df)

    assert len(cleaned) == 5
    assert rep.total_removed == 0  # nothing was wrong; gap is data, not dirt
    diff = cleaned["timestamp"].diff().iloc[2]
    assert diff == pd.Timedelta(hours=2)  # the hole is still there


def test_empty_frame_handled() -> None:
    cleaned, rep = DataCleaner().clean(pd.DataFrame())
    assert len(cleaned) == 0
    assert rep.rows_in == 0
