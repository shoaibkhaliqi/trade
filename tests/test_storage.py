"""Tests for DataStorage: Parquet roundtrip, lineage metadata, safety checks."""

from __future__ import annotations

import pytest
from pandas.testing import assert_frame_equal

from darwin.data.storage import DataStorage


@pytest.fixture
def storage(tmp_path) -> DataStorage:
    return DataStorage(tmp_path / "processed")


def test_roundtrip_preserves_values_and_dtypes(storage, make_ohlcv) -> None:
    df = make_ohlcv(5)

    path = storage.save(df, "SOLUSDT", "1h")
    loaded = storage.load("SOLUSDT", "1h")

    assert path.exists()
    assert path.name == "SOLUSDT_1h.parquet"
    assert_frame_equal(loaded, df)


def test_lineage_metadata_is_embedded_and_readable(storage, make_ohlcv) -> None:
    df = make_ohlcv(3)
    storage.save(df, "SOLUSDT", "15m", metadata={"source": "bybit", "days_back": 180})

    lineage = storage.read_lineage("SOLUSDT", "15m")

    assert lineage["symbol"] == "SOLUSDT"
    assert lineage["timeframe"] == "15m"
    assert lineage["source"] == "bybit"
    assert lineage["days_back"] == 180
    assert "saved_at_utc" in lineage


def test_save_refuses_non_canonical_frames(storage, make_ohlcv) -> None:
    df = make_ohlcv(3).drop(columns=["volume"])

    with pytest.raises(ValueError, match="missing columns"):
        storage.save(df, "SOLUSDT", "1h")


def test_load_missing_dataset_raises_clear_error(storage) -> None:
    with pytest.raises(FileNotFoundError, match="BTCUSDT_1m"):
        storage.load("BTCUSDT", "1m")


def test_loaded_data_sorted_even_if_saved_unsorted(storage, make_ohlcv) -> None:
    df = make_ohlcv(6).sample(frac=1.0, random_state=7).reset_index(drop=True)
    storage.save(df, "SOLUSDT", "1h")

    loaded = storage.load("SOLUSDT", "1h")

    assert loaded["timestamp"].is_monotonic_increasing
