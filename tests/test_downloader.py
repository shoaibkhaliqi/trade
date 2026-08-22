"""Tests for the Bybit downloader: parsing, pagination, error handling."""

from __future__ import annotations

import pandas as pd
import pytest

from darwin.data.downloader import (
    TIMEFRAME_TO_BYBIT_INTERVAL,
    DataDownloader,
    ExchangeError,
    parse_klines,
)
from darwin.data.schema import OHLCV_COLUMNS


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


class TestParseKlines:
    def test_parses_strings_reverses_order_drops_turnover(self) -> None:
        # raw Bybit rows arrive newest-first as strings, turnover included
        rows = [
            ["1704067200000", "101", "102", "100", "101.5", "5000", "507500"],
            ["1704063600000", "100", "101", "99", "100.5", "4000", "402000"],
        ]
        df = parse_klines(rows)

        assert list(df.columns) == OHLCV_COLUMNS
        assert df["timestamp"].tolist() == sorted(df["timestamp"].tolist())
        assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert df["close"].dtype == "float64"
        assert df.iloc[0]["close"] == pytest.approx(100.5)
        assert "turnover" not in df.columns

    def test_empty_rows_gives_canonical_empty_frame(self) -> None:
        df = parse_klines([])
        assert df.empty
        assert list(df.columns) == OHLCV_COLUMNS


class TestFetcherPagination:
    @staticmethod
    def _fake_exchange(candles: list[tuple[int, float]], calls: list[dict]):
        """Transport mimicking Bybit: newest-first, at most `limit` in window."""

        def transport(url: str, params: dict) -> dict:
            calls.append(dict(params))
            start, end = params["start"], params["end"]
            in_window = [c for c in candles if start <= c[0] <= end]
            page = in_window[-params["limit"] :]  # newest N
            rows = [
                [str(t), f"{o:.4f}", f"{o + 1:.4f}", f"{o - 1:.4f}", f"{o + 0.5:.4f}",
                 "12345.6", "999999"]
                for t, o in reversed(page)
            ]
            return {"retCode": 0, "retMsg": "OK", "result": {"list": rows}}

        return transport

    def test_paginates_backwards_until_start(self) -> None:
        # six hourly candles; fake exchange serves only 3 per request
        t0 = pd.Timestamp("2024-01-01", tz="UTC")
        candles = [(t0 + pd.Timedelta(hours=i)).value // 1_000_000 for i in range(6)]
        candles = [(m, 100.0 + i * 0.5) for i, m in enumerate(candles)]
        calls: list[dict] = []
        dl = DataDownloader(
            transport=self._fake_exchange(candles, calls),
            sleep_s=0.0,
            page_limit=1000,
        )
        # force pagination by shrinking the fake's page size via monkeypatched limit
        dl.page_limit = 3

        df = dl.fetch(
            "SOLUSDT",
            "1h",
            t0,
            t0 + pd.Timedelta(hours=5),
        )

        assert len(calls) == 2, f"expected 2 pages, got {len(calls)}"
        # second page must look strictly backwards from the oldest candle seen
        assert calls[1]["end"] < calls[0]["end"]
        assert len(df) == 6
        assert df["timestamp"].is_monotonic_increasing
        assert df.iloc[0]["timestamp"] == t0
        assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"

    def test_window_bounds_are_respected(self) -> None:
        t0 = pd.Timestamp("2024-01-01", tz="UTC")
        candles = [(_ms(t0) + i * 3_600_000, 50.0 + i) for i in range(10)]
        calls: list[dict] = []
        dl = DataDownloader(transport=self._fake_exchange(candles, calls), sleep_s=0.0)

        df = dl.fetch("SOLUSDT", "1h", t0 + pd.Timedelta(hours=3), t0 + pd.Timedelta(hours=6))

        assert df["timestamp"].min() == t0 + pd.Timedelta(hours=3)
        assert df["timestamp"].max() == t0 + pd.Timedelta(hours=6)


class TestErrorPaths:
    def test_exchange_error_code_raises(self) -> None:
        def transport(url: str, params: dict) -> dict:
            return {"retCode": 10001, "retMsg": "params error", "result": {}}

        dl = DataDownloader(transport=transport, sleep_s=0.0)
        with pytest.raises(ExchangeError, match="10001"):
            dl.fetch("SOLUSDT", "1h", "2024-01-01", "2024-01-02")

    def test_empty_history_returns_canonical_empty(self) -> None:
        def transport(url: str, params: dict) -> dict:
            return {"retCode": 0, "retMsg": "OK", "result": {"list": []}}

        dl = DataDownloader(transport=transport, sleep_s=0.0)
        df = dl.fetch("SOLUSDT", "1h", "2024-01-01", "2024-01-02")
        assert df.empty
        assert list(df.columns) == OHLCV_COLUMNS

    def test_unsupported_timeframe_rejected(self) -> None:
        dl = DataDownloader(transport=lambda u, p: {"retCode": 0}, sleep_s=0.0)
        with pytest.raises(ValueError, match="unsupported timeframe"):
            dl.fetch("SOLUSDT", "2h", "2024-01-01", "2024-01-02")

    def test_timeframe_map_covers_project_timeframes(self) -> None:
        for tf in ("1m", "5m", "15m", "1h"):
            assert tf in TIMEFRAME_TO_BYBIT_INTERVAL
