"""OHLCV download from Bybit's public v5 REST API (no API key needed).

Pagination model: /market/kline returns at most `limit` candles, newest first,
within [start, end]. We walk backwards: each page's oldest candle becomes the
next page's upper bound (minus 1ms), until we reach the requested start or the
exchange runs out of history.

The HTTP layer is injected (`transport` callable) so tests run fully offline
and deterministically - network is a detail, parsing/pagination logic is not.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import requests

from darwin.data.schema import OHLCV_COLUMNS, empty_ohlcv, to_canonical_timestamps

TIMEFRAME_TO_BYBIT_INTERVAL = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
}

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class ExchangeError(RuntimeError):
    """Raised when the exchange replies with an error code."""


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    """Normalize str / datetime / ms-int / Timestamp to a UTC pandas Timestamp."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.Timestamp(int(value), unit="ms", tz="UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def parse_klines(rows: list[list[str]]) -> pd.DataFrame:
    """Parse raw Bybit kline rows (newest-first strings) into canonical form."""
    if not rows:
        return empty_ohlcv()
    df = pd.DataFrame(
        rows,
        columns=["start", "open", "high", "low", "close", "volume", "turnover"],
    ).drop(columns="turnover")
    for col in ("open", "high", "low", "close", "volume"):
        # all-integral strings like "5000" would infer int64; contract is float64
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df.insert(
        0,
        "timestamp",
        to_canonical_timestamps(df.pop("start").astype("int64"), unit="ms"),
    )
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .reset_index(drop=True)
    )
    return df[OHLCV_COLUMNS]


def requests_transport(base_url: str, session: requests.Session | None = None) -> Transport:
    """Default transport: plain HTTPS GET returning parsed JSON."""
    sess = session or requests.Session()

    def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
        response = sess.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def transport(path_or_url: str, params: dict[str, Any]) -> dict[str, Any]:
        url = path_or_url if path_or_url.startswith("http") else f"{base_url}{path_or_url}"
        return _get(url, params)

    return transport


class DataDownloader:
    """Downloads OHLCV klines from Bybit and returns canonical DataFrames."""

    MAX_PAGES = 20_000

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        category: str = "spot",
        page_limit: int = 1000,
        sleep_s: float = 0.15,
    ) -> None:
        self.transport = transport or requests_transport("https://api.bybit.com")
        self.category = category
        self.page_limit = page_limit
        self.sleep_s = sleep_s

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = self.transport(path, params)
        if int(payload.get("retCode", -1)) != 0:
            msg = f"exchange error {payload.get('retCode')}: {payload.get('retMsg')}"
            raise ExchangeError(msg)
        return payload

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: Any,
        end: Any,
    ) -> pd.DataFrame:
        """Fetch all candles with open time in [start, end], ascending."""
        try:
            interval = TIMEFRAME_TO_BYBIT_INTERVAL[timeframe]
        except KeyError:
            supported = ", ".join(sorted(TIMEFRAME_TO_BYBIT_INTERVAL))
            msg = f"unsupported timeframe '{timeframe}'. Supported: {supported}"
            raise ValueError(msg) from None

        start_ms = int(_to_utc_timestamp(start).value // 1_000_000)
        end_ms = int(_to_utc_timestamp(end).value // 1_000_000)

        pages: list[pd.DataFrame] = []
        cursor_end = end_ms
        for _ in range(self.MAX_PAGES):
            payload = self._get(
                "/v5/market/kline",
                {
                    "category": self.category,
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "start": start_ms,
                    "end": cursor_end,
                    "limit": min(self.page_limit, 1000),
                },
            )
            rows = payload.get("result", {}).get("list", [])
            if not rows:
                break
            batch = parse_klines(rows)
            if batch.empty:
                break
            pages.append(batch)

            oldest_ms = int(batch["timestamp"].min().value // 1_000_000)
            if oldest_ms <= start_ms:
                break
            next_cursor = oldest_ms - 1
            if next_cursor >= cursor_end:  # defensive: exchange made no progress
                break
            cursor_end = next_cursor
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        if not pages:
            return empty_ohlcv()
        df = (
            pd.concat(pages, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="first")
        )
        lo = pd.Timestamp(start_ms * 1_000_000, tz="UTC")
        hi = pd.Timestamp(end_ms * 1_000_000, tz="UTC")
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
        return df.reset_index(drop=True)

    def fetch_recent(self, symbol: str, timeframe: str, days_back: float | None) -> pd.DataFrame:
        """Convenience wrapper: last `days_back` days, or all history if None."""
        end = pd.Timestamp.now(tz="UTC").floor("min")
        start = (
            end - pd.Timedelta(days=days_back)
            if days_back is not None
            else pd.Timestamp("2017-01-01", tz="UTC")  # pre-listing => full history
        )
        return self.fetch(symbol, timeframe, start, end)
