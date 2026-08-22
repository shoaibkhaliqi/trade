"""Deterministic repair of known OHLCV defects.

Design decisions (deliberate, documented, testable):
- Sort ascending. Rolling windows and backtests assume chronological order.
- Drop duplicate timestamps keeping the FIRST row. The downloader paginates
  with overlapping windows, so duplicates are re-fetches of identical candles;
  'first' is stable and deterministic.
- Drop rows with impossible prices/volumes or NaN values. We never invent
  prices: fabricating candles would silently poison every feature derived
  from them. A shorter honest dataset beats a longer fake one.
- Gaps are NOT filled here. Gap-filling is a modeling decision (e.g. forward-
  fill only within limits) that belongs to later milestones, made explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from darwin.data.schema import (
    OHLCV_COLUMNS,
    PRICE_COLUMNS,
    TIMESTAMP_COL,
    to_canonical_timestamps,
)


@dataclass(frozen=True)
class CleaningReport:
    rows_in: int
    rows_out: int
    nan_rows_removed: int
    duplicates_removed: int
    unsorted_fixed: bool
    invalid_rows_removed: int

    @property
    def total_removed(self) -> int:
        return self.rows_in - self.rows_out


class DataCleaner:
    """Applies the deterministic cleaning pipeline to an OHLCV frame."""

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        if df.empty:
            return df.copy(), CleaningReport(0, 0, 0, 0, False, 0)

        out = df[OHLCV_COLUMNS].copy()
        rows_in = len(out)

        # 1) coerce numerics: strings like "13.5209" -> float64, bad -> NaN
        numeric_cols = PRICE_COLUMNS + ["volume"]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # 2) drop rows with missing measurements (can't be repaired honestly)
        nan_mask = out[numeric_cols].isna().any(axis=1)
        n_nan = int(nan_mask.sum())
        out = out[~nan_mask]

        # 3) normalize timestamps to UTC; unparseable stamps -> NaT -> dropped
        ts = to_canonical_timestamps(out[TIMESTAMP_COL])
        bad_ts = ts.isna()
        n_nan += int(bad_ts.sum())
        out = out[~bad_ts]
        out.loc[:, TIMESTAMP_COL] = ts[~bad_ts]

        # 4) sort ascending
        was_sorted = bool(out[TIMESTAMP_COL].is_monotonic_increasing)
        out = out.sort_values(TIMESTAMP_COL, kind="stable")

        # 5) drop duplicate timestamps (keep first)
        n_dupes = int(out[TIMESTAMP_COL].duplicated().sum())
        out = out.drop_duplicates(subset=TIMESTAMP_COL, keep="first")

        # 6) drop physically impossible rows
        o, h, low, c = (
            out["open"],
            out["high"],
            out["low"],
            out["close"],
        )
        impossible = (
            out[PRICE_COLUMNS].le(0).any(axis=1)
            | (h < o)
            | (h < c)
            | (low > o)
            | (low > c)
            | (low > h)
            | (out["volume"] < 0)
        )
        n_impossible = int(impossible.sum())
        out = out[~impossible]

        cleaned = out.reset_index(drop=True)
        report = CleaningReport(
            rows_in=rows_in,
            rows_out=len(cleaned),
            nan_rows_removed=n_nan,
            duplicates_removed=n_dupes,
            unsorted_fixed=not was_sorted,
            invalid_rows_removed=n_impossible,
        )
        return cleaned, report
