"""Structural validation of OHLCV frames.

The validator never mutates data - it reports. Separating *detection*
(here) from *repair* (cleaner.py) keeps both testable and auditable.

Severity policy:
- errors   : must be fixed before any research use (duplicates, unsorted,
             tz problems, impossible prices/volumes)
- warnings : situational facts worth knowing (gaps are often legitimate -
             exchange downtime, listing events)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from darwin.data.schema import OHLCV_COLUMNS, PRICE_COLUMNS, expected_interval

MAX_GAP_RANGES_STORED = 50


@dataclass(frozen=True)
class ValidationReport:
    """Immutable summary of every check applied to one dataset."""

    n_rows: int
    n_duplicates: int
    monotonic_index: bool
    timezone_utc: bool
    n_missing_candles: int
    n_invalid_ohlc: int
    n_negative_volume: int
    n_nan_rows: int
    gap_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class DataValidator:
    """Checks an OHLCV frame against the structural rules research requires."""

    def validate(self, df: pd.DataFrame, timeframe: str) -> ValidationReport:
        interval = expected_interval(timeframe)
        errors: list[str] = []
        warnings: list[str] = []

        missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"missing columns: {missing_cols}")
            return self._report(df, errors, warnings)

        ts = df["timestamp"]
        tz_ok = isinstance(ts.dtype, pd.DatetimeTZDtype) and str(ts.dt.tz) == "UTC"
        if not isinstance(ts.dtype, pd.DatetimeTZDtype):
            errors.append(
                f"timestamp must be tz-aware datetime64[ns, UTC]; got dtype={ts.dtype}"
            )
        elif str(ts.dt.tz) != "UTC":
            errors.append(f"timestamps must be UTC; got tz={ts.dt.tz}")

        n_duplicates = int(ts.duplicated().sum())
        monotonic = bool(ts.is_monotonic_increasing)
        if n_duplicates:
            errors.append(f"{n_duplicates} duplicate timestamp(s)")
        if not monotonic and not n_duplicates:
            errors.append("timestamps are not sorted ascending")

        gap_ranges, n_missing = self._find_gaps(ts, interval)
        if n_missing:
            warnings.append(
                f"{n_missing} missing candle(s) across {len(gap_ranges)} gap(s)"
            )

        nan_mask = df[PRICE_COLUMNS + ["volume"]].isna().any(axis=1)
        n_nan = int(nan_mask.sum())

        o, h, low, c = df["open"], df["high"], df["low"], df["close"]
        bad_ohlc = (
            df[PRICE_COLUMNS].le(0).any(axis=1)
            | (h < o)
            | (h < c)
            | (low > o)
            | (low > c)
            | (low > h)
        )
        # rows with NaN would also fail comparisons; count them separately above
        bad_ohlc &= ~nan_mask
        n_bad_ohlc = int(bad_ohlc.sum())
        if n_bad_ohlc:
            errors.append(f"{n_bad_ohlc} row(s) violate OHLC relationships / positive prices")

        neg_vol = df["volume"].notna() & (df["volume"] < 0)
        n_neg_vol = int(neg_vol.sum())
        if n_neg_vol:
            errors.append(f"{n_neg_vol} row(s) with negative volume")
        if n_nan:
            errors.append(f"{n_nan} row(s) contain NaN in price/volume columns")

        return self._report(
            df,
            errors,
            warnings,
            duplicates=n_duplicates,
            monotonic=monotonic,
            tz=tz_ok,
            gaps=gap_ranges,
            n_missing=n_missing,
            bad_ohlc=n_bad_ohlc,
            neg_vol=n_neg_vol,
            nan_rows=n_nan,
        )

    def _find_gaps(
        self, ts: pd.Series, interval: pd.Timedelta
    ) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], int]:
        diffs = ts.diff().dropna()
        gap_mask = diffs > interval
        ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        n_missing = 0
        for idx in diffs[gap_mask].index:
            prev_ts = ts.loc[idx - 1]
            missing = int(diffs.loc[idx] // interval) - 1
            n_missing += missing
            if len(ranges) < MAX_GAP_RANGES_STORED:
                ranges.append((prev_ts, ts.loc[idx]))
        return ranges, n_missing

    def _report(
        self,
        df: pd.DataFrame,
        errors: list[str],
        warnings: list[str],
        *,
        duplicates: int = 0,
        monotonic: bool = False,
        tz: bool = False,
        gaps: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        n_missing: int = 0,
        bad_ohlc: int = 0,
        neg_vol: int = 0,
        nan_rows: int = 0,
    ) -> ValidationReport:
        return ValidationReport(
            n_rows=len(df),
            n_duplicates=duplicates,
            monotonic_index=monotonic,
            timezone_utc=tz,
            n_missing_candles=n_missing,
            gap_ranges=gaps or [],
            n_invalid_ohlc=bad_ohlc,
            n_negative_volume=neg_vol,
            n_nan_rows=nan_rows,
            errors=errors,
            warnings=warnings,
        )
