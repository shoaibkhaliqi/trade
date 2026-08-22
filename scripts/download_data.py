"""Download SOLUSDT OHLCV from Bybit, validate, clean, and store as Parquet.

Usage:
    .venv\\Scripts\\python.exe scripts\\download_data.py                 # all configured timeframes
    .venv\\Scripts\\python.exe scripts\\download_data.py --timeframes 1h # subset
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from darwin.config import load_config
from darwin.data.cleaner import DataCleaner
from darwin.data.downloader import DataDownloader
from darwin.data.storage import DataStorage
from darwin.data.validator import DataValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development", help="config name in configs/")
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="subset of timeframes to download, e.g. --timeframes 1h 15m",
    )
    parser.add_argument(
        "--days-back",
        type=float,
        default=None,
        help="override the configured history window for all timeframes",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=None,
        help="override delay between API pages (rate-limit tuning)",
    )
    return parser.parse_args()


def report_lines(label: str, rep) -> list[str]:
    lines = [
        f"  {label}: rows={rep.n_rows} dupes={rep.n_duplicates} "
        f"sorted={rep.monotonic_index} utc={rep.timezone_utc}",
        f"          gaps={rep.n_missing_candles} bad_ohlc={rep.n_invalid_ohlc} "
        f"neg_vol={rep.n_negative_volume} nan={rep.n_nan_rows} ok={rep.ok}",
    ]
    for err in rep.errors:
        lines.append(f"          ERROR  {err}")
    for warn in rep.warnings:
        lines.append(f"          WARN   {warn}")
    return lines


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    symbol = data_cfg["symbol"]
    timeframes = args.timeframes or list(data_cfg["days_back"].keys())
    dl_cfg = cfg.get("download", {})

    downloader = DataDownloader(
        category=data_cfg.get("category", "spot"),
        page_limit=dl_cfg.get("page_limit", 1000),
        sleep_s=args.sleep_s if args.sleep_s is not None else dl_cfg.get("sleep_s", 0.15),
    )
    validator = DataValidator()
    cleaner = DataCleaner()
    storage = DataStorage(data_cfg["processed_dir"])

    exit_code = 0
    for tf in timeframes:
        days_back = args.days_back if args.days_back is not None else data_cfg["days_back"][tf]
        print(f"\n=== {symbol} {tf} (days_back={days_back}) ===")
        raw = downloader.fetch_recent(symbol, tf, days_back)
        if raw.empty:
            print("  no data returned - check connectivity/symbol; SKIPPED")
            exit_code = 1
            continue

        raw_rep = validator.validate(raw, tf)
        print("\n".join(report_lines("raw     ", raw_rep)))

        cleaned, clean_rep = cleaner.clean(raw)
        print(
            f"  cleaning: removed={clean_rep.total_removed} "
            f"(nan={clean_rep.nan_rows_removed}, dupes={clean_rep.duplicates_removed}, "
            f"impossible={clean_rep.invalid_rows_removed}) sorted_fixed={clean_rep.unsorted_fixed}"
        )

        final_rep = validator.validate(cleaned, tf)
        print("\n".join(report_lines("cleaned ", final_rep)))
        if not final_rep.ok:
            print("  still failing after cleaning - NOT SAVED")
            exit_code = 1
            continue

        path = storage.save(
            cleaned,
            symbol,
            tf,
            metadata={
                "source": data_cfg.get("source", "bybit"),
                "category": data_cfg.get("category", "spot"),
                "days_back": days_back,
                "config": args.config,
            },
        )
        span_lo = cleaned["timestamp"].iloc[0].date()
        span_hi = cleaned["timestamp"].iloc[-1].date()
        print(f"  saved {len(cleaned)} candles -> {path} ({span_lo} .. {span_hi})")

    return exit_code


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    sys.exit(main())
