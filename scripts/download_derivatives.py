"""Download funding + open-interest history for the configured symbol.

Usage:
    .venv\\Scripts\\python.exe scripts\\download_derivatives.py
    .venv\\Scripts\\python.exe scripts\\download_derivatives.py --symbol SOLUSDT
"""

from __future__ import annotations

import argparse

import pandas as pd

from darwin.data.downloader import DataDownloader
from darwin.data.storage import DataStorage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="PAXGUSDT")
    parser.add_argument("--oi-interval", default="1h",
                        choices=["15min", "30min", "1h", "4h", "1d"])
    parser.add_argument("--sleep-s", type=float, default=0.05)
    args = parser.parse_args()

    dl = DataDownloader(category="linear", sleep_s=args.sleep_s)
    storage = DataStorage("data/processed")

    end = pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp("2022-01-01", tz="UTC")

    funding = dl.fetch_funding_history(args.symbol, start, end)
    if funding.empty:
        print("no funding history returned")
        return 1
    fpath = storage.save(
        funding, args.symbol, "funding",
        metadata={"kind": "funding", "source": "bybit", "rows": len(funding)},
        required_columns=["timestamp", "funding_rate"],
    )
    print(f"funding: {len(funding)} settlements "
          f"({funding['timestamp'].iloc[0].date()} .. {funding['timestamp'].iloc[-1].date()})"
          f" -> {fpath}")

    oi = dl.fetch_open_interest(args.symbol, start, end, interval=args.oi_interval)
    if oi.empty:
        print("no open-interest history returned")
        return 1
    opath = storage.save(
        oi, args.symbol, f"oi_{args.oi_interval}",
        metadata={"kind": "open_interest", "source": "bybit",
                  "interval": args.oi_interval, "rows": len(oi)},
        required_columns=["timestamp", "open_interest"],
    )
    print(f"open interest: {len(oi)} snapshots "
          f"({oi['timestamp'].iloc[0].date()} .. {oi['timestamp'].iloc[-1].date()})"
          f" -> {opath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
