"""Load stored datasets, print summary statistics, save exploration figures.

Usage:
    .venv\\Scripts\\python.exe scripts\\explore_data.py --timeframes 1h
"""

from __future__ import annotations

import argparse
from pathlib import Path

from darwin.config import load_config
from darwin.data.schema import expected_interval
from darwin.data.storage import DataStorage
from darwin.visualization.plots import plot_overview, plot_returns_histogram

FIGURES_DIR = Path("experiments/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframes", nargs="*", default=None)
    return parser.parse_args()


def candles_per_year(timeframe: str) -> float:
    seconds_per_year = 365.0 * 24 * 3600
    return seconds_per_year / expected_interval(timeframe).total_seconds()


def describe(df, timeframe: str, symbol: str) -> None:
    returns = df["close"].pct_change().dropna()
    cummax = df["close"].cummax()
    max_drawdown = float((df["close"] / cummax - 1.0).min())
    ann_vol = float(returns.std() * candles_per_year(timeframe) ** 0.5)
    print(
        f"  rows={len(df)}  span={df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}\n"
        f"  close: min={df['close'].min():.2f} max={df['close'].max():.2f} "
        f"last={df['close'].iloc[-1]:.2f}\n"
        f"  returns/candle: mean={returns.mean():+.6f} std={returns.std():.5f} "
        f"| annualized vol~{ann_vol:.1%}\n"
        f"  max drawdown over window (close-to-close): {max_drawdown:.2%}"
    )
    print(f"  [{symbol} {timeframe}]")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]
    timeframes = args.timeframes or list(data_cfg["days_back"].keys())
    storage = DataStorage(data_cfg["processed_dir"])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        try:
            df = storage.load(symbol, tf)
        except FileNotFoundError:
            print(f"[{symbol} {tf}] no dataset - run download_data.py first")
            continue
        if df.empty:
            print(f"[{symbol} {tf}] dataset is empty")
            continue

        print(f"\n=== {symbol} {tf} ===")
        describe(df, tf, symbol)
        lineage = storage.read_lineage(symbol, tf)
        print(f"  lineage: source={lineage.get('source')} saved={lineage.get('saved_at_utc')}")

        base = FIGURES_DIR / f"{symbol}_{tf}"
        plot_overview(
            df,
            timeframe=tf,
            title=f"{symbol} {tf} — price / volume / returns",
            save_path=base.with_name(base.name + "_overview.png"),
        )
        plot_returns_histogram(
            df,
            title=f"{symbol} {tf} — distribution of per-candle returns",
            save_path=base.with_name(base.name + "_returns_hist.png"),
        )
        print(f"  figures -> {base}_overview.png, {base}_returns_hist.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
