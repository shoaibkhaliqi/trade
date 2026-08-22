"""Compute and persist feature matrices from stored OHLCV datasets.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_features.py                  # all timeframes
    .venv\\Scripts\\python.exe scripts\\build_features.py --timeframes 1h  # subset
"""

from __future__ import annotations

import argparse

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.features.engine import FeatureEngine
from darwin.features.schema import ALL_FEATURES, FEATURE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframes", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]
    timeframes = args.timeframes or list(data_cfg["days_back"].keys())

    src = DataStorage(data_cfg["processed_dir"])
    dst = DataStorage(data_cfg["features_dir"])
    engine = FeatureEngine()
    required = ["timestamp", *ALL_FEATURES]

    exit_code = 0
    for tf in timeframes:
        try:
            df = src.load(symbol, tf)
        except FileNotFoundError:
            print(f"[{symbol} {tf}] no dataset - run download_data.py first")
            exit_code = 1
            continue

        feats = engine.build_feature_matrix(df)
        path = dst.save(
            feats,
            symbol,
            tf,
            metadata={
                "kind": "features",
                "feature_version": FEATURE_VERSION,
                "source_rows": len(df),
                "config": args.config,
            },
            required_columns=required,
        )
        n_complete = int(feats[required].notna().all(axis=1).sum())
        print(
            f"[{symbol} {tf}] rows={len(feats)} cols={feats.shape[1]} "
            f"feature_v{FEATURE_VERSION} fully-valid={n_complete} -> {path}"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
