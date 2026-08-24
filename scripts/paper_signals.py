"""Live paper-trading signal generator for Ichimoku + meta-filter.

Fetches the latest candles from Bybit, computes features, runs the
meta-filter, and outputs the current trading signal. This is the signal
engine - not a full execution system. It tells you WHAT to do, not HOW.

Usage:
    .venv\\Scripts\\python.exe scripts\\paper_signals.py --symbol SOLUSDT
    .venv\\Scripts\\python.exe scripts\\paper_signals.py --symbol SOLUSDT --loop 300
"""
# ruff: noqa: E702
import argparse
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action

STABLE = ["upper_wick_pct", "vol_change", "range_pct",
          "lower_wick_pct", "rel_vol_20"]


def fetch_latest_candles(symbol, timeframe="1h", limit=500):
    url = "https://api.bybit.com/v5/market/kline"
    resp = requests.get(url, params={
        "category": "linear", "symbol": symbol,
        "interval": {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}[timeframe],
        "limit": min(limit, 1000),
    }, timeout=30)
    data = resp.json()
    if data.get("retCode") != 0:
        msg = f"Bybit API error: {data.get('retMsg')}"
        raise RuntimeError(msg)
    rows = data["result"]["list"]
    records = []
    for row in reversed(rows):  # oldest first
        records.append({
            "timestamp": pd.Timestamp(int(row[0]), unit="ms", tz="UTC"),
            "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]),
            "volume": float(row[5]),
        })
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def compute_features(ohlcv):
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")
    open_ = ohlcv["open"].astype("float64")
    volume = ohlcv["volume"].astype("float64")

    feats = pd.DataFrame(index=ohlcv.index)
    feats["ret_1"] = close.pct_change()
    feats["body_pct"] = (close - open_) / close
    feats["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    feats["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    feats["range_pct"] = (high - low) / close
    feats["vol_change"] = volume.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    feats["rel_vol_20"] = volume / volume.rolling(20).mean().replace(0.0, np.nan)

    # regime features
    trend_ret = (close / close.shift(96) - 1.0).fillna(0.0)
    vol = np.log(close / close.shift(1)).rolling(96).std().fillna(0.0)
    feats["trend_ret_96"] = trend_ret
    feats["vol_96"] = vol
    return feats


def train_meta_model(ohlcv, feats):
    """Train the meta-filter on historical data."""
    close = ohlcv["close"].astype("float64")
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")

    # generate Ichimoku signals
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    n = len(ohlcv)
    actions = [Action.HOLD] * n
    pos = 0
    for t in range(78, n):
        above = close.iloc[t] > max(senkou_a.iloc[t], senkou_b.iloc[t])
        below = close.iloc[t] < min(senkou_a.iloc[t], senkou_b.iloc[t])
        tk = tenkan.iloc[t] > kijun.iloc[t]
        if pos != 0:
            if pos > 0 and below:
                actions[t] = Action.CLOSE; pos = 0
            elif pos < 0 and above:
                actions[t] = Action.CLOSE; pos = 0
            continue
        if above and tk:
            actions[t] = Action.LONG; pos = 1
        elif below and not tk:
            actions[t] = Action.SHORT; pos = -1

    # meta labels
    H = 24
    close_np = close.to_numpy()
    signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]
    meta_label = np.full(n, np.nan)
    for i in signal_bars:
        if i + H >= n:
            continue
        d = 1 if actions[i] == Action.LONG else -1
        meta_label[i] = 1.0 if (close_np[i + H] - close_np[i]) * d > 0 else 0.0

    # use only the 5 stable features + regime (computed from ohlcv)
    trend_ret = (close / close.shift(96) - 1.0).fillna(0.0)
    vol = np.log(close / close.shift(1)).rolling(96).std().fillna(0.0)
    X = pd.DataFrame({
        "upper_wick_pct": feats["upper_wick_pct"],
        "vol_change": feats["vol_change"],
        "range_pct": feats["range_pct"],
        "lower_wick_pct": feats["lower_wick_pct"],
        "rel_vol_20": feats["rel_vol_20"],
        "trend_ret_96": trend_ret,
        "vol_96": vol,
    }).fillna(0.0)
    valid = ~np.isnan(meta_label)
    y = meta_label

    train_idx = np.arange(n)
    train_idx = train_idx[valid[train_idx]]
    if len(train_idx) < 200:
        msg = f"insufficient training data: {len(train_idx)} labeled signals"
        raise ValueError(msg)

    model = LGBMClassifier(
        n_estimators=150, num_leaves=15, learning_rate=0.03,
        min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
        verbose=-1, random_state=42)
    model.fit(X.iloc[train_idx], y[train_idx])
    return model, actions


def generate_signal(symbol, timeframe="1h"):
    cfg = load_config("development")
    dst = DataStorage(cfg["data"]["features_dir"])

    # fetch latest candles
    ohlcv = fetch_latest_candles(symbol, timeframe, limit=500)
    feats = compute_features(ohlcv)

    # train meta-model on historical data
    hist_ohlcv = dst and src.load(symbol, timeframe)
    hist_feats = dst.load(symbol, timeframe)
    model, hist_actions = train_meta_model(hist_ohlcv, hist_feats)

    # generate Ichimoku signal on latest data
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    t = len(ohlcv) - 1
    above = close.iloc[t] > max(senkou_a.iloc[t], senkou_b.iloc[t])
    below = close.iloc[t] < min(senkou_a.iloc[t], senkou_b.iloc[t])
    tk = tenkan.iloc[t] > kijun.iloc[t]

    if above and tk:
        primary = "LONG"
    elif below and not tk:
        primary = "SHORT"
    else:
        primary = "HOLD"

    # meta-filter: P(win)
    X_latest = feats[STABLE + ["trend_ret_96", "vol_96"]].iloc[[t]].fillna(0.0)
    if primary != "HOLD":
        prob = float(model.predict_proba(X_latest)[0, 1])
        signal = primary if prob >= 0.55 else "HOLD (filtered)"
    else:
        prob = float(model.predict_proba(X_latest)[0, 1])
        signal = "HOLD"

    # current position state
    last_signal = None
    for a in reversed(hist_actions):
        if a != Action.HOLD:
            last_signal = a.value
            break

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": str(ohlcv["timestamp"].iloc[t]),
        "close": float(close.iloc[t]),
        "primary": primary,
        "p_win": round(prob, 3),
        "signal": signal,
        "last_position": last_signal,
    }


src = DataStorage("data/processed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--loop", type=int, default=0,
                        help="poll interval in seconds (0 = run once)")
    args = parser.parse_args()

    if args.loop > 0:
        print(f"polling every {args.loop}s... Ctrl+C to stop")
        while True:
            try:
                sig = generate_signal(args.symbol, args.timeframe)
                print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] "
                      f"{sig['symbol']} close={sig['close']:.2f} | "
                      f"primary={sig['primary']} P(win)={sig['p_win']:.3f} | "
                      f"SIGNAL: {sig['signal']}")
            except Exception as e:
                print(f"error: {e}")
            time.sleep(args.loop)
    else:
        sig = generate_signal(args.symbol, args.timeframe)
        for k, v in sig.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
