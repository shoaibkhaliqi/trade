"""Standalone signal generator for GitHub Actions.

Loads committed historical data, fetches the latest candle from
CryptoCompare (works from datacenter IPs), runs the meta-filter,
and appends the signal to forward_log.md.
"""
# ruff: noqa: E702
# ruff: noqa: E702
import os

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier

STABLE = ["upper_wick_pct", "vol_change", "range_pct",
          "lower_wick_pct", "rel_vol_20"]
THRESHOLD = 0.55
HORIZON = 24

COIN_MAP = {"SOLUSDT": "SOL", "ETHUSDT": "ETH", "BTCUSDT": "BTC"}


def fetch_latest_binance(symbol, limit=48):
    """Try Binance API (largest exchange, usually works from datacenters)."""
    pair = symbol  # SOLUSDT, ETHUSDT, BTCUSDT
    url = "https://api.binance.com/api/v3/klines"
    resp = requests.get(url, params={
        "symbol": pair, "interval": "1h", "limit": limit}, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    records = []
    for r in rows:
        records.append({
            "timestamp": pd.Timestamp(r[0], unit="ms", tz="UTC"),
            "open": float(r[1]), "high": float(r[2]),
            "low": float(r[3]), "close": float(r[4]),
            "volume": float(r[5])})
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_latest_cryptocompare(symbol, limit=48):
    """Fallback: CryptoCompare API."""
    fsym = COIN_MAP.get(symbol, symbol.replace("USDT", ""))
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    resp = requests.get(url, params={
        "fsym": fsym, "tsym": "USDT", "limit": limit}, timeout=15)
    data = resp.json()
    if data.get("Response") != "Success":
        msg = f"CryptoCompare error: {data.get('Message', 'unknown')}"
        raise RuntimeError(msg)
    rows = data["Data"]["Data"]
    records = []
    for r in rows:
        records.append({
            "timestamp": pd.Timestamp(r["time"], unit="s", tz="UTC"),
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volumefrom"])})
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_latest_okx(symbol, limit=48):
    """Fallback: OKX API."""
    inst = symbol.replace("USDT", "-USDT")
    url = "https://www.okx.com/api/v5/market/candles"
    resp = requests.get(url, params={
        "instId": inst, "bar": "1H", "limit": limit}, timeout=15)
    data = resp.json()
    if data.get("code") != "0":
        msg = f"OKX error: {data.get('msg', 'unknown')}"
        raise RuntimeError(msg)
    rows = data["data"]  # newest first
    records = []
    for r in reversed(rows):
        records.append({
            "timestamp": pd.Timestamp(int(r[0]), unit="ms", tz="UTC"),
            "open": float(r[1]), "high": float(r[2]),
            "low": float(r[3]), "close": float(r[4]),
            "volume": float(r[5])})
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_live(symbol, limit=48):
    """Try multiple sources in order. Returns (df, source_name)."""
    sources = [
        ("binance", fetch_latest_binance),
        ("okx", fetch_latest_okx),
        ("cryptocompare", fetch_latest_cryptocompare),
    ]
    for name, fn in sources:
        try:
            df = fn(symbol, limit)
            if len(df) > 10:
                return df, name
        except Exception as e:
            print(f"  {name} failed: {e}")
    return None, "all sources failed"


def compute_features(ohlcv):
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")
    open_ = ohlcv["open"].astype("float64")
    volume = ohlcv["volume"].astype("float64")

    feats = pd.DataFrame(index=ohlcv.index)
    feats["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    feats["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    feats["range_pct"] = (high - low) / close
    feats["vol_change"] = volume.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    feats["rel_vol_20"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    feats["trend_ret_96"] = (close / close.shift(96) - 1.0).fillna(0.0)
    feats["vol_96"] = np.log(close / close.shift(1)).rolling(96).std().fillna(0.0)
    return feats


def ichimoku_signal(ohlcv):
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    n = len(ohlcv)
    actions = ["HOLD"] * n
    pos = 0
    for t in range(78, n):
        above = close.iloc[t] > max(senkou_a.iloc[t], senkou_b.iloc[t])
        below = close.iloc[t] < min(senkou_a.iloc[t], senkou_b.iloc[t])
        tk = tenkan.iloc[t] > kijun.iloc[t]
        if pos != 0:
            if pos > 0 and below:
                actions[t] = "CLOSE"; pos = 0
            elif pos < 0 and above:
                actions[t] = "CLOSE"; pos = 0
            continue
        if above and tk:
            actions[t] = "LONG"; pos = 1
        elif below and not tk:
            actions[t] = "SHORT"; pos = -1
    return actions


def main():
    lines = ["## Forward Signals — " +
             pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")]

    for symbol in ("SOLUSDT", "ETHUSDT", "BTCUSDT"):
        data_path = f"assets/data/{symbol}_1h.parquet"
        if not os.path.exists(data_path):
            lines.append(f"\n### {symbol}\n⚠️ historical data not found")
            continue

        hist = pd.read_parquet(data_path)

        # try live data, fall back to committed data
        live, source = fetch_live(symbol, 48)
        if live is None:
            # use last 48 bars from committed data as "live" (slightly stale but safe)
            live = hist.iloc[-48:].copy().reset_index(drop=True)
            source = "committed data (stale)"
            lines.append(f"\n### {symbol}\n⚠️ using stale committed data")

        live_feats = compute_features(live)

        # generate Ichimoku signals on HISTORICAL data for meta-model training
        hist_actions = ichimoku_signal(hist)

        # generate Ichimoku signal on LIVE data for current bar
        live_actions = ichimoku_signal(live)
        primary = live_actions[-1].upper()

        # meta-model: train on historical signals + labels
        hist_feats = compute_features(hist)
        signal_bars = [i for i, a in enumerate(hist_actions) if a != "HOLD"]
        H = HORIZON
        close_np = hist["close"].astype("float64").to_numpy()
        meta_label = np.full(len(hist), np.nan)
        for i in signal_bars:
            if i + H >= len(hist):
                continue
            d = 1 if hist_actions[i] == "LONG" else -1
            meta_label[i] = 1.0 if (close_np[i + H] - close_np[i]) * d > 0 else 0.0

        feature_cols = STABLE + ["trend_ret_96", "vol_96"]
        X_hist = hist_feats[feature_cols].fillna(0.0)
        valid = ~np.isnan(meta_label)
        y = meta_label

        train_idx = np.arange(len(hist))
        train_idx = train_idx[valid[train_idx]]
        if len(train_idx) < 200:
            lines.append(f"\n### {symbol}\n⚠️ insufficient training data")
            continue

        model = LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.03,
            min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
            verbose=-1, random_state=42)
        model.fit(X_hist.iloc[train_idx], y[train_idx])

        X_latest = live_feats[feature_cols].fillna(0.0)
        prob = float(model.predict_proba(X_latest)[0, 1])
        signal = primary if (primary != "HOLD" and prob >= THRESHOLD) else "HOLD"

        close_price = float(live["close"].iloc[-1])
        bar_time = str(live["timestamp"].iloc[-1])

        lines.append(
            f"\n### {symbol} (source: {source})\n"
            f"| | |\n|---|---|\n"
            f"| Close | ${close_price:,.2f} |\n"
            f"| Primary | {primary} |\n"
            f"| P(win) | {prob:.3f} |\n"
            f"| **Signal** | **{signal}** |\n"
            f"| Bar | {bar_time} |")

    output = "\n".join(lines)
    print(output)

    with open("forward_log.md", "a") as f:
        f.write("\n" + output + "\n\n---\n")


if __name__ == "__main__":
    main()
