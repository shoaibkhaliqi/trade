"""Forward tracker: live signal logging + hypothetical position tracking.

Run daily (or hourly) to:
1. Fetch latest candles from Bybit
2. Generate Ichimoku + meta-filter signal
3. Log to SQLite (signals + trades tables)
4. Track hypothetical position P&L
5. Compare forward performance vs backtest expectations

Usage:
    .venv\\Scripts\\python.exe scripts\\forward_tracker.py --symbol SOLUSDT
    .venv\\Scripts\\python.exe scripts\\forward_tracker.py --symbol SOLUSDT --report
"""
# ruff: noqa: E702, F401, F841
import argparse
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action

DB_PATH = "experiments/forward_tracking.sqlite"
STABLE = ["upper_wick_pct", "vol_change", "range_pct",
          "lower_wick_pct", "rel_vol_20"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    close REAL NOT NULL,
    primary_signal TEXT NOT NULL,
    p_win REAL NOT NULL,
    final_signal TEXT NOT NULL,
    UNIQUE(symbol, bar_time)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_time TEXT,
    exit_price REAL,
    pnl_pct REAL,
    bars_held INTEGER,
    status TEXT DEFAULT 'open',
    UNIQUE(symbol, entry_time)
);
"""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def fetch_candles(symbol, timeframe="1h", limit=500):
    url = "https://api.bybit.com/v5/market/kline"
    resp = requests.get(url, params={
        "category": "linear", "symbol": symbol,
        "interval": {"1h": "60", "15m": "15"}[timeframe],
        "limit": min(limit, 1000)}, timeout=30)
    data = resp.json()
    if data.get("retCode") != 0:
        msg = f"Bybit error: {data.get('retMsg')}"
        raise RuntimeError(msg)
    rows = data["result"]["list"]
    records = []
    for row in reversed(rows):
        records.append({
            "timestamp": pd.Timestamp(int(row[0]), unit="ms", tz="UTC"),
            "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]),
            "volume": float(row[5])})
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
    feats["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    feats["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    feats["range_pct"] = (high - low) / close
    feats["vol_change"] = volume.pct_change().fillna(0).replace(
        [np.inf, -np.inf], 0.0)
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
    return actions


def train_and_predict(ohlcv_hist, feats_hist, feats_latest, ohlcv_latest):
    close = ohlcv_hist["close"].astype("float64")
    actions = ichimoku_signal(ohlcv_hist)
    n = len(ohlcv_hist)
    signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]
    H = 24
    close_np = close.to_numpy()
    meta_label = np.full(n, np.nan)
    for i in signal_bars:
        if i + H >= n:
            continue
        d = 1 if actions[i] == Action.LONG else -1
        meta_label[i] = 1.0 if (close_np[i + H] - close_np[i]) * d > 0 else 0.0

    feature_cols = STABLE + ["trend_ret_96", "vol_96"]
    # compute regime features from ohlcv_hist (not stored in parquet)
    close_s = ohlcv_hist["close"].astype("float64")
    trend_ret = (close_s / close_s.shift(96) - 1.0).fillna(0.0)
    vol = np.log(close_s / close_s.shift(1)).rolling(96).std().fillna(0.0)
    X_hist = feats_hist[STABLE].fillna(0.0).copy()
    X_hist["trend_ret_96"] = trend_ret.to_numpy()
    X_hist["vol_96"] = vol.to_numpy()
    valid = ~np.isnan(meta_label)
    y = meta_label

    train_idx = np.arange(n)
    train_idx = train_idx[valid[train_idx]]
    if len(train_idx) < 200:
        msg = f"insufficient training data: {len(train_idx)}"
        raise ValueError(msg)

    model = LGBMClassifier(
        n_estimators=150, num_leaves=15, learning_rate=0.03,
        min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
        verbose=-1, random_state=42)
    model.fit(X_hist.iloc[train_idx], y[train_idx])

    X_latest = feats_latest[feature_cols].fillna(0.0)
    prob = float(model.predict_proba(X_latest)[0, 1])
    # also compute the primary signal from the latest live data
    primary = "HOLD"
    high_l = ohlcv_latest["high"].astype("float64")
    low_l = ohlcv_latest["low"].astype("float64")
    close_l = ohlcv_latest["close"].astype("float64")
    tenkan = (high_l.rolling(9).max() + low_l.rolling(9).min()) / 2
    kijun = (high_l.rolling(26).max() + low_l.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high_l.rolling(52).max() + low_l.rolling(52).min()) / 2).shift(26)
    t = len(ohlcv_latest) - 1
    above = close_l.iloc[t] > max(senkou_a.iloc[t], senkou_b.iloc[t])
    below = close_l.iloc[t] < min(senkou_a.iloc[t], senkou_b.iloc[t])
    tk = tenkan.iloc[t] > kijun.iloc[t]
    if above and tk:
        primary = "LONG"
    elif below and not tk:
        primary = "SHORT"

    return prob, actions, primary


def generate_and_log(symbol):
    cfg = load_config("development")
    src = DataStorage("data/processed")
    dst = DataStorage("data/features")

    # historical data for training (thousands of bars)
    hist_ohlcv = src.load(symbol, "1h")
    hist_feats = dst.load(symbol, "1h")

    # live data for the latest signal
    live_ohlcv = fetch_candles(symbol, "1h", 500)
    live_feats = compute_features(live_ohlcv)

    bar_time = str(live_ohlcv["timestamp"].iloc[-1])
    close_price = float(live_ohlcv["close"].iloc[-1])

    prob, hist_actions, primary = train_and_predict(
        hist_ohlcv, hist_feats, live_feats, live_ohlcv)

    # final signal after meta-filter
    final = primary if (primary != "HOLD" and prob >= 0.55) else "HOLD"

    conn = get_db()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR IGNORE INTO signals "
        "(timestamp, symbol, bar_time, close, primary_signal, p_win, final_signal) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (now, symbol, bar_time, close_price, primary, prob, final))
    conn.commit()

    # update hypothetical position
    open_trade = conn.execute(
        "SELECT id, direction, entry_price FROM trades "
        "WHERE symbol = ? AND status = 'open'",
        (symbol,)).fetchone()

    if final in ("LONG", "SHORT") and open_trade is None:
        conn.execute(
            "INSERT INTO trades (symbol, direction, entry_time, entry_price, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            (symbol, final, bar_time, close_price))
        print(f"  OPEN {final} @ {close_price}")
    elif open_trade is not None:
        direction = open_trade[1]
        entry_price = open_trade[2]
        if (final == "HOLD" and (
                (direction == "LONG" and primary == "SHORT")
                or (direction == "SHORT" and primary == "LONG"))):
            pnl = (close_price - entry_price) / entry_price
            if direction == "SHORT":
                pnl = -pnl
            conn.execute(
                "UPDATE trades SET exit_time = ?, exit_price = ?, "
                "pnl_pct = ?, status = 'closed' WHERE id = ?",
                (bar_time, close_price, pnl, open_trade[0]))
            print(f"  CLOSE {direction} @ {close_price} PnL={pnl:+.2%}")
        elif final != "HOLD" and final != direction:
            pnl = (close_price - entry_price) / entry_price
            if direction == "SHORT":
                pnl = -pnl
            conn.execute(
                "UPDATE trades SET exit_time = ?, exit_price = ?, "
                "pnl_pct = ?, status = 'closed' WHERE id = ?",
                (bar_time, close_price, pnl, open_trade[0]))
            conn.execute(
                "INSERT INTO trades (symbol, direction, entry_time, entry_price, status) "
                "VALUES (?, ?, ?, ?, 'open')",
                (symbol, final, bar_time, close_price))
            print(f"  FLIP to {final} @ {close_price} (closed PnL={pnl:+.2%})")
        else:
            unrealized = (close_price - entry_price) / entry_price
            if direction == "SHORT":
                unrealized = -unrealized
            print(f"  HOLD {direction} @ {entry_price} "
                  f"unrealized={unrealized:+.2%}")
    conn.commit()
    conn.close()

    print(f"  [{symbol}] bar={bar_time} close={close_price:.2f} "
          f"primary={primary} P(win)={prob:.3f} signal={final}")


def report(symbol):
    conn = get_db()
    signals = conn.execute(
        "SELECT * FROM signals WHERE symbol = ? ORDER BY bar_time",
        (symbol,)).fetchall()
    trades = conn.execute(
        "SELECT * FROM trades WHERE symbol = ? AND status = 'closed' "
        "ORDER BY entry_time", (symbol,)).fetchall()
    open_trades = conn.execute(
        "SELECT * FROM trades WHERE symbol = ? AND status = 'open'",
        (symbol,)).fetchall()
    conn.close()

    print(f"\n=== Forward Report: {symbol} ===")
    print(f"signals logged: {len(signals)}")
    print(f"closed trades: {len(trades)}")
    if open_trades:
        ot = open_trades[0]
        print(f"open position: {ot[2]} @ {ot[4]} since {ot[3]}")

    if trades:
        pnls = [t[6] for t in trades if t[6] is not None]
        if pnls:
            pnls = np.array(pnls)
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            wr = len(wins) / len(pnls) if pnls.size else 0
            pf = (wins.sum() / abs(losses.sum())
                  if losses.size and losses.sum() != 0 else float("inf"))
            total = pnls.sum()
            print(f"forward PnL: {total:+.2%} | WR: {wr:.1%} | "
                  f"PF: {pf:.2f} | avg: {pnls.mean():+.3%}")
            print(f"  wins: {len(wins)} ({wins.mean():+.3%} avg) | "
                  f"losses: {len(losses)} ({losses.mean():+.3%} avg)")
    else:
        print("no closed trades yet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--report", action="store_true",
                        help="only show report, don't generate new signal")
    args = parser.parse_args()

    if args.report:
        report(args.symbol)
    else:
        generate_and_log(args.symbol)
        report(args.symbol)


if __name__ == "__main__":
    main()
