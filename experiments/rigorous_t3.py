"""Rigorous T3-cross meta-filter: per-fold feature selection + cross-venue.

Fixes EH-v6's feature-selection leakage: instead of selecting top-8 features
globally (using all data including test), each walk-forward fold:
1. trains a full-feature model on its TRAINING data only
2. extracts top-8 features from THAT model's importance
3. retrains on those 8 features only
4. predicts the fold's test window

Then runs the same methodology on BTC and ETH for cross-venue validation.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from darwin.agents.strategies import T3Strategy
from darwin.config import load_config
from darwin.data.schema import TIMESTAMP_COL
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport, format_row
from darwin.evolution.baseline import directional_baseline

cfg = load_config("development")
sim_cfg = SimulatorConfig(
    initial_capital=1000.0, taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
    slippage_pct=cfg["simulator"]["slippage_pct"],
    position_size_pct=cfg["simulator"]["fixed_position_size_pct"])
src = DataStorage("data/processed")
dst = DataStorage("data/features")


def meta_setup(ohlcv, feats, actions, horizon=24):
    n = len(ohlcv)
    signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]
    close = ohlcv["close"].astype("float64").to_numpy()
    meta_label = np.full(n, np.nan)
    for i in signal_bars:
        if i + horizon >= n:
            continue
        d = 1 if actions[i] == Action.LONG else -1
        meta_label[i] = 1.0 if (close[i + horizon] - close[i]) * d > 0 else 0.0
    X = feats.drop(columns=[TIMESTAMP_COL]).fillna(0.0)
    valid = ~np.isnan(meta_label)
    return X, meta_label, valid, signal_bars


def expanding_filter_rigorous(X, y, valid, actions, n,
                               min_train=10000, test_bars=4000,
                               embargo=48, threshold=0.50,
                               top_k=8):
    """Per-fold feature selection: no global leakage."""
    filtered = [Action.HOLD] * n
    test_start = min_train + embargo
    selected_log = []

    while test_start + test_bars <= n:
        train_end = test_start - embargo
        train_idx = np.arange(0, train_end)
        train_idx = train_idx[valid[train_idx]]
        test_idx = np.arange(test_start, min(test_start + test_bars, n))

        if len(train_idx) < 200:
            test_start += test_bars
            continue

        # step 1: full-feature model on TRAINING data only -> importance
        model_full = LGBMClassifier(
            n_estimators=100, num_leaves=15, verbose=-1, random_state=42)
        model_full.fit(X.iloc[train_idx], y[train_idx])
        importance = pd.Series(model_full.feature_importances_,
                               index=X.columns)
        top_features = importance.nlargest(top_k).index.tolist()
        selected_log.append(top_features)

        # step 2: retrain on top-k features only, same training data
        model = LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.03,
            min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
            verbose=-1, random_state=42)
        model.fit(X[top_features].iloc[train_idx], y[train_idx])

        # step 3: predict test window
        probs = model.predict_proba(X[top_features].iloc[test_idx])[:, 1]
        for j, idx in enumerate(test_idx):
            if actions[idx] != Action.HOLD and probs[j] >= threshold:
                filtered[idx] = actions[idx]

        test_start += test_bars

    return filtered, selected_log


def evaluate(label, ohlcv, filtered, tf="1h"):
    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    rep = MetricsReport.from_result(res, tf)
    n = len(ohlcv)
    val_end = int(n * 0.85)
    tail = res.equity_curve.iloc[val_end + 5000:].reset_index(drop=True)
    tail_rep = MetricsReport.from_parts(tail, (), tf)
    baseline, side = directional_baseline(
        ohlcv.iloc[val_end + 5000:].reset_index(drop=True), sim_cfg)
    tail_pass = tail_rep.total_return > baseline
    yearly = []
    for yr in sorted(ohlcv["timestamp"].dt.year.unique()):
        mask = (ohlcv["timestamp"].dt.year == yr).to_numpy()
        y_f = [filtered[i] for i in range(n) if mask[i]]
        y_res = TradingSimulator(sim_cfg).run(
            ohlcv[mask].reset_index(drop=True), y_f)
        y_rep = MetricsReport.from_result(y_res, tf)
        yearly.append(f"{yr}:{y_rep.total_return:+.0%}({y_rep.n_trades}t)")
    print(format_row(label, rep))
    print(f"  trades={rep.n_trades} PF={rep.profit_factor:.2f} "
          f"avg={rep.avg_trade_net:+.2f} | tail={tail_rep.total_return:+.2%} "
          f"vs {baseline:+.2%} PASS={tail_pass}")
    print(f"  yearly: {' | '.join(yearly)}")
    return rep, tail_pass


# ====================================================================
print("=" * 60)
print("RIGOROUS PER-FOLD FEATURE SELECTION + CROSS-VENUE")
print("=" * 60)

results = {}
for symbol, tf in [("SOLUSDT", "1h"), ("BTCUSDT", "1h"), ("ETHUSDT", "1h")]:
    print(f"\n--- {symbol} {tf} ---")
    ohlcv = src.load(symbol, tf)
    feats = dst.load(symbol, tf)
    n = len(ohlcv)

    actions = T3Strategy(mode="cross", period=8, slow_period=21).generate_actions(ohlcv)
    X, y, valid, signal_bars = meta_setup(ohlcv, feats, actions)
    print(f"signals: {len(signal_bars)} | bars: {n}")

    filtered, sel_log = expanding_filter_rigorous(X, y, valid, actions, n)
    rep, tail_pass = evaluate(f"{symbol[:3]}_rigorous", ohlcv, filtered, tf)
    results[symbol] = {"rep": rep, "tail_pass": tail_pass}

    # feature stability: how consistent are the selected features across folds?
    from collections import Counter
    all_selected = [f for fold in sel_log for f in fold]
    freq = Counter(all_selected)
    print(f"  feature stability (top-5 most selected): "
          f"{freq.most_common(5)}")

# ====================================================================
print("\n" + "=" * 60)
print("CROSS-VENUE SUMMARY")
print("=" * 60)
print(f"{'Symbol':<12}{'return':>9}{'DD':>8}{'PF':>7}{'tail_pass':>10}")
for symbol, r in results.items():
    rep = r["rep"]
    print(f"{symbol:<12}{rep.total_return:>9.2%}{rep.max_drawdown:>8.2%}"
          f"{rep.profit_factor:>7.2f}{str(r['tail_pass']):>10}")
positive = sum(1 for r in results.values() if r["rep"].total_return > 0)
tail_passes = sum(1 for r in results.values() if r["tail_pass"])
print(f"\npositive: {positive}/3 | tail passes: {tail_passes}/3")
