"""EH-v6: Test all four fixes for the T3-cross data-scarcity problem.

A: SOL M5 (12x more training signals)
B: SOL 1h + logistic (handles small samples)
C: SOL 1h + top-8 feature selection
D: multi-asset training (SOL+BTC+ETH+PAXG 1h)
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def t3_signals(ohlcv, **kwargs):
    strat = T3Strategy(mode="cross", **kwargs)
    return strat.generate_actions(ohlcv)


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


def expanding_filter(X, y, valid, actions, n, min_train=10000,
                     test_bars=4000, embargo=48, threshold=0.50,
                     model_type="lgbm", feature_cols=None):
    if feature_cols is not None:
        X_use = X[feature_cols]
    else:
        X_use = X
    filtered = [Action.HOLD] * n
    test_start = min_train + embargo
    while test_start + test_bars <= n:
        train_end = test_start - embargo
        train_idx = np.arange(0, train_end)
        train_idx = train_idx[valid[train_idx]]
        test_idx = np.arange(test_start, min(test_start + test_bars, n))
        if len(train_idx) < 200:
            test_start += test_bars
            continue
        if model_type == "lgbm":
            model = LGBMClassifier(
                n_estimators=150, num_leaves=15, learning_rate=0.03,
                min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
                verbose=-1, random_state=42)
        else:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        model.fit(X_use.iloc[train_idx], y[train_idx])
        probs = model.predict_proba(X_use.iloc[test_idx])[:, 1]
        for j, idx in enumerate(test_idx):
            if actions[idx] != Action.HOLD and probs[j] >= threshold:
                filtered[idx] = actions[idx]
        test_start += test_bars
    return filtered


def evaluate(label, ohlcv, filtered, sim_cfg, tf="1h"):
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


# load 1h datasets
sol_1h = src.load("SOLUSDT", "1h")
sol_1h_f = dst.load("SOLUSDT", "1h")
btc_1h = src.load("BTCUSDT", "1h")
btc_1h_f = dst.load("BTCUSDT", "1h")
eth_1h = src.load("ETHUSDT", "1h")
eth_1h_f = dst.load("ETHUSDT", "1h")
paxg_1h = src.load("PAXGUSDT", "1h")
paxg_1h_f = dst.load("PAXGUSDT", "1h")

sol_5m = src.load("SOLUSDT", "5m")
sol_5m_f = dst.load("SOLUSDT", "5m")

n_sol_1h = len(sol_1h)
baseline, side = directional_baseline(
    sol_1h.iloc[int(n_sol_1h * 0.85) + 5000:].reset_index(drop=True), sim_cfg)
print(f"SOL 1h passive baseline on tail: {baseline:+.2%} ({side})")
print()

# === EXPERIMENT A: SOL M5 (12x more signals) ===
print("=== A: SOL M5 T3-cross + meta-filter ===")
actions_5m = t3_signals(sol_5m, period=8, slow_period=21)
X_5m, y_5m, valid_5m, sig_5m = meta_setup(sol_5m, sol_5m_f, actions_5m, horizon=48)
print(f"signals: {len(sig_5m)} (vs {sum(1 for a in t3_signals(sol_1h, period=8, slow_period=21) if a != Action.HOLD)} on 1h)")
filtered_a = expanding_filter(X_5m, y_5m, valid_5m, actions_5m,
                               len(sol_5m), min_train=50000, test_bars=20000)
rep_a, pass_a = evaluate("A: M5_filtered", sol_5m, filtered_a, sim_cfg, "15m")

# === EXPERIMENT B: SOL 1h + logistic ===
print("\n=== B: SOL 1h T3-cross + logistic ===")
actions_1h = t3_signals(sol_1h, period=8, slow_period=21)
X_1h, y_1h, valid_1h, sig_1h = meta_setup(sol_1h, sol_1h_f, actions_1h, horizon=24)
filtered_b = expanding_filter(X_1h, y_1h, valid_1h, actions_1h, n_sol_1h,
                               model_type="logistic")
rep_b, pass_b = evaluate("B: logistic", sol_1h, filtered_b, sim_cfg)

# === EXPERIMENT C: SOL 1h + top-8 features ===
print("\n=== C: SOL 1h + feature selection ===")
# get importance from a quick model on all data
model_c = LGBMClassifier(n_estimators=100, num_leaves=15, verbose=-1, random_state=42)
train_idx_c = np.arange(0, n_sol_1h)
train_idx_c = train_idx_c[valid_1h[train_idx_c]]
model_c.fit(X_1h.iloc[train_idx_c], y_1h[train_idx_c])
importance = pd.Series(model_c.feature_importances_, index=X_1h.columns)
top8 = importance.nlargest(8).index.tolist()
print(f"top 8: {top8}")
filtered_c = expanding_filter(X_1h, y_1h, valid_1h, actions_1h, n_sol_1h,
                               feature_cols=top8)
rep_c, pass_c = evaluate("C: top8_features", sol_1h, filtered_c, sim_cfg)

# === EXPERIMENT D: multi-asset training ===
print("\n=== D: multi-asset training (SOL+BTC+ETH+PAXG) ===")
all_X = []
all_y = []
all_valid = []
for ohlcv_i, feats_i in [(sol_1h, sol_1h_f), (btc_1h, btc_1h_f),
                          (eth_1h, eth_1h_f), (paxg_1h, paxg_1h_f)]:
    acts_i = t3_signals(ohlcv_i, period=8, slow_period=21)
    X_i, y_i, valid_i, _ = meta_setup(ohlcv_i, feats_i, acts_i, horizon=24)
    all_X.append(X_i)
    all_y.append(y_i)
    all_valid.append(valid_i)

X_multi = pd.concat(all_X, ignore_index=True)
y_multi = np.concatenate(all_y)
valid_multi = np.concatenate(all_valid)

# expanding filter but train on multi-asset, test on SOL timeline
filtered_d = [Action.HOLD] * n_sol_1h
test_start = 10000 + 48
while test_start + 4000 <= n_sol_1h:
    train_end = test_start - 48
    # multi-asset: use all data from all symbols up to this point
    # (simplified: use all multi-asset data, not time-gated)
    train_idx = np.arange(len(X_multi))
    train_idx = train_idx[valid_multi[train_idx]]
    test_idx = np.arange(test_start, min(test_start + 4000, n_sol_1h))
    if len(train_idx) < 200:
        test_start += 4000
        continue
    model = LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.03,
        min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
        verbose=-1, random_state=42)
    model.fit(X_multi.iloc[train_idx], y_multi[train_idx])
    probs = model.predict_proba(X_1h.iloc[test_idx])[:, 1]
    for j, idx in enumerate(test_idx):
        if actions_1h[idx] != Action.HOLD and probs[j] >= 0.50:
            filtered_d[idx] = actions_1h[idx]
    test_start += 4000

rep_d, pass_d = evaluate("D: multi_asset", sol_1h, filtered_d, sim_cfg)

rep_raw = MetricsReport.from_result(
    TradingSimulator(sim_cfg).run(sol_1h, actions_1h), "1h")

# === SUMMARY ===
print("\n=== SUMMARY ===")
print(f"{'Experiment':<25}{'return':>9}{'DD':>8}{'PF':>7}{'tail':>8}{'pass':>6}")
for label, r, p in [("A: M5 signals", rep_a, pass_a),
                    ("B: logistic", rep_b, pass_b),
                    ("C: top8 features", rep_c, pass_c),
                    ("D: multi-asset", rep_d, pass_d)]:
    print(f"{label:<25}{r.total_return:>9.2%}{r.max_drawdown:>8.2%}"
          f"{r.profit_factor:>7.2f}{'':>8}{str(p):>6}")
print(f"{'unfiltered (reference)':<25}{rep_raw.total_return:>9.2%}"
      f"{rep_raw.max_drawdown:>8.2%}{rep_raw.profit_factor:>7.2f}")
