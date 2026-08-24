"""Meta-label the Ichimoku primary on SOL 1h and run the full verification.

This is the first time the meta-filter is applied to a POSITIVE primary.
The T3-cross filter proved the MECHANISM works (-16.2% -> -0.75%/fold).
Now we test: does it AMPLIFY a +115% strategy?
"""
# ruff: noqa: E702
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

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

ohlcv = src.load("SOLUSDT", "1h")
feats = dst.load("SOLUSDT", "1h")
n = len(ohlcv)

# --- generate Ichimoku signals ---
high = ohlcv["high"].astype("float64")
low = ohlcv["low"].astype("float64")
close = ohlcv["close"].astype("float64")
tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

actions = [Action.HOLD] * n
pos = 0
for t in range(78, n):
    above = close.iloc[t] > max(senkou_a.iloc[t], senkou_b.iloc[t])
    below = close.iloc[t] < min(senkou_a.iloc[t], senkou_b.iloc[t])
    tk = tenkan_sen.iloc[t] > kijun_sen.iloc[t]
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

# --- meta labels ---
H = 24
close_np = close.to_numpy()
signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]
meta_label = np.full(n, np.nan)
for i in signal_bars:
    if i + H >= n:
        continue
    d = 1 if actions[i] == Action.LONG else -1
    meta_label[i] = 1.0 if (close_np[i + H] - close_np[i]) * d > 0 else 0.0

# --- features: 5 stable + regime + all features ---
STABLE = ["upper_wick_pct", "vol_change", "range_pct", "lower_wick_pct", "rel_vol_20"]
c = close
trend_ret = (c / c.shift(96) - 1.0).fillna(0.0).to_numpy()
vol = np.log(c / c.shift(1)).rolling(96).std().fillna(0.0).to_numpy()

X_all = feats.drop(columns=[TIMESTAMP_COL]).fillna(0.0).copy()
X_all["trend_ret_96"] = trend_ret
X_all["vol_96"] = vol
X_stable = feats[STABLE].fillna(0.0).copy()
X_stable["trend_ret_96"] = trend_ret
X_stable["vol_96"] = vol

valid = ~np.isnan(meta_label)
y = meta_label


def expanding_filter(X, actions, min_train, threshold, top_k=None):
    filtered = [Action.HOLD] * n
    test_start = min_train + 48
    while test_start + 4000 <= n:
        train_end = test_start - 48
        train_idx = np.arange(0, train_end)
        train_idx = train_idx[valid[train_idx]]
        test_idx = np.arange(test_start, min(test_start + 4000, n))
        if len(train_idx) < 200:
            test_start += 4000
            continue
        if top_k is not None:
            m = LGBMClassifier(n_estimators=100, num_leaves=15, verbose=-1,
                               random_state=42)
            m.fit(X.iloc[train_idx], y[train_idx])
            imp = pd.Series(m.feature_importances_, index=X.columns)
            cols = imp.nlargest(top_k).index.tolist()
            X_train = X[cols].iloc[train_idx]
            X_test = X[cols].iloc[test_idx]
            model = LGBMClassifier(
                n_estimators=150, num_leaves=15, learning_rate=0.03,
                min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
                verbose=-1, random_state=42)
        else:
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            model = LGBMClassifier(
                n_estimators=150, num_leaves=15, learning_rate=0.03,
                min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
                verbose=-1, random_state=42)
        model.fit(X_train, y[train_idx])
        probs = model.predict_proba(X_test)[:, 1]
        for j, idx in enumerate(test_idx):
            if actions[idx] != Action.HOLD and probs[j] >= threshold:
                filtered[idx] = actions[idx]
        test_start += 4000
    return filtered


def evaluate(label, filtered):
    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    rep = MetricsReport.from_result(res, "1h")
    val_end = int(n * 0.85)
    tail = res.equity_curve.iloc[val_end + 5000:].reset_index(drop=True)
    tail_rep = MetricsReport.from_parts(tail, (), "1h")
    baseline, side = directional_baseline(
        ohlcv.iloc[val_end + 5000:].reset_index(drop=True), sim_cfg)
    tail_pass = tail_rep.total_return > baseline
    yearly = []
    for yr in sorted(ohlcv["timestamp"].dt.year.unique()):
        mask = (ohlcv["timestamp"].dt.year == yr).to_numpy()
        y_f = [filtered[i] for i in range(n) if mask[i]]
        y_res = TradingSimulator(sim_cfg).run(
            ohlcv[mask].reset_index(drop=True), y_f)
        y_rep = MetricsReport.from_result(y_res, "1h")
        yearly.append(f"{yr}:{y_rep.total_return:+.0%}({y_rep.n_trades}t)")
    print(format_row(label, rep))
    print(f"  trades={rep.n_trades} PF={rep.profit_factor:.2f} "
          f"avg={rep.avg_trade_net:+.2f} fees={rep.fees_paid:.0f}")
    print(f"  tail={tail_rep.total_return:+.2%} vs {baseline:+.2%} "
          f"({side}) PASS={tail_pass}")
    print(f"  yearly: {' | '.join(yearly)}")
    return rep, tail_pass


# --- run ---
res_raw = TradingSimulator(sim_cfg).run(ohlcv, actions)
rep_raw = MetricsReport.from_result(res_raw, "1h")
print("=== ICHIMOKU META-FILTER ON SOL 1h ===")
print(format_row("unfiltered     ", rep_raw))
for threshold in (0.50, 0.55):
    filtered_stable = expanding_filter(X_stable, actions, 10000, threshold)
    rep_s, pass_s = evaluate(f"stable@{threshold}   ", filtered_stable)
    filtered_top8 = expanding_filter(X_all, actions, 10000, threshold, top_k=8)
    rep_t8, pass_t8 = evaluate(f"top8@{threshold}    ", filtered_top8)

# summary
print("\n=== COMPARISON ===")
print(f"{'Version':<25}{'return':>9}{'DD':>8}{'PF':>7}{'trades':>8}")
print(f"{'unfiltered':<25}{rep_raw.total_return:>9.2%}"
      f"{rep_raw.max_drawdown:>8.2%}{rep_raw.profit_factor:>7.2f}"
      f"{rep_raw.n_trades:>8d}")
for label, r, _p in [
    ("stable@0.50", rep_s, pass_s), ("top8@0.50", rep_t8, pass_t8)]:
    print(f"{label:<25}{r.total_return:>9.2%}{r.max_drawdown:>8.2%}"
          f"{r.profit_factor:>7.2f}{r.n_trades:>8d}")
