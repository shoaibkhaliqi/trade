"""T3-fix: expanding-window walk-forward with regime features + minimum-training sweep."""
import numpy as np
from lightgbm import LGBMClassifier

from darwin.agents.strategies import T3Strategy
from darwin.config import load_config
from darwin.data.schema import TIMESTAMP_COL
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport
from darwin.evolution.baseline import directional_baseline

cfg = load_config("development")
sim_cfg = SimulatorConfig(
    initial_capital=1000.0, taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
    slippage_pct=cfg["simulator"]["slippage_pct"],
    position_size_pct=cfg["simulator"]["fixed_position_size_pct"])
ohlcv = DataStorage("data/processed").load("SOLUSDT", "1h")
feats = DataStorage("data/features").load("SOLUSDT", "1h")
n = len(ohlcv)
strat = T3Strategy(mode="cross", period=8, slow_period=21)
actions = strat.generate_actions(ohlcv)
signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]
H = 24
close = ohlcv["close"].astype("float64").to_numpy()
meta_label = np.full(n, np.nan)
for i in signal_bars:
    if i + H >= n:
        continue
    d = 1 if actions[i] == Action.LONG else -1
    meta_label[i] = 1.0 if (close[i + H] - close[i]) * d > 0 else 0.0

c = ohlcv["close"].astype("float64")
trend_ret = (c / c.shift(96) - 1.0).fillna(0.0).to_numpy()
vol = np.log(c / c.shift(1)).rolling(96).std().fillna(0.0).to_numpy()
X = feats.drop(columns=[TIMESTAMP_COL]).fillna(0.0).copy()
X["trend_ret_96"] = trend_ret
X["vol_96"] = vol
valid = ~np.isnan(meta_label)
y = meta_label

for MIN_TRAIN in (5000, 10000, 15000):
    TEST_BARS = 4000
    EMBARGO = 48
    THRESHOLD = 0.50
    filtered = [Action.HOLD] * n
    test_start = MIN_TRAIN + EMBARGO
    while test_start + TEST_BARS <= n:
        train_end = test_start - EMBARGO
        train_idx = np.arange(0, train_end)
        train_idx = train_idx[valid[train_idx]]
        test_idx = np.arange(test_start, min(test_start + TEST_BARS, n))
        if len(train_idx) < 200:
            test_start += TEST_BARS
            continue
        model = LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.03,
            min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
            verbose=-1, random_state=42)
        model.fit(X.iloc[train_idx], y[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        for j, idx in enumerate(test_idx):
            if actions[idx] != Action.HOLD and probs[j] >= THRESHOLD:
                filtered[idx] = actions[idx]
        test_start += TEST_BARS

    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    rep = MetricsReport.from_result(res, "1h")
    val_end = int(n * 0.85)
    tail = res.equity_curve.iloc[val_end + 5000:].reset_index(drop=True)
    tail_rep = MetricsReport.from_parts(tail, (), "1h")
    baseline, side = directional_baseline(
        ohlcv.iloc[val_end + 5000:].reset_index(drop=True), sim_cfg)
    yearly = []
    for yr in sorted(ohlcv["timestamp"].dt.year.unique()):
        mask = (ohlcv["timestamp"].dt.year == yr).to_numpy()
        y_f = [filtered[i] for i in range(n) if mask[i]]
        y_res = TradingSimulator(sim_cfg).run(ohlcv[mask].reset_index(drop=True), y_f)
        y_rep = MetricsReport.from_result(y_res, "1h")
        yearly.append(f"{yr}:{y_rep.total_return:+.1%}({y_rep.n_trades}t)")

    tail_pass = tail_rep.total_return > baseline
    print(f"\nMIN_TRAIN={MIN_TRAIN}: ret={rep.total_return:+.2%} dd={rep.max_drawdown:.2%} "
          f"sharpe={rep.sharpe:.3f} PF={rep.profit_factor:.2f} trades={rep.n_trades}")
    print(f"  tail: {tail_rep.total_return:+.2%} vs passive {baseline:+.2%} ({side})"
          f" | PASS={tail_pass}")
    print(f"  yearly: {' | '.join(yearly)}")
