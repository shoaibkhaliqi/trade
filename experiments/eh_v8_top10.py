"""EH-v8: Ichimoku + meta-filter across the Bybit Top 10.

Tests the exact configuration that produced our best results on SOL
(stable@0.55: 5 stable features + regime, expanding window, threshold 0.55)
across all 10 assets. Reports a single comparison table.
"""
# ruff: noqa: E702, F401, F541
import numpy as np
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

STABLE = ["upper_wick_pct", "vol_change", "range_pct",
          "lower_wick_pct", "rel_vol_20"]
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "SUIUSDT",
          "HYPEUSDT", "AAVEUSDT", "SANDUSDT", "ZECUSDT"]


def ichimoku_signals(ohlcv):
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


def run_asset(symbol, threshold=0.55):
    ohlcv = src.load(symbol, "1h")
    feats = dst.load(symbol, "1h")
    n = len(ohlcv)
    actions = ichimoku_signals(ohlcv)
    signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]

    H = 24
    close_s = ohlcv["close"].astype("float64")
    close = close_s.to_numpy()
    meta_label = np.full(n, np.nan)
    for i in signal_bars:
        if i + H >= n:
            continue
        d = 1 if actions[i] == Action.LONG else -1
        meta_label[i] = 1.0 if (close[i + H] - close[i]) * d > 0 else 0.0

    trend_ret = (close_s / close_s.shift(96) - 1.0).fillna(0.0).to_numpy()
    vol = np.log(close_s / close_s.shift(1)).rolling(96).std().fillna(0.0).to_numpy()
    X = feats[STABLE].fillna(0.0).copy()
    X["trend_ret_96"] = trend_ret
    X["vol_96"] = vol
    valid = ~np.isnan(meta_label)
    y = meta_label

    # unfiltered
    res_raw = TradingSimulator(sim_cfg).run(ohlcv, actions)
    rep_raw = MetricsReport.from_result(res_raw, "1h")

    # filtered: expanding window @ threshold
    filtered = [Action.HOLD] * n
    MIN_TRAIN = 10000; TEST_BARS = 4000; EMBARGO = 48
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
            if actions[idx] != Action.HOLD and probs[j] >= threshold:
                filtered[idx] = actions[idx]
        test_start += TEST_BARS

    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    rep = MetricsReport.from_result(res, "1h")
    val_end = int(n * 0.85)
    tail_start = val_end + 5000
    if tail_start < n:
        tail = res.equity_curve.iloc[tail_start:].reset_index(drop=True)
        tail_rep = MetricsReport.from_parts(tail, (), "1h")
        baseline, side = directional_baseline(
            ohlcv.iloc[tail_start:].reset_index(drop=True), sim_cfg)
        tail_pass = tail_rep.total_return > baseline
    else:
        tail_rep = None
        baseline = 0.0
        side = "n/a"
        tail_pass = False

    yearly = []
    for yr in sorted(ohlcv["timestamp"].dt.year.unique()):
        mask = (ohlcv["timestamp"].dt.year == yr).to_numpy()
        y_f = [filtered[i] for i in range(n) if mask[i]]
        y_res = TradingSimulator(sim_cfg).run(
            ohlcv[mask].reset_index(drop=True), y_f)
        y_rep = MetricsReport.from_result(y_res, "1h")
        yearly.append(f"{yr}:{y_rep.total_return:+.0%}({y_rep.n_trades}t)")

    return {
        "symbol": symbol, "raw": rep_raw, "filtered": rep,
        "tail_rep": tail_rep, "baseline": baseline, "side": side,
        "tail_pass": tail_pass, "yearly": yearly,
        "signals": len(signal_bars),
    }


print("=" * 70)
print("EH-v8: ICHIMOKU + META-FILTER ACROSS BYBIT TOP 10")
print("stable@0.55, expanding window, per-asset meta-model")
print("=" * 70)

all_results = []
for symbol in ASSETS:
    print(f"\n--- {symbol} ---")
    r = run_asset(symbol)
    all_results.append(r)
    print(format_row("unfiltered ", r["raw"]))
    print(format_row("filtered@.55", r["filtered"]))
    print(f"  signals={r['signals']} trades={r['filtered'].n_trades} "
          f"PF={r['filtered'].profit_factor:.2f} "
          f"avg={r['filtered'].avg_trade_net:+.2f} "
          f"sharpe={r['filtered'].sharpe:.3f}")
    tail_str = (f"{r['tail_rep'].total_return:+.2%} vs "
                f"{r['baseline']:+.2%} ({r['side']}) PASS={r['tail_pass']}"
                ) if r["tail_rep"] is not None else "n/a (dataset too short)"
    print(f"  tail={tail_str}")
    print(f"  yearly: {' | '.join(r['yearly'])}")

# summary table
print("\n" + "=" * 70)
print("SUMMARY (sorted by filtered Sharpe)")
print("=" * 70)
print(f"{'Asset':<12}{'unfilt':>8}{'filtered':>10}{'DD':>8}{'PF':>7}"
      f"{'Sharpe':>8}{'MAR':>7}{'trades':>8}{'tail':>8}{'pass':>6}")

sorted_results = sorted(all_results,
                        key=lambda r: r["filtered"].sharpe, reverse=True)
for r in sorted_results:
    rep = r["filtered"]
    mar = abs(rep.total_return / rep.max_drawdown) if rep.max_drawdown != 0 else 0
    tail_str = (f"{r['tail_rep'].total_return:>8.1%}"
                ) if r["tail_rep"] is not None else f"{'n/a':>8}"
    print(f"{r['symbol'][:8]:<12}{r['raw'].total_return:>8.1%}"
          f"{rep.total_return:>10.1%}{rep.max_drawdown:>8.1%}"
          f"{rep.profit_factor:>7.2f}{rep.sharpe:>8.3f}{mar:>7.1f}"
          f"{rep.n_trades:>8d}{tail_str}"
          f"{str(r['tail_pass']):>6}")

positive = sum(1 for r in all_results if r["filtered"].total_return > 0)
beats_raw = sum(1 for r in all_results
                if r["filtered"].total_return > r["raw"].total_return)
tail_passes = sum(1 for r in all_results if r["tail_pass"])
sharpe_above_1 = sum(1 for r in all_results if r["filtered"].sharpe > 1.0)
sharpe_above_08 = sum(1 for r in all_results if r["filtered"].sharpe > 0.8)
print(f"\npositive: {positive}/{len(all_results)} | "
      f"beats unfiltered: {beats_raw}/{len(all_results)} | "
      f"tail passes: {tail_passes}/{len(all_results)} | "
      f"sharpe>1.0: {sharpe_above_1}/{len(all_results)} | "
      f"sharpe>0.8: {sharpe_above_08}/{len(all_results)}")

best = sorted_results[:3]
print(f"\ntop 3 by Sharpe:")
for r in best:
    rep = r["filtered"]
    print(f"  {r['symbol']:<12} Sharpe={rep.sharpe:.3f} "
          f"ret={rep.total_return:+.1%} DD={rep.max_drawdown:.1%} "
          f"PF={rep.profit_factor:.2f}")
