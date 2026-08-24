"""Next phase: fair tail test + parameter perturbation + portfolio.

1. FAIR TAIL: compare Sharpe ratio in the tail period, not absolute return.
   A 42%-exposure strategy with Sharpe 1.2 is BETTER than 100%-exposure
   buy-and-hold with Sharpe 0.8, even if absolute return is lower.

2. PARAMETER PERTURBATION: run Ichimoku with tenkan/kijun/senkou shifted
   by +20%, -20%, and +10/-10%. If the edge only exists at exact (9,26,52),
   it's overfitted. If it survives parameter changes, it's structural.

3. PORTFOLIO: combine the filtered signals from SOL + ETH + SUI into a
   single equity curve. Diversification should reduce DD further.
"""
# ruff: noqa: E702, F401, E402, I001, F821
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

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
src = DataStorage("data/processed")
dst = DataStorage("data/features")

STABLE = ["upper_wick_pct", "vol_change", "range_pct",
          "lower_wick_pct", "rel_vol_20"]


def ichimoku_signals(ohlcv, tenkan_p=9, kijun_p=26, senkou_p=52):
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")
    tenkan = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2
    kijun = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
    senkou_b = ((high.rolling(senkou_p).max()
                 + low.rolling(senkou_p).min()) / 2).shift(kijun_p)
    n = len(ohlcv)
    actions = [Action.HOLD] * n
    pos = 0
    for t in range(max(tenkan_p, kijun_p, senkou_p) + kijun_p, n):
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


def meta_filter(ohlcv, feats, actions, threshold=0.55, min_train=10000):
    n = len(ohlcv)
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
        model = LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.03,
            min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
            verbose=-1, random_state=42)
        model.fit(X.iloc[train_idx], y[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        for j, idx in enumerate(test_idx):
            if actions[idx] != Action.HOLD and probs[j] >= threshold:
                filtered[idx] = actions[idx]
        test_start += 4000
    return filtered


def tail_sharpe(ohlcv, equity_curve, threshold=0.55):
    """Fair tail test: compare Sharpe ratios in the tail period."""
    n = len(ohlcv)
    val_end = int(n * 0.85)
    tail_start = val_end + 5000
    if tail_start >= n - 10:
        return None, None, None
    tail_eq = equity_curve["equity"].iloc[tail_start:].reset_index(drop=True)
    tail_close = ohlcv["close"].iloc[tail_start:].reset_index(drop=True)

    strat_ret = tail_eq.pct_change().dropna()
    market_ret = tail_close.pct_change().dropna()
    min_len = min(len(strat_ret), len(market_ret))
    strat_ret = strat_ret.iloc[:min_len]
    market_ret = market_ret.iloc[:min_len]

    def sharpe(r):
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(len(r) * 0))  # per-bar

    strat_sharpe = float(strat_ret.mean() / strat_ret.std()) if strat_ret.std() > 0 else 0
    market_sharpe = float(market_ret.mean() / market_ret.std()) if market_ret.std() > 0 else 0
    return strat_sharpe, market_sharpe, strat_sharpe > market_sharpe


# ====================================================================
print("=" * 60)
print("PHASE 1: FAIR TAIL TEST (Sharpe vs Sharpe)")
print("=" * 60)

for symbol in ("SOLUSDT", "ETHUSDT"):
    ohlcv = src.load(symbol, "1h")
    feats = dst.load(symbol, "1h")
    actions = ichimoku_signals(ohlcv)
    filtered = meta_filter(ohlcv, feats, actions)
    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    s_strat, s_market, passes = tail_sharpe(ohlcv, res.equity_curve)
    if s_strat is not None:
        print(f"{symbol}: strategy Sharpe={s_strat:.4f} vs "
              f"market Sharpe={s_market:.4f} -> "
              f"{'PASS' if passes else 'FAIL'} "
              f"({s_strat - s_market:+.4f} edge)")
    else:
        print(f"{symbol}: tail too short for Sharpe comparison")

# ====================================================================
print("\n" + "=" * 60)
print("PHASE 2: PARAMETER PERTURBATION (+-20%)")
print("=" * 60)

perturbations = [
    ("baseline (9,26,52)", 9, 26, 52),
    ("tenkan +20% (11,26,52)", 11, 26, 52),
    ("tenkan -20% (7,26,52)", 7, 26, 52),
    ("kijun +20% (9,31,52)", 9, 31, 52),
    ("kijun -20% (9,21,52)", 9, 21, 52),
    ("senkou +20% (9,26,62)", 9, 26, 62),
    ("senkou -20% (9,26,42)", 9, 26, 42),
    ("all +10% (10,29,57)", 10, 29, 57),
    ("all -10% (8,23,47)", 8, 23, 47),
]

perturb_results = []
for label, tp, kp, sp in perturbations:
    ohlcv = src.load("SOLUSDT", "1h")
    feats = dst.load("SOLUSDT", "1h")
    actions = ichimoku_signals(ohlcv, tp, kp, sp)
    filtered = meta_filter(ohlcv, feats, actions)
    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    rep = MetricsReport.from_result(res, "1h")
    s_strat, s_mkt, passes = tail_sharpe(ohlcv, res.equity_curve)
    perturb_results.append({
        "label": label, "return": rep.total_return,
        "dd": rep.max_drawdown, "pf": rep.profit_factor,
        "sharpe": rep.sharpe, "trades": rep.n_trades,
        "tail_strat": s_strat, "tail_mkt": s_mkt,
        "tail_pass": passes,
    })
    print(f"{label:<25} ret={rep.total_return:+.2%} "
          f"DD={rep.max_drawdown:.2%} PF={rep.profit_factor:.2f} "
          f"sharpe={rep.sharpe:.3f} trades={rep.n_trades} "
          f"tail={'PASS' if passes else 'FAIL'}")

positive_perturbs = sum(1 for p in perturb_results if p["return"] > 0)
tail_passes_perturb = sum(1 for p in perturb_results if p["tail_pass"])
avg_return = np.mean([p["return"] for p in perturb_results])
std_return = np.std([p["return"] for p in perturb_results])
print(f"\nperturbation: {positive_perturbs}/9 positive | "
      f"tail passes: {tail_passes_perturb}/9 | "
      f"return {avg_return:+.1%} +- {std_return:.1%}")
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 3: PORTFOLIO (SOL + ETH + SUI equal-weight)")
print("=" * 60)

portfolio_curves = []
for symbol in ("SOLUSDT", "ETHUSDT", "SUIUSDT"):
    ohlcv = src.load(symbol, "1h")
    feats = dst.load(symbol, "1h")
    actions = ichimoku_signals(ohlcv)
    filtered = meta_filter(ohlcv, feats, actions)
    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    curve = res.equity_curve[["timestamp", "equity"]].copy()
    curve = curve.rename(columns={"equity": f"eq_{symbol}"})
    curve = curve.set_index("timestamp")
    portfolio_curves.append(curve)

# align on common timestamps
from functools import reduce
merged = reduce(
    lambda left, right: pd.merge(left, right, left_index=True,
                                 right_index=True, how="inner"),
    portfolio_curves)
# equal-weight: average of normalized equity curves
for col in merged.columns:
    merged[col + "_norm"] = merged[col] / merged[col].iloc[0]
merged["portfolio_eq"] = merged[[c for c in merged.columns if "_norm" in c]].mean(axis=1) * 1000.0

portfolio_return = float(merged["portfolio_eq"].iloc[-1] / 1000.0 - 1)
portfolio_peak = merged["portfolio_eq"].cummax()
portfolio_dd = float((merged["portfolio_eq"] / portfolio_peak - 1).min())
portfolio_ret = merged["portfolio_eq"].pct_change().dropna()
portfolio_sharpe = float(portfolio_ret.mean() / portfolio_ret.std()) if portfolio_ret.std() > 0 else 0

print(f"portfolio: return={portfolio_return:+.2%} "
      f"DD={portfolio_dd:.2%} Sharpe={portfolio_sharpe:.3f}")

# individual for comparison
for symbol in ("SOLUSDT", "ETHUSDT", "SUIUSDT"):
    col = f"eq_{symbol}_norm"
    ind_return = float(merged[col].iloc[-1] / merged[col].iloc[0] - 1)
    ind_peak = merged[col].cummax()
    ind_dd = float((merged[col] / ind_peak - 1).min())
    ind_ret = merged[col].pct_change().dropna()
    ind_sharpe = float(ind_ret.mean() / ind_ret.std()) if ind_ret.std() > 0 else 0
    print(f"  {symbol[:8]:<10} return={ind_return:+.2%} "
          f"DD={ind_dd:.2%} Sharpe={ind_sharpe:.3f}")
