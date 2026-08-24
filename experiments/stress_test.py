"""Step 2: Stress test the Ichimoku+meta-filter on SOL and ETH.

Tests:
1. Fee stress: 2x, 3x current taker fees
2. Slippage stress: 2x, 5x current slippage
3. Combined worst-case: 3x fees + 5x slippage
4. Monte Carlo: 1000 bootstrap resamples of trade returns
"""
# ruff: noqa: E702, F401
import numpy as np
from lightgbm import LGBMClassifier

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport

cfg = load_config("development")
src = DataStorage("data/processed")
dst = DataStorage("data/features")

STABLE = ["upper_wick_pct", "vol_change", "range_pct",
          "lower_wick_pct", "rel_vol_20"]


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


def meta_filter(ohlcv, feats, actions, threshold=0.55):
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
    test_start = 10000 + 48
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


for symbol in ("SOLUSDT", "ETHUSDT"):
    ohlcv = src.load(symbol, "1h")
    feats = dst.load(symbol, "1h")
    actions = ichimoku_signals(ohlcv)
    filtered = meta_filter(ohlcv, feats, actions)

    base_fee = cfg["simulator"]["taker_fee_pct"]
    base_slip = cfg["simulator"]["slippage_pct"]

    stress_tests = [
        ("baseline",       base_fee, base_slip),
        ("2x fees",        base_fee * 2, base_slip),
        ("3x fees",        base_fee * 3, base_slip),
        ("2x slippage",    base_fee, base_slip * 2),
        ("5x slippage",    base_fee, base_slip * 5),
        ("3x fees + 5x slip", base_fee * 3, base_slip * 5),
    ]

    print(f"\n=== {symbol}: STRESS TEST ===")
    print(f"{'scenario':<22}{'return':>9}{'DD':>8}{'PF':>7}{'sharpe':>8}")
    mc_trades = None
    for name, fee, slip in stress_tests:
        stress_cfg = SimulatorConfig(
            initial_capital=1000.0, taker_fee_pct=fee,
            slippage_pct=slip,
            position_size_pct=cfg["simulator"]["fixed_position_size_pct"])
        res = TradingSimulator(stress_cfg).run(ohlcv, filtered)
        rep = MetricsReport.from_result(res, "1h")
        print(f"{name:<22}{rep.total_return:>9.2%}{rep.max_drawdown:>8.2%}"
              f"{rep.profit_factor:>7.2f}{rep.sharpe:>8.3f}")
        if name == "baseline":
            mc_trades = [t.net_pnl for t in res.trades]

    # Monte Carlo bootstrap
    if mc_trades and len(mc_trades) > 10:
        trades = np.array(mc_trades)
        rng = np.random.default_rng(42)
        n_sims = 5000
        n_trades = len(trades)
        boot_returns = []
        for _ in range(n_sims):
            sample = rng.choice(trades, size=n_trades, replace=True)
            boot_returns.append(np.sum(sample))
        boot_returns = np.array(boot_returns)
        pct_5 = np.percentile(boot_returns, 5)
        pct_25 = np.percentile(boot_returns, 25)
        pct_50 = np.percentile(boot_returns, 50)
        pct_75 = np.percentile(boot_returns, 75)
        pct_95 = np.percentile(boot_returns, 95)
        prob_loss = float((boot_returns < 0).mean())
        print(f"\n  Monte Carlo ({n_sims} bootstrap resamples of {n_trades} trades):")
        print(f"  5th percentile: {pct_5:+.0f}  25th: {pct_25:+.0f}  "
              f"median: {pct_50:+.0f}  75th: {pct_75:+.0f}  95th: {pct_95:+.0f}")
        print(f"  P(total loss) = {prob_loss:.1%}")
