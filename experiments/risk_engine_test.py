"""Step 1: Risk engine on the Ichimoku+meta-filter candidates (SOL + ETH).

Tests multiple SL/TP configurations to find the optimal risk overlay.
The Ichimoku cloud-breach exit is already a dynamic exit; the risk engine
adds a hard safety net on top.
"""
# ruff: noqa: E702, F401, F541
import numpy as np
from lightgbm import LGBMClassifier

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.env import TradingEnv
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport, format_row
from darwin.execution.risk import RiskConfig, RiskManager

cfg = load_config("development")
sim_cfg = SimulatorConfig(
    initial_capital=1000.0, taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
    slippage_pct=cfg["simulator"]["slippage_pct"],
    position_size_pct=cfg["simulator"]["fixed_position_size_pct"])
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


ACTION_MAP = {"hold": 0, "long": 1, "short": 2, "close": 3}


def run_with_risk(ohlcv, feats, filtered, risk_cfg):
    env = TradingEnv(ohlcv, feats, config=sim_cfg,
                     risk=RiskManager(risk_cfg))
    obs, _ = env.reset(seed=42)
    done = False
    while not done:
        idx = env.sim._i
        if idx < len(filtered):
            act = ACTION_MAP[filtered[idx].value]
        else:
            act = 0
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return MetricsReport.from_result(env.last_result, "1h")


for symbol in ("SOLUSDT", "ETHUSDT"):
    ohlcv = src.load(symbol, "1h")
    feats = dst.load(symbol, "1h")
    n = len(ohlcv)
    actions = ichimoku_signals(ohlcv)
    filtered = meta_filter(ohlcv, feats, actions)

    # baseline without risk engine
    res = TradingSimulator(sim_cfg).run(ohlcv, filtered)
    rep = MetricsReport.from_result(res, "1h")
    print(f"\n=== {symbol}: Risk Engine A/B ===")
    print(format_row("no_risk      ", rep))

    configs = [
        ("sl5_tp15_dd20", RiskConfig(
            stop_loss_pct=5.0, take_profit_pct=15.0,
            max_drawdown_pct=20.0, cooldown_bars=4,
            max_position_size_pct=25.0)),
        ("sl3_tp10_dd15", RiskConfig(
            stop_loss_pct=3.0, take_profit_pct=10.0,
            max_drawdown_pct=15.0, cooldown_bars=4,
            max_position_size_pct=25.0)),
        ("sl7_tp20_dd20", RiskConfig(
            stop_loss_pct=7.0, take_profit_pct=20.0,
            max_drawdown_pct=20.0, cooldown_bars=4,
            max_position_size_pct=25.0)),
        ("sl5_noTP_dd15", RiskConfig(
            stop_loss_pct=5.0, take_profit_pct=None,
            max_drawdown_pct=15.0, cooldown_bars=4,
            max_position_size_pct=25.0)),
        ("sl3_noTP_dd20", RiskConfig(
            stop_loss_pct=3.0, take_profit_pct=None,
            max_drawdown_pct=20.0, cooldown_bars=4,
            max_position_size_pct=25.0)),
    ]

    for name, rc in configs:
        rep = run_with_risk(ohlcv, feats, filtered, rc)
        print(format_row(name + "  ", rep))

    # yearly for the best config
    best_cfg = RiskConfig(
        stop_loss_pct=5.0, take_profit_pct=None,
        max_drawdown_pct=15.0, cooldown_bars=4,
        max_position_size_pct=25.0)
    rep = run_with_risk(ohlcv, feats, filtered, best_cfg)
    print(f"\n  yearly (sl5_noTP_dd15):")
    for yr in sorted(ohlcv["timestamp"].dt.year.unique()):
        mask = (ohlcv["timestamp"].dt.year == yr).to_numpy()
        y_f = [filtered[i] for i in range(n) if mask[i]]
        env = TradingEnv(ohlcv[mask].reset_index(drop=True),
                        feats[mask].reset_index(drop=True),
                        config=sim_cfg, risk=RiskManager(best_cfg))
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            idx = env.sim._i
            act = ACTION_MAP[filtered[idx].value] if idx < len(filtered) else 0
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        y_rep = MetricsReport.from_result(env.last_result, "1h")
        print(f"    {yr}: {y_rep.total_return:+.2%} dd={y_rep.max_drawdown:.2%} "
              f"trades={y_rep.n_trades}")
