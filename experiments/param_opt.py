"""Walk-forward parameter optimization for Ichimoku + meta-filter.

For each fold:
1. Run Optuna (40 trials) on the TRAIN window only
2. Objective: Sharpe on the last 20% of train (validation)
3. Take the best parameters
4. Apply to TEST window
5. Record: best params, test performance, parameter stability

Answers: is (9,26,52) robust, lucky, or suboptimal?
"""
# ruff: noqa: E702, F401, F541, F841
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport

optuna.logging.set_verbosity(optuna.logging.WARNING)

cfg = load_config("development")
sim_cfg = SimulatorConfig(
    initial_capital=1000.0, taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
    slippage_pct=cfg["simulator"]["slippage_pct"],
    position_size_pct=cfg["simulator"]["fixed_position_size_pct"])
src = DataStorage("data/processed")
dst = DataStorage("data/features")

STABLE = ["upper_wick_pct", "vol_change", "range_pct",
          "lower_wick_pct", "rel_vol_20"]

ohlcv = src.load("SOLUSDT", "1h")
feats = dst.load("SOLUSDT", "1h")
n = len(ohlcv)
close_s = ohlcv["close"].astype("float64")
close = close_s.to_numpy()
high = ohlcv["high"].astype("float64").to_numpy()
low = ohlcv["low"].astype("float64").to_numpy()

trend_ret = (close_s / close_s.shift(96) - 1.0).fillna(0.0).to_numpy()
vol = np.log(close_s / close_s.shift(1)).rolling(96).std().fillna(0.0).to_numpy()
X_base = feats[STABLE].fillna(0.0).copy()
X_base["trend_ret_96"] = trend_ret
X_base["vol_96"] = vol


def ichimoku_signals(close_arr, high_arr, low_arr, tp, kp, sp):
    high_s = pd.Series(high_arr)
    low_s = pd.Series(low_arr)
    close_s2 = pd.Series(close_arr)
    tenkan = (high_s.rolling(tp).max() + low_s.rolling(tp).min()) / 2
    kijun = (high_s.rolling(kp).max() + low_s.rolling(kp).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kp)
    senkou_b = ((high_s.rolling(sp).max() + low_s.rolling(sp).min()) / 2).shift(kp)
    acts = [Action.HOLD] * len(close_arr)
    pos = 0
    for t in range(max(tp, kp, sp) + kp, len(close_arr)):
        above = close_arr[t] > max(senkou_a.iloc[t], senkou_b.iloc[t])
        below = close_arr[t] < min(senkou_a.iloc[t], senkou_b.iloc[t])
        tk = tenkan.iloc[t] > kijun.iloc[t]
        if pos != 0:
            if pos > 0 and below:
                acts[t] = Action.CLOSE; pos = 0
            elif pos < 0 and above:
                acts[t] = Action.CLOSE; pos = 0
            continue
        if above and tk:
            acts[t] = Action.LONG; pos = 1
        elif below and not tk:
            acts[t] = Action.SHORT; pos = -1
    return acts


def evaluate_params(train_end, params):
    """Evaluate a parameter set on the train window. Returns Sharpe."""
    tp = params["tenkan_p"]
    kp = params["kijun_p"]
    sp = params["senkou_p"]
    threshold = params["threshold"]
    n_est = params["n_estimators"]
    leaves = params["num_leaves"]
    lr = params["learning_rate"]
    mcs = params["min_child_samples"]

    actions = ichimoku_signals(close[:train_end], high[:train_end],
                               low[:train_end], tp, kp, sp)
    signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]
    H = 24
    meta_label = np.full(train_end, np.nan)
    for i in signal_bars:
        if i + H >= train_end:
            continue
        d = 1 if actions[i] == Action.LONG else -1
        meta_label[i] = 1.0 if (close[i + H] - close[i]) * d > 0 else 0.0

    X = X_base.iloc[:train_end]
    valid = ~np.isnan(meta_label)
    y = meta_label

    filtered = [Action.HOLD] * train_end
    # use last 80% of train for fitting, last 20% for validation Sharpe
    val_start = int(train_end * 0.8)
    fit_idx = np.arange(0, val_start)
    fit_idx = fit_idx[valid[fit_idx]]
    if len(fit_idx) < 200:
        return -10.0

    model = LGBMClassifier(n_estimators=n_est, num_leaves=leaves,
                           learning_rate=lr, min_child_samples=mcs,
                           subsample=0.9, colsample_bytree=0.9,
                           verbose=-1, random_state=42)
    model.fit(X.iloc[fit_idx], y[fit_idx])

    val_idx = np.arange(val_start, train_end)
    probs = model.predict_proba(X.iloc[val_idx])[:, 1]
    for j, idx in enumerate(val_idx):
        if actions[idx] != Action.HOLD and probs[j] >= threshold:
            filtered[idx] = actions[idx]

    # simulate on validation portion
    val_ohlcv = ohlcv.iloc[val_start:train_end].reset_index(drop=True)
    val_filtered = filtered[val_start:train_end]
    if len(val_ohlcv) < 100:
        return -10.0
    res = TradingSimulator(sim_cfg).run(val_ohlcv, val_filtered)
    rep = MetricsReport.from_result(res, "1h")
    if rep.n_trades < 5:
        return -5.0
    return rep.sharpe


def optimize_fold(train_end, n_trials=40):
    def objective(trial):
        params = {
            "tenkan_p": trial.suggest_int("tenkan_p", 5, 20),
            "kijun_p": trial.suggest_int("kijun_p", 15, 40),
            "senkou_p": trial.suggest_int("senkou_p", 30, 80),
            "threshold": trial.suggest_float("threshold", 0.45, 0.70),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 5, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
        }
        if params["kijun_p"] <= params["tenkan_p"]:
            raise optuna.TrialPruned()
        if params["senkou_p"] <= params["kijun_p"]:
            raise optuna.TrialPruned()
        return evaluate_params(train_end, params)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# walk-forward optimization
MIN_TRAIN = 10000
TEST_BARS = 4000
EMBARGO = 48
N_TRIALS = 40

fold_results = []
test_start = MIN_TRAIN + EMBARGO
fold_idx = 0

while test_start + TEST_BARS <= n:
    train_end = test_start - EMBARGO
    print(f"\n--- fold {fold_idx}: train=[0,{train_end}] "
          f"test=[{test_start},{test_start+TEST_BARS}] ---")

    best_params, best_val_sharpe = optimize_fold(train_end, N_TRIALS)
    print(f"  best val sharpe: {best_val_sharpe:.3f}")
    print(f"  best params: tenkan={best_params['tenkan_p']} "
          f"kijun={best_params['kijun_p']} senkou={best_params['senkou_p']} "
          f"th={best_params['threshold']:.2f} "
          f"leaves={best_params['num_leaves']} "
          f"lr={best_params['learning_rate']:.4f}")

    # apply best params to TEST window
    tp = best_params["tenkan_p"]
    kp = best_params["kijun_p"]
    sp = best_params["senkou_p"]
    actions = ichimoku_signals(close, high, low, tp, kp, sp)
    signal_bars = [i for i, a in enumerate(actions) if a != Action.HOLD]
    H = 24
    meta_label = np.full(n, np.nan)
    for i in signal_bars:
        if i + H >= n:
            continue
        d = 1 if actions[i] == Action.LONG else -1
        meta_label[i] = 1.0 if (close[i + H] - close[i]) * d > 0 else 0.0

    X = X_base.copy()
    X["threshold_weight"] = 0.0  # placeholder, not used by model
    valid = ~np.isnan(meta_label)
    y = meta_label

    filtered = [Action.HOLD] * n
    train_idx = np.arange(0, train_end)
    train_idx = train_idx[valid[train_idx]]
    test_idx = np.arange(test_start, min(test_start + TEST_BARS, n))
    model = LGBMClassifier(
        n_estimators=best_params["n_estimators"],
        num_leaves=best_params["num_leaves"],
        learning_rate=best_params["learning_rate"],
        min_child_samples=best_params["min_child_samples"],
        subsample=0.9, colsample_bytree=0.9,
        verbose=-1, random_state=42)
    model.fit(X.iloc[train_idx], y[train_idx])
    probs = model.predict_proba(X.iloc[test_idx])[:, 1]
    for j, idx in enumerate(test_idx):
        if actions[idx] != Action.HOLD and probs[j] >= best_params["threshold"]:
            filtered[idx] = actions[idx]

    test_ohlcv = ohlcv.iloc[test_start:test_start + TEST_BARS].reset_index(drop=True)
    test_filtered = filtered[test_start:test_start + TEST_BARS]
    res = TradingSimulator(sim_cfg).run(test_ohlcv, test_filtered)
    rep = MetricsReport.from_result(res, "1h")
    print(f"  TEST: ret={rep.total_return:+.2%} sharpe={rep.sharpe:.3f} "
          f"trades={rep.n_trades} PF={rep.profit_factor:.2f}")

    fold_results.append({
        "fold": fold_idx,
        "params": best_params,
        "val_sharpe": best_val_sharpe,
        "test_return": rep.total_return,
        "test_sharpe": rep.sharpe,
        "test_trades": rep.n_trades,
    })
    test_start += TEST_BARS
    fold_idx += 1

# summary
print("\n" + "=" * 60)
print("PARAMETER STABILITY ACROSS FOLDS")
print("=" * 60)
for fr in fold_results:
    p = fr["params"]
    print(f"fold {fr['fold']}: tenkan={p['tenkan_p']} kijun={p['kijun_p']} "
          f"senkou={p['senkou_p']} th={p['threshold']:.2f} | "
          f"val_sharpe={fr['val_sharpe']:.3f} -> "
          f"test_ret={fr['test_return']:+.2%} test_sharpe={fr['test_sharpe']:.3f}")

tenkans = [fr["params"]["tenkan_p"] for fr in fold_results]
kijuns = [fr["params"]["kijun_p"] for fr in fold_results]
senkous = [fr["params"]["senkou_p"] for fr in fold_results]
print(f"\ntenkan range: {min(tenkans)}-{max(tenkans)} (std={np.std(tenkans):.1f})")
print(f"kijun range: {min(kijuns)}-{max(kijuns)} (std={np.std(kijuns):.1f})")
print(f"senkou range: {min(senkous)}-{max(senkous)} (std={np.std(senkous):.1f})")

tuned_positive = sum(1 for fr in fold_results if fr["test_return"] > 0)
tuned_total_ret = np.prod([1 + fr["test_return"] for fr in fold_results]) - 1
print(f"\ntuned: {tuned_positive}/{len(fold_results)} positive folds | "
      f"compound return: {tuned_total_ret:+.2%}")
print(f"default (9,26,52) compound: +129.63% (from previous run)")
