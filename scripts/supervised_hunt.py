"""Walk-forward supervised hunt: GBDT/logistic on triple-barrier labels.

The EH-v4 paradigm: models estimate P(upper barrier first | features); a
threshold rule converts probabilities to target positions; the signals run
through the SAME simulator/risk/metrics arena as every other candidate.

Superpower vs RL: fitting takes seconds, so the model RETRAINS on every
walk-forward fold - genuine out-of-sample discipline at negligible cost.

Protocol (identical to scripts/walk_forward.py):
- indicators are precomputed causal features (M2 contract)
- per fold: fit on train rows (complete labels only), predict test rows
- decisions outside test windows are forced HOLD; one simulator run total
- per-fold metrics scored on the out-of-sample equity portion only
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from darwin.config import load_config
from darwin.data.schema import TIMESTAMP_COL
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport
from darwin.evolution.baseline import directional_baseline
from darwin.evolution.behavior import summarize_behavior
from darwin.experiments.splits import walk_forward_splits
from darwin.features.labels import LabelConfig, triple_barrier_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--features-suffix", default="")
    parser.add_argument("--model", choices=["lightgbm", "logistic"],
                        default="lightgbm")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--tp-sigma", type=float, default=2.0)
    parser.add_argument("--sl-sigma", type=float, default=2.0)
    parser.add_argument("--label-vol-window", type=int, default=96)
    parser.add_argument("--threshold-long", type=float, default=0.60)
    parser.add_argument("--threshold-short", type=float, default=0.40)
    parser.add_argument("--train-bars", type=int, default=30_000)
    parser.add_argument("--test-bars", type=int, default=5_000)
    parser.add_argument("--embargo-bars", type=int, default=64)
    parser.add_argument("--lgbm-estimators", type=int, default=150)
    parser.add_argument("--lgbm-leaves", type=int, default=15)
    parser.add_argument("--lgbm-lr", type=float, default=0.03)
    parser.add_argument("--lgbm-min-child", type=int, default=500)
    return parser.parse_args()


def make_model(args: argparse.Namespace):
    if args.model == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=args.lgbm_estimators,
            num_leaves=args.lgbm_leaves,
            learning_rate=args.lgbm_lr,
            min_child_samples=args.lgbm_min_child,
            subsample=0.9,
            colsample_bytree=0.9,
            verbose=-1,
            random_state=42,
        )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    symbol = args.symbol or data_cfg["symbol"]

    src = DataStorage(data_cfg["processed_dir"])
    dst = DataStorage(data_cfg["features_dir"])
    ohlcv = src.load(symbol, args.timeframe)
    feats = dst.load(symbol, args.timeframe, args.features_suffix)
    n = len(ohlcv)

    label_cfg = LabelConfig(horizon=args.horizon, tp_sigma=args.tp_sigma,
                            sl_sigma=args.sl_sigma,
                            vol_window=args.label_vol_window)
    labels = triple_barrier_labels(ohlcv["close"], label_cfg)

    X = feats.drop(columns=[TIMESTAMP_COL]).fillna(0.0)
    valid = labels.notna().to_numpy()
    y = labels.to_numpy()

    folds = walk_forward_splits(n, train_bars=args.train_bars,
                                test_bars=args.test_bars,
                                embargo_bars=args.embargo_bars)

    actions: list[Action] = [Action.HOLD] * n
    probs_out = np.full(n, np.nan)
    fold_rows: list[dict] = []
    feature_importance: dict[str, float] | None = None

    for fold_idx, (train_seg, test_seg) in enumerate(folds):
        train_idx = np.arange(train_seg.start, train_seg.end)
        train_idx = train_idx[valid[train_idx]]
        if len(train_idx) < 500:
            print(f"fold {fold_idx}: insufficient labeled train rows - skipped")
            continue

        model = make_model(args)
        model.fit(X.iloc[train_idx], y[train_idx])
        test_idx = np.arange(test_seg.start, test_seg.end)
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        probs_out[test_idx] = probs

        # overconfidence diagnostics: a calibrated model rarely leaves (0.2, 0.8)
        if fold_idx % 8 == 0 or fold_idx == len(folds) - 1:
            print(f"fold {fold_idx} probs: mean={probs.mean():.3f} "
                  f">th_long={float((probs >= args.threshold_long).mean()):.0%} "
                  f"<th_short={float((probs <= args.threshold_short).mean()):.0%}")

        for j, idx in enumerate(test_idx):
            if probs[j] >= args.threshold_long:
                actions[idx] = Action.LONG
            elif probs[j] <= args.threshold_short:
                actions[idx] = Action.SHORT

        if args.model == "lightgbm" and fold_idx == len(folds) - 1:
            importance = model.feature_importances_
            feature_importance = dict(sorted(
                zip(X.columns, importance, strict=True),
                key=lambda kv: kv[1], reverse=True)[:10])

    sim_cfg = SimulatorConfig(
        initial_capital=cfg["simulator"]["initial_capital"],
        taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
        slippage_pct=cfg["simulator"]["slippage_pct"],
        position_size_pct=cfg["simulator"]["fixed_position_size_pct"],
    )
    sim = TradingSimulator(sim_cfg)
    result = sim.run(ohlcv, actions)

    print(f"\n{symbol} {args.timeframe} | model={args.model} "
          f"horizon={args.horizon} barriers=+-{args.tp_sigma}sigma "
          f"thresholds=({args.threshold_long}/{args.threshold_short}) "
          f"| {len(folds)} folds")
    print(f"passive baseline: {directional_baseline(ohlcv, sim_cfg)}")

    header = (f"{'fold':>5}{'test_start':>22}{'return':>9}{'dd':>9}"
              f"{'trades':>8}{'long_sig':>9}{'short_sig':>10}")
    print(header)
    for fold_idx, (_train_seg, test_seg) in enumerate(folds):
        local = test_seg.start
        curve = result.equity_curve.iloc[local:test_seg.end].reset_index(drop=True)
        report = MetricsReport.from_parts(curve, (), args.timeframe)
        n_long = sum(1 for a in actions[local:test_seg.end] if a == Action.LONG)
        n_short = sum(1 for a in actions[local:test_seg.end] if a == Action.SHORT)
        fold_rows.append({"fold": fold_idx, "return": report.total_return})
        print(f"{fold_idx:>5}{str(ohlcv[TIMESTAMP_COL].iloc[test_seg.start])[:16]:>22}"
              f"{report.total_return:>9.2%}{report.max_drawdown:>9.2%}"
              f"{report.n_trades:>8d}{n_long:>9d}{n_short:>10d}")

    rets = pd.Series([r["return"] for r in fold_rows])
    print(f"\naggregate: mean {rets.mean():+.2%} std {rets.std():.2%} "
          f"pos {float((rets > 0).mean()):.0%} of folds | "
          f"full-run return {result.final_equity / sim_cfg.initial_capital - 1:+.2%}")

    trades_by_fold: dict[int, int] = {}
    ts_index = ohlcv[TIMESTAMP_COL].reset_index(drop=True)
    for trade in result.trades:
        entry_local = int(ts_index.searchsorted(trade.entry_ts))
        fold_of_trade = next((i for i, (_t, seg) in enumerate(folds)
                              if seg.start <= entry_local < seg.end), -1)
        trades_by_fold[fold_of_trade] = trades_by_fold.get(fold_of_trade, 0) + 1
    print("trades per fold:", [trades_by_fold.get(i, 0)
                              for i in range(len(folds))])

    positions = result.equity_curve["position_qty"]
    behavior = summarize_behavior(
        [int(a.value == "long") and 1 or (int(a.value == "short") and 2 or 0)
         for a in actions[:-1]],
        positions.tolist()[:-1],
    )
    print(f"behavior: long={behavior['pos_long_frac']:.1%} "
          f"short={behavior['pos_short_frac']:.1%} "
          f"flat={behavior['pos_flat_frac']:.1%}")

    if feature_importance:
        print("top features (last fold):")
        for name, imp in feature_importance.items():
            print(f"  {name:<20} {imp:>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
