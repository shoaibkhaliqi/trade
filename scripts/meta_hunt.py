"""Meta-labeling hunt: predict whether the PRIMARY strategy's bet will WIN.

Paradigm (Lopez de Prado's meta-labeling, adapted):
1. a simple PRIMARY strategy supplies the side each bar (here: EMA 5/20 sign)
2. the meta label at bar t = 1 if a triple-barrier bet in the primary's
   direction starting at t would WIN (upper barrier for longs / lower for
   shorts), else 0
3. walk-forward: fit a classifier on (features -> meta label), then each test
   bar takes the primary side ONLY when P(win) >= threshold - otherwise HOLD

This separates direction-finding (proven hard) from bet-selection (untested).
The primary alone, the meta-filtered strategy, and passive baselines all run
through the same simulator arena.
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
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="min P(win) to take the primary's side")
    parser.add_argument("--train-bars", type=int, default=30_000)
    parser.add_argument("--test-bars", type=int, default=5_000)
    parser.add_argument("--embargo-bars", type=int, default=64)
    return parser.parse_args()


def make_model(args: argparse.Namespace):
    if args.model == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.03,
            min_child_samples=500, subsample=0.9, colsample_bytree=0.9,
            verbose=-1, random_state=42,
        )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000))])


def primary_side(feats: pd.DataFrame) -> pd.Series:
    """EMA 5/20 relative position: +1 long, -1 short, NaN warmup."""
    diff = feats["ema_5"] - feats["ema_20"]
    side = pd.Series(np.nan, index=feats.index)
    side[diff > 0] = 1.0
    side[diff < 0] = -1.0
    return side


def meta_labels(side: pd.Series, outcome: pd.Series) -> pd.Series:
    """1 if the barrier outcome AGREES with the primary side, else 0."""
    label = pd.Series(np.nan, index=side.index)
    valid = side.notna() & outcome.notna()
    label[valid] = ((side[valid] > 0) == (outcome[valid] > 0)).astype(float)
    return label


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

    outcome = triple_barrier_labels(
        ohlcv["close"],
        LabelConfig(horizon=args.horizon, tp_sigma=args.tp_sigma,
                    sl_sigma=args.sl_sigma,
                    vol_window=args.label_vol_window))
    side = primary_side(feats)
    meta = meta_labels(side, outcome)

    X = feats.drop(columns=[TIMESTAMP_COL]).fillna(0.0)
    valid_rows = meta.notna().to_numpy()
    y = meta.to_numpy()

    folds = walk_forward_splits(n, train_bars=args.train_bars,
                                test_bars=args.test_bars,
                                embargo_bars=args.embargo_bars)

    meta_actions: list[Action] = [Action.HOLD] * n
    primary_actions: list[Action] = [Action.HOLD] * n
    fold_rows: list[dict] = []
    n_train_last = 0

    for _fold_idx, (train_seg, test_seg) in enumerate(folds):
        train_idx = np.arange(train_seg.start, train_seg.end)
        train_idx = train_idx[valid_rows[train_idx]]
        n_train_last = len(train_idx)
        if len(train_idx) < 300:
            continue
        model = make_model(args)
        model.fit(X.iloc[train_idx], y[train_idx])

        test_idx = np.arange(test_seg.start, test_seg.end)
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]

        for j, idx in enumerate(test_idx):
            s = side.iloc[idx]
            if np.isnan(s):
                continue
            primary = Action.LONG if s > 0 else Action.SHORT
            primary_actions[idx] = primary
            if probs[j] >= args.threshold:
                meta_actions[idx] = primary

    sim_cfg = SimulatorConfig(
        initial_capital=cfg["simulator"]["initial_capital"],
        taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
        slippage_pct=cfg["simulator"]["slippage_pct"],
        position_size_pct=cfg["simulator"]["fixed_position_size_pct"],
    )
    sim = TradingSimulator(sim_cfg)
    meta_result = sim.run(ohlcv, meta_actions)
    primary_result = TradingSimulator(sim_cfg).run(ohlcv, primary_actions)

    print(f"\n{symbol} {args.timeframe} | meta-labeling | model={args.model} "
          f"horizon={args.horizon} threshold={args.threshold} "
          f"| last-fold train rows={n_train_last}")
    print(f"passive baseline: {directional_baseline(ohlcv, sim_cfg)}")

    header = (f"{'fold':>5}{'test_start':>22}{'meta_ret':>10}{'prim_ret':>10}"
              f"{'meta_tr':>9}{'prim_tr':>9}")
    print(header)
    for fold_idx, (_train_seg, test_seg) in enumerate(folds):
        lo, hi = test_seg.start, test_seg.end
        m_curve = meta_result.equity_curve.iloc[lo:hi].reset_index(drop=True)
        p_curve = primary_result.equity_curve.iloc[lo:hi].reset_index(drop=True)
        m_rep = MetricsReport.from_parts(m_curve, (), args.timeframe)
        p_rep = MetricsReport.from_parts(p_curve, (), args.timeframe)
        fold_rows.append({"fold": fold_idx, "meta": m_rep.total_return,
                          "primary": p_rep.total_return})
        print(f"{fold_idx:>5}{str(ohlcv[TIMESTAMP_COL].iloc[lo])[:16]:>22}"
              f"{m_rep.total_return:>10.2%}{p_rep.total_return:>10.2%}"
              f"{m_rep.n_trades:>9d}{p_rep.n_trades:>9d}")

    m_rets = pd.Series([r["meta"] for r in fold_rows])
    p_rets = pd.Series([r["primary"] for r in fold_rows])
    print(f"\nmeta    : mean {m_rets.mean():+.2%} std {m_rets.std():.2%} "
          f"pos {float((m_rets > 0).mean()):.0%} of folds | "
          f"full-run {meta_result.final_equity / sim_cfg.initial_capital - 1:+.2%}")
    print(f"primary : mean {p_rets.mean():+.2%} std {p_rets.std():.2%} "
          f"pos {float((p_rets > 0).mean()):.0%} of folds | "
          f"full-run {primary_result.final_equity / sim_cfg.initial_capital - 1:+.2%}")

    positions = meta_result.equity_curve["position_qty"]
    behavior = summarize_behavior(
        [1 if a == Action.LONG else 2 if a == Action.SHORT else 0
         for a in meta_actions[:-1]],
        positions.tolist()[:-1])
    print(f"meta behavior: long={behavior['pos_long_frac']:.1%} "
          f"short={behavior['pos_short_frac']:.1%} "
          f"flat={behavior['pos_flat_frac']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
