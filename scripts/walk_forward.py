"""Walk-forward evaluation of benchmark strategies.

Protocol per fold:
1. Hand the strategy a slice spanning train+embargo+test, so its indicator
   warmup consumes only PRE-test history (legitimate information).
2. Force every decision inside the train/embargo region to HOLD - nothing may
   trade there - then simulate once. The account is therefore flat when the
   test window opens, and all trades/fees come purely from out-of-sample data.
3. Score metrics on the test-window portion of the equity curve only.

Aggregates report mean/std/min/max across folds - dispersion is the point.
"""

from __future__ import annotations

import argparse

import pandas as pd

from darwin.agents import default_benchmarks
from darwin.config import load_config
from darwin.data.schema import TIMESTAMP_COL
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport
from darwin.experiments.splits import walk_forward_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--train-bars", type=int, default=30_000)
    parser.add_argument("--test-bars", type=int, default=5_000)
    parser.add_argument("--embargo-bars", type=int, default=64)
    return parser.parse_args()


def sim_config_from_yaml(cfg: dict) -> SimulatorConfig:
    s = cfg["simulator"]
    return SimulatorConfig(
        initial_capital=s["initial_capital"],
        taker_fee_pct=s["taker_fee_pct"],
        slippage_pct=s["slippage_pct"],
        position_size_pct=s["fixed_position_size_pct"],
    )


def evaluate_fold(
    candles: pd.DataFrame,
    test_local_start: int,
    strategy,
    sim: TradingSimulator,
    timeframe: str,
) -> MetricsReport:
    """Score one strategy on one fold under the leak-safe protocol."""
    slice_df = candles.reset_index(drop=True)

    actions = strategy.generate_actions(slice_df)

    # Tail-align decisions to the fold: the bar immediately BEFORE test open
    # is the first one whose fill lands out-of-sample. Everything earlier is
    # forced HOLD, while indicators were still warmed by full prior history.
    # Action lists are target-STATE series, so sampling the tail is exact.
    first_decision = max(test_local_start - 1, 0)
    guarded = [Action.HOLD] * first_decision + list(actions[first_decision:])
    assert all(a == Action.HOLD for a in guarded[:first_decision])

    result = sim.run(slice_df, guarded)

    # score ONLY the out-of-sample portion of the equity path
    test_curve = result.equity_curve.iloc[test_local_start:].reset_index(drop=True)
    return MetricsReport.from_parts(test_curve, result.trades, timeframe)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]

    ohlcv = DataStorage(data_cfg["processed_dir"]).load(symbol, args.timeframe)
    folds = walk_forward_splits(
        len(ohlcv),
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        embargo_bars=args.embargo_bars,
    )
    print(f"\n{symbol} {args.timeframe}: {len(folds)} walk-forward folds "
          f"(train={args.train_bars}, test={args.test_bars}, embargo={args.embargo_bars})")

    strategies = default_benchmarks(seed=42)
    sim_cfg = sim_config_from_yaml(cfg)
    sim = TradingSimulator(sim_cfg)

    rows: list[dict[str, object]] = []
    for fold_idx, (train_seg, test_seg) in enumerate(folds):
        window = ohlcv.iloc[train_seg.start : test_seg.end].reset_index(drop=True)
        test_local_start = len(train_seg)
        for strat in strategies:
            report = evaluate_fold(
                window, test_local_start, strat, sim, args.timeframe
            )
            rows.append({
                "fold": fold_idx,
                "test_start": str(ohlcv[TIMESTAMP_COL].iloc[test_seg.start]),
                "strategy": strat.name,
                "total_return": report.total_return,
                "max_drawdown": report.max_drawdown,
                "sharpe": report.sharpe,
                "sortino": report.sortino,
                "profit_factor": report.profit_factor,
                "n_trades": report.n_trades,
            })

    frame = pd.DataFrame(rows)
    csv_path = f"experiments/walk_forward_{symbol}_{args.timeframe}.csv"
    frame.to_csv(csv_path, index=False)

    print("\n=== AGGREGATES ACROSS FOLDS ===")
    header = (
        f"{'strategy':<16}{'ret_mean':>10}{'ret_std':>10}{'ret_min':>10}{'ret_max':>10}"
        f"{'sharpe_mu':>11}{'dd_worst':>10}{'trades':>8}"
    )
    print(header)
    for strat in strategies:
        sub = frame[frame["strategy"] == strat.name]
        rets = sub["total_return"].astype(float)
        print(
            f"{strat.name:<16}{rets.mean():>10.2%}{rets.std():>10.2%}"
            f"{rets.min():>10.2%}{rets.max():>10.2%}"
            f"{sub['sharpe'].astype(float).mean():>11.3f}"
            f"{sub['max_drawdown'].astype(float).min():>10.2%}"
            f"{int(sub['n_trades'].sum()):>8d}"
        )
    print(f"\nper-fold details -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
