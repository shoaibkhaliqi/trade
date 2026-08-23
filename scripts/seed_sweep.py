"""Multi-seed variance sweep: how much of a 'result' is initialization luck?

Trains identical configurations under different seeds and reports the
dispersion of TEST metrics. A strategy is only interesting if its WORST seed
still clears the bar - single-seed numbers are lottery tickets.
"""

from __future__ import annotations

import argparse

import numpy as np

from darwin.evaluation.metrics import MetricsReport, format_row
from darwin.execution.risk import RiskConfig
from darwin.experiments.tracker import record_experiment
from darwin.experiments.training import (
    git_commit,
    load_frames,
    sim_config_from_yaml,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--eval-window", type=int, default=3_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, symbol, ohlcv, feats = load_frames(args.config, args.timeframe)
    n = len(ohlcv)
    train_end = int(n * args.train_frac)
    val_end = int(n * (args.train_frac + 0.15))

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])

    print(f"seed sweep: {len(args.seeds)} seeds x {args.timesteps} steps "
          f"| {symbol} {args.timeframe} | test rows={n - val_end}")
    results: list[tuple[int, MetricsReport]] = []
    for seed in args.seeds:
        _path, report, _behavior = train_and_evaluate(
            seed=seed, ohlcv=ohlcv, feats=feats,
            timeframe=args.timeframe,
            sim_cfg=sim_cfg, risk_cfg=risk_cfg,
            train_end=train_end, val_end=val_end,
            timesteps=args.timesteps, eval_window=args.eval_window,
        )
        results.append((seed, report))
        print(format_row(f"seed_{seed}", report))

    rets = np.array([r.total_return for _, r in results])
    sharpes = np.array([r.sharpe for _, r in results])
    dds = np.array([r.max_drawdown for _, r in results])
    print("\n=== DISPERSION ACROSS SEEDS ===")
    print(f"total_return: mean {rets.mean():+.2%}  std {rets.std():.2%}  "
          f"range [{rets.min():+.2%}, {rets.max():+.2%}]")
    print(f"sharpe      : mean {sharpes.mean():.3f}  std {sharpes.std():.3f}")
    print(f"max_dd      : worst {dds.min():.2%}")

    record_experiment("seed_sweep", {
        "symbol": symbol,
        "timeframe": args.timeframe,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "git_commit": git_commit(),
        "returns": rets.tolist(),
        "sharpes": sharpes.tolist(),
        "max_drawdowns": dds.tolist(),
        "per_seed": [
            {"seed": s, **{k: v for k, v in vars(r).items()}} for s, r in results
        ],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

