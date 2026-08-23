"""Train ONE RL agent (PPO/MLP) on PAXGUSDT and judge it honestly.

Protocol (implementation in darwin.experiments.training):
- Chronological split TRAIN / VAL / TEST (70/15/15 by default).
- Training sees TRAIN only; VAL drives the eval callback via a quick proxy
  window; TEST is touched exactly ONCE at the end - by the agent AND by the
  M4 benchmarks, in the same arena, under identical costs.
- Risk engine always attached (M6): agents cannot bypass limits.
"""

from __future__ import annotations

import argparse

from darwin.agents import default_benchmarks
from darwin.environment.simulator import TradingSimulator
from darwin.evaluation.metrics import MetricsReport, format_header, format_row
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
    parser.add_argument("--timesteps", type=int, default=40_960)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--eval-window", type=int, default=3_000)
    parser.add_argument("--out", default="experiments/runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, symbol, ohlcv, feats = load_frames(args.config, args.timeframe)
    n = len(ohlcv)
    train_end = int(n * args.train_frac)
    val_end = int(n * (args.train_frac + args.val_frac))
    print(f"{symbol} {args.timeframe}: {n} rows | train<={train_end} "
          f"val<={val_end} test={val_end}..{n - 1}")

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])

    model_path, agent_report, _behavior = train_and_evaluate(
        seed=args.seed,
        ohlcv=ohlcv,
        feats=feats,
        timeframe=args.timeframe,
        sim_cfg=sim_cfg,
        risk_cfg=risk_cfg,
        train_end=train_end,
        val_end=val_end,
        timesteps=args.timesteps,
        eval_window=args.eval_window,
        out_dir=args.out,
    )
    print(f"\nmodel saved -> {model_path}")

    # ------------------------------------------------------------------
    # TEST ARENA: agent first, then benchmarks - same slice, same costs
    # ------------------------------------------------------------------
    test_candles = ohlcv.iloc[val_end:].reset_index(drop=True)

    print("\n=== TEST ARENA (unseen during training) ===")
    print(format_header())
    print(format_row("ppo_agent", agent_report))
    for strategy in default_benchmarks(seed=args.seed):
        result = TradingSimulator(sim_cfg).run(
            test_candles, strategy.generate_actions(test_candles)
        )
        print(format_row(
            strategy.name,
            MetricsReport.from_result(result, args.timeframe),
        ))

    exp_id = record_experiment("train_agent", {
        "model": "PPO/MlpPolicy",
        "symbol": symbol,
        "timeframe": args.timeframe,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "split": {"train_end": train_end, "val_end": val_end, "test_rows": n - val_end},
        "sim_config": str(sim_cfg),
        "risk_config": str(risk_cfg),
        "git_commit": git_commit(),
        "model_path": model_path,
        "test_metrics": vars(agent_report),
    })
    print(f"\nexperiment recorded: {exp_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
