"""Instantiate a population, train+score every agent identically, rank them.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_population.py --size 8 --timesteps 5000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from darwin.agents import BuyAndHoldStrategy
from darwin.environment.simulator import TradingSimulator
from darwin.evaluation.metrics import format_header, format_row
from darwin.evolution.fitness import preset
from darwin.evolution.population import Population
from darwin.evolution.survival import SurvivalConfig
from darwin.execution.risk import RiskConfig
from darwin.experiments.tracker import (
    get_agents,
    record_experiment,
)
from darwin.experiments.training import (
    git_commit,
    load_frames,
    sim_config_from_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--master-seed", type=int, default=2026)
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--score-window", type=int, default=5_000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--eval-window", type=int, default=3_000)
    parser.add_argument("--fitness", default="spec")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, symbol, ohlcv, feats = load_frames(args.config, args.timeframe)
    n = len(ohlcv)
    train_end = int(n * args.train_frac)
    val_end = int(n * (args.train_frac + 0.15))

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])

    # fitness compass + its baseline: buy&hold on the identical score window
    val_end = int(n * (args.train_frac + 0.15))
    window = ohlcv.iloc[val_end : val_end + args.score_window].reset_index(drop=True)
    bh = TradingSimulator(sim_cfg).run(
        window, BuyAndHoldStrategy().generate_actions(window)
    )
    baseline = float(
        bh.equity_curve["equity"].iloc[-1] / bh.equity_curve["equity"].iloc[0] - 1.0
    )
    fcfg = preset(args.fitness, baseline_return=baseline)
    survival_cfg = SurvivalConfig()
    print(f"fitness={args.fitness} | buy&hold baseline on score window: {baseline:+.2%}")

    pop = Population(size=args.size)
    agents = pop.initialize(master_seed=args.master_seed)
    print(f"population: {len(agents)} agents | {symbol} {args.timeframe} | "
          f"{args.timesteps} steps each | score window={args.score_window} test bars")

    started = time.time()
    already_done = {
        r["agent_id"] for r in get_agents(db_path=pop.db_path) if r["metrics"]
    }
    for i, agent in enumerate(agents, start=1):
        if agent.agent_id in already_done:
            print(f"[{i:>3}/{len(agents)}] {agent.agent_id} already evaluated - skipped")
            continue
        t0 = time.time()
        report, metrics, verdict = pop.evaluate_agent(
            agent,
            ohlcv=ohlcv, feats=feats, timeframe=args.timeframe,
            sim_cfg=sim_cfg, risk_cfg=risk_cfg,
            fitness_cfg=fcfg, survival_cfg=survival_cfg,
            train_end=train_end, val_end=val_end,
            timesteps=args.timesteps, eval_window=args.eval_window,
            score_window=args.score_window,
        )
        symbol_map = {"alive": "+", "weak": "~", "dead": "x"}
        print(f"[{i:>3}/{len(agents)}] {agent.agent_id} "
              f"ret={report.total_return:+7.2%} dd={report.max_drawdown:>7.2%} "
              f"sharpe={report.sharpe:>7.3f} trades={report.n_trades:>4d} "
              f"fit={metrics['fitness']:+7.3f} [{symbol_map[verdict.status]}] "
              f"({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------
    # leaderboard: THIS run's cohort only - the DB accumulates rosters
    # ------------------------------------------------------------------
    cohort_ids = {a.agent_id for a in agents}
    rows = [r for r in get_agents(db_path=pop.db_path)
            if r["agent_id"] in cohort_ids
            and r["metrics"] and "fitness" in r["metrics"]]
    rows.sort(key=lambda r: r["metrics"]["fitness"], reverse=True)

    print(f"\n=== LEADERBOARD (by fitness: {args.fitness}) ===")
    print(format_header() + "     fitness  status")
    for r in rows:
        m = r["metrics"]
        report = type("R", (), m)()  # MetricsReport-like shim for formatting
        print(format_row(r["agent_id"][-8:], report)
              + f"  {m['fitness']:+8.3f}  {r['status']:>5}")

    rets = np.array([r["metrics"]["total_return"] for r in rows])
    print(f"\nspread: mean {rets.mean():+.2%} | std {rets.std():.2%} | "
          f"best {rets.max():+.2%} | worst {rets.min():+.2%} | "
          f"wall {time.time() - started:.0f}s")

    record_experiment("population", {
        "size": len(agents),
        "master_seed": args.master_seed,
        "symbol": symbol,
        "timeframe": args.timeframe,
        "timesteps": args.timesteps,
        "score_window": args.score_window,
        "fitness": args.fitness,
        "baseline_return": baseline,
        "git_commit": git_commit(),
        "returns": rets.tolist(),
        "agent_ids": [r["agent_id"] for r in rows],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
