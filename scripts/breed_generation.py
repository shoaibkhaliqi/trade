"""Breed the next generation from the current roster, optionally train them.

Usage:
    .venv\\Scripts\\python.exe scripts\\breed_generation.py
    .venv\\Scripts\\python.exe scripts\\breed_generation.py --no-train-children
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from darwin.evolution.fitness import preset
from darwin.evolution.population import Population
from darwin.evolution.reproduction import (
    ReproductionConfig,
    reproduce,
    select_parents,
    summarize_mutations,
)
from darwin.evolution.survival import SurvivalConfig
from darwin.execution.risk import RiskConfig
from darwin.experiments.tracker import get_agents, record_experiment
from darwin.experiments.training import git_commit, load_frames, sim_config_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--offspring-per-rank", type=int, nargs="+", default=[2, 1, 1])
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--mutation-intensity", type=float, default=0.25)
    parser.add_argument("--master-seed", type=int, default=2026)
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--score-window", type=int, default=5_000)
    parser.add_argument("--fitness", default="spec")
    parser.add_argument("--inherit-weights", action="store_true",
                        help="children fine-tune the parent's trained policy")
    parser.add_argument("--no-train-children", action="store_true",
                        help="only create children; skip their evaluation")
    parser.add_argument("--immigrants", type=int, default=1,
                        help="fresh random genomes injected this generation")
    parser.add_argument("--intensity-decay", type=float, default=1.0,
                        help="per-generation multiplier on mutation intensity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, symbol, ohlcv, feats = load_frames(args.config, args.timeframe)
    n = len(ohlcv)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])
    survival_cfg = SurvivalConfig()

    # baseline on the identical score window (same recipe as run_population)
    from darwin.agents import BuyAndHoldStrategy
    from darwin.environment.simulator import TradingSimulator

    window = ohlcv.iloc[val_end : val_end + args.score_window].reset_index(drop=True)
    bh = TradingSimulator(sim_cfg).run(
        window, BuyAndHoldStrategy().generate_actions(window)
    )
    baseline = float(
        bh.equity_curve["equity"].iloc[-1] / bh.equity_curve["equity"].iloc[0] - 1.0
    )
    fcfg = preset(args.fitness, baseline_return=baseline)

    rows = get_agents()
    current_gen = max((r["generation"] for r in rows), default=0)
    next_gen = current_gen + 1

    rcfg = ReproductionConfig(
        offspring_per_rank=tuple(args.offspring_per_rank),
        mutation_rate=args.mutation_rate,
        mutation_intensity=args.mutation_intensity,
        intensity_decay=args.intensity_decay,
        immigrants_per_generation=args.immigrants,
    )
    parents = select_parents(rows, rcfg)
    print(f"generation {current_gen} -> {next_gen} | fitness={args.fitness} "
          f"(b&h baseline {baseline:+.2%})")
    print("selected parents:")
    for parent_row, n_children in parents:
        print(f"  {parent_row['agent_id']}  fitness={parent_row['metrics']['fitness']:+.3f} "
              f"status={parent_row['status']} -> {n_children} child(ren)")

    rng = np.random.default_rng(args.master_seed + next_gen)
    pop = Population(size=2, db_path="experiments/metadata.sqlite")

    from darwin.evolution.diversity import format_diversity, population_diversity
    from darwin.experiments.tracker import get_genome

    roster_genomes = [
        get_genome(r["genome_id"], db_path=pop.db_path)["values"]
        for r in rows
        if get_genome(r["genome_id"], db_path=pop.db_path)
    ]
    print("\ndiversity BEFORE breeding:")
    print(format_diversity(population_diversity(roster_genomes)))

    children = reproduce(pop, parents, rng, rcfg,
                         generation=next_gen, master_seed=args.master_seed)

    after_genomes = roster_genomes + [c.genome.values for c in children]
    print(f"\ndiversity AFTER breeding (+{len(children)} children, "
          f"{rcfg.immigrants_per_generation} immigrant(s)):")
    print(format_diversity(population_diversity(after_genomes)))

    print("\nbirth records:")
    for m in summarize_mutations(children):
        print(f"  {m.gene:<20} {m.old:>10.4f} -> {m.new:>10.4f}")
    print(f"{len(children)} children created")

    if args.no_train_children:
        print("(skipping child training)")
        record_experiment("breed", {
            "generation": next_gen, "children": len(children),
            "parent_ids": [p[0]["agent_id"] for p in parents],
            "git_commit": git_commit(),
        })
        return 0

    print()
    model_paths = {r["agent_id"]: r["model_path"] for r in rows}
    # children were created in parent order; rebuild that order to wire
    # weight inheritance (and bookkeeping) to the right parent rows
    ordered_parents = [p for p, k in parents for _ in range(k)]
    results = []
    for _i, (child, parent_row) in enumerate(zip(children, ordered_parents, strict=True)):
        t0 = time.time()
        inherit = model_paths.get(parent_row["agent_id"]) if args.inherit_weights else None
        report, metrics, verdict = pop.evaluate_agent(
            child,
            ohlcv=ohlcv, feats=feats, timeframe=args.timeframe,
            sim_cfg=sim_cfg, risk_cfg=risk_cfg,
            fitness_cfg=fcfg, survival_cfg=survival_cfg,
            train_end=train_end, val_end=val_end,
            timesteps=args.timesteps,
            score_window=args.score_window,
            inherit_weights_from=inherit,
        )
        results.append((child, metrics, verdict))
        print(f"[g{next_gen}] {child.agent_id} "
              f"ret={report.total_return:+7.2%} dd={report.max_drawdown:>7.2%} "
              f"trades={report.n_trades:>4d} fit={metrics['fitness']:+7.3f} "
              f"[{verdict.status}] ({time.time() - t0:.0f}s)")

    print(f"\n=== GENERATION {next_gen} LEADERBOARD ===")
    for child, metrics, _v in sorted(results, key=lambda t: t[1]["fitness"], reverse=True):
        print(f"{child.agent_id:<18} fitness={metrics['fitness']:+.3f} "
              f"ret={metrics['total_return']:+.2%}")

    record_experiment("breed", {
        "generation": next_gen,
        "children": len(children),
        "parent_ids": [p[0]["agent_id"] for p in parents],
        "inherit_weights": args.inherit_weights,
        "child_fitness": [m["fitness"] for _c, m, _v in results],
        "git_commit": git_commit(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
