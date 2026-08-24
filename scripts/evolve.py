"""Run the generational evolution loop and chart what happened.

Usage:
    .venv\\Scripts\\python.exe scripts\\evolve.py --generations 3
    .venv\\Scripts\\python.exe scripts\\evolve.py --generations 50 --timesteps 8000
"""

from __future__ import annotations

import argparse
import time

from darwin.agents import BuyAndHoldStrategy
from darwin.environment.simulator import TradingSimulator
from darwin.evolution.diversity import population_diversity
from darwin.evolution.fitness import preset
from darwin.evolution.generation import (
    cohort_rows,
    cohort_stats,
    run_generation,
)
from darwin.evolution.population import Population
from darwin.evolution.reproduction import ReproductionConfig
from darwin.evolution.survival import SurvivalConfig
from darwin.execution.risk import RiskConfig
from darwin.experiments.tracker import (
    get_agents,
    get_generations,
    get_genome,
    record_experiment,
    record_generation_stats,
)
from darwin.experiments.training import (
    git_commit,
    load_frames,
    sim_config_from_yaml,
)
from darwin.visualization.evolution import plot_evolution

DB = "experiments/metadata.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--master-seed", type=int, default=2026)
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--score-window", type=int, default=5_000)
    parser.add_argument("--fitness", default="spec")
    parser.add_argument("--offspring-per-rank", type=int, nargs="+", default=[2, 1, 1])
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--mutation-intensity", type=float, default=0.25)
    parser.add_argument("--immigrants", type=int, default=1)
    parser.add_argument("--intensity-decay", type=float, default=1.0)
    parser.add_argument("--seeds-per-agent", type=int, default=1)
    parser.add_argument("--reward-baseline-weight", type=float, default=0.0)
    parser.add_argument("--charts-dir", default="experiments/figures")
    return parser.parse_args()


def backfill_history() -> None:
    """Record stats for cohorts that predate the generations table."""
    recorded = {g["generation"] for g in get_generations(db_path=DB)}
    all_gens = sorted({r["generation"] for r in get_agents(db_path=DB)})
    for g in all_gens:
        if g in recorded:
            continue
        rows = cohort_rows(g, db_path=DB)
        stats = cohort_stats(rows)
        genomes = []
        for r in rows:
            genome = get_genome(r["genome_id"], db_path=DB)
            if genome:
                genomes.append(genome["values"])
        diversity = population_diversity(genomes) if genomes else None
        record_generation_stats(
            g,
            n_agents=stats["n_agents"],
            best_fitness=stats["best_fitness"],
            median_fitness=stats["median_fitness"],
            worst_fitness=stats["worst_fitness"],
            best_return=stats["best_return"],
            mean_drawdown=stats["mean_drawdown"],
            n_alive=stats["n_alive"],
            n_weak=stats["n_weak"],
            n_dead=stats["n_dead"],
            diversity_mean=diversity.mean_pairwise if diversity else None,
            diversity_min=diversity.min_pairwise if diversity else None,
            n_immigrants=0,
            db_path=DB,
        )
        print(f"backfilled history for generation {g}")


def main() -> int:
    args = parse_args()
    cfg, symbol, ohlcv, feats = load_frames(args.config, args.timeframe)
    n = len(ohlcv)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])
    survival_cfg = SurvivalConfig()

    window = ohlcv.iloc[val_end : val_end + args.score_window].reset_index(drop=True)
    bh = TradingSimulator(sim_cfg).run(
        window, BuyAndHoldStrategy().generate_actions(window)
    )
    baseline = float(
        bh.equity_curve["equity"].iloc[-1] / bh.equity_curve["equity"].iloc[0] - 1.0
    )
    fcfg = preset(args.fitness, baseline_return=baseline)

    pop = Population(size=2, db_path=DB)
    if not get_agents(db_path=DB):
        pop.initialize(master_seed=args.master_seed)
        print(f"bootstrapped generation 0 with {len(pop.agents)} founders")

    backfill_history()

    rcfg = ReproductionConfig(
        offspring_per_rank=tuple(args.offspring_per_rank),
        mutation_rate=args.mutation_rate,
        mutation_intensity=args.mutation_intensity,
        intensity_decay=args.intensity_decay,
        immigrants_per_generation=args.immigrants,
    )

    def evaluator(agent):
        return pop.evaluate_agent(
            agent,
            ohlcv=ohlcv, feats=feats, timeframe=args.timeframe,
            sim_cfg=sim_cfg, risk_cfg=risk_cfg,
            fitness_cfg=fcfg, survival_cfg=survival_cfg,
            train_end=train_end, val_end=val_end,
            timesteps=args.timesteps, score_window=args.score_window,
            n_seeds=args.seeds_per_agent,
            reward_baseline_weight=args.reward_baseline_weight,
        )

    started = time.time()
    ran: list[int] = []
    for _ in range(args.generations):
        pending = [r for r in get_agents(db_path=DB) if r["metrics"] is None]
        if not pending:
            print("no pending agents - nothing to evaluate; stopping")
            break
        target = min(r["generation"] for r in pending)
        print(f"\n--- generation {target}: evaluating "
              f"{len([r for r in pending if r['generation'] == target])} agent(s) ---")
        summary = run_generation(
            generation=target,
            population=pop,
            rcfg=rcfg,
            master_seed=args.master_seed,
            evaluator=evaluator,
            db_path=DB,
        )
        ran.append(target)
        s = summary["stats"]
        print(
            f"gen {target}: best={_fmt(s['best_fitness'])} "
            f"median={_fmt(s['median_fitness'])} worst={_fmt(s['worst_fitness'])} "
            f"| alive/weak/dead {s['n_alive']}/{s['n_weak']}/{s['n_dead']} "
            f"| diversity {summary['diversity_mean']:.3f} "
            f"| bred {summary['children_bred']} into gen {target + 1}"
        )

    history = get_generations(db_path=DB)
    if history:
        prefix = f"{args.charts_dir}/evolution_{symbol}_{args.timeframe}"
        paths = plot_evolution(history, prefix)
        for p in paths:
            print(f"chart -> {p}")

    best = max(
        (r for r in get_agents(db_path=DB)
         if r["metrics"] and r["metrics"].get("fitness") is not None),
        key=lambda r: r["metrics"]["fitness"],
        default=None,
    )
    if best:
        print(f"\nhall of fame: {best['agent_id']} "
              f"fitness={best['metrics']['fitness']:+.3f} "
              f"ret={best['metrics']['total_return']:+.2%} "
              f"(gen {best['generation']})")

    record_experiment("evolve", {
        "generations_run": ran,
        "timesteps": args.timesteps,
        "fitness": args.fitness,
        "rcfg": str(rcfg),
        "wall_s": round(time.time() - started, 1),
        "git_commit": git_commit(),
    })
    return 0


def _fmt(value: float | None) -> str:
    return f"{value:+.3f}" if value is not None else "  n/a"


if __name__ == "__main__":
    raise SystemExit(main())
