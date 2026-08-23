"""One-generation orchestrator: evaluate cohort, record stats, breed the next.

Semantics (deliberate):
- A COHORT is the set of agents bred into generation g (children+immigrants).
  Per-generation statistics describe the cohort; the full roster remains as
  ancestry and hall-of-fame material.
- The evaluator is injected: production passes Population.evaluate_agent
  (the full training funnel); tests inject fakes, making loop logic
  exhaustively testable in milliseconds instead of GPU-hours.
- Selection breeds from the just-evaluated cohort - generational replacement,
  with the roster preserving every ancestor's record.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from darwin.evolution.diversity import population_diversity
from darwin.evolution.reproduction import (
    ReproductionConfig,
    reproduce,
    select_parents,
)
from darwin.evolution.survival import STATUS_DEAD
from darwin.experiments.tracker import (
    get_agents,
    get_genome,
    record_generation_stats,
)


def cohort_rows(generation: int, db_path: Any = "experiments/metadata.sqlite") -> list[dict]:
    """All agents bred into ``generation`` (children + immigrants)."""
    return [r for r in get_agents(db_path=db_path) if r["generation"] == generation]


def cohort_stats(rows: list[dict]) -> dict[str, Any]:
    """Fitness/health summary over evaluated cohort rows."""
    scored = [r for r in rows if r["metrics"] and r["metrics"].get("fitness") is not None]
    if not scored:
        return {
            "n_agents": len(rows), "best_fitness": None, "median_fitness": None,
            "worst_fitness": None, "best_return": None, "mean_drawdown": None,
            "n_alive": 0, "n_weak": 0, "n_dead": 0,
        }
    fits = sorted(float(r["metrics"]["fitness"]) for r in scored)
    n = len(fits)
    median = fits[n // 2] if n % 2 else (fits[n // 2 - 1] + fits[n // 2]) / 2
    dds = [abs(float(r["metrics"]["max_drawdown"])) for r in scored]
    statuses = [r["status"] for r in scored]
    return {
        "n_agents": len(rows),
        "best_fitness": fits[-1],
        "median_fitness": median,
        "worst_fitness": fits[0],
        "best_return": max(float(r["metrics"]["total_return"]) for r in scored),
        "mean_drawdown": sum(dds) / len(dds),
        "n_alive": statuses.count("alive"),
        "n_weak": statuses.count("weak"),
        "n_dead": statuses.count(STATUS_DEAD),
    }


def cohort_genomes(rows: list[dict], db_path: Any) -> list[dict[str, float]]:
    genomes: list[dict[str, float]] = []
    for r in rows:
        g = get_genome(r["genome_id"], db_path=db_path)
        if g is not None:
            genomes.append(g["values"])
    return genomes


Evaluator = Callable[[Any], tuple[Any, dict, Any]]


def run_generation(
    *,
    generation: int,
    population: Any,
    rcfg: ReproductionConfig,
    master_seed: int,
    evaluator: Evaluator,
    db_path: Any = "experiments/metadata.sqlite",
) -> dict[str, Any]:
    """Evaluate generation ``g``'s pending cohort, record stats, breed g+1.

    Returns a summary dict with cohort stats, verdicts and the children bred.
    """
    rows = cohort_rows(generation, db_path=db_path)
    pending = [r for r in rows if r["metrics"] is None]

    from darwin.evolution.genome import Genome
    from darwin.evolution.population import AgentSpec
    from darwin.experiments.tracker import mark_agent_status

    verdicts: dict[str, str] = {}
    for row in pending:
        genome_row = get_genome(row["genome_id"], db_path=db_path)
        assert genome_row is not None
        agent = AgentSpec(
            agent_id=row["agent_id"],
            genome=Genome(values=genome_row["values"],
                          genome_id=row["genome_id"]),
            seed=row["seed"],
            generation=row["generation"],
        )
        # contract: evaluator returns (report, metrics, verdict); the
        # orchestrator OWNS persistence - fakes stay pure, production's
        # internal write is an idempotent repeat. model_path stays untouched:
        # the evaluator already recorded the real artifact location.
        _report, metrics, verdict = evaluator(agent)
        population.record_result(agent.agent_id, metrics=metrics)
        mark_agent_status(
            agent.agent_id, verdict.status,
            reason="; ".join(verdict.reasons) if verdict.reasons else None,
            fitness=metrics.get("fitness"),
            max_drawdown=metrics.get("max_drawdown"),
            db_path=db_path,
        )
        verdicts[agent.agent_id] = verdict.status

    # re-fetch: evaluation persisted metrics; the pre-eval snapshot is stale
    rows = cohort_rows(generation, db_path=db_path)
    stats = cohort_stats(rows)

    diversity = None
    genomes = cohort_genomes(rows, db_path)
    if genomes:
        diversity = population_diversity(genomes)

    record_generation_stats(
        generation,
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
        n_immigrants=rcfg.immigrants_per_generation,
        db_path=db_path,
    )

    # breed the next cohort from THIS cohort (generational replacement)
    fresh_rows = cohort_rows(generation, db_path=db_path)
    parents = select_parents(fresh_rows, rcfg)
    rng = np.random.default_rng(master_seed + generation + 1)
    children = reproduce(population, parents, rng, rcfg,
                         generation=generation + 1,
                         master_seed=master_seed)

    return {
        "generation": generation,
        "stats": stats,
        "verdicts": verdicts,
        "diversity_mean": diversity.mean_pairwise if diversity else None,
        "diversity_min": diversity.min_pairwise if diversity else None,
        "children_bred": len(children),
        "parents_selected": [p[0]["agent_id"] for p in parents],
    }
