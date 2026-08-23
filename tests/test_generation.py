"""Generational-loop tests - cohort semantics with an injected fake evaluator."""

from __future__ import annotations

import pytest

from darwin.evolution.generation import (
    cohort_rows,
    cohort_stats,
    run_generation,
)
from darwin.evolution.population import Population
from darwin.evolution.reproduction import ReproductionConfig
from darwin.evolution.survival import Verdict


class FakeReport:
    def __init__(self, ret: float, dd: float) -> None:
        self.total_return = ret
        self.max_drawdown = dd
        self.sharpe = 0.5
        self.sortino = 0.5
        self.profit_factor = 1.0
        self.win_rate = 0.5
        self.n_trades = 10
        self.fees_paid = 1.0
        self.avg_trade_net = 0.1
        self.exposure = 0.5
        self.periods_per_year = 35040.0


def _fake_evaluator(seed_scores: dict[int, float]):
    """Deterministic fake: fitness derived from the agent's training seed."""

    def evaluate(agent):
        score = seed_scores.get(agent.seed, 0.0)
        metrics = vars(FakeReport(ret=score, dd=0.05))
        metrics["fitness"] = score
        return FakeReport(ret=score, dd=0.05), metrics, Verdict("alive", ())

    return evaluate


@pytest.fixture
def world(tmp_path):
    pop = Population(size=6, db_path=tmp_path / "m.sqlite")
    pop.initialize(master_seed=42)
    db = tmp_path / "m.sqlite"
    rcfg = ReproductionConfig(offspring_per_rank=(2, 1),
                              immigrants_per_generation=1)
    return pop, db, rcfg


class TestCohortStats:
    def test_best_median_worst_math(self) -> None:
        rows = [
            {"metrics": {"fitness": 0.5, "total_return": 0.1,
                         "max_drawdown": -0.02}, "status": "alive"},
            {"metrics": {"fitness": -0.2, "total_return": -0.01,
                         "max_drawdown": -0.08}, "status": "weak"},
            {"metrics": {"fitness": 0.1, "total_return": 0.03,
                         "max_drawdown": -0.04}, "status": "alive"},
        ]
        stats = cohort_stats(rows)
        assert stats["best_fitness"] == 0.5
        assert stats["median_fitness"] == 0.1
        assert stats["worst_fitness"] == -0.2
        assert stats["best_return"] == 0.1
        assert stats["mean_drawdown"] == pytest.approx((0.02 + 0.08 + 0.04) / 3)
        assert stats["n_alive"] == 2 and stats["n_weak"] == 1

    def test_even_count_median_averages_middle(self) -> None:
        rows = [
            {"metrics": {"fitness": 1.0, "total_return": 0, "max_drawdown": 0},
             "status": "alive"},
            {"metrics": {"fitness": 0.0, "total_return": 0, "max_drawdown": 0},
             "status": "alive"},
        ]
        assert cohort_stats(rows)["median_fitness"] == 0.5

    def test_empty_cohort_is_safe(self) -> None:
        stats = cohort_stats([])
        assert stats["best_fitness"] is None
        assert stats["n_agents"] == 0


class TestRunGeneration:
    def test_full_cycle_with_fake_evaluator(self, world) -> None:
        pop, db, rcfg = world
        scores = {a.seed: float(i) * 0.1 for i, a in enumerate(pop.agents)}

        summary = run_generation(
            generation=0,
            population=pop,
            rcfg=rcfg,
            master_seed=42,
            evaluator=_fake_evaluator(scores),
            db_path=db,
        )

        # founders evaluated
        assert summary["stats"]["n_agents"] == 6
        assert summary["stats"]["best_fitness"] == 0.5
        # bred 2+1 children + 1 immigrant = 4 into generation 1
        assert summary["children_bred"] == 4
        assert len(cohort_rows(1, db_path=db)) == 4

        # generation stats persisted
        from darwin.experiments.tracker import get_generations

        gens = get_generations(db_path=db)
        assert len(gens) == 1
        assert gens[0]["generation"] == 0
        assert gens[0]["n_immigrants"] == 1

    def test_second_generation_evaluates_only_pending(self, world) -> None:
        pop, db, rcfg = world
        base_scores = {a.seed: 0.0 for a in pop.agents}
        run_generation(generation=0, population=pop, rcfg=rcfg,
                       master_seed=42, evaluator=_fake_evaluator(base_scores),
                       db_path=db)

        cohort1 = cohort_rows(1, db_path=db)
        child_scores = {r["seed"]: 0.9 for r in cohort1}
        summary = run_generation(generation=1, population=pop, rcfg=rcfg,
                                 master_seed=42,
                                 evaluator=_fake_evaluator(child_scores),
                                 db_path=db)

        assert summary["stats"]["best_fitness"] == 0.9
        assert summary["stats"]["n_agents"] == len(cohort1)

        from darwin.experiments.tracker import get_generations

        assert [g["generation"] for g in get_generations(db_path=db)] == [0, 1]

    def test_selection_uses_cohort_not_whole_roster(self, world) -> None:
        """A gen-0 founder with huge fitness must NOT parent gen-2."""
        pop, db, rcfg = world
        founder_scores = {a.seed: 10.0 for a in pop.agents}
        run_generation(generation=0, population=pop, rcfg=rcfg, master_seed=42, evaluator=_fake_evaluator(founder_scores), db_path=db)

        # generation 1 children are mediocre
        cohort1 = cohort_rows(1, db_path=db)
        child_scores = {r["seed"]: -1.0 for r in cohort1}
        summary = run_generation(generation=1, population=pop, rcfg=rcfg, master_seed=42, evaluator=_fake_evaluator(child_scores), db_path=db)

        # parents for gen 2 came from cohort 1 (fitness -1), not the founders
        assert summary["parents_selected"][0].startswith("a00042-g01")
