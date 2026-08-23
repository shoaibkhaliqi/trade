"""Reproduction & selection tests - fast, training-free, lineage-focused."""

from __future__ import annotations

import numpy as np
import pytest

from darwin.evolution.genome import GENE_SPECS
from darwin.evolution.population import Population
from darwin.evolution.reproduction import (
    ReproductionConfig,
    reproduce,
    select_parents,
)
from darwin.experiments.tracker import get_agents, get_genome


def _row(agent_id: str, status: str, fitness: float | None, genome_id: str) -> dict:
    metrics = None if fitness is None else {"fitness": fitness, "total_return": fitness}
    return {
        "agent_id": agent_id, "status": status, "metrics": metrics,
        "genome_id": genome_id, "seed": 1, "generation": 0,
    }


class TestSelection:
    def test_dead_excluded_and_fitness_ranks(self) -> None:
        rows = [
            _row("champ", "alive", 0.5, "g1"),
            _row("ghost", "dead", 0.9, "g2"),      # dead: out of the gene pool
            _row("mid", "weak", 0.2, "g3"),
            _row("noneval", "alive", None, "g4"),  # unevaluated: ineligible
        ]
        pairs = select_parents(rows, ReproductionConfig(offspring_per_rank=(2, 1, 1)))

        assert [p[0]["agent_id"] for p in pairs] == ["champ", "mid"]
        assert pairs[0][1] == 2
        assert pairs[1][1] == 1

    def test_fewer_eligible_than_ranks_truncates(self) -> None:
        rows = [_row("only", "alive", 0.1, "g1")]
        pairs = select_parents(rows, ReproductionConfig(offspring_per_rank=(2, 1, 1)))
        assert len(pairs) == 1

    def test_tie_breaks_stable_by_input_order(self) -> None:
        rows = [_row("a", "alive", 0.5, "g1"), _row("b", "alive", 0.5, "g2")]
        pairs = select_parents(rows, ReproductionConfig(offspring_per_rank=(1, 1)))
        assert [p[0]["agent_id"] for p in pairs] == ["a", "b"]


class TestConfigValidation:
    def test_rejects_nonsense(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ReproductionConfig(offspring_per_rank=(2, -1))
        with pytest.raises(ValueError, match="zero total"):
            ReproductionConfig(offspring_per_rank=(0, 0))
        with pytest.raises(ValueError, match="mutation_rate"):
            ReproductionConfig(mutation_rate=1.5)


class TestReproduce:
    @pytest.fixture
    def setup(self, tmp_path):
        pop = Population(size=4, db_path=tmp_path / "m.sqlite")
        agents = pop.initialize(master_seed=42)
        # attach fitness to two founders
        pop.record_result(agents[0].agent_id,
                          metrics={"fitness": 0.8, "total_return": 0.05},
                          model_path="runs/p0.zip")
        pop.record_result(agents[1].agent_id,
                          metrics={"fitness": -0.2, "total_return": -0.01},
                          model_path="runs/p1.zip")
        from darwin.experiments.tracker import mark_agent_status

        mark_agent_status(agents[0].agent_id, "alive", db_path=pop.db_path)
        mark_agent_status(agents[1].agent_id, "weak", db_path=pop.db_path)
        rows = get_agents(db_path=pop.db_path)
        return pop, agents, rows

    def test_children_created_with_lineage(self, setup) -> None:
        pop, agents, rows = setup
        cfg = ReproductionConfig(offspring_per_rank=(2, 1))
        parents = select_parents(rows, cfg)

        children = reproduce(pop, parents, np.random.default_rng(7), cfg,
                             generation=1, master_seed=42)

        assert len(children) == 3
        assert all(c.generation == 1 for c in children)
        assert len({c.agent_id for c in children}) == 3
        assert len({c.seed for c in children}) == 3
        # parent links flow through genomes
        champ_id = agents[0].genome.genome_id
        weak_id = agents[1].genome.genome_id
        champ_children = [c for c in children if c.genome.parent_id == champ_id]
        weak_children = [c for c in children if c.genome.parent_id == weak_id]
        assert len(champ_children) == 2
        assert len(weak_children) == 1

    def test_children_within_bounds_and_persisted(self, setup, tmp_path) -> None:
        pop, agents, rows = setup
        cfg = ReproductionConfig(offspring_per_rank=(1,), mutation_rate=1.0,
                                 mutation_intensity=0.4)
        children = reproduce(pop, select_parents(rows, cfg),
                             np.random.default_rng(3), cfg,
                             generation=1, master_seed=42)
        db = tmp_path / "m.sqlite"

        for child in children:
            for name, value in child.genome.values.items():
                spec = GENE_SPECS[name]
                assert spec.low <= value <= spec.high
            row = get_agents(db_path=db, status="alive")
            assert any(r["agent_id"] == child.agent_id for r in row)
            g = get_genome(child.genome.genome_id, db_path=db)
            assert g is not None
            assert g["generation"] == 1
            assert g["parent_id"] == agents[0].genome.genome_id
            assert len(g["mutations"]) >= 1  # rate=1.0 mutated every gene

    def test_deterministic_given_rng(self, setup) -> None:
        pop, agents, rows = setup
        cfg = ReproductionConfig(offspring_per_rank=(1,))

        c1 = reproduce(pop, select_parents(rows, cfg), np.random.default_rng(9),
                       cfg, generation=1, master_seed=42)
        # fresh population copy to avoid id collisions
        pop2 = Population(size=4, db_path=pop.db_path)
        c2 = reproduce(pop2, select_parents(rows, cfg), np.random.default_rng(9),
                       cfg, generation=1, master_seed=42)

        assert [c.genome.values for c in c1] == [c.genome.values for c in c2]
        assert [c.seed for c in c1] == [c.seed for c in c2]

    def test_zero_mutation_rate_clones_parent_genome(self, setup) -> None:
        pop, agents, rows = setup
        cfg = ReproductionConfig(offspring_per_rank=(1,), mutation_rate=0.0)
        children = reproduce(pop, select_parents(rows, cfg),
                             np.random.default_rng(1), cfg,
                             generation=1, master_seed=42)

        parent_values = agents[0].genome.values
        assert children[0].genome.values == parent_values
