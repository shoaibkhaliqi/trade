"""Population tests - roster fairness and persistence (training-free, fast)."""

from __future__ import annotations

import pytest

from darwin.evolution.genome import GENE_SPECS, Genome
from darwin.evolution.population import Population
from darwin.experiments.tracker import get_agents, get_genome


@pytest.fixture
def pop(tmp_path):
    return Population(size=6, db_path=tmp_path / "meta.sqlite")


class TestInitialization:
    def test_creates_distinct_genomes_and_seeds(self, pop) -> None:
        agents = pop.initialize(master_seed=42)

        assert len(agents) == 6
        genome_ids = {a.genome.genome_id for a in agents}
        seeds = {a.seed for a in agents}
        agent_ids = {a.agent_id for a in agents}
        assert len(genome_ids) == 6   # no clones at birth
        assert len(seeds) == 6
        assert len(agent_ids) == 6
        assert all(a.generation == 0 for a in agents)

    def test_deterministic_from_master_seed(self, tmp_path) -> None:
        a = Population(4, db_path=tmp_path / "x.sqlite").initialize(master_seed=7)
        b = Population(4, db_path=tmp_path / "y.sqlite").initialize(master_seed=7)
        assert [s.genome.values for s in a] == [s.genome.values for s in b]
        assert [s.seed for s in a] == [s.seed for s in b]

    def test_genomes_within_bounds_and_persisted(self, pop, tmp_path) -> None:
        agents = pop.initialize(master_seed=1)
        db = tmp_path / "meta.sqlite"

        for agent in agents:
            for name, value in agent.genome.values.items():
                spec = GENE_SPECS[name]
                assert spec.low <= value <= spec.high
            row = get_genome(agent.genome.genome_id, db_path=db)
            assert row is not None
            assert row["values"] == agent.genome.values
            assert row["generation"] == 0

    def test_agent_rows_written_before_evaluation(self, pop, tmp_path) -> None:
        pop.initialize(master_seed=3)
        rows = get_agents(db_path=tmp_path / "meta.sqlite")
        assert len(rows) == 6
        assert all(r["status"] == "alive" for r in rows)
        assert all(r["metrics"] is None for r in rows)

    def test_record_result_updates_row(self, pop, tmp_path) -> None:
        agents = pop.initialize(master_seed=5)
        db = tmp_path / "meta.sqlite"

        pop.record_result(
            agents[0].agent_id,
            metrics={"total_return": 0.12, "sharpe": 1.5},
            model_path="runs/x.zip",
        )

        rows = {r["agent_id"]: r for r in get_agents(db_path=db)}
        done = rows[agents[0].agent_id]
        assert done["status"] == "evaluated"
        assert done["metrics"]["total_return"] == pytest.approx(0.12)
        assert done["model_path"] == "runs/x.zip"
        assert rows[agents[1].agent_id]["status"] == "alive"


class TestGuards:
    def test_size_below_two_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="rivals"):
            Population(1, db_path=tmp_path / "z.sqlite")


def test_genome_reconstruction_from_values(tmp_path) -> None:
    """Population must be able to rebuild Genome objects from stored rows."""
    pop = Population(3, db_path=tmp_path / "m.sqlite")
    agents = pop.initialize(master_seed=11)
    rebuilt = Genome(values=agents[0].genome.values,
                     genome_id=agents[0].genome.genome_id)
    assert rebuilt.values == agents[0].genome.values
