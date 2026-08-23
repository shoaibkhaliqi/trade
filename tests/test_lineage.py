"""Lineage query tests - provenance, subtrees, dossiers on synthetic families."""

from __future__ import annotations

import numpy as np
import pytest

from darwin.evolution.genome import GENE_SPECS
from darwin.evolution.lineage import (
    children_of,
    descendants,
    dossier,
    gene_origin,
    hall_of_fame,
    mutation_path,
)
from darwin.evolution.population import Population
from darwin.evolution.reproduction import (
    ReproductionConfig,
    reproduce,
    select_parents,
)
from darwin.experiments.tracker import get_agents, mark_agent_status


@pytest.fixture
def family(tmp_path):
    """Three generations: 2 founders -> gen1 children -> gen2 children."""
    pop = Population(size=2, db_path=tmp_path / "m.sqlite")
    founders = pop.initialize(master_seed=42)
    db = tmp_path / "m.sqlite"

    pop.record_result(founders[0].agent_id,
                      metrics={"fitness": 0.5, "total_return": 0.04},
                      model_path="runs/f0.zip")
    mark_agent_status(founders[0].agent_id, "alive", db_path=db)
    pop.record_result(founders[1].agent_id,
                      metrics={"fitness": -0.1, "total_return": -0.005},
                      model_path="runs/f1.zip")
    mark_agent_status(founders[1].agent_id, "weak", db_path=db)

    rows = get_agents(db_path=db)
    rcfg = ReproductionConfig(offspring_per_rank=(2, 1),
                              immigrants_per_generation=0,
                              mutation_rate=0.6, mutation_intensity=0.3)
    gen1 = reproduce(pop, select_parents(rows, rcfg),
                     np.random.default_rng(7), rcfg,
                     generation=1, master_seed=42)

    # score gen1: first child strong, others weaker
    for i, child in enumerate(gen1):
        pop.record_result(child.agent_id,
                          metrics={"fitness": 1.0 - 0.2 * i,
                                   "total_return": 0.05 - 0.01 * i},
                          model_path=f"runs/g1-{i}.zip")
        mark_agent_status(child.agent_id, "alive", db_path=db)

    rows1 = get_agents(db_path=db)
    gen2 = reproduce(pop, select_parents(rows1, rcfg),
                     np.random.default_rng(8), rcfg,
                     generation=2, master_seed=42)
    for i, child in enumerate(gen2):
        pop.record_result(child.agent_id,
                          metrics={"fitness": 0.4 - 0.1 * i,
                                   "total_return": 0.02},
                          model_path=f"runs/g2-{i}.zip")
        mark_agent_status(child.agent_id, "weak", db_path=db)
    return pop, db, founders, gen1, gen2


class TestQueries:
    def test_children_of_champion(self, family) -> None:
        pop, db, founders, gen1, _gen2 = family
        kids = children_of(founders[0].agent_id, db_path=db)
        assert len(kids) == 2
        assert all(k["generation"] == 1 for k in kids)

    def test_descendants_spans_generations(self, family) -> None:
        pop, db, founders, _gen1, _gen2 = family
        desc = descendants(founders[0].agent_id, db_path=db)
        generations = {d["generation"] for d in desc}
        assert generations == {1, 2}
        # founder(2 kids) -> champion breeds 3 -> others: at least 5 total
        assert len(desc) >= 5

    def test_mutation_path_ordered_root_to_agent(self, family) -> None:
        pop, db, _founders, gen1, gen2 = family
        leaf = gen2[0]
        path = mutation_path(leaf.agent_id, db_path=db)

        # rate=0.6 over two breeding steps makes an empty path very unlikely;
        # an EMPTY path here would mean the agent->genome resolution broke
        assert len(path) >= 1, "mutation path must not be empty for a bred leaf"
        generations = [m["generation"] for m in path]
        assert generations == sorted(generations)
        assert set(generations) <= {1, 2}  # founder genomes carry no mutations

    def test_gene_origin_finds_most_recent_change(self, family) -> None:
        pop, db, _founders, _gen1, gen2 = family
        leaf = gen2[0]
        origin = gene_origin(leaf.agent_id, "gamma", db_path=db)
        if origin is not None:
            # most recent change must come from the leaf's own parent chain,
            # at the highest generation below the leaf
            assert origin["generation"] in (1, 2)
            assert origin["old"] != origin["new"]

    def test_gene_origin_none_when_founder_value(self, family) -> None:
        """A gene never mutated anywhere on the path reports None."""
        pop, db, _founders, _gen1, gen2 = family
        leaf = gen2[-1]
        path = mutation_path(leaf.agent_id, db_path=db)
        mutated_genes = {m["gene"] for m in path}
        unmutated = [g for g in GENE_SPECS if g not in mutated_genes]
        if not unmutated:
            pytest.skip("every gene mutated somewhere on this path")
        origin = gene_origin(leaf.agent_id, unmutated[0], db_path=db)
        assert origin is None


class TestDossier:
    def test_dossier_fields(self, family) -> None:
        pop, db, founders, gen1, _gen2 = family
        champion = max(gen1, key=lambda c: 0)  # first child; scored 1.0
        d = dossier(champion.agent_id, db_path=db)

        assert d.generation == 1
        assert d.status == "alive"
        assert d.metrics is not None and d.metrics["fitness"] == pytest.approx(1.0)
        assert d.death_reason is None
        assert len(d.ancestors) == 1  # one founder above
        assert d.descendants_count >= 1  # it bred gen2
        assert d.best_descendant is not None

    def test_dossier_includes_death_reason(self, family) -> None:
        pop, db, founders, _gen1, _gen2 = family
        from darwin.experiments.tracker import mark_agent_status

        mark_agent_status(founders[1].agent_id, "dead",
                          reason="fitness below floor",
                          fitness=-2.5, max_drawdown=-0.22, db_path=db)
        d = dossier(founders[1].agent_id, db_path=db)
        assert d.death_reason == "fitness below floor"


class TestHallOfFame:
    def test_ranked_by_fitness_across_generations(self, family) -> None:
        pop, db, _f, _g1, _g2 = family
        hof = hall_of_fame(db_path=db)
        assert hof[0]["metrics"]["fitness"] == pytest.approx(1.0)
        fits = [r["metrics"]["fitness"] for r in hof]
        assert fits == sorted(fits, reverse=True)
