"""Genome, mutation, and lineage tests - evolution's audit trail."""

from __future__ import annotations

import numpy as np
import pytest

from darwin.evolution.genome import GENE_SPECS, Genome
from darwin.execution.risk import RiskConfig
from darwin.experiments.tracker import get_genome, lineage_of, record_genome
from darwin.experiments.training import risk_config_from_genome


class TestGeneSpecs:
    def test_registry_is_complete_and_ordered(self) -> None:
        expected = {
            "position_size_pct", "stop_loss_pct", "take_profit_pct",
            "cooldown_bars", "max_trades_per_day",
            "learning_rate", "ent_coef", "gamma",
        }
        assert set(GENE_SPECS) == expected

    def test_invalid_spec_rejected(self) -> None:
        from darwin.evolution.genome import GeneSpec

        with pytest.raises(ValueError):
            GeneSpec("x", 5.0, 5.0)
        with pytest.raises(ValueError):
            GeneSpec("y", 1.0, 10.0, integer=True, log_scale=True)


class TestRandomGenome:
    def test_all_genes_within_bounds(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            g = Genome.random(rng)
            for name, value in g.values.items():
                spec = GENE_SPECS[name]
                assert spec.low <= value <= spec.high

    def test_same_seed_same_genome(self) -> None:
        a = Genome.random(np.random.default_rng(42))
        b = Genome.random(np.random.default_rng(42))
        assert a.values == b.values

    def test_integer_genes_are_whole_numbers(self) -> None:
        g = Genome.random(np.random.default_rng(3))
        assert g["cooldown_bars"] == int(g["cooldown_bars"])
        assert g["max_trades_per_day"] == int(g["max_trades_per_day"])

    def test_out_of_bounds_value_rejected(self) -> None:
        values = {name: (spec.low + spec.high) / 2 for name, spec in GENE_SPECS.items()}
        values["gamma"] = 1.5  # outside [0.90, 0.999]
        with pytest.raises(ValueError, match="outside"):
            Genome(values=values)


class TestMutation:
    def test_rate_zero_is_identity_with_no_records(self) -> None:
        parent = Genome.random(np.random.default_rng(1))
        child = parent.mutate(np.random.default_rng(2), rate=0.0)

        assert child.values == parent.values
        assert child.mutations == ()
        assert child.generation == parent.generation + 1
        assert child.parent_id == parent.genome_id or parent.genome_id == ""

    def test_mutations_are_recorded_exactly(self) -> None:
        rng = np.random.default_rng(7)
        parent = Genome.random(rng)
        child = parent.mutate(rng, rate=0.6, intensity=0.3, genome_id="child1")

        mutated_genes = [m.gene for m in child.mutations]
        # at most one record per gene
        assert len(mutated_genes) == len(set(mutated_genes))
        for rec in child.mutations:
            assert rec.old == parent[rec.gene]
            assert rec.new == child[rec.gene]
        # untouched genes are bit-identical to the parent
        for name in GENE_SPECS:
            if name not in mutated_genes:
                assert child[name] == parent[name]

    def test_bounds_are_never_violated_even_at_edges(self) -> None:
        rng = np.random.default_rng(9)
        edge_values = {name: spec.high for name, spec in GENE_SPECS.items()}
        parent = Genome(values=edge_values)
        child = parent.mutate(rng, rate=1.0, intensity=5.0)  # absurd sigma
        for name, value in child.values.items():
            spec = GENE_SPECS[name]
            assert spec.low <= value <= spec.high

    def test_deterministic_given_seed(self) -> None:
        parent = Genome.random(np.random.default_rng(4))
        c1 = parent.mutate(np.random.default_rng(11), rate=1.0, intensity=0.3)
        c2 = parent.mutate(np.random.default_rng(11), rate=1.0, intensity=0.3)
        assert c1.values == c2.values
        assert list(c1.mutations) == list(c2.mutations)

    def test_lineage_fields_flow_to_child(self) -> None:
        parent = Genome.random(np.random.default_rng(5), genome_id="parent-001")
        child = parent.mutate(np.random.default_rng(6), genome_id="child-002",
                              generation=17)

        assert child.parent_id == "parent-001"
        assert child.generation == 17


class TestPersistenceAndLineage:
    def test_genome_roundtrip_and_lineage_walk(self, tmp_path) -> None:
        db = tmp_path / "meta.sqlite"
        rng = np.random.default_rng(8)

        grand = Genome.random(rng, genome_id="g-grand")
        record_genome(grand.values, genome_id=grand.genome_id, parent_id=None,
                      generation=0, mutations=[], db_path=db)

        parent = grand.mutate(rng, genome_id="g-parent")
        record_genome(parent.values, genome_id=parent.genome_id,
                      parent_id=parent.parent_id, generation=parent.generation,
                      mutations=[m.__dict__ for m in parent.mutations], db_path=db)

        child = parent.mutate(rng, genome_id="g-child")
        record_genome(child.values, genome_id=child.genome_id,
                      parent_id=child.parent_id, generation=child.generation,
                      mutations=[m.__dict__ for m in child.mutations], db_path=db)

        loaded = get_genome("g-child", db_path=db)
        assert loaded is not None
        assert loaded["values"] == child.values
        assert loaded["parent_id"] == "g-parent"
        assert len(loaded["mutations"]) == len(child.mutations)

        chain = lineage_of("g-child", db_path=db)
        assert [node["genome_id"] for node in chain] == ["g-child", "g-parent", "g-grand"]
        assert all(node["generation"] == i for i, node in enumerate(reversed(chain)))

    def test_missing_genome_returns_none(self, tmp_path) -> None:
        assert get_genome("nope", db_path=tmp_path / "db.sqlite") is None


class TestGenomeToRiskConfig:
    def test_behavioral_genes_override_only_their_fields(self) -> None:
        base = RiskConfig()  # yaml defaults incl. kill-switch etc.
        rng = np.random.default_rng(12)
        genome = Genome.random(rng, genome_id="g-test")

        effective = risk_config_from_genome(base, genome)

        assert effective.stop_loss_pct == genome["stop_loss_pct"]
        assert effective.take_profit_pct == genome["take_profit_pct"]
        assert effective.cooldown_bars == int(genome["cooldown_bars"])
        assert effective.max_trades_per_day == int(genome["max_trades_per_day"])
        # protocol constants must survive evolution untouched
        assert effective.max_drawdown_pct == base.max_drawdown_pct
        assert effective.max_leverage == base.max_leverage
        assert effective.max_daily_loss_pct == base.max_daily_loss_pct

    def test_none_genome_returns_base(self) -> None:
        base = RiskConfig(stop_loss_pct=1.234)
        assert risk_config_from_genome(base, None) is base
