"""Diversity measurement tests - normalized distances, convergence detection."""

from __future__ import annotations

import pytest

from darwin.evolution.diversity import (
    format_diversity,
    genome_distance,
    population_diversity,
)
from darwin.evolution.genome import GENE_SPECS, Genome


def _mid() -> dict[str, float]:
    return {name: (spec.low + spec.high) / 2 for name, spec in GENE_SPECS.items()}


class TestGenomeDistance:
    def test_identical_genomes_are_zero(self) -> None:
        a = Genome.random(__import__("numpy").random.default_rng(1))
        assert genome_distance(a.values, a.values) == pytest.approx(0.0)

    def test_opposite_corners_are_one(self) -> None:
        lo = {name: spec.low for name, spec in GENE_SPECS.items()}
        hi = {name: spec.high for name, spec in GENE_SPECS.items()}
        assert genome_distance(lo, hi) == pytest.approx(1.0)

    def test_normalization_makes_genes_comparable(self) -> None:
        """A 10% move in ONE gene must cost the same regardless of gene scale."""
        base = _mid()
        moved_pos = {**base, "position_size_pct": base["position_size_pct"]
                     + 0.1 * GENE_SPECS["position_size_pct"].span}
        moved_gamma = {**base, "gamma": base["gamma"]
                       + 0.1 * GENE_SPECS["gamma"].span}

        d_pos = genome_distance(base, moved_pos)
        d_gamma = genome_distance(base, moved_gamma)
        assert d_pos == pytest.approx(d_gamma)
        assert d_pos == pytest.approx(0.1 / 8)  # one gene moved 10% of its span

    def test_symmetry(self) -> None:
        a = Genome.random(__import__("numpy").random.default_rng(2))
        b = Genome.random(__import__("numpy").random.default_rng(3))
        assert genome_distance(a.values, b.values) == pytest.approx(
            genome_distance(b.values, a.values)
        )


class TestPopulationDiversity:
    def test_identical_population_is_converged(self) -> None:
        g = _mid()
        report = population_diversity([g, g, g])
        assert report.n_unique == 1
        assert report.mean_pairwise == 0.0
        assert report.converging

    def test_spread_population_is_not_converged(self) -> None:
        rng = __import__("numpy").random.default_rng(5)
        genomes = [Genome.random(rng).values for _ in range(8)]
        report = population_diversity(genomes)
        assert report.n_unique == 8
        assert report.mean_pairwise > 0.3  # random genomes spread wide
        assert not report.converging

    def test_adding_random_genome_widens_coverage(self) -> None:
        """Coverage (max pairwise distance) is monotonic under additions;
        mean pairwise is NOT - a point near the centroid dilutes it."""
        rng = __import__("numpy").random.default_rng(6)
        base = [Genome.random(rng).values for _ in range(3)]
        before = population_diversity(base)
        after = population_diversity([*base, Genome.random(rng).values])

        assert after.max_pairwise >= before.max_pairwise
        assert after.n_genomes == before.n_genomes + 1

    def test_empty_population_rejected(self) -> None:
        with pytest.raises(ValueError):
            population_diversity([])

    def test_per_gene_std_identifies_converged_gene(self) -> None:
        rng = __import__("numpy").random.default_rng(7)
        genomes = [Genome.random(rng).values for _ in range(6)]
        same_gamma = {**genomes[0]}
        clones = [dict(g) for g in genomes]
        for c in clones:
            c["gamma"] = same_gamma["gamma"]
        report = population_diversity(clones)
        assert report.per_gene_std["gamma"] == pytest.approx(0.0, abs=1e-12)
        assert "gamma" in format_diversity(report)  # flagged as converged
