"""Population diversity: measuring whether evolution is still exploring.

Every gene is normalized by its span before comparison, so a 0.001 shift in
``gamma`` counts exactly as much as a 1.0 shift in ``position_size_pct`` -
raw units would let big-range genes fake diversity while small-range genes
silently converge.

Premature convergence shows up here FIRST: mean pairwise distance collapsing
toward zero while fitness plateaus means the population stopped searching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from darwin.evolution.genome import GENE_SPECS

CONVERGENCE_WARNING_THRESHOLD = 0.05


def genome_distance(a: Any, b: Any) -> float:
    """Mean per-gene |difference| / span across all genes. 0 = identical."""
    total = 0.0
    for name, spec in GENE_SPECS.items():
        total += abs(float(a[name]) - float(b[name])) / spec.span
    return total / len(GENE_SPECS)


@dataclass(frozen=True)
class DiversityReport:
    n_genomes: int
    n_unique: int
    mean_pairwise: float
    min_pairwise: float
    max_pairwise: float
    per_gene_std: dict[str, float]

    @property
    def converging(self) -> bool:
        return self.n_unique <= 1 or self.mean_pairwise < CONVERGENCE_WARNING_THRESHOLD


def population_diversity(genomes: Sequence[Any]) -> DiversityReport:
    """Pairwise-distance report over a set of genome value-mappings."""
    if not genomes:
        msg = "cannot measure diversity of an empty population"
        raise ValueError(msg)

    n = len(genomes)
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(genome_distance(genomes[i], genomes[j]))

    per_gene_std: dict[str, float] = {}
    for name, spec in GENE_SPECS.items():
        values = [float(g[name]) for g in genomes]
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        per_gene_std[name] = (var**0.5) / spec.span  # normalized std

    unique = {tuple(float(g[name]) for name in GENE_SPECS) for g in genomes}
    return DiversityReport(
        n_genomes=n,
        n_unique=len(unique),
        mean_pairwise=sum(distances) / len(distances) if distances else 0.0,
        min_pairwise=min(distances) if distances else 0.0,
        max_pairwise=max(distances) if distances else 0.0,
        per_gene_std=per_gene_std,
    )


def format_diversity(report: DiversityReport) -> str:
    lines = [
        f"genomes={report.n_genomes} unique={report.n_unique} "
        f"pairwise mean={report.mean_pairwise:.3f} "
        f"min={report.min_pairwise:.3f} max={report.max_pairwise:.3f}"
    ]
    if report.converging:
        lines.append(
            f"WARNING: mean pairwise distance < "
            f"{CONVERGENCE_WARNING_THRESHOLD} - population converging"
        )
    converged = sorted(
        (name for name, std in report.per_gene_std.items() if std < 0.02),
    )
    if converged:
        lines.append(f"nearly-converged genes: {', '.join(converged)}")
    return "\n".join(lines)
