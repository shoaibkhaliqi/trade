"""Agent genomes: bounded genes, controlled mutation, recorded lineage.

A genome is a flat map of named genes. Each gene is declared once in
GENE_SPECS with hard bounds - mutation may never leave them. Bounds encode
prior engineering knowledge (e.g. a 200% position size is not a strategy,
it is an accident), which is what makes this EVOLUTION rather than noise
search.

Every mutation event is captured in a MutationRecord (gene, old, new, sigma);
callers persist these alongside the child genome so any value in the
population can be traced back through its ancestors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeneSpec:
    """Declaration of one evolvable parameter."""

    name: str
    low: float
    high: float
    integer: bool = False     # genes that are counts
    log_scale: bool = False   # mutate multiplicatively in log space

    def __post_init__(self) -> None:
        if not self.low < self.high:
            msg = f"gene '{self.name}': low must be < high"
            raise ValueError(msg)
        if self.integer and self.log_scale:
            msg = f"gene '{self.name}': cannot be both integer and log_scale"
            raise ValueError(msg)

    @property
    def span(self) -> float:
        return self.high - self.low

    def clamp(self, value: float) -> float:
        v = min(max(value, self.low), self.high)
        if self.integer:
            v = float(round(v))
        return float(v)


GENE_SPECS: dict[str, GeneSpec] = {
    # --- behavioral genes: bound the risk layer -----------------------
    "position_size_pct": GeneSpec("position_size_pct", 5.0, 50.0),
    "stop_loss_pct": GeneSpec("stop_loss_pct", 0.5, 8.0),
    "take_profit_pct": GeneSpec("take_profit_pct", 0.75, 15.0),
    "cooldown_bars": GeneSpec("cooldown_bars", 0.0, 24.0, integer=True),
    "max_trades_per_day": GeneSpec("max_trades_per_day", 2.0, 80.0, integer=True),
    # --- learning genes: bound the trainer ----------------------------
    "learning_rate": GeneSpec("learning_rate", 1e-4, 3e-3, log_scale=True),
    "ent_coef": GeneSpec("ent_coef", 0.0, 0.03),
    "gamma": GeneSpec("gamma", 0.90, 0.999),
}


@dataclass(frozen=True)
class MutationRecord:
    gene: str
    old: float
    new: float
    sigma: float


@dataclass(frozen=True)
class Genome:
    values: dict[str, float]
    genome_id: str = ""
    parent_id: str | None = None
    generation: int = 0
    mutations: tuple[MutationRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        unknown = set(self.values) - set(GENE_SPECS)
        if unknown:
            msg = f"unknown genes: {sorted(unknown)}"
            raise ValueError(msg)
        missing = set(GENE_SPECS) - set(self.values)
        if missing:
            msg = f"missing genes: {sorted(missing)}"
            raise ValueError(msg)
        for name, value in self.values.items():
            spec = GENE_SPECS[name]
            if not spec.low <= value <= spec.high:
                msg = f"gene '{name}'={value} outside [{spec.low}, {spec.high}]"
                raise ValueError(msg)

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    # ------------------------------------------------------------------
    @classmethod
    def random(cls, rng: Any, genome_id: str = "") -> Genome:
        """Uniform sample inside every gene's bounds."""
        values = {}
        for name, spec in GENE_SPECS.items():
            raw = rng.uniform(spec.low, spec.high)
            if spec.log_scale:
                raw = math.exp(rng.uniform(math.log(spec.low), math.log(spec.high)))
            values[name] = spec.clamp(raw)
        return cls(values=values, genome_id=genome_id)

    def mutate(
        self,
        rng: Any,
        *,
        rate: float = 0.35,
        intensity: float = 0.25,
        genome_id: str = "",
        generation: int | None = None,
    ) -> Genome:
        """Gaussian per-gene mutation of THIS genome, clamped, fully recorded."""
        if not 0 <= rate <= 1:
            msg = "mutation rate must be within [0, 1]"
            raise ValueError(msg)
        if intensity < 0:
            msg = "mutation intensity must be >= 0"
            raise ValueError(msg)

        new_values = dict(self.values)
        records: list[MutationRecord] = []
        next_generation = self.generation + 1 if generation is None else generation
        for name, spec in GENE_SPECS.items():
            if rng.random() >= rate:
                continue
            old = new_values[name]
            if spec.log_scale:
                # mutate multiplicatively: sigma proportional to the LOG-range,
                # so a given intensity applies comparable relative pressure to
                # magnitudes of any size
                lo, hi = math.log(spec.low), math.log(spec.high)
                sigma = intensity * (hi - lo)
                clamped = min(max(math.log(old) + rng.normal(0.0, sigma), lo), hi)
                new_values[name] = spec.clamp(math.exp(clamped))
            else:
                sigma = intensity * spec.span
                new_values[name] = spec.clamp(old + rng.normal(0.0, sigma))
            records.append(
                MutationRecord(gene=name, old=old, new=new_values[name], sigma=sigma)
            )
        return Genome(
            values=new_values,
            genome_id=genome_id,
            parent_id=self.genome_id or None,
            generation=next_generation,
            mutations=tuple(records),
        )
