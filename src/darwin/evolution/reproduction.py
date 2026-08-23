"""Reproduction: fitness selects parents; children inherit mutated genomes.

Selection rules:
- DEAD agents are out of the gene pool (M11 status is the first filter).
- Eligible parents (alive/weak with a fitness score) are ranked; the top
  ``len(offspring_per_rank)`` breed, rank k producing offspring_per_rank[k]
  children. Weak parents may breed: mutation variance is their recovery path.
- Every child gets: fresh agent_id, its own training seed, generation+1, a
  mutated genome whose birth record (mutations + parent link) is persisted
  BEFORE the child ever trains.

Inheritance of knowledge (parent weights) is an explicit opt-in flag at the
training layer - default children start from fresh policies under their
mutated genomes, keeping generation-1 exploration honest.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from darwin.evolution.genome import Genome, MutationRecord
from darwin.evolution.population import AgentSpec, Population
from darwin.experiments.tracker import record_agent, record_genome

STATUS_DEAD = "dead"


@dataclass(frozen=True)
class ReproductionConfig:
    offspring_per_rank: tuple[int, ...] = (2, 1, 1)
    mutation_rate: float = 0.35
    mutation_intensity: float = 0.25
    # diversity policy (M13): decay exploration pressure over generations but
    # never below the floor; inject fresh random genomes every generation
    intensity_decay: float = 1.0      # 1.0 = constant intensity
    min_intensity: float = 0.05
    immigrants_per_generation: int = 1

    def __post_init__(self) -> None:
        if any(n < 0 for n in self.offspring_per_rank):
            msg = "offspring counts must be non-negative"
            raise ValueError(msg)
        if sum(self.offspring_per_rank) == 0:
            msg = "reproduction with zero total offspring does nothing"
            raise ValueError(msg)
        if not 0 <= self.mutation_rate <= 1:
            msg = "mutation_rate must be within [0, 1]"
            raise ValueError(msg)
        if self.mutation_intensity < 0:
            msg = "mutation_intensity must be >= 0"
            raise ValueError(msg)
        if not 0 < self.intensity_decay <= 1:
            msg = "intensity_decay must be within (0, 1]"
            raise ValueError(msg)
        if not 0 < self.min_intensity <= self.mutation_intensity:
            msg = "min_intensity must be within (0, mutation_intensity]"
            raise ValueError(msg)
        if self.immigrants_per_generation < 0:
            msg = "immigrants_per_generation must be >= 0"
            raise ValueError(msg)


def effective_intensity(cfg: ReproductionConfig, generation: int) -> float:
    """Mutation pressure for a generation: decays, floored, never zero."""
    decayed = cfg.mutation_intensity * (cfg.intensity_decay**generation)
    return max(cfg.min_intensity, decayed)


def select_parents(
    agents: list[dict],
    cfg: ReproductionConfig,
) -> list[tuple[dict, int]]:
    """Rank eligible agents by fitness; pair top ranks with offspring counts.

    ``agents`` rows are tracker dicts (agent_id, status, metrics, ...).
    """
    eligible = [
        a for a in agents
        if a["status"] in ("alive", "weak")
        and a.get("metrics")
        and a["metrics"].get("fitness") is not None
    ]
    eligible.sort(key=lambda a: a["metrics"]["fitness"], reverse=True)

    pairs: list[tuple[dict, int]] = []
    for rank, n_children in enumerate(cfg.offspring_per_rank):
        if rank >= len(eligible) or n_children == 0:
            continue
        pairs.append((eligible[rank], n_children))
    return pairs


def reproduce(
    population: Population,
    parents: list[tuple[dict, int]],
    rng: np.random.Generator,
    cfg: ReproductionConfig,
    *,
    generation: int,
    master_seed: int,
) -> list[AgentSpec]:
    """Create and persist all children (bred + immigrant); returns AgentSpecs."""
    children: list[AgentSpec] = []
    intensity = effective_intensity(cfg, generation)
    for parent_row, n_children in parents:
        parent_genome_row = _parent_genome(population, parent_row)
        parent = Genome(
            values=parent_genome_row["values"],
            genome_id=parent_row["genome_id"],
        )
        for _ in range(n_children):
            children.append(
                _make_child(
                    population, parent.mutate(
                        rng,
                        rate=cfg.mutation_rate,
                        intensity=intensity,
                        genome_id=uuid.uuid4().hex[:10],
                        generation=generation,
                    ),
                    rng, generation, master_seed, len(children),
                )
            )

    # fresh blood: random immigrants with no parent - the anti-inbreeding
    # workhorse; they reseed exploration wherever the family tree thinned
    for _ in range(cfg.immigrants_per_generation):
        genome_id = uuid.uuid4().hex[:10]
        immigrant = Genome.random(rng, genome_id=genome_id)
        # Genome.random has no generation field; rebuild with lineage metadata
        immigrant = Genome(
            values=immigrant.values,
            genome_id=genome_id,
            parent_id=None,
            generation=generation,
        )
        children.append(
            _make_child(population, immigrant, rng, generation, master_seed,
                        len(children))
        )
    return children


def _make_child(
    population: Population,
    genome: Genome,
    rng: np.random.Generator,
    generation: int,
    master_seed: int,
    index: int,
) -> AgentSpec:
    record_genome(
        genome.values,
        genome_id=genome.genome_id,
        parent_id=genome.parent_id,
        generation=generation,
        mutations=[m.__dict__ for m in genome.mutations],
        db_path=population.db_path,
    )
    agent_id = f"a{master_seed:05d}-g{generation:02d}-{index:03d}"
    seed = int(rng.integers(0, 2**31 - 1))
    record_agent(
        agent_id,
        genome_id=genome.genome_id,
        seed=seed,
        generation=generation,
        db_path=population.db_path,
    )
    return AgentSpec(agent_id=agent_id, genome=genome, seed=seed,
                     generation=generation)


def _parent_genome(population: Population, parent_row: dict) -> dict[str, Any]:
    from darwin.experiments.tracker import get_genome

    row = get_genome(parent_row["genome_id"], db_path=population.db_path)
    if row is None:
        msg = f"parent genome missing: {parent_row['genome_id']}"
        raise RuntimeError(msg)
    return row


def summarize_mutations(children: list[AgentSpec]) -> list[MutationRecord]:
    """Flat view of every mutation that happened in this birth wave."""
    return [m for c in children for m in c.genome.mutations]
