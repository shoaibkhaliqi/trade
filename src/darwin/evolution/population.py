"""Population: N agents with distinct genomes/seeds, evaluated identically.

Fairness rules encoded here:
- every agent gets the same data slices, costs, risk baseline, timesteps and
  scoring window - the ONLY differences are its genome and its training seed
- every agent's genome and result row is persisted before/after evaluation,
  so a crashed run still leaves a complete roster behind
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from darwin.evolution.genome import Genome
from darwin.experiments.tracker import (
    record_agent,
    record_genome,
    update_agent_result,
)


@dataclass(frozen=True)
class AgentSpec:
    """One roster entry: identity + genome + training seed."""

    agent_id: str
    genome: Genome
    seed: int
    generation: int = 0


class Population:
    """Creates and tracks a generation-0 roster of agents."""

    def __init__(self, size: int, db_path: Any = "experiments/metadata.sqlite") -> None:
        if size < 2:
            msg = "population size must be >= 2 (selection needs rivals)"
            raise ValueError(msg)
        self.size = size
        self.db_path = db_path
        self.agents: list[AgentSpec] = []

    def initialize(self, master_seed: int) -> list[AgentSpec]:
        """Create the roster deterministically from one master seed."""
        rng = np.random.default_rng(master_seed)
        self.agents = []
        for i in range(self.size):
            genome_id = uuid.uuid4().hex[:10]
            genome = Genome.random(rng, genome_id=genome_id)
            record_genome(
                genome.values,
                genome_id=genome_id,
                parent_id=None,
                generation=0,
                mutations=[],
                db_path=self.db_path,
            )
            agent_id = f"a{master_seed:05d}-{i:03d}"
            seed = int(rng.integers(0, 2**31 - 1))
            record_agent(
                agent_id,
                genome_id=genome_id,
                seed=seed,
                generation=0,
                db_path=self.db_path,
            )
            self.agents.append(
                AgentSpec(agent_id=agent_id, genome=genome, seed=seed)
            )
        return self.agents

    def record_result(
        self,
        agent_id: str,
        metrics: dict,
        model_path: str,
        status: str = "evaluated",
    ) -> None:
        update_agent_result(
            agent_id, metrics=metrics, model_path=model_path,
            status=status, db_path=self.db_path,
        )
