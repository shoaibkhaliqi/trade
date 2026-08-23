"""Lineage queries: ancestry, descendants, mutation provenance, dossiers.

Genealogy turns evolution from a scoreboard into a story: which ancestor
contributed the champion's genes, where branches thrived or died, and which
mutations actually mattered. All queries are read-only over the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.experiments.tracker import (
    get_agents,
    get_children_of,
    get_deaths,
    get_genome,
    lineage_of,
)


@dataclass(frozen=True)
class LineageDossier:
    agent_id: str
    genome_id: str
    generation: int
    status: str
    metrics: dict[str, Any] | None
    death_reason: str | None
    ancestors: list[dict]                 # root -> immediate parent
    mutation_path: list[dict]             # ordered root -> this agent
    children: list[dict]                  # immediate children (agent rows)
    descendants_count: int
    best_descendant: dict | None = None   # highest-fitness descendant, if any
    gene_origins: dict[str, dict] = field(default_factory=dict)


def children_of(agent_id: str, db_path: Any = "experiments/metadata.sqlite") -> list[dict]:
    agent = _agent_row(agent_id, db_path)
    if agent is None:
        msg = f"unknown agent: {agent_id}"
        raise ValueError(msg)
    return get_children_of(agent["genome_id"], db_path=db_path)


def descendants(
    agent_id: str,
    db_path: Any = "experiments/metadata.sqlite",
    max_depth: int = 20,
) -> list[dict]:
    """All agents below this one (immediate children, grandchildren, ...)."""
    agent = _agent_row(agent_id, db_path)
    if agent is None:
        msg = f"unknown agent: {agent_id}"
        raise ValueError(msg)

    found: list[dict] = []
    frontier = [agent["genome_id"]]
    seen: set[str] = set()
    for _depth in range(max_depth):
        next_frontier: list[str] = []
        for genome_id in frontier:
            if genome_id in seen:
                continue
            seen.add(genome_id)
            for child in get_children_of(genome_id, db_path=db_path):
                found.append(child)
                next_frontier.append(child["genome_id"])
        if not next_frontier:
            break
        frontier = next_frontier
    return found


def mutation_path(
    agent_id: str,
    db_path: Any = "experiments/metadata.sqlite",
) -> list[dict]:
    """Every mutation from the root ancestor down to this agent, in order."""
    chain = lineage_of(_genome_id_of(agent_id, db_path), db_path=db_path)  # agent genome -> ... -> root
    path: list[dict] = []
    for node in reversed(chain):  # root -> ... -> agent
        for m in node["mutations"]:
            path.append({
                "generation": node["generation"],
                "genome_id": node["genome_id"],
                **m,
            })
    return path


def gene_origin(
    agent_id: str,
    gene: str,
    db_path: Any = "experiments/metadata.sqlite",
) -> dict | None:
    """The most recent ancestor event that SET ``gene`` to its current value.

    Returns {generation, genome_id, old, new} or None if the gene was never
    mutated along this lineage (inherited unchanged from the founder).
    """
    chain = lineage_of(_genome_id_of(agent_id, db_path), db_path=db_path)  # this genome first
    for node in chain:  # walk upward: the FIRST hit is the most recent change
        for m in node["mutations"]:
            if m["gene"] == gene:
                return {
                    "generation": node["generation"],
                    "genome_id": node["genome_id"],
                    "old": m["old"],
                    "new": m["new"],
                }
    return None


def dossier(
    agent_id: str,
    db_path: Any = "experiments/metadata.sqlite",
) -> LineageDossier:
    """Everything worth knowing about one agent's family story."""
    agent = _agent_row(agent_id, db_path)
    if agent is None:
        msg = f"unknown agent: {agent_id}"
        raise ValueError(msg)

    chain = lineage_of(_genome_id_of(agent_id, db_path), db_path=db_path)  # agent genome -> ... -> root
    # ancestors as AGENT rows (root -> immediate parent); lineage walks
    # genome space, but humans think in agent names
    agents_by_genome = {r["genome_id"]: r for r in get_agents(db_path=db_path)}
    ancestors = [
        agents_by_genome[n["genome_id"]]
        for n in reversed(chain[1:])
        if n["genome_id"] in agents_by_genome
    ]

    kids = get_children_of(agent["genome_id"], db_path=db_path)
    desc = descendants(agent_id, db_path=db_path)
    scored_desc = [d for d in desc if d["metrics"] and d["metrics"].get("fitness") is not None]
    best_desc = (
        max(scored_desc, key=lambda d: d["metrics"]["fitness"])
        if scored_desc else None
    )

    deaths = {d["agent_id"]: d["reason"] for d in get_deaths(db_path=db_path)}

    origins = {
        gene: gene_origin(agent_id, gene, db_path=db_path)
        for gene in agent_metrics_genes(agent)
    }

    return LineageDossier(
        agent_id=agent["agent_id"],
        genome_id=agent["genome_id"],
        generation=agent["generation"],
        status=agent["status"],
        metrics=agent["metrics"],
        death_reason=deaths.get(agent_id),
        ancestors=ancestors,
        mutation_path=mutation_path(agent_id, db_path=db_path),
        children=kids,
        descendants_count=len(desc),
        best_descendant=best_desc,
        gene_origins=origins,
    )


def agent_metrics_genes(agent: dict) -> list[str]:
    """Gene names for origin tracing - from the agent's genome row."""
    genome = get_genome(agent["genome_id"])
    return list(genome["values"]) if genome else []


def hall_of_fame(limit: int = 5, db_path: Any = "experiments/metadata.sqlite") -> list[dict]:
    """Top evaluated agents across all generations."""
    scored = [
        r for r in get_agents(db_path=db_path)
        if r["metrics"] and r["metrics"].get("fitness") is not None
    ]
    scored.sort(key=lambda r: r["metrics"]["fitness"], reverse=True)
    return scored[:limit]


def _agent_row(agent_id: str, db_path: Any) -> dict | None:
    for row in get_agents(db_path=db_path):
        if row["agent_id"] == agent_id:
            return row
    return None


def _genome_id_of(agent_id: str, db_path: Any) -> str:
    """Resolve an agent id to its genome id (lineage walks genome space)."""
    row = _agent_row(agent_id, db_path)
    if row is None:
        msg = f"unknown agent: {agent_id}"
        raise ValueError(msg)
    return row["genome_id"]
