"""Minimal SQLite experiment tracking.

Every training run records its identity (seed, data slices, config, results)
so any number can be traced back to exactly how it was produced. This is the
humble beginning of M7's full tracking system - deliberately simple, already
non-negotiable.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from darwin.evolution.survival import STATUS_DEAD

DEFAULT_DB = Path("experiments/metadata.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genomes (
    genome_id TEXT PRIMARY KEY,
    parent_id TEXT,
    generation INTEGER NOT NULL,
    values_json TEXT NOT NULL,
    mutations_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL,
    model_path TEXT,
    metrics_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deaths (
    death_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    fitness REAL,
    max_drawdown REAL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    generation INTEGER PRIMARY KEY,
    n_agents INTEGER NOT NULL,
    best_fitness REAL,
    median_fitness REAL,
    worst_fitness REAL,
    best_return REAL,
    mean_drawdown REAL,
    n_alive INTEGER,
    n_weak INTEGER,
    n_dead INTEGER,
    diversity_mean REAL,
    diversity_min REAL,
    n_immigrants INTEGER,
    created_at TEXT NOT NULL
);
"""


def record_genome(
    values: dict[str, float],
    *,
    genome_id: str,
    parent_id: str | None,
    generation: int,
    mutations: list[dict] | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> str:
    """Persist one genome with its full birth record; returns its id."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO genomes "
            "(genome_id, parent_id, generation, values_json, mutations_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                genome_id,
                parent_id,
                generation,
                json.dumps(values, sort_keys=True),
                json.dumps(mutations or [], sort_keys=True),
                created,
            ),
        )
        conn.commit()
    return genome_id


def get_genome(genome_id: str, db_path: Path | str = DEFAULT_DB) -> dict | None:
    """Load one genome row (values + mutation record + lineage fields)."""
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT genome_id, parent_id, generation, values_json, mutations_json "
            "FROM genomes WHERE genome_id = ?",
            (genome_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "genome_id": row[0],
        "parent_id": row[1],
        "generation": row[2],
        "values": json.loads(row[3]),
        "mutations": json.loads(row[4]),
    }


def lineage_of(genome_id: str, db_path: Path | str = DEFAULT_DB) -> list[dict]:
    """Walk parents from a genome back to its root ancestor (inclusive)."""
    chain: list[dict] = []
    current: str | None = genome_id
    seen: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        node = get_genome(current, db_path)
        if node is None:
            break
        chain.append(node)
        current = node["parent_id"]
    return chain


def record_agent(
    agent_id: str,
    *,
    genome_id: str,
    seed: int,
    generation: int = 0,
    status: str = "alive",
    db_path: Path | str = DEFAULT_DB,
) -> str:
    """Register an agent in the population roster."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO agents "
            "(agent_id, genome_id, seed, generation, status, model_path, metrics_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, NULL, ?) "
            "ON CONFLICT(agent_id) DO NOTHING",
            (agent_id, genome_id, seed, generation, status, created),
        )
        conn.commit()
    return agent_id


def update_agent_result(
    agent_id: str,
    *,
    metrics: dict,
    model_path: str,
    status: str = "evaluated",
    db_path: Path | str = DEFAULT_DB,
) -> None:
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "UPDATE agents SET metrics_json = ?, model_path = ?, status = ? "
            "WHERE agent_id = ?",
            (json.dumps(metrics, sort_keys=True, default=str), model_path, status, agent_id),
        )
        conn.commit()


def mark_agent_status(
    agent_id: str,
    status: str,
    *,
    reason: str | None = None,
    fitness: float | None = None,
    max_drawdown: float | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> None:
    """Set an agent's status; deaths additionally receive a certificate row.

    History is NEVER deleted: the agents row keeps its metrics; the deaths
    table only ADDS the cause-of-death record.
    """
    path = Path(db_path)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "UPDATE agents SET status = ? WHERE agent_id = ?",
            (status, agent_id),
        )
        if status == STATUS_DEAD and reason is not None:
            conn.execute(
                "INSERT INTO deaths (agent_id, reason, fitness, max_drawdown, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    agent_id,
                    reason,
                    fitness,
                    max_drawdown,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
        conn.commit()


def merge_agent_metrics(
    agent_id: str,
    fields: dict,
    db_path: Path | str = DEFAULT_DB,
) -> None:
    """Merge keys into an agent's stored metrics (e.g. backfilling fitness)."""
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT metrics_json FROM agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            msg = f"unknown agent: {agent_id}"
            raise ValueError(msg)
        metrics = json.loads(row[0]) if row[0] else {}
        metrics.update(fields)
        conn.execute(
            "UPDATE agents SET metrics_json = ? WHERE agent_id = ?",
            (json.dumps(metrics, sort_keys=True, default=str), agent_id),
        )
        conn.commit()


def record_generation_stats(
    generation: int,
    *,
    n_agents: int,
    best_fitness: float | None,
    median_fitness: float | None,
    worst_fitness: float | None,
    best_return: float | None,
    mean_drawdown: float | None,
    n_alive: int,
    n_weak: int,
    n_dead: int,
    diversity_mean: float | None,
    diversity_min: float | None,
    n_immigrants: int,
    db_path: Path | str = DEFAULT_DB,
) -> None:
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO generations "
            "(generation, n_agents, best_fitness, median_fitness, worst_fitness, "
            " best_return, mean_drawdown, n_alive, n_weak, n_dead, "
            " diversity_mean, diversity_min, n_immigrants, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                generation, n_agents, best_fitness, median_fitness, worst_fitness,
                best_return, mean_drawdown, n_alive, n_weak, n_dead,
                diversity_mean, diversity_min, n_immigrants,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def get_generations(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        rows = conn.execute(
            "SELECT generation, n_agents, best_fitness, median_fitness, "
            "worst_fitness, best_return, mean_drawdown, n_alive, n_weak, "
            "n_dead, diversity_mean, diversity_min, n_immigrants "
            "FROM generations ORDER BY generation"
        ).fetchall()
    keys = [
        "generation", "n_agents", "best_fitness", "median_fitness",
        "worst_fitness", "best_return", "mean_drawdown", "n_alive",
        "n_weak", "n_dead", "diversity_mean", "diversity_min", "n_immigrants",
    ]
    return [dict(zip(keys, row, strict=True)) for row in rows]


def get_children_of(
    parent_genome_id: str,
    db_path: Path | str = DEFAULT_DB,
) -> list[dict]:
    """Agents whose genome's parent is ``parent_genome_id`` (immediate children)."""
    query = (
        "SELECT a.agent_id, a.genome_id, a.seed, a.generation, a.status, "
        "a.model_path, a.metrics_json "
        "FROM agents a JOIN genomes g ON a.genome_id = g.genome_id "
        "WHERE g.parent_id = ? ORDER BY a.created_at, a.agent_id"
    )
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        rows = conn.execute(query, (parent_genome_id,)).fetchall()
    return [
        {
            "agent_id": r[0], "genome_id": r[1], "seed": r[2],
            "generation": r[3], "status": r[4], "model_path": r[5],
            "metrics": json.loads(r[6]) if r[6] else None,
        }
        for r in rows
    ]


def get_deaths(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    """All death certificates, oldest first."""
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        rows = conn.execute(
            "SELECT agent_id, reason, fitness, max_drawdown, recorded_at "
            "FROM deaths ORDER BY death_id"
        ).fetchall()
    return [
        {
            "agent_id": r[0],
            "reason": r[1],
            "fitness": r[2],
            "max_drawdown": r[3],
            "recorded_at": r[4],
        }
        for r in rows
    ]


def get_agents(db_path: Path | str = DEFAULT_DB, status: str | None = None) -> list[dict]:
    """All agent rows (optionally filtered), oldest first."""
    query = (
        "SELECT agent_id, genome_id, seed, generation, status, model_path, metrics_json "
        "FROM agents"
    )
    params: tuple = ()
    if status is not None:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY created_at, agent_id"
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "agent_id": r[0],
            "genome_id": r[1],
            "seed": r[2],
            "generation": r[3],
            "status": r[4],
            "model_path": r[5],
            "metrics": json.loads(r[6]) if r[6] else None,
        }
        for r in rows
    ]


def record_experiment(kind: str, payload: dict, db_path: Path | str = DEFAULT_DB) -> str:
    """Persist one experiment record and return its unique id."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exp_id = uuid.uuid4().hex[:12]
    created = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO experiments (id, created_at, kind, payload) VALUES (?, ?, ?, ?)",
            (exp_id, created, kind, json.dumps(payload, sort_keys=True, default=str)),
        )
        conn.commit()
    return exp_id


def load_experiment(exp_id: str, db_path: Path | str = DEFAULT_DB) -> dict | None:
    """Return one experiment payload by id, or None."""
    with sqlite3.connect(Path(db_path)) as conn:
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT kind, payload FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
    if row is None:
        return None
    return {"id": exp_id, "kind": row[0], **json.loads(row[1])}
