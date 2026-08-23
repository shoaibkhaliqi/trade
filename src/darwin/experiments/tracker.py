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
