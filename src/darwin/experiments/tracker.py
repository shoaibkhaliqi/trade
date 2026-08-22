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
"""


def record_experiment(kind: str, payload: dict, db_path: Path | str = DEFAULT_DB) -> str:
    """Persist one experiment record and return its unique id."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exp_id = uuid.uuid4().hex[:12]
    created = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(path) as conn:
        conn.execute(_SCHEMA)
        conn.execute(
            "INSERT INTO experiments (id, created_at, kind, payload) VALUES (?, ?, ?, ?)",
            (exp_id, created, kind, json.dumps(payload, sort_keys=True, default=str)),
        )
        conn.commit()
    return exp_id


def load_experiment(exp_id: str, db_path: Path | str = DEFAULT_DB) -> dict | None:
    """Return one experiment payload by id, or None."""
    with sqlite3.connect(Path(db_path)) as conn:
        conn.execute(_SCHEMA)
        row = conn.execute(
            "SELECT kind, payload FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
    if row is None:
        return None
    return {"id": exp_id, "kind": row[0], **json.loads(row[1])}
