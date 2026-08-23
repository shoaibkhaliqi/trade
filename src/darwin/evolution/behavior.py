"""Behavioral diversity: what agents DO, not what their genomes say.

M16 exposed the monoculture: distinct genomes, identical always-LONG behavior.
Genome distance (M13) measures recipes; this module measures the dish.

An agent's behavioral signature is its per-bar POSITION STATE sequence
(flat / long / short) - LONG-while-long is behaviorally a HOLD, so raw
action counts lie. Signatures give us:
- exact clone detection (md5 of the state sequence)
- pairwise distance via position-fraction vectors
- population-level reports with a convergence tripwire that actually works
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

BEHAVIOR_CONVERGENCE_THRESHOLD = 0.05


def summarize_behavior(
    actions: Sequence[int],
    position_qty: Sequence[float],
) -> dict[str, Any]:
    """Fingerprint one deterministic episode.

    ``actions``: Discrete(4) indices per decision (0=hold 1=long 2=short 3=close)
    ``position_qty``: signed position after each candle (same length).
    """
    if len(actions) != len(position_qty):
        msg = "actions and positions must be aligned"
        raise ValueError(msg)

    n = len(actions)
    counts = {name: 0 for name in ("hold", "long", "short", "close")}
    for a in actions:
        counts[["hold", "long", "short", "close"][int(a)]] += 1

    states = np.sign(np.asarray(position_qty, dtype="float64"))
    n_flat = int((states == 0).sum())
    n_long = int((states > 0).sum())
    n_short = int((states < 0).sum())
    signature = hashlib.md5(states.tobytes()).hexdigest()[:12]

    return {
        "actions": counts,
        "pos_flat_frac": n_flat / n if n else 0.0,
        "pos_long_frac": n_long / n if n else 0.0,
        "pos_short_frac": n_short / n if n else 0.0,
        "behavior_hash": signature,
    }


def behavior_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """0..1 distance between two behavior summaries.

    Identical position-state sequences are distance 0 regardless of raw
    action counts (two agents holding long are the same strategy even if
    one re-submits LONG every bar).
    """
    if a["behavior_hash"] == b["behavior_hash"]:
        return 0.0
    return 0.5 * (
        abs(a["pos_long_frac"] - b["pos_long_frac"])
        + abs(a["pos_short_frac"] - b["pos_short_frac"])
        + abs(a["pos_flat_frac"] - b["pos_flat_frac"])
    )


@dataclass(frozen=True)
class BehaviorReport:
    n_agents: int
    n_unique_behaviors: int
    mean_pairwise: float
    min_pairwise: float
    max_pairwise: float
    mean_long_frac: float
    std_long_frac: float

    @property
    def monoculture(self) -> bool:
        return self.n_unique_behaviors == 1 or self.mean_pairwise < (
            BEHAVIOR_CONVERGENCE_THRESHOLD
        )


def behavior_diversity(behaviors: Sequence[dict[str, Any]]) -> BehaviorReport:
    if not behaviors:
        msg = "cannot measure behavior of an empty population"
        raise ValueError(msg)

    distances = [
        behavior_distance(behaviors[i], behaviors[j])
        for i in range(len(behaviors))
        for j in range(i + 1, len(behaviors))
    ]
    longs = [float(b["pos_long_frac"]) for b in behaviors]
    unique = {b["behavior_hash"] for b in behaviors}
    return BehaviorReport(
        n_agents=len(behaviors),
        n_unique_behaviors=len(unique),
        mean_pairwise=sum(distances) / len(distances) if distances else 0.0,
        min_pairwise=min(distances) if distances else 0.0,
        max_pairwise=max(distances) if distances else 0.0,
        mean_long_frac=sum(longs) / len(longs),
        std_long_frac=float(np.std(longs)),
    )


def format_behavior_report(report: BehaviorReport) -> str:
    lines = [
        f"agents={report.n_agents} unique_behaviors={report.n_unique_behaviors} "
        f"pairwise mean={report.mean_pairwise:.3f} "
        f"min={report.min_pairwise:.3f} max={report.max_pairwise:.3f}"
    ]
    if report.monoculture:
        lines.append("WARNING: behavioral monoculture detected")
    return "\n".join(lines)
