"""Evolution history charts (fitness curves + diversity track)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_evolution(generations: list[dict[str, Any]], out_prefix: str | Path) -> list[Path]:
    """Two charts: fitness best/median/worst, and diversity track."""
    if not generations:
        msg = "no generation records to plot"
        raise ValueError(msg)

    gens = [g["generation"] for g in generations]
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color in (
        ("best_fitness", "best", "#2ca02c"),
        ("median_fitness", "median", "#1f77b4"),
        ("worst_fitness", "worst", "#d62728"),
    ):
        ys = [g[key] for g in generations]
        if any(y is not None for y in ys):
            ax.plot(gens, ys, marker="o", ms=3, lw=1.2, label=label, color=color)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.6)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (spec compass)")
    ax.set_title("Fitness across generations")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = Path(f"{out_prefix}_fitness.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(10, 5))
    mean = [g["diversity_mean"] for g in generations]
    low = [g["diversity_min"] for g in generations]
    if any(v is not None for v in mean):
        ax.plot(gens, mean, marker="o", ms=3, lw=1.2, color="#1f77b4",
                label="mean pairwise")
        ax.plot(gens, low, marker=".", ms=3, lw=0.9, color="#ff7f0e",
                label="min pairwise")
        ax.axhline(0.05, color="red", lw=0.8, ls="--", alpha=0.7,
                   label="convergence warning")
    ax.set_xlabel("generation")
    ax.set_ylabel("genome distance")
    ax.set_title("Population diversity across generations")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p2 = Path(f"{out_prefix}_diversity.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)

    return paths
