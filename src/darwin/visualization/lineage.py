"""Lineage visualization: ASCII family trees and PNG subtree plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darwin.experiments.tracker import get_agents, get_children_of, get_genome

STATUS_SYMBOL = {"alive": "+", "weak": "~", "dead": "x", "evaluated": "+"}


def _agent_map(db_path: Any) -> dict[str, dict]:
    return {r["agent_id"]: r for r in get_agents(db_path=db_path)}


def render_tree(
    root_agent_id: str,
    db_path: Any = "experiments/metadata.sqlite",
) -> str:
    """ASCII family tree rooted at (and including) ``root_agent_id``."""
    agents = _agent_map(db_path)
    if root_agent_id not in agents:
        msg = f"unknown agent: {root_agent_id}"
        raise ValueError(msg)

    lines: list[str] = []

    def render(agent_id: str, prefix: str, child_prefix: str) -> None:
        agent = agents[agent_id]
        genome = get_genome(agent["genome_id"], db_path=db_path)
        metrics = agent["metrics"] or {}
        fit = metrics.get("fitness")
        fit_txt = f"{fit:+.3f}" if fit is not None else "  n/a"
        sym = STATUS_SYMBOL.get(agent["status"], "?")
        n_mut = len(genome["mutations"]) if genome else 0
        lines.append(
            f"{prefix}{sym} {agent_id} g{agent['generation']} "
            f"fit={fit_txt} mut={n_mut} [{agent['status']}]"
        )
        kids = get_children_of(agent["genome_id"], db_path=db_path)
        for i, child in enumerate(kids):
            last = i == len(kids) - 1
            render(child["agent_id"],
                   child_prefix + ("└─ " if last else "├─ "),
                   child_prefix + ("   " if last else "│  "))

    render(root_agent_id, "", "")
    return "\n".join(lines)


def plot_subtree(
    root_agent_id: str,
    out_path: str | Path,
    db_path: Any = "experiments/metadata.sqlite",
) -> Path:
    """PNG of the descendant subtree; node color = status, y = generation."""
    agents = _agent_map(db_path)
    if root_agent_id not in agents:
        msg = f"unknown agent: {root_agent_id}"
        raise ValueError(msg)

    nodes: dict[str, dict] = {root_agent_id: agents[root_agent_id]}
    edges: list[tuple[str, str]] = []

    def collect(agent_id: str) -> None:
        for child in get_children_of(agents[agent_id]["genome_id"], db_path=db_path):
            if child["agent_id"] not in nodes:
                nodes[child["agent_id"]] = child
                edges.append((agent_id, child["agent_id"]))
                collect(child["agent_id"])

    collect(root_agent_id)

    # assign x by DFS order, y by generation (inverted so root on top)
    order: dict[str, int] = {}

    def assign_x(agent_id: str, counter: list[int]) -> None:
        kids = [c for a, c in edges if a == agent_id]
        for k in kids:
            assign_x(k, counter)
        order[agent_id] = counter[0]
        counter[0] += 1

    assign_x(root_agent_id, [0])

    color_map = {"alive": "#2ca02c", "weak": "#ff7f0e", "dead": "#d62728",
                 "evaluated": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(12, max(4, 1.1 * len(nodes))))
    for parent_id, child_id in edges:
        ax.plot(
            [order[parent_id], order[child_id]],
            [agents[parent_id]["generation"], agents[child_id]["generation"]],
            color="gray", lw=0.8, zorder=1,
        )
    for agent_id, node in nodes.items():
        fit = node["metrics"]["fitness"] if node["metrics"] and node["metrics"].get("fitness") is not None else None
        label = f"{agent_id[-8:]}" + (f"\n{fit:+.2f}" if fit is not None else "")
        ax.scatter(
            order[agent_id], node["generation"],
            s=600, zorder=2,
            color=color_map.get(node["status"], "#888888"),
            edgecolors="black", linewidths=0.6,
        )
        ax.annotate(label, (order[agent_id], node["generation"]),
                    fontsize=7, ha="center", va="center", zorder=3)
    ax.set_xlabel("subtree order")
    ax.set_ylabel("generation")
    ax.set_title(f"Lineage subtree of {root_agent_id}")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
