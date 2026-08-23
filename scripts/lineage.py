"""Lineage CLI: dossiers, family trees, gene provenance.

Usage:
    .venv\\Scripts\\python.exe scripts\\lineage.py --agent a02026-g01-000
    .venv\\Scripts\\python.exe scripts\\lineage.py --tree a02026-000
    .venv\\Scripts\\python.exe scripts\\lineage.py --tree a02026-000 --plot
    .venv\\Scripts\\python.exe scripts\\lineage.py --agent ID --gene stop_loss_pct
    .venv\\Scripts\\python.exe scripts\\lineage.py --hof
"""

from __future__ import annotations

import argparse

from darwin.evolution.lineage import (
    dossier,
    gene_origin,
    hall_of_fame,
)
from darwin.visualization.lineage import plot_subtree, render_tree

DB = "experiments/metadata.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default=None, help="agent id for dossier")
    parser.add_argument("--tree", default=None, help="ASCII tree rooted at agent")
    parser.add_argument("--plot", action="store_true",
                        help="with --tree: also save a PNG subtree plot")
    parser.add_argument("--gene", default=None,
                        help="with --agent: trace where a gene came from")
    parser.add_argument("--hof", action="store_true", help="hall of fame")
    return parser.parse_args()


def print_dossier(agent_id: str, gene: str | None) -> None:
    d = dossier(agent_id, db_path=DB)
    print(f"agent      : {d.agent_id} (generation {d.generation}, {d.status})")
    if d.death_reason:
        print(f"death      : {d.death_reason}")
    if d.metrics:
        m = d.metrics
        print(f"performance: fitness={m.get('fitness', float('nan')):+.3f} "
              f"ret={m.get('total_return', 0):+.2%} "
              f"dd={m.get('max_drawdown', 0):.2%} trades={m.get('n_trades', 0)}")
    print("\nancestry (root -> parent):")
    for a in d.ancestors:
        print(f"  g{a['generation']} {a['agent_id']} [{a['status']}]")
    if not d.ancestors:
        print("  (founder)")

    print(f"\nmutation path ({len(d.mutation_path)} events, root -> this agent):")
    for m in d.mutation_path:
        print(f"  gen{m['generation']} {m['gene']:<20} "
              f"{m['old']:>10.4f} -> {m['new']:>10.4f}")

    if gene is not None:
        origin = gene_origin(agent_id, gene, db_path=DB)
        if origin:
            print(f"\ngene origin [{gene}]: set in generation {origin['generation']} "
                  f"({origin['old']:.4f} -> {origin['new']:.4f})")
        else:
            print(f"\ngene origin [{gene}]: never mutated - founder value")

    print(f"\nchildren: {len(d.children)} | total descendants: {d.descendants_count}")
    if d.best_descendant:
        bd = d.best_descendant
        print(f"best descendant: {bd['agent_id']} "
              f"fitness={bd['metrics']['fitness']:+.3f} "
              f"ret={bd['metrics']['total_return']:+.2%}")


def main() -> int:
    args = parse_args()
    if args.hof:
        print("=== HALL OF FAME ===")
        for rank, row in enumerate(hall_of_fame(db_path=DB), start=1):
            m = row["metrics"]
            print(f"{rank}. {row['agent_id']:<18} gen{row['generation']} "
                  f"[{row['status']:<5}] fitness={m['fitness']:+.3f} "
                  f"ret={m['total_return']:+.2%}")
        return 0

    if args.tree:
        print(render_tree(args.tree, db_path=DB))
        if args.plot:
            path = plot_subtree(
                args.tree,
                f"experiments/figures/lineage_{args.tree}.png",
                db_path=DB,
            )
            print(f"\ntree plot -> {path}")
        return 0

    if args.agent:
        print_dossier(args.agent, args.gene)
        return 0

    print("nothing to do - pass --agent, --tree, or --hof "
          "(see --help)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
