"""Print the diversity report for the stored agent roster."""

from __future__ import annotations

import argparse

from darwin.evolution.diversity import format_diversity, population_diversity
from darwin.experiments.tracker import get_agents, get_genome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="experiments/metadata.sqlite")
    args = parser.parse_args()

    genomes = []
    for row in get_agents(db_path=args.db):
        g = get_genome(row["genome_id"], db_path=args.db)
        if g is not None:
            genomes.append(g["values"])

    if not genomes:
        print("no genomes found - initialize a population first")
        return 1

    print(format_diversity(population_diversity(genomes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
