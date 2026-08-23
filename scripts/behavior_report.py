"""Behavioral fingerprint report for the stored roster.

Usage:
    .venv\\Scripts\\python.exe scripts\\behavior_report.py                # all agents
    .venv\\Scripts\\python.exe scripts\\behavior_report.py --prefix a424242
"""

from __future__ import annotations

import argparse

from darwin.evolution.behavior import behavior_diversity, format_behavior_report
from darwin.experiments.tracker import get_agents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="experiments/metadata.sqlite")
    parser.add_argument("--prefix", default=None,
                        help="only agents whose id starts with this prefix")
    args = parser.parse_args()

    agents = [
        a for a in get_agents(db_path=args.db)
        if a["metrics"] and "behavior" in (a["metrics"] or {})
        and (args.prefix is None or a["agent_id"].startswith(args.prefix))
    ]
    if not agents:
        print("no agents with behavior fingerprints found")
        return 1

    print(f"{'agent':<20}{'long%':>7}{'short%':>8}{'flat%':>7}{'trades':>8}"
          f"{'L/S/C/H actions':>18}{'hash':>14}")
    behaviors = []
    for a in agents:
        b = a["metrics"]["behavior"]
        acts = b["actions"]
        behaviors.append(b)
        print(f"{a['agent_id']:<20}{b['pos_long_frac']:>7.1%}"
              f"{b['pos_short_frac']:>8.1%}{b['pos_flat_frac']:>7.1%}"
              f"{int(a['metrics']['n_trades']):>8d}"
              f"{acts['long']:>5d}{acts['short']:>4d}{acts['close']:>4d}{acts['hold']:>5d}"
              f"{b['behavior_hash']:>14}")

    print()
    print(format_behavior_report(behavior_diversity(behaviors)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
