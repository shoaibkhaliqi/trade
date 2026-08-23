"""Print the experiment history recorded in experiments metadata DB."""

from __future__ import annotations

import argparse
import json
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="experiments/metadata.sqlite")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        rows = conn.execute(
            "SELECT id, created_at, kind, payload FROM experiments "
            "ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()

    header = (
        f"{'id':<13}{'created':<21}{'kind':<12}{'symbol':<10}{'tf':<5}"
        f"{'seed':>6}{'steps':>8}{'test_ret':>10}{'sharpe':>9}"
    )
    print(header)
    for exp_id, created, kind, payload_json in rows:
        p = json.loads(payload_json)
        metrics = p.get("test_metrics", {})
        ret = metrics.get("total_return")
        sharpe = metrics.get("sharpe")
        if "returns" in p:  # aggregate rows (e.g. seed sweeps)
            ret = sum(p["returns"]) / len(p["returns"])
            sharpe = sum(p["sharpes"]) / len(p["sharpes"])
        print(
            f"{exp_id:<13}{created:<21}{kind:<12}"
            f"{str(p.get('symbol', '')):<10}{str(p.get('timeframe', '')):<5}"
            f"{str(p.get('seed', '')):>6}{str(p.get('timesteps', '')):>8}"
            f"{ret:>10.2%}"
            f"{sharpe:>9.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
