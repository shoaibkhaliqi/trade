"""One-shot: rescore the 1h cohort under the hardened passive-max baseline."""
import json
import sqlite3

from darwin.evolution.fitness import compute_fitness, preset

BASELINE = 0.0230  # always_short measured on the 1h score window

conn = sqlite3.connect("experiments/metadata.sqlite")
rows = conn.execute(
    "SELECT agent_id, metrics_json FROM agents WHERE agent_id LIKE 'a515000-%'"
).fetchall()
rescored = []
for agent_id, mj in rows:
    m = json.loads(mj)
    old = m.get("fitness")
    m["fitness"] = compute_fitness(m, preset("spec", baseline_return=BASELINE)).total
    conn.execute(
        "UPDATE agents SET metrics_json = ? WHERE agent_id = ?",
        (json.dumps(m, sort_keys=True), agent_id),
    )
    rescored.append((m["fitness"], old, agent_id, m["total_return"], m["n_trades"]))
conn.commit()
rescored.sort(reverse=True)
print(f"rescored {len(rescored)} agents under hardened baseline (naive-short {BASELINE:+.2%})")
header = f"{'agent':<18}{'ret':>8}{'trades':>8}{'old_fit':>9}{'new_fit':>9}"
print(header)
for fit, old, aid, ret, trades in rescored:
    print(f"{aid:<18}{ret:>8.2%}{trades:>8}{old:>+9.3f}{fit:>+9.3f}")
