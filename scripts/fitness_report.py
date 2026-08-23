"""Re-rank the stored population under different fitness compasses.

No retraining: reads each agent's metrics from the DB, computes the buy&hold
baseline on the identical scoring window, and prints one leaderboard per
fitness preset - demonstrating that the fitness function IS the objective.
"""

from __future__ import annotations

import argparse

from darwin.agents import BuyAndHoldStrategy
from darwin.environment.simulator import TradingSimulator
from darwin.evolution.fitness import compute_fitness, preset
from darwin.experiments.tracker import get_agents
from darwin.experiments.training import (
    load_frames,
    sim_config_from_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--score-window", type=int, default=5_000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--db", default="experiments/metadata.sqlite")
    parser.add_argument("--apply", action="store_true",
                        help="persist survival verdicts to the roster")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, symbol, ohlcv, _feats = load_frames(args.config, args.timeframe)
    n = len(ohlcv)
    val_end = int(n * (args.train_frac + 0.15))
    window = ohlcv.iloc[val_end : val_end + args.score_window].reset_index(drop=True)

    sim_cfg = sim_config_from_yaml(cfg)
    bh = TradingSimulator(sim_cfg).run(
        window, BuyAndHoldStrategy().generate_actions(window)
    )
    baseline_return = float(bh.equity_curve["equity"].iloc[-1]
                            / bh.equity_curve["equity"].iloc[0] - 1.0)
    print(f"{symbol} {args.timeframe} | score window rows={len(window)} | "
          f"buy&hold baseline = {baseline_return:+.2%}\n")

    agents = [a for a in get_agents(db_path=args.db) if a["metrics"]]
    if not agents:
        print("no evaluated agents found - run scripts/run_population.py first")
        return 1

    for preset_name in ("spec", "pure_return", "risk_parity", "conservative"):
        fcfg = preset(preset_name, baseline_return=baseline_return)
        scored = []
        for a in agents:
            m = dict(a["metrics"])
            m.setdefault("initial_capital_proxy", 1000.0)
            b = compute_fitness(m, fcfg)
            scored.append((b.total, a["agent_id"], m))
        scored.sort(reverse=True, key=lambda t: t[0])

        print(f"=== fitness: {preset_name} ===")
        print(f"{'rank':<5}{'agent':<12}{'return':>9}{'trades':>8}{'dd':>9}{'fitness':>9}")
        for rank, (fit, agent_id, m) in enumerate(scored, start=1):
            print(f"{rank:<5}{agent_id[-8:]:<12}"
                  f"{m['total_return']:>9.2%}{int(m['n_trades']):>8d}"
                  f"{m['max_drawdown']:>9.2%}{fit:>+9.3f}")
        print()

    # ------------------------------------------------------------------
    # survival audit under the spec compass (+ default survival config)
    # ------------------------------------------------------------------
    from darwin.evolution.survival import SurvivalConfig, evaluate_survival

    spec_cfg = preset("spec", baseline_return=baseline_return)
    surv_cfg = SurvivalConfig()
    print("=== SURVIVAL AUDIT (spec compass, default thresholds) ===")
    for a in agents:
        m = dict(a["metrics"])
        m.setdefault("initial_capital_proxy", 1000.0)
        fit = compute_fitness(m, spec_cfg).total
        verdict = evaluate_survival(m, fit, surv_cfg)
        reasons = "; ".join(verdict.reasons) if verdict.reasons else "-"
        print(f"{a['agent_id'][-8:]:<12} {verdict.status:>5}  {reasons}")
    if args.apply:
        from darwin.experiments.tracker import mark_agent_status, merge_agent_metrics

        for a in agents:
            m = dict(a["metrics"])
            m.setdefault("initial_capital_proxy", 1000.0)
            fit = compute_fitness(m, spec_cfg).total
            verdict = evaluate_survival(m, fit, surv_cfg)
            merge_agent_metrics(a["agent_id"], {"fitness": fit}, db_path=args.db)
            mark_agent_status(
                a["agent_id"], verdict.status,
                reason="; ".join(verdict.reasons) if verdict.reasons else None,
                fitness=fit, max_drawdown=m["max_drawdown"], db_path=args.db,
            )
        print("\nverdicts persisted to the roster (--apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
