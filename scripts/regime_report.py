"""Per-regime performance of the top agents (regenerated on the score window).

Usage:
    .venv\\Scripts\\python.exe scripts\\regime_report.py --top 3
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from darwin.config import load_config  # noqa: F401 - parity with other CLIs
from darwin.environment.env import TradingEnv
from darwin.evaluation.regimes import (
    RegimeConfig,
    format_regime_table,
    regime_performance,
    regime_timeline,
)
from darwin.execution.risk import RiskConfig, RiskManager
from darwin.experiments.tracker import get_agents
from darwin.experiments.training import load_frames, sim_config_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--score-window", type=int, default=5_000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--db", default="experiments/metadata.sqlite")
    parser.add_argument("--trend-window", type=int, default=96)
    parser.add_argument("--vol-window", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, symbol, ohlcv, feats = load_frames(args.config, args.timeframe)
    n = len(ohlcv)
    val_end = int(n * (args.train_frac + 0.15))

    window = ohlcv.iloc[val_end : val_end + args.score_window].reset_index(drop=True)
    window_feats = feats.iloc[val_end : val_end + args.score_window].reset_index(drop=True)

    rcfg = RegimeConfig(trend_window=args.trend_window, vol_window=args.vol_window)
    timeline = regime_timeline(window, rcfg)

    post = timeline[timeline["combined"] != "warmup"]
    print(f"{symbol} {args.timeframe} | score window bars={len(window)}")
    print("\nwindow regime composition (post-warmup):")
    shares = post["combined"].value_counts(normalize=True).sort_index()
    for label, share in shares.items():
        print(f"  {label:<22} {share:>6.1%}")

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])

    # fidelity: reproduce each agent's ORIGINAL evaluation config - its genes
    # override sizing/risk exactly like Population.evaluate_agent does
    from darwin.evolution.genome import Genome
    from darwin.experiments.training import risk_config_from_genome

    agents = [
        a for a in get_agents(db_path=args.db)
        if a["metrics"] and a["metrics"].get("fitness") is not None
        and a["model_path"]
    ]
    agents.sort(key=lambda a: a["metrics"]["fitness"], reverse=True)
    agents = agents[: args.top]
    if not agents:
        print("\nno evaluated agents with saved models found")
        return 1

    for agent in agents:
        genome_row = None
        from darwin.experiments.tracker import get_genome

        genome_row = get_genome(agent["genome_id"], db_path=args.db)
        genome = None
        if genome_row is not None:
            genome = Genome(values=genome_row["values"],
                            genome_id=agent["genome_id"])
        eff_risk = risk_config_from_genome(risk_cfg, genome)
        eff_sim = sim_cfg
        if genome is not None:
            from darwin.environment.simulator import SimulatorConfig

            eff_sim = SimulatorConfig(
                initial_capital=sim_cfg.initial_capital,
                taker_fee_pct=sim_cfg.taker_fee_pct,
                slippage_pct=sim_cfg.slippage_pct,
                position_size_pct=min(sim_cfg.position_size_pct,
                                      genome["position_size_pct"]),
                qty_decimals=sim_cfg.qty_decimals,
                close_at_end=sim_cfg.close_at_end,
            )

        model = PPO.load(agent["model_path"], device="cpu")
        env = TradingEnv(window, window_feats, config=eff_sim,
                         risk=RiskManager(eff_risk))
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        assert env.last_result is not None
        equity = env.last_result.equity_curve["equity"].reset_index(drop=True)

        perf = regime_performance(equity, timeline)
        m = agent["metrics"]
        print(f"\n=== {agent['agent_id']} (gen {agent['generation']}, "
              f"fitness {m['fitness']:+.3f}, window return "
              f"{equity.iloc[-1] / equity.iloc[0] - 1:+.2%}) ===")
        print(format_regime_table(perf))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
