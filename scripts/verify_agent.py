"""Out-of-sample verification for a saved agent.

Evaluates the artifact on slices it was NOT selected on:
  score : the selection window itself (in-sample reference)
  tail  : test rows AFTER the score window (truly untouched)
  full  : the whole test slice
  all   : entire post-warmup history of the symbol (cross-venue mode)

With --symbol XAUUSDT the candidate faces a different venue trading the same
underlying - the strictest generalization test we have.

Usage:
    .venv\\Scripts\\python.exe scripts\\verify_agent.py --agent a515000-001 --timeframe 1h --slice tail
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from darwin.data.storage import DataStorage
from darwin.environment.env import TradingEnv
from darwin.environment.simulator import SimulatorConfig
from darwin.evaluation.metrics import MetricsReport, format_header, format_row
from darwin.evaluation.regimes import (
    RegimeConfig,
    format_regime_table,
    regime_performance,
    regime_timeline,
)
from darwin.execution.risk import RiskConfig, RiskManager
from darwin.experiments.tracker import get_agents, get_genome
from darwin.experiments.training import load_frames, sim_config_from_yaml
from darwin.features.engine import FeatureEngine
from darwin.features.schema import ALL_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--slice", choices=["score", "tail", "full", "all"],
                        default="tail")
    parser.add_argument("--symbol", default=None,
                        help="cross-venue override (features auto-built)")
    parser.add_argument("--score-window", type=int, default=5_000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--db", default="experiments/metadata.sqlite")
    parser.add_argument("--regimes", action="store_true")
    return parser.parse_args()


def load_symbol_frames(symbol: str, timeframe: str, cfg_name: str):
    cfg, _default_symbol, ohlcv, feats = load_frames(cfg_name, timeframe)
    if symbol is None or symbol == _default_symbol:
        return cfg, ohlcv, feats
    src = DataStorage(cfg["data"]["processed_dir"])
    ohlcv = src.load(symbol, timeframe)
    dst = DataStorage(cfg["data"]["features_dir"])
    if not dst.path_for(symbol, timeframe).exists():
        built = FeatureEngine().build_feature_matrix(ohlcv)
        dst.save(built, symbol, timeframe,
                 metadata={"kind": "features", "note": f"built for verify_agent {symbol}"},
                 required_columns=["timestamp", *ALL_FEATURES])
        feats = built
    else:
        feats = dst.load(symbol, timeframe)
    return cfg, ohlcv, feats


def main() -> int:
    args = parse_args()
    cfg, ohlcv, feats = load_symbol_frames(args.symbol, args.timeframe, args.config)
    symbol = args.symbol or cfg["data"]["symbol"]
    n = len(ohlcv)
    # match the training pipeline's split exactly: score window = first
    # score_window bars AFTER validation end
    val_end = int(n * (args.train_frac + args.val_frac))
    score_end = min(val_end + args.score_window, n)

    bounds = {
        "score": (val_end, score_end),
        "tail": (score_end, n),
        "full": (val_end, n),
        "all": (200, n),  # post-warmup; cross-venue data is all unseen
    }
    lo, hi = bounds[args.slice]

    agent = next((a for a in get_agents(db_path=args.db)
                  if a["agent_id"] == args.agent), None)
    if agent is None or not agent["model_path"]:
        print(f"agent not found or has no model: {args.agent}")
        return 1
    genome_row = get_genome(agent["genome_id"], db_path=args.db)
    assert genome_row is not None

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])
    from darwin.evolution.genome import Genome
    from darwin.experiments.training import risk_config_from_genome

    genome = Genome(values=genome_row["values"], genome_id=agent["genome_id"])
    eff_risk = risk_config_from_genome(risk_cfg, genome)
    eff_sim = SimulatorConfig(
        initial_capital=sim_cfg.initial_capital,
        taker_fee_pct=sim_cfg.taker_fee_pct,
        slippage_pct=sim_cfg.slippage_pct,
        position_size_pct=min(sim_cfg.position_size_pct,
                              genome["position_size_pct"]),
        qty_decimals=sim_cfg.qty_decimals,
        close_at_end=sim_cfg.close_at_end,
    )

    window = ohlcv.iloc[lo:hi].reset_index(drop=True)
    window_feats = feats.iloc[lo:hi].reset_index(drop=True)
    if len(window) < 250:
        print(f"slice too small ({len(window)} bars) for a meaningful episode")
        return 1

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

    report = MetricsReport.from_result(env.last_result, args.timeframe)
    behavior = agent["metrics"].get("behavior", {}) if agent["metrics"] else {}
    print(f"verify {args.agent} | {symbol} {args.timeframe} | slice={args.slice} "
          f"[{window['timestamp'].iloc[0]} .. {window['timestamp'].iloc[-1]}] "
          f"| {len(window)} bars")
    print(format_header())
    print(format_row("candidate", report))
    if behavior:
        print(f"behavior: long={behavior.get('pos_long_frac', 0):.1%} "
              f"short={behavior.get('pos_short_frac', 0):.1%} "
              f"flat={behavior.get('pos_flat_frac', 0):.1%} "
              f"(selection-time fingerprint; current run may differ)")
    if args.regimes and len(window) > 400:
        timeline = regime_timeline(window, RegimeConfig(trend_window=96,
                                                        vol_window=96))
        perf = regime_performance(
            env.last_result.equity_curve["equity"].reset_index(drop=True),
            timeline,
        )
        print(format_regime_table(perf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
