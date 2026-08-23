"""Train ONE RL agent (PPO/MLP) on PAXGUSDT and judge it honestly.

Protocol:
- Chronological split TRAIN / VAL / TEST (70/15/15 by default).
- Training sees TRAIN only; VAL drives the periodic eval callback (quick
  proxy windows); TEST is touched exactly ONCE at the end - by the agent AND
  by the M4 benchmarks, in the same arena, under identical costs.
"""

from __future__ import annotations

import argparse
import subprocess

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from darwin.agents import default_benchmarks
from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.env import TradingEnv
from darwin.environment.simulator import SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport, format_header, format_row
from darwin.execution.risk import RiskConfig, RiskManager
from darwin.experiments.tracker import record_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--timesteps", type=int, default=40_960)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--eval-window", type=int, default=3_000,
                        help="recent VAL rows used by the quick eval callback")
    parser.add_argument("--out", default="experiments/runs")
    return parser.parse_args()


def sim_config_from_yaml(cfg: dict) -> SimulatorConfig:
    s = cfg["simulator"]
    return SimulatorConfig(
        initial_capital=s["initial_capital"],
        taker_fee_pct=s["taker_fee_pct"],
        slippage_pct=s["slippage_pct"],
        position_size_pct=s["fixed_position_size_pct"],
    )


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]

    ohlcv = DataStorage(data_cfg["processed_dir"]).load(symbol, args.timeframe)
    feats = DataStorage(data_cfg["features_dir"]).load(symbol, args.timeframe)

    n = len(ohlcv)
    train_end = int(n * args.train_frac)
    val_end = int(n * (args.train_frac + args.val_frac))
    print(f"{symbol} {args.timeframe}: {n} rows | train<={train_end} "
          f"val<= {val_end} test={val_end}..{n - 1}")

    sim_cfg = sim_config_from_yaml(cfg)
    risk_cfg = RiskConfig(**cfg["risk"])

    def make_risk() -> RiskManager:
        # separate instance per environment: latch/day state must not leak
        return RiskManager(risk_cfg)

    def make_train():
        return TradingEnv(
            ohlcv.iloc[:train_end].reset_index(drop=True),
            feats.iloc[:train_end].reset_index(drop=True),
            config=sim_cfg,
            risk=make_risk(),
        )

    def make_val_quick():
        lo = max(train_end, val_end - args.eval_window)
        return TradingEnv(
            ohlcv.iloc[lo:val_end].reset_index(drop=True),
            feats.iloc[lo:val_end].reset_index(drop=True),
            config=sim_cfg,
            risk=make_risk(),
        )

    train_vec = DummyVecEnv([make_train])
    eval_env = make_val_quick()

    model = PPO(
        "MlpPolicy",
        train_vec,
        seed=args.seed,
        device="cuda",
        verbose=1,
        n_steps=2048,
        batch_size=256,
    )
    callback = EvalCallback(
        eval_env,
        eval_freq=args.eval_window // 2,
        n_eval_episodes=1,
        deterministic=True,
        warn=False,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    out_path = f"{args.out}/ppo_mlp_{symbol}_{args.timeframe}_s{args.seed}"
    model.save(out_path)
    print(f"\nmodel saved -> {out_path}.zip")

    # ------------------------------------------------------------------
    # TEST: touched exactly once - agent first...
    # ------------------------------------------------------------------
    test_candles = ohlcv.iloc[val_end:].reset_index(drop=True)
    test_feats = feats.iloc[val_end:].reset_index(drop=True)
    test_env = TradingEnv(test_candles, test_feats, config=sim_cfg, risk=make_risk())
    obs, _ = test_env.reset(seed=args.seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(int(action))
        done = terminated or truncated

    assert test_env.last_result is not None
    agent_report = MetricsReport.from_result(test_env.last_result, args.timeframe)

    # ...then the benchmark roster, identical slice and costs
    print("\n=== TEST ARENA (unseen during training) ===")
    print(format_header())
    print(format_row("ppo_agent", agent_report))
    for strategy in default_benchmarks(seed=args.seed):
        result = TradingSimulator(sim_cfg).run(
            test_candles, strategy.generate_actions(test_candles)
        )
        report = MetricsReport.from_result(result, args.timeframe)
        print(format_row(strategy.name, report))

    exp_id = record_experiment("train_agent", {
        "model": "PPO/MlpPolicy",
        "symbol": symbol,
        "timeframe": args.timeframe,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "split": {"train_end": train_end, "val_end": val_end, "test_rows": n - val_end},
        "sim_config": str(sim_cfg),
        "risk_config": str(risk_cfg),
        "git_commit": git_commit(),
        "model_path": out_path + ".zip",
        "test_metrics": vars(agent_report),
    })
    print(f"\nexperiment recorded: {exp_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
