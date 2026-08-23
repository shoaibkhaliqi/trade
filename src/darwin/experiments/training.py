"""Reusable single-agent training & evaluation protocol.

Home of the M5/M7 protocol so both CLI scripts (and future evolution loops)
share one implementation:
- chronological split boundaries computed by caller via experiments.splits
- train PPO/MLP on TRAIN slice with risk attached and VAL-proxy eval callback
- exactly ONE deterministic scoring pass over TEST
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.env import TradingEnv
from darwin.environment.simulator import SimulatorConfig
from darwin.evaluation.metrics import MetricsReport
from darwin.execution.risk import RiskConfig, RiskManager


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sim_config_from_yaml(cfg: dict) -> SimulatorConfig:
    s = cfg["simulator"]
    return SimulatorConfig(
        initial_capital=s["initial_capital"],
        taker_fee_pct=s["taker_fee_pct"],
        slippage_pct=s["slippage_pct"],
        position_size_pct=s["fixed_position_size_pct"],
    )


def load_frames(config_name: str, timeframe: str):
    """Return (cfg, symbol, ohlcv, features) for stored datasets."""
    cfg = load_config(config_name)
    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]
    ohlcv = DataStorage(data_cfg["processed_dir"]).load(symbol, timeframe)
    feats = DataStorage(data_cfg["features_dir"]).load(symbol, timeframe)
    return cfg, symbol, ohlcv, feats


def train_and_evaluate(
    *,
    seed: int,
    ohlcv: pd.DataFrame,
    feats: pd.DataFrame,
    timeframe: str,
    sim_cfg: SimulatorConfig,
    risk_cfg: RiskConfig,
    train_end: int,
    val_end: int,
    timesteps: int,
    eval_window: int = 3_000,
    out_dir: str = "experiments/runs",
) -> tuple[str, MetricsReport]:
    """Train on [0, train_end), validate near VAL tail, score once on TEST."""

    def make_train() -> TradingEnv:
        return TradingEnv(
            ohlcv.iloc[:train_end].reset_index(drop=True),
            feats.iloc[:train_end].reset_index(drop=True),
            config=sim_cfg,
            risk=RiskManager(risk_cfg),
        )

    def make_val_quick() -> TradingEnv:
        lo = max(train_end, val_end - eval_window)
        return TradingEnv(
            ohlcv.iloc[lo:val_end].reset_index(drop=True),
            feats.iloc[lo:val_end].reset_index(drop=True),
            config=sim_cfg,
            risk=RiskManager(risk_cfg),
        )

    model = PPO(
        "MlpPolicy",
        DummyVecEnv([make_train]),
        seed=seed,
        device="cuda",
        verbose=0,
        n_steps=2048,
        batch_size=256,
    )
    callback = EvalCallback(
        make_val_quick(),
        eval_freq=max(eval_window // 2, 256),
        n_eval_episodes=1,
        deterministic=True,
        warn=False,
    )
    model.learn(total_timesteps=timesteps, callback=callback)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{out_dir}/ppo_mlp_seed{seed}_t{timesteps}"
    model.save(model_path)

    test_candles = ohlcv.iloc[val_end:].reset_index(drop=True)
    test_feats = feats.iloc[val_end:].reset_index(drop=True)
    report = evaluate_agent_on_test(
        model_path, test_candles, test_feats,
        sim_cfg=sim_cfg, risk_cfg=risk_cfg, seed=seed, timeframe=timeframe,
    )
    return f"{model_path}.zip", report


def evaluate_agent_on_test(
    model_path: str,
    test_candles: pd.DataFrame,
    test_feats: pd.DataFrame,
    *,
    sim_cfg: SimulatorConfig,
    risk_cfg: RiskConfig,
    seed: int,
    timeframe: str,
) -> MetricsReport:
    """One deterministic pass over TEST - the only scoring that counts."""
    model = PPO.load(model_path, device="cpu")
    env = TradingEnv(test_candles, test_feats, config=sim_cfg,
                     risk=RiskManager(risk_cfg))
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    assert env.last_result is not None
    return MetricsReport.from_result(env.last_result, timeframe)
