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
from typing import Any

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.env import TradingEnv
from darwin.environment.simulator import SimulatorConfig
from darwin.evaluation.metrics import MetricsReport
from darwin.evolution.genome import Genome
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


def risk_config_from_genome(base: RiskConfig, genome: Genome | None) -> RiskConfig:
    """Overlay a genome's behavioral genes onto the yaml risk baseline."""
    if genome is None:
        return base
    return RiskConfig(
        max_position_size_pct=base.max_position_size_pct,
        max_leverage=base.max_leverage,
        max_risk_per_trade_pct=base.max_risk_per_trade_pct,
        stop_loss_pct=genome["stop_loss_pct"],
        take_profit_pct=genome["take_profit_pct"],
        max_daily_loss_pct=base.max_daily_loss_pct,
        # the kill-switch stays a PROTOCOL constant: evolution must never be
        # able to breed its way out of the emergency brake
        max_drawdown_pct=base.max_drawdown_pct,
        cooldown_bars=int(genome["cooldown_bars"]),
        max_trades_per_day=int(genome["max_trades_per_day"]),
    )


def ppo_kwargs_from_genome(genome: Genome | None) -> dict[str, Any]:
    if genome is None:
        return {}
    return {
        "learning_rate": genome["learning_rate"],
        "ent_coef": genome["ent_coef"],
        "gamma": genome["gamma"],
    }


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
    genome: Genome | None = None,
    score_window_bars: int | None = None,
    init_from_model_path: str | None = None,
    reward_baseline_weight: float = 0.0,
) -> tuple[str, MetricsReport, dict]:
    """Train on [0, train_end), validate near VAL tail, score once on TEST.

    Returns (model_path, report, behavior fingerprint).

    When ``genome`` is provided it overrides stop/TP/cooldown/trade-cap in the
    risk layer and PPO learning genes; entry sizing uses its position gene.
    ``score_window_bars`` optionally limits the TEST scoring to the first N
    test rows (population-scale scoring); None means the full test slice.
    ``init_from_model_path`` continues training from a parent's weights
    (inheritance of knowledge) instead of starting from random policy init.
    """
    effective_risk = risk_config_from_genome(risk_cfg, genome)
    episode_sim_cfg = sim_cfg
    if genome is not None:
        episode_sim_cfg = SimulatorConfig(
            initial_capital=sim_cfg.initial_capital,
            taker_fee_pct=sim_cfg.taker_fee_pct,
            slippage_pct=sim_cfg.slippage_pct,
            position_size_pct=min(sim_cfg.position_size_pct,
                                  genome["position_size_pct"]),
            qty_decimals=sim_cfg.qty_decimals,
            close_at_end=sim_cfg.close_at_end,
        )

    def make_train() -> TradingEnv:
        return TradingEnv(
            ohlcv.iloc[:train_end].reset_index(drop=True),
            feats.iloc[:train_end].reset_index(drop=True),
            config=episode_sim_cfg,
            risk=RiskManager(effective_risk),
            reward_baseline_weight=reward_baseline_weight,
        )

    def make_val_quick() -> TradingEnv:
        lo = max(train_end, val_end - eval_window)
        return TradingEnv(
            ohlcv.iloc[lo:val_end].reset_index(drop=True),
            feats.iloc[lo:val_end].reset_index(drop=True),
            config=episode_sim_cfg,
            risk=RiskManager(effective_risk),
            reward_baseline_weight=reward_baseline_weight,
        )

    learning_kwargs = ppo_kwargs_from_genome(genome)
    if init_from_model_path is not None:
        # inheritance of knowledge: start from the parent's trained weights,
        # override the child's mutated learning genes, keep training
        model = PPO.load(init_from_model_path, device="cpu")
        model.set_env(DummyVecEnv([make_train]))
        for key, value in learning_kwargs.items():
            setattr(model, key, value)
    else:
        model = PPO(
            "MlpPolicy",
            DummyVecEnv([make_train]),
            seed=seed,
            # measured on this stack (M9): CPU 748 fps vs CUDA 303 fps for a
            # ~5k-param MLP fed by a single Python env - the GPU starves
            # waiting for env steps. Revisit when policies/parallel envs grow.
            device="cpu",
            verbose=0,
            n_steps=2048,
            batch_size=256,
            **learning_kwargs,
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
    if genome is not None and genome.genome_id:
        model_path += f"_g{genome.genome_id}"
    model.save(model_path)

    test_candles = ohlcv.iloc[val_end:].reset_index(drop=True)
    test_feats = feats.iloc[val_end:].reset_index(drop=True)
    if score_window_bars is not None:
        keep = min(score_window_bars, len(test_candles))
        test_candles = test_candles.iloc[:keep].reset_index(drop=True)
        test_feats = test_feats.iloc[:keep].reset_index(drop=True)
    report, behavior = evaluate_agent_on_test(
        model_path, test_candles, test_feats,
        sim_cfg=episode_sim_cfg, risk_cfg=effective_risk,
        seed=seed, timeframe=timeframe,
        reward_baseline_weight=reward_baseline_weight,
    )
    return f"{model_path}.zip", report, behavior


def evaluate_agent_on_test(
    model_path: str,
    test_candles: pd.DataFrame,
    test_feats: pd.DataFrame,
    *,
    sim_cfg: SimulatorConfig,
    risk_cfg: RiskConfig,
    seed: int,
    timeframe: str,
    reward_baseline_weight: float = 0.0,
) -> tuple[MetricsReport, dict]:
    """One deterministic pass over TEST - the only scoring that counts.

    Returns (report, behavior): behavior is the agent's position-state
    fingerprint (see evolution.behavior) - captured here because the episode
    is already being stepped; measuring behavior costs nothing extra.
    """
    from darwin.evolution.behavior import summarize_behavior

    model = PPO.load(model_path, device="cpu")
    env = TradingEnv(test_candles, test_feats, config=sim_cfg,
                     risk=RiskManager(risk_cfg),
                     reward_baseline_weight=reward_baseline_weight)
    obs, _ = env.reset(seed=seed)
    actions: list[int] = []
    positions: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        positions.append(float(info["position_qty"]))
    assert env.last_result is not None
    report = MetricsReport.from_result(env.last_result, timeframe)
    behavior = summarize_behavior(actions, positions)
    return report, behavior
