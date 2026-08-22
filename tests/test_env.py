"""TradingEnv tests: contracts, determinism, and no-look-ahead guarantees."""

from __future__ import annotations

import numpy as np
import pytest

from darwin.environment.env import TradingEnv, action_to_signal
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.features.engine import FeatureEngine


@pytest.fixture
def frames(make_ohlcv):
    candles = make_ohlcv(n=300)
    features = FeatureEngine().build_feature_matrix(candles)
    return candles, features


def _no_cost_cfg() -> SimulatorConfig:
    return SimulatorConfig(
        initial_capital=1000.0,
        taker_fee_pct=0.0,
        slippage_pct=0.0,
        position_size_pct=25.0,
        close_at_end=False,
    )


class TestContract:
    def test_observation_and_action_spaces(self, frames) -> None:
        candles, features = frames
        env = TradingEnv(candles, features)
        obs, _ = env.reset(seed=0)

        assert env.action_space.n == 4
        assert obs.shape == (27 + 3,)
        assert obs.dtype == np.float32

    def test_warmup_guard_defaults_to_first_valid_row(self, frames) -> None:
        candles, features = frames
        env = TradingEnv(candles, features)
        # ema_200 becomes valid at index 199 on a 300-row frame
        assert env._start == 199

    def test_start_before_warmup_rejected(self, frames) -> None:
        candles, features = frames
        with pytest.raises(ValueError, match="warmup"):
            TradingEnv(candles, features, start_idx=50)


class TestExecutionSemantics:
    def test_action_mapping_roundtrip(self) -> None:
        assert action_to_signal(1) == Action.LONG
        assert action_to_signal(2) == Action.SHORT
        assert action_to_signal(3) == Action.CLOSE
        assert action_to_signal(0) == Action.HOLD

    def test_long_then_close_flows_through_simulator(self, frames) -> None:
        candles, features = frames
        cfg = _no_cost_cfg()
        env = TradingEnv(candles, features, config=cfg)

        obs, _ = env.reset(seed=0)
        flat_sign = obs[-3]
        assert flat_sign == 0.0

        for _ in range(3):
            obs, *_ = env.step(0)  # HOLD
        obs_after_long, _, _, _, info = env.step(1)  # LONG -> fills next open

        assert info["position_qty"] > 0
        assert obs_after_long[-3] == 1.0

        obs_after_close, _, _, _, info2 = env.step(3)  # CLOSE -> fills next open
        assert info2["position_qty"] == 0.0
        assert obs_after_close[-3] == 0.0


class TestReward:
    def test_reward_equals_log_equity_ratio(self, frames) -> None:
        """Rewards must equal log ratios of the simulator's marked equity."""
        candles, features = frames
        start, end = 210, 240
        cfg = _no_cost_cfg()
        env = TradingEnv(candles, features, config=cfg, start_idx=start, end_idx=end)

        actions = [1, 0, 0, 3, 2, 0] * ((end - start) // 6 + 1)
        obs, _ = env.reset(seed=0)

        rewards = []
        done = False
        i = 0
        while not done:
            _, r, terminated, truncated, _ = env.step(int(actions[i]))
            rewards.append(r)
            done = terminated or truncated
            i += 1

        scripted = [action_to_signal(a) for a in actions[:i]]
        # batch API wants one entry per candle; the trailing decision slot
        # never executes, so padding with HOLD reproduces the episode exactly
        scripted.append(Action.HOLD)
        result = TradingSimulator(cfg).run(
            candles.iloc[start : end + 1].reset_index(drop=True), scripted
        )
        curve = result.equity_curve["equity"].to_numpy()
        expected = [float(np.log(curve[k + 1] / curve[k])) for k in range(len(curve) - 1)]

        assert len(rewards) == len(expected)
        for got, want in zip(rewards, expected, strict=True):
            assert got == pytest.approx(want, abs=1e-12)


class TestNoLookAhead:
    def test_future_feature_perturbation_cannot_change_past_observations(self, frames) -> None:
        candles, features = frames
        k = 250
        perturbed = features.copy()
        perturbed.loc[k:, "rsi_14"] = 99.0
        perturbed.loc[k:, "close"] = 123456.0

        env_a = TradingEnv(candles, features, start_idx=210, end_idx=260)
        env_b = TradingEnv(candles, perturbed, start_idx=210, end_idx=260)

        obs_a, _ = env_a.reset(seed=0)
        obs_b, _ = env_b.reset(seed=0)
        assert np.array_equal(obs_a, obs_b)

        for step in range(k - 210 - 5):
            a, *_ = env_a.step(0)
            b, *_ = env_b.step(0)
            assert np.array_equal(a, b), f"observation leaked future info at step {step}"


class TestDeterminismAndTermination:
    def test_identical_seeds_and_actions_bit_identical(self, frames) -> None:
        candles, features = frames

        def rollout(env):
            obs, _ = env.reset(seed=42)
            seen = [obs.copy()]
            rewards = []
            done = False
            rng = np.random.default_rng(7)
            while not done:
                action = int(rng.integers(0, 4))
                obs, r, terminated, truncated, info = env.step(action)
                seen.append(obs.copy())
                rewards.append(r)
                done = terminated or truncated
            return seen, rewards, info

        cfg = SimulatorConfig()  # default costs ON - determinism includes them
        trace_a = rollout(TradingEnv(*frames, config=cfg))
        trace_b = rollout(TradingEnv(*frames, config=cfg))

        for xa, xb in zip(trace_a[0], trace_b[0], strict=True):
            assert np.array_equal(xa, xb)
        assert trace_a[1] == trace_b[1]

    def test_episode_ends_flat_via_close_at_end(self, frames) -> None:
        candles, features = frames
        env = TradingEnv(candles, features, start_idx=220, end_idx=280)
        env.reset(seed=1)

        rng = np.random.default_rng(3)
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 4)))
            done = terminated or truncated
            steps += 1

        assert steps == 280 - 220  # one decision per candle after the first
        assert env.last_result is not None
        assert env.last_result.closed_at_end
        assert float(env.last_result.equity_curve["position_qty"].iloc[-1]) == 0.0


class TestBlownAccount:
    def test_log_reward_branches(self, frames) -> None:
        from darwin.environment.env import TradingEnv as _Env

        # normal growth: exact log ratio
        assert _Env._log_reward(100.0, 110.0) == pytest.approx(np.log(1.1))
        # equity wiped or negative: the guard fires instead of math error
        assert _Env._log_reward(1.0, 0.0) == -10.0
        assert _Env._log_reward(1.0, -5.0) == -10.0
        # flat: zero reward (no free signal for doing nothing)
        assert _Env._log_reward(100.0, 100.0) == pytest.approx(0.0)
