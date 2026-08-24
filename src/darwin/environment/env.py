"""Gymnasium environment wrapping the stepping simulator.

Contracts honored (they are the whole point of this file):
- Observation at step t contains ONLY information from candles <= t:
  27 M2 features (NaN -> 0.0 after warmup, meaning "no signal") plus three
  account terms [position_sign, unrealized/equity, current_drawdown].
- The agent's discrete action is submitted AFTER the current candle closes;
  the simulator fills it at the NEXT candle's open. Look-ahead execution is
  structurally impossible because the simulator forbids it (M3).
- Reward = log(equity_t / equity_{t-1}) using the simulator's marked equity,
  so maximizing reward literally maximizes (log) account growth. On the final
  step the reward includes the close_at_end liquidation so episode totals are
  honest about exit costs.
- Determinism: same seed + same action sequence => bit-identical trajectory.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from darwin.environment.simulator import (
    Action,
    SimResult,
    SimulatorConfig,
    TradingSimulator,
)
from darwin.execution.risk import RiskContext, RiskManager
from darwin.features.schema import ALL_FEATURES

N_ACCOUNT_FEATURES = 3


def action_to_signal(action: int) -> Action:
    """Map Discrete(4) index to the simulator action vocabulary."""
    return [Action.HOLD, Action.LONG, Action.SHORT, Action.CLOSE][action]


class TradingEnv(gym.Env):
    """Single-asset trading episode over a chronological candle slice."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        candles: Any,
        features: Any,
        config: SimulatorConfig | None = None,
        start_idx: int | None = None,
        end_idx: int | None = None,
        risk: RiskManager | None = None,
        reward_baseline_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if len(candles) != len(features):
            msg = "candles and features must be row-aligned"
            raise ValueError(msg)

        feats_only = features[list(ALL_FEATURES)]
        valid_mask = feats_only.notna().all(axis=1).to_numpy()
        first_valid = int(np.argmax(valid_mask)) if valid_mask.any() else len(feats_only)
        if not valid_mask.any():
            msg = "features never become fully valid - frame too short for warmup"
            raise ValueError(msg)

        self._start = start_idx if start_idx is not None else first_valid
        self._end = end_idx if end_idx is not None else len(candles) - 1
        if not (0 <= self._start <= self._end < len(candles)):
            msg = f"invalid episode bounds [{self._start}, {self._end}]"
            raise ValueError(msg)
        if self._start < first_valid:
            msg = f"start_idx {self._start} precedes feature warmup ({first_valid})"
            raise ValueError(msg)

        # NaN -> 0.0 is a documented modeling choice: after warmup, a NaN
        # feature means "undefined because market inactive", which we feed as
        # neutral zero rather than fabricating activity.
        window = slice(self._start, self._end + 1)
        self._feat_rows = feats_only.iloc[window].fillna(0.0).to_numpy(dtype="float32")
        self._candles_window = candles.iloc[window].reset_index(drop=True)
        self._span = self._end - self._start + 1

        self.sim = TradingSimulator(config or SimulatorConfig())
        # The risk layer is part of the environment, not the agent: any policy
        # trained/evaluated here physically cannot submit an unfiltered action.
        self.risk = risk
        # Reward shaping (M17b): r = log(eq ratio) - w * log(price ratio).
        # w=0 reproduces the historical reward exactly; w>0 makes sitting
        # flat through a move costly INSIDE the reward - the flat attractor's
        # plateau becomes a slope. Baseline = the window's own close series.
        self.reward_baseline_weight = float(reward_baseline_weight)
        closes = self._candles_window["close"].to_numpy(dtype="float64")
        price_logret = np.diff(np.log(closes), prepend=0.0)
        self._baseline_logret = price_logret.astype("float64")
        obs_dim = self._feat_rows.shape[1] + N_ACCOUNT_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(list(Action)))

        self._equity_peak: float | None = None
        self._flat_bars: int | None = 0  # bars since last exit; None while positioned
        self.last_result: SimResult | None = None

    # ------------------------------------------------------------------
    def _account_terms(self) -> np.ndarray:
        wallet = self.sim.wallet
        mark = float(self.sim._closes[self.sim._i])  # noqa: SLF001 - same package contract
        equity = wallet.equity(mark)
        sign_pos = float(np.sign(wallet.qty))
        unreal_ratio = float(np.clip(wallet.unrealized(mark) / equity, -1.0, 1.0))
        assert self._equity_peak is not None
        dd = float(equity / self._equity_peak - 1.0)
        return np.array([sign_pos, unreal_ratio, dd], dtype=np.float32)

    def _observe(self, info: dict[str, object]) -> np.ndarray:
        idx = int(info["index"])  # type: ignore[arg-type]
        # simulator indices are LOCAL to the episode window
        market = self._feat_rows[idx]
        return np.concatenate([market, self._account_terms()]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.last_result = None
        info = self.sim.prepare(self._candles_window)
        self._equity_peak = float(info["equity"])  # type: ignore[arg-type]
        return self._observe(info), {"index": int(info["index"])}  # type: ignore[arg-type]

    def _risk_context(self, mark: float) -> RiskContext:
        wallet = self.sim.wallet
        peak = self._equity_peak if self._equity_peak is not None else wallet.equity(mark)
        return RiskContext(
            timestamp=pd.Timestamp(self.sim._timestamps[self.sim._i]),  # noqa: SLF001
            bar_index=self.sim._i,  # noqa: SLF001
            mark=mark,
            equity=wallet.equity(mark),
            peak_equity=peak,
            position_qty=wallet.qty,
            entry_price=wallet.entry if wallet.has_position else float("nan"),
            total_trades=self.sim.n_trades,
            bars_since_exit=self._flat_bars,
        )

    @staticmethod
    def _log_reward(prev_equity: float, equity: float) -> float:
        """Per-step reward: log growth of marked equity; account blow-up => -10."""
        if equity <= 0.0:
            return -10.0
        return float(np.log(equity / prev_equity))

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._equity_peak is None:
            msg = "step() called before reset()"
            raise RuntimeError(msg)
        i = self.sim._i  # noqa: SLF001 - same-package contract with the simulator
        mark = float(self.sim._closes[i])  # noqa: SLF001
        prev_equity = float(self.sim.wallet.equity(mark))

        proposed = action_to_signal(int(action))
        size_pct: float | None = None
        if self.risk is not None:
            ctx = self._risk_context(mark)
            stop_distance = (
                self.risk.cfg.stop_loss_pct / 100.0
                if self.risk.cfg.stop_loss_pct is not None
                else None
            )
            proposed, size_pct = self.risk.apply(
                proposed,
                ctx,
                base_size_pct=self.sim.cfg.position_size_pct,
                stop_distance_pct=stop_distance,
            )
        self.sim.submit(proposed, size_pct=size_pct)
        info = self.sim.step()

        if self.sim.wallet.has_position:
            self._flat_bars = None
        else:
            self._flat_bars = 0 if self._flat_bars is None else self._flat_bars + 1
        equity = float(info["equity"])  # type: ignore[arg-type]

        idx = int(info["index"])  # type: ignore[arg-type]
        terminated = idx >= self._span - 1
        truncated = False

        if terminated:
            # fold close_at_end liquidation into the final reward so the
            # agent pays its exit costs inside the episode it caused them
            self.last_result = self.sim.result()
            equity = self.last_result.final_equity

        self._equity_peak = max(self._equity_peak, equity)
        reward = self._log_reward(prev_equity, equity)
        if self.reward_baseline_weight:
            reward -= self.reward_baseline_weight * float(self._baseline_logret[idx])

        obs = self._observe(info) if not terminated else self._observe_final()
        return obs, reward, terminated, truncated, {
            "index": idx,
            "timestamp": info["timestamp"],
            "equity": equity,
            "position_qty": info["position_qty"],
        }

    def _observe_final(self) -> np.ndarray:
        """Post-liquidation observation (flat account at final marks)."""
        market = self._feat_rows[-1]
        return np.concatenate([market, np.zeros(N_ACCOUNT_FEATURES, dtype="float32")]).astype(
            np.float32
        )

    def render(self) -> None:  # pragma: no cover - research console only
        print(f"index={self.sim._i} equity={self.sim.wallet.cash}")
