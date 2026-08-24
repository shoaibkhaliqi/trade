"""Benchmark strategies: deliberately simple reference traders.

Contract (matches the simulator): ``generate_actions(ohlcv) -> list[Action]``,
one decision per candle, decided AFTER that candle closes; the simulator fills
each decision at the NEXT candle's open.

Design rule: indicator-based strategies call FeatureEngine instead of
re-implementing math - benchmarks must flow through the exact code paths
M2 proved shift-safe.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from darwin.environment.simulator import Action
from darwin.features.engine import FeatureEngine


class Strategy(Protocol):
    name: str

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]: ...


def _validate(actions: list[Action], n: int) -> list[Action]:
    if len(actions) != n:
        msg = f"strategy produced {len(actions)} actions for {n} candles"
        raise RuntimeError(msg)
    return actions


def _signs_to_actions(signs: np.ndarray) -> list[Action]:
    """Map {-1, 0, +1} signals to actions; NaN (warmup) becomes HOLD."""
    out: list[Action] = []
    for raw in signs:
        if np.isnan(raw):
            out.append(Action.HOLD)
        elif raw > 0:
            out.append(Action.LONG)
        elif raw < 0:
            out.append(Action.SHORT)
        else:
            out.append(Action.HOLD)
    return out


class BuyAndHoldStrategy:
    """Target state: long for the entire horizon.

    Emits a persistent target-state series (LONG everywhere) rather than a
    single entry event - repeats are no-ops in the simulator, and fold-based
    evaluation can sample any bar of this list and read the true intent.
    """

    name = "buy_and_hold"

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        return [Action.LONG] * len(ohlcv)


class AlwaysShortStrategy:
    """Target state: short for the entire horizon (mirror of buy_and_hold).

    Exists as a CONTROL: any agent whose performance merely matches this in
    a falling window has a directional bias, not an edge. The fitness
    baselines use max(long, short) passive returns for exactly that reason.
    """

    name = "always_short"

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        return [Action.SHORT] * len(ohlcv)


class RandomTraderStrategy:
    """Seeded noise trader - fully deterministic for a given seed."""

    name = "random"

    def __init__(self, seed: int = 42, p_hold: float = 0.6, p_long: float = 0.2) -> None:
        if not (0 <= p_hold <= 1 and 0 <= p_long <= 1):
            msg = "probabilities must be within [0, 1]"
            raise ValueError(msg)
        if p_hold + p_long > 1:
            msg = "p_hold + p_long must not exceed 1"
            raise ValueError(msg)
        self.seed = seed
        self.p_hold = p_hold
        self.p_long = p_long

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(3, size=len(ohlcv), p=[self.p_hold, self.p_long,
                                                1 - self.p_hold - self.p_long])
        mapping = [Action.HOLD, Action.LONG, Action.SHORT]
        return [mapping[i] for i in idx]


class MovingAverageCrossStrategy:
    """LONG while fast EMA above slow EMA, SHORT below; HOLD through warmup."""

    name = "ema_cross"

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        if fast >= slow:
            msg = "fast window must be smaller than slow window"
            raise ValueError(msg)
        self.fast = fast
        self.slow = slow

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        feats = FeatureEngine().compute(ohlcv)
        diff = feats[f"ema_{self.fast}"] - feats[f"ema_{self.slow}"]
        return _validate(_signs_to_actions(np.sign(diff.to_numpy())), len(ohlcv))


class RSIMeanReversionStrategy:
    """LONG when oversold, SHORT when overbought, HOLD inside the band."""

    name = "rsi_reversion"

    def __init__(self, lower: float = 30.0, upper: float = 70.0) -> None:
        if not 0 <= lower < upper <= 100:
            msg = "require 0 <= lower < upper <= 100"
            raise ValueError(msg)
        self.lower = lower
        self.upper = upper

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        rsi = FeatureEngine().compute(ohlcv)["rsi_14"].to_numpy()
        signs = np.where(
            np.isnan(rsi),
            np.nan,
            np.where(rsi < self.lower, 1.0, np.where(rsi > self.upper, -1.0, 0.0)),
        )
        return _validate(_signs_to_actions(signs), len(ohlcv))


class VWAPMeanReversionStrategy:
    """Fade stretches away from session VWAP beyond a relative threshold."""

    name = "vwap_reversion"

    def __init__(self, threshold: float = 0.004) -> None:
        if threshold <= 0:
            msg = "threshold must be positive"
            raise ValueError(msg)
        self.threshold = threshold

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        dist = FeatureEngine().compute(ohlcv)["vwap_dist"].to_numpy()
        signs = np.where(
            np.isnan(dist),
            np.nan,
            np.where(dist > self.threshold, -1.0,
                     np.where(dist < -self.threshold, 1.0, 0.0)),
        )
        return _validate(_signs_to_actions(signs), len(ohlcv))


def default_benchmarks(seed: int = 42) -> list[Strategy]:
    """The standard benchmark roster used by scripts and comparisons."""
    return [
        BuyAndHoldStrategy(),
        AlwaysShortStrategy(),
        RandomTraderStrategy(seed=seed),
        MovingAverageCrossStrategy(),
        RSIMeanReversionStrategy(),
        VWAPMeanReversionStrategy(),
    ]
