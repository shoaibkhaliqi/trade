"""Regime labeling tests - causality, known trends, per-label accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.evaluation.regimes import (
    RegimeConfig,
    format_regime_table,
    regime_performance,
    regime_timeline,
)


def _closes(n: int, drift: float, noise: float, seed: int, base: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, noise, n)
    return base * np.exp(np.cumsum(steps))


def _candles(closes: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    n = len(closes)
    ts = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamp": ts,
        "open": c,
        "high": c * 1.001,
        "low": c * 0.999,
        "close": c,
        "volume": 10.0,
    })


CFG = RegimeConfig(trend_window=48, vol_window=48)


class TestCausality:
    def test_future_perturbation_cannot_change_past_labels(self) -> None:
        closes = _closes(400, 0.0, 0.001, seed=1)
        candles = _candles(closes)
        perturbed = candles.copy()
        k = 250
        perturbed.loc[k:, "close"] *= 3.0

        base = regime_timeline(candles, CFG)
        pert = regime_timeline(perturbed, CFG)

        same = (
            base["trend"].to_numpy()[:k] == pert["trend"].to_numpy()[:k]
        ).all() and (
            base["vol_regime"].to_numpy()[:k] == pert["vol_regime"].to_numpy()[:k]
        ).all()
        assert same, "regime labels leaked future information"

    def test_warmup_bars_are_labeled_warmup(self) -> None:
        closes = _closes(200, 0.0, 0.001, seed=2)
        timeline = regime_timeline(_candles(closes), CFG)
        assert (timeline["combined"].iloc[:48] == "warmup").all()
        assert (timeline["combined"].iloc[48:] != "warmup").all()


class TestTrendLabels:
    def test_rising_market_labeled_bull(self) -> None:
        closes = _closes(300, 0.002, 0.0005, seed=3)  # ~0.2%/bar up
        timeline = regime_timeline(_candles(closes), CFG)
        post = timeline.iloc[48:]["trend"]
        assert (post == "strong_bull").mean() > 0.6

    def test_falling_market_labeled_bear(self) -> None:
        closes = _closes(300, -0.002, 0.0005, seed=4)
        timeline = regime_timeline(_candles(closes), CFG)
        post = timeline.iloc[48:]["trend"]
        assert (post == "strong_bear").mean() > 0.6

    def test_flat_market_labeled_sideways(self) -> None:
        closes = _closes(300, 0.0, 0.0002, seed=5)
        timeline = regime_timeline(_candles(closes), CFG)
        post = timeline.iloc[48:]["trend"]
        assert (post == "sideways").mean() > 0.6


class TestVolatilityLabels:
    def test_volatility_spike_labeled_high(self) -> None:
        rng = np.random.default_rng(6)
        calm = rng.normal(0.0, 0.0005, 200)
        wild = rng.normal(0.0, 0.006, 200)  # 12x the noise
        closes = 100.0 * np.exp(np.cumsum(np.concatenate([calm, wild])))
        timeline = regime_timeline(_candles(closes), CFG)

        late_wild = timeline.iloc[250:400]
        assert (late_wild["vol_regime"] == "high").mean() > 0.5


class TestRegimePerformance:
    def test_per_label_returns_reconstruct_the_whole(self) -> None:
        """Sum of per-label equity log-returns == total log-return."""
        n = 300
        closes = _closes(n, 0.001, 0.001, seed=7)
        candles = _candles(closes)
        timeline = regime_timeline(candles, CFG)

        equity = pd.Series(1000.0 * np.exp(np.cumsum(
            np.random.default_rng(8).normal(0.0005, 0.002, n))))
        equity.iloc[0] = 1000.0

        perf = regime_performance(equity, timeline)
        total_log = float(np.log(equity.iloc[-1] / equity.iloc[0]))
        assert perf["log_return"].sum() == pytest.approx(total_log, abs=1e-9)
        # warmup is reported as its own bucket - no bar goes missing silently
        assert (perf["regime"] == "warmup").any()

    def test_misaligned_frames_rejected(self) -> None:
        candles = _candles(_closes(200, 0.0, 0.001, seed=9))
        timeline = regime_timeline(candles, CFG)
        with pytest.raises(ValueError, match="misaligned"):
            regime_performance(pd.Series(np.ones(100)), timeline)

    def test_formatting_is_stable(self) -> None:
        frame = pd.DataFrame([
            {"regime": "sideways/low", "bars": 10, "time_share": 0.5,
             "log_return": 0.0, "return": 0.0},
        ])
        text = format_regime_table(frame)
        assert "sideways/low" in text
