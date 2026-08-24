"""Triple-barrier label tests - crafted paths resolve exactly as specified."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.features.labels import LabelConfig, triple_barrier_labels


def _series(moves: list[float], base: float = 100.0) -> pd.Series:
    """Close series from consecutive percentage moves."""
    steps = np.array([m / 100.0 for m in moves])
    return pd.Series(base * np.exp(np.cumsum(steps)))


def _quiet(n: int, base: float = 100.0) -> pd.Series:
    """Alternating +-0.05% - nonzero but tiny volatility (never exactly 0)."""
    return _series([0.05 if i % 2 == 0 else -0.05 for i in range(n)], base)


def _drift(n: int, per_bar: float, base: float = 100.0) -> pd.Series:
    """Net drift per bar with alternating noise around it (sigma stays ~0.05%)."""
    return _series([per_bar + 0.05 if i % 2 == 0 else per_bar - 0.05
                    for i in range(n)], base)


CFG = LabelConfig(horizon=16, vol_window=96)


class TestResolution:
    def test_immediate_up_move_hits_upper_first(self) -> None:
        s = pd.concat([_quiet(120), _series([5.0] + [0.0] * 20)], ignore_index=True)
        labels = triple_barrier_labels(s, CFG)
        assert labels.iloc[119] == 1.0

    def test_immediate_down_move_hits_lower_first(self) -> None:
        s = pd.concat([_quiet(120), _series([-5.0] + [0.0] * 20)], ignore_index=True)
        labels = triple_barrier_labels(s, CFG)
        assert labels.iloc[119] == 0.0

    def test_lower_barrier_wins_when_hit_before_upper(self) -> None:
        # +1% (inside the 2-sigma band after quiet history? no: breaks upper)
        # construct explicitly: small up then big down within the scan window
        s = pd.concat([_quiet(120), _series([0.04, -0.2] + [0.0] * 20)],
                      ignore_index=True)
        labels = triple_barrier_labels(s, CFG)
        assert labels.iloc[119] == 0.0

    def test_timeout_labels_by_sign(self) -> None:
        # drift 0.002%/bar with 0.05% noise: 16-bar drift ~0.03% << 2-sigma
        # band (~0.1%) -> no breach -> timeout labels by end-vs-entry sign
        up = _drift(200, 0.002)
        down = _drift(200, -0.002)
        assert triple_barrier_labels(up, CFG).iloc[100] == 1.0
        assert triple_barrier_labels(down, CFG).iloc[100] == 0.0


class TestCausalityAndTail:
    def test_tail_horizon_bars_are_nan(self) -> None:
        labels = triple_barrier_labels(_quiet(150), CFG)
        assert labels.iloc[-16:].isna().all()
        assert labels.iloc[-17:-16].notna().all()

    def test_warmup_vol_is_nan(self) -> None:
        labels = triple_barrier_labels(_quiet(150), CFG)
        assert labels.iloc[:96].isna().all()

    def test_perturbing_far_future_cannot_change_earlier_labels(self) -> None:
        s = pd.concat([_quiet(120), _series([5.0] + [0.0] * 80)],
                      ignore_index=True)
        a = triple_barrier_labels(s, CFG)
        perturbed = s.copy()
        perturbed.iloc[160:] *= 10.0
        b = triple_barrier_labels(perturbed, CFG)
        assert (a.iloc[:124].dropna() == b.iloc[:124].dropna()).all()

    def test_vol_scaled_barriers_widen_with_volatility(self) -> None:
        """Same +1% jump: breach under quiet history, timeout-down under wild."""
        calm = pd.concat([_quiet(120), _series([1.0] + [0.0] * 20)],
                         ignore_index=True)
        # wild history: +-1% alternating (sigma ~1%, 2-sigma band ~2%)
        wild = _series([1.0 if i % 2 == 0 else -1.0 for i in range(120)]
                       + [1.0] + [-0.08] * 20)

        assert triple_barrier_labels(calm, CFG).iloc[119] == 1.0
        assert triple_barrier_labels(wild, CFG).iloc[119] == 0.0


class TestValidation:
    def test_bad_config_rejected(self) -> None:
        with pytest.raises(ValueError):
            LabelConfig(horizon=0)
        with pytest.raises(ValueError):
            LabelConfig(tp_sigma=-1.0)
