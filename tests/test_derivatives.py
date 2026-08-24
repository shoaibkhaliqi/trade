"""Derivatives feature tests - causal event merging, warmup honesty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darwin.features.derivatives import merge_derivatives_features


def _ts(hours: list[int]) -> pd.Series:
    return pd.Series([pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=h)
                      for h in hours])


def _base(n: int = 48) -> pd.DataFrame:
    ts = _ts(list(range(n)))
    return pd.DataFrame({"timestamp": ts, "f": np.arange(n, dtype=float)})


def _funding(hours_rates: list[tuple[int, float]]) -> pd.DataFrame:
    ts = _ts([h for h, _ in hours_rates])
    return pd.DataFrame({
        "timestamp": ts,
        "funding_rate": [r for _, r in hours_rates],
    })


class TestFundingMerge:
    def test_bar_sees_latest_past_settlement(self) -> None:
        base = _base(10)
        funding = _funding([(2, 0.0001), (6, 0.0002), (7, -0.0001)])

        out = merge_derivatives_features(base, funding, None)

        assert out["fnd_rate"].iloc[1].item() is np.nan or np.isnan(out["fnd_rate"].iloc[1])
        assert out["fnd_rate"].iloc[2] == pytest.approx(0.0001)
        assert out["fnd_rate"].iloc[5] == pytest.approx(0.0001)   # before 6:00
        assert out["fnd_rate"].iloc[6] == pytest.approx(0.0002)
        assert out["fnd_rate"].iloc[9] == pytest.approx(-0.0001)

    def test_settlement_exactly_at_bar_is_visible(self) -> None:
        base = _base(10)
        funding = _funding([(4, 0.0003)])
        out = merge_derivatives_features(base, funding, None)
        assert out["fnd_rate"].iloc[4] == pytest.approx(0.0003)

    def test_future_settlement_cannot_leak(self) -> None:
        base = _base(10)
        funding = _funding([(2, 0.0001), (20, 0.0009)])  # far future
        out = merge_derivatives_features(base, funding, None)
        assert out["fnd_rate"].iloc[2:20].eq(0.0001).all()
        assert not out["fnd_rate"].iloc[2:20].eq(0.0009).any()
        assert out["fnd_rate"].iloc[:2].isna().all()

    def test_trailing_stats_use_past_events_only(self) -> None:
        base = _base(48)
        funding = _funding([(h, 0.0001 * (h // 4 + 1)) for h in range(0, 40, 4)])
        out = merge_derivatives_features(base, funding, None, z_window=6)
        # settlements: h=0 -> 0.0001, h=4 -> 0.0002, h=8 -> 0.0003
        row3 = out.loc[out["timestamp"] == _ts([8]).iloc[0], "fnd_ma3"]
        assert row3.iloc[0] == pytest.approx((0.0001 + 0.0002 + 0.0003) / 3)


class TestOpenInterest:
    def test_oi_and_change_merge(self) -> None:
        base = _base(48)
        oi = pd.DataFrame({
            "timestamp": _ts(list(range(0, 48))),
            "open_interest": np.arange(48, dtype=float) * 100.0,
        })
        out = merge_derivatives_features(base, None, oi, oi_change_bars=24)
        assert out["oi"].iloc[30] == pytest.approx(3000.0)
        # change at h=30 vs h=6: 3000/600 - 1 = 4.0
        assert out["oi_chg"].iloc[30] == pytest.approx(4.0)
        assert np.isnan(out["oi_chg"].iloc[10])  # before lookback fills

    def test_missing_series_yield_nan_columns(self) -> None:
        base = _base(10)
        out = merge_derivatives_features(base, None, None)
        for col in ("fnd_rate", "fnd_ma3", "fnd_z", "oi", "oi_chg"):
            assert out[col].isna().all()


class TestCausality:
    def test_future_oi_perturbation_cannot_leak(self) -> None:
        base = _base(48)
        oi = pd.DataFrame({
            "timestamp": _ts(list(range(48))),
            "open_interest": np.arange(48, dtype=float) * 100.0,
        })
        perturbed = oi.copy()
        perturbed.loc[30:, "open_interest"] = 999999.0

        a = merge_derivatives_features(base, None, oi)
        b = merge_derivatives_features(base, None, perturbed)
        mask = base["timestamp"] < _ts([30]).iloc[0]
        assert (a.loc[mask, "oi"] == b.loc[mask, "oi"]).all()
