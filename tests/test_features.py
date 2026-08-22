"""Feature engine tests - correctness plus the two no-look-ahead guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from darwin.features.engine import FeatureEngine
from darwin.features.schema import ALL_FEATURES, ATR_PERIOD, RSI_PERIOD


@pytest.fixture
def engine() -> FeatureEngine:
    return FeatureEngine()


def _matrix(engine: FeatureEngine, make_ohlcv, n: int) -> pd.DataFrame:
    return engine.build_feature_matrix(make_ohlcv(n=n))


def _frame_from_closes(
    closes: list[float], vols: list[float] | None = None
) -> pd.DataFrame:
    """Build a canonical OHLCV frame where each candle opens at the prior close."""
    c = pd.Series(closes, dtype="float64")
    o = c.shift(1).fillna(closes[0])
    ts = pd.date_range("2024-01-01", periods=len(c), freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": o,
            "high": np.maximum(o, c) + 0.5,
            "low": np.minimum(o, c) - 0.5,
            "close": c,
            "volume": vols if vols is not None else np.full(len(c), 100.0),
        }
    )
    return df


class TestContract:
    def test_shape_columns_dtypes(self, engine, make_ohlcv) -> None:
        m = _matrix(engine, make_ohlcv, 300)
        assert len(m) == 300
        assert list(m.columns) == ["timestamp", *ALL_FEATURES]
        assert all(m[c].dtype == "float64" for c in ALL_FEATURES)

    def test_missing_columns_rejected(self, engine, make_ohlcv) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            engine.compute(make_ohlcv(5).drop(columns=["volume"]))

    def test_deterministic_repeat_calls(self, engine, make_ohlcv) -> None:
        a = _matrix(engine, make_ohlcv, 250)
        b = _matrix(engine, make_ohlcv, 250)
        assert_frame_equal(a, b)

    def test_no_infinite_values_anywhere(self, engine, make_ohlcv) -> None:
        m = _matrix(engine, make_ohlcv, 400)
        assert not np.isinf(m[list(ALL_FEATURES)].to_numpy()).any()


class TestNoLookAhead:
    """The two guards that make the feature engine trustworthy."""

    @pytest.mark.parametrize("cut", [210, 260, 299])
    def test_truncation_invariance(self, engine, make_ohlcv, cut: int) -> None:
        full = _matrix(engine, make_ohlcv, 320)
        part = _matrix(engine, make_ohlcv, cut)
        assert_frame_equal(part, full.iloc[:cut], check_exact=True)

    def test_future_perturbation_leaves_past_identical(self, engine, make_ohlcv) -> None:
        base = make_ohlcv(300)
        k = 200
        perturbed = base.copy()
        perturbed.loc[k:, ["open", "high", "low", "close"]] *= 3.0
        perturbed.loc[k:, "volume"] = perturbed.loc[k:, "volume"] * 7.0 + 1.0

        a = engine.build_feature_matrix(base)
        b = engine.build_feature_matrix(perturbed)

        assert_frame_equal(a.iloc[:k], b.iloc[:k], check_exact=True)


class TestKnownValues:
    def test_simple_and_log_returns(self, engine) -> None:
        df = _frame_from_closes([100.0, 110.0, 99.0])
        out = engine.compute(df)

        assert np.isnan(out["ret_1"].iloc[0])
        assert out["ret_1"].iloc[1] == pytest.approx(0.10)
        assert out["ret_1"].iloc[2] == pytest.approx(-0.10)
        assert out["log_ret_1"].iloc[1] == pytest.approx(np.log(1.1))
        assert out["log_ret_1"].iloc[2] == pytest.approx(np.log(0.9))

    def test_body_wicks_range_on_one_candle(self, engine) -> None:
        # single bull candle: o=100 c=110 h=115 l=98
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [115.0],
                "low": [98.0],
                "close": [110.0],
                "volume": [10.0],
            }
        )
        out = engine.compute(df)

        assert out["body_pct"].iloc[0] == pytest.approx(10.0 / 110.0)
        assert out["upper_wick_pct"].iloc[0] == pytest.approx(5.0 / 110.0)
        assert out["lower_wick_pct"].iloc[0] == pytest.approx(2.0 / 110.0)
        assert out["range_pct"].iloc[0] == pytest.approx(17.0 / 110.0)


class TestIndicators:
    def test_constant_market_degenerate_values(self, engine) -> None:
        n = 60
        ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": 50.0,
                "high": 50.0,
                "low": 50.0,
                "close": 50.0,
                "volume": 10.0,
            }
        )
        out = engine.compute(df)

        valid_rsi = out["rsi_14"].dropna()
        assert (valid_rsi == 50.0).all()
        assert (out["atr_14"].dropna() == 0.0).all()
        assert (out["atr_pct"].dropna() == 0.0).all()
        assert (out["ret_1"].dropna().abs() < 1e-15).all()
        assert (out["ema_dist_5"].dropna().abs() < 1e-12).all()
        assert (out["rel_vol_20"].dropna() == 1.0).all()

    def test_rsi_extremes(self, engine) -> None:
        up = _frame_from_closes([100.0 + i for i in range(40)])
        down = _frame_from_closes([200.0 - i for i in range(40)])

        rsi_up = engine.compute(up)["rsi_14"]
        rsi_down = engine.compute(down)["rsi_14"]

        finite_up = rsi_up.dropna()
        finite_down = rsi_down.dropna()
        assert len(finite_up) > 0 and len(finite_down) > 0
        assert (finite_up == 100.0).all()
        assert (finite_down == 0.0).all()

    def test_indicator_warmup_is_nan_not_garbage(self, engine, make_ohlcv) -> None:
        out = engine.compute(make_ohlcv(320))

        assert out["rsi_14"].iloc[: RSI_PERIOD - 1].isna().all()
        assert out["rsi_14"].iloc[RSI_PERIOD - 1 :].notna().any()
        assert out["atr_14"].iloc[: ATR_PERIOD - 1].isna().all()
        assert out["ema_200"].iloc[:199].isna().all()
        assert out["ema_200"].iloc[199:].notna().all()
        assert out["rv_20"].iloc[:19].isna().all()

    def test_atr_matches_wilder_recursion(self, engine) -> None:
        rng = np.random.default_rng(7)
        n = 40
        closes = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
        df = _frame_from_closes(closes.tolist())
        out = engine.compute(df)

        h, low, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([h - low, (h - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)

        alpha = 1.0 / ATR_PERIOD
        expected = tr.iloc[0]
        for t in range(1, n):
            expected = alpha * tr.iloc[t] + (1 - alpha) * expected

        # last value must match an independently coded recursion of the same rule
        assert out["atr_14"].iloc[-1] == pytest.approx(expected, rel=1e-12)

    def test_session_vwap_resets_at_utc_midnight(self, engine) -> None:
        n_per_day = 24
        closes = [100.0] * n_per_day + [200.0] * n_per_day
        df = _frame_from_closes(closes)
        out = engine.compute(df)

        # Leak detector: day-2 VWAP must stay inside day-2's EXPANDING price
        # band [cummin(low), cummax(high)]. A leak from day-1 (~100 level)
        # would drag vwap below every later candle's low.
        day2_mask = df["timestamp"] >= pd.Timestamp("2024-01-02", tz="UTC")
        valid = day2_mask & out["vwap_dist"].notna()
        session = df["timestamp"].dt.normalize()
        band_lo = df.groupby(session)["low"].transform("cummin")
        band_hi = df.groupby(session)["high"].transform("cummax")
        vwap = df["close"] / (1.0 + out["vwap_dist"])
        assert (vwap[valid] >= band_lo[valid] - 1e-9).all()
        assert (vwap[valid] <= band_hi[valid] + 1e-9).all()
        # cumulative mean converges TOWARD the constant level; the single
        # straddling first candle keeps it slightly below forever - a leak
        # would instead show dist ~ -0.33
        assert abs(out["vwap_dist"].iloc[-1]) < 0.01

    def test_vwap_stays_within_session_expanding_extremes(self, engine, make_ohlcv) -> None:
        df = make_ohlcv(240)
        out = engine.compute(df)

        session = df["timestamp"].dt.normalize()
        sess_low = df.groupby(session)["low"].transform("cummin")
        sess_high = df.groupby(session)["high"].transform("cummax")

        vwap = df["close"] / (1.0 + out["vwap_dist"])
        valid = out["vwap_dist"].notna()
        inside = (vwap >= sess_low * 0.999) & (vwap <= sess_high * 1.001)
        assert inside[valid].all()

    def test_zero_volume_produces_nan_not_crash(self, engine) -> None:
        df = _frame_from_closes([100.0, 101.0, 100.5], vols=[0.0, 0.0, 500.0])
        out = engine.compute(df)

        assert out["vwap_dist"].isna().iloc[:2].all()
        assert np.isfinite(out["vwap_dist"].iloc[2])

    def test_vwma_known_small_case(self, engine) -> None:
        closes = list(np.full(25, 50.0))
        vols = list(np.full(25, 1.0))
        # overwrite final bars with known values; window still contains earlier bars
        closes[-3:] = [10.0, 20.0, 30.0]
        vols[-3:] = [1.0, 3.0, 100.0]
        df = _frame_from_closes(closes, vols=vols)

        out = engine.compute(df)
        c_arr = np.asarray(closes)
        v_arr = np.asarray(vols)
        expected = float((c_arr[-20:] * v_arr[-20:]).sum() / v_arr[-20:].sum())
        vwma_last = df["close"].iloc[-1] / (1.0 + out["vwma_dist_20"].iloc[-1])

        assert vwma_last == pytest.approx(expected)
