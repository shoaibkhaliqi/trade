"""Deterministic, shift-safe technical features on closed candles.

CONTRACT - the most important docstring in the project:

- One input row == one CLOSED candle, stamped with its OPEN time.
- The feature value at row t uses ONLY candles with timestamp <= t.
- Downstream rule (enforced by the simulator from M3): a decision made after
  candle t closes executes at the open of candle t+1. Features never decide
  fills inside their own candle.
- Warmup honesty: indicators needing N observations emit NaN until N candles
  exist (via min_periods), instead of unstable seed values.
- Determinism: pure vectorized pandas - same input frame => identical output,
  no RNG, no global state, no hidden normalization over future data.

Two machine guards prove the no-look-ahead property (tests/test_features.py):
1. truncation invariance      - features(prefix) == features(full)[:len]
2. future-perturbation change - editing later candles leaves earlier rows bit-
                                 identical.
Any new feature must pass both before it may feed an agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from darwin.data.schema import OHLCV_COLUMNS, TIMESTAMP_COL
from darwin.features.schema import (
    ALL_FEATURES,
    ATR_PERIOD,
    EMA_SPANS,
    FEATURE_VERSION,
    RSI_PERIOD,
    RV_WINDOW,
    VOL_MA_WINDOW,
    VWMA_WINDOW,
)


def _no_inf(s: pd.Series) -> pd.Series:
    """Zero-volume denominators can produce +-inf; research data forbids them."""
    return s.replace([np.inf, -np.inf], np.nan)


class FeatureEngine:
    """Computes the registered feature set from a canonical OHLCV frame."""

    VERSION = FEATURE_VERSION

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one column per feature, aligned row-for-row with ``df``."""
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            msg = f"input frame is missing columns: {missing}"
            raise ValueError(msg)

        o = df["open"].astype("float64")
        h = df["high"].astype("float64")
        low = df["low"].astype("float64")
        c = df["close"].astype("float64")
        v = df["volume"].astype("float64")

        feats: dict[str, pd.Series] = {}
        ts = pd.Series(df[TIMESTAMP_COL].to_numpy(), index=df.index)
        feats.update(self._price_features(o, h, low, c))
        feats.update(self._trend_features(c))
        feats.update(self._session_vwap_features(ts, h, low, c, v))
        feats.update(self._vwma_features(c, v))
        feats.update(self._momentum_features(c))
        feats.update(self._volatility_features(h, low, c))
        feats.update(self._volume_features(v))
        return pd.DataFrame(feats, index=df.index)[sorted(feats)]

    def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features plus the timestamp column - the persisted artifact shape."""
        feats = self.compute(df)
        feats.insert(0, TIMESTAMP_COL, df[TIMESTAMP_COL].to_numpy())
        return feats[[TIMESTAMP_COL, *ALL_FEATURES]]

    # ------------------------------------------------------------------
    def _price_features(
        self, o: pd.Series, h: pd.Series, low: pd.Series, c: pd.Series
    ) -> dict[str, pd.Series]:
        return {
            "ret_1": c.pct_change(fill_method=None),
            "log_ret_1": _no_inf(np.log(c / c.shift(1))),
            "body_pct": (c - o) / c,
            "upper_wick_pct": (h - np.maximum(o, c)) / c,
            # min(o,c) >= low on validated data, so the ratio is already >= 0
            "lower_wick_pct": (np.minimum(o, c) - low) / c,
            "range_pct": (h - low) / c,
        }

    def _trend_features(self, c: pd.Series) -> dict[str, pd.Series]:
        feats: dict[str, pd.Series] = {}
        for n in EMA_SPANS:
            ema = c.ewm(span=n, adjust=False, min_periods=n).mean()
            feats[f"ema_{n}"] = ema
            feats[f"ema_dist_{n}"] = c / ema - 1.0
        return feats

    def _session_vwap_features(
        self,
        ts: pd.Series,
        h: pd.Series,
        low: pd.Series,
        c: pd.Series,
        v: pd.Series,
    ) -> dict[str, pd.Series]:
        tp = (h + low + c) / 3.0
        session = ts.dt.normalize()  # UTC midnight => one session per calendar day
        cum_tp_v = (tp * v).groupby(session).cumsum()
        cum_v = v.groupby(session).cumsum()
        vwap = _no_inf(cum_tp_v / cum_v)
        return {"vwap_dist": c / vwap - 1.0}

    def _vwma_features(self, c: pd.Series, v: pd.Series) -> dict[str, pd.Series]:
        num = (c * v).rolling(VWMA_WINDOW).sum()
        den = v.rolling(VWMA_WINDOW).sum()
        vwma = _no_inf(num / den)
        return {"vwma_dist_20": c / vwma - 1.0}

    def _momentum_features(self, c: pd.Series) -> dict[str, pd.Series]:
        delta = c.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()

        rsi = pd.Series(np.nan, index=c.index, dtype="float64")
        has_vals = avg_gain.notna() & avg_loss.notna()
        both_zero = has_vals & (avg_gain == 0) & (avg_loss == 0)
        loss_zero = has_vals & (avg_loss == 0) & (avg_gain > 0)
        normal = has_vals & ~both_zero & ~loss_zero
        rsi[normal] = 100.0 - 100.0 / (1.0 + avg_gain[normal] / avg_loss[normal])
        rsi[loss_zero] = 100.0
        rsi[both_zero] = 50.0  # perfectly flat market carries no momentum signal
        return {"rsi_14": rsi}

    def _volatility_features(
        self, h: pd.Series, low: pd.Series, c: pd.Series
    ) -> dict[str, pd.Series]:
        prev_c = c.shift(1)
        tr = pd.concat(
            [h - low, (h - prev_c).abs(), (low - prev_c).abs()], axis=1
        ).max(axis=1)  # row 0 falls back to (h - l) since prev terms are NaN
        atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
        log_ret = np.log(c / c.shift(1))
        return {
            "atr_14": atr,
            "atr_pct": atr / c,
            "rv_20": log_ret.rolling(RV_WINDOW).std(),
        }

    def _volume_features(self, v: pd.Series) -> dict[str, pd.Series]:
        vol_ma = v.rolling(VOL_MA_WINDOW).mean()
        rel_vol = _no_inf(v / vol_ma)
        vol_change = _no_inf(v.pct_change(fill_method=None))
        return {"vol_change": vol_change, "vol_ma_20": vol_ma, "rel_vol_20": rel_vol}
