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


class GoldTrendPullbackStrategy:
    """H4 trend + H1 VWAP/EMA50 pullback with ATR stops (rules-based).

    Faithful implementation of the community 'H1/H4 trend-pullback' template
    within this lab's documented constraints:
    - H4 regime computed by resampling the incoming H1 candles (causal: an
      H4 bar is usable only after it closes, open+4h)
    - ADX(14) computed locally (Wilder), not part of the feature registry
    - exits: 1.5xATR initial stop, breakeven trail after +1R, EMA50 trail
      after +1R, 3R take-profit. PARTIAL profit-taking is not supported by
      the simulator (fixed sizing at entry) - the runner variant is tested.
    - 3 consecutive losses pause trading for the UTC day
    - spread/news filters omitted: no data
    """

    name = "gold_trend_pullback"

    def __init__(
        self,
        adx_threshold: float = 20.0,
        sl_atr_mult: float = 1.5,
        tp_r_multiple: float = 3.0,
        pullback_atr_frac: float = 0.5,
        max_consecutive_losses: int = 3,
    ) -> None:
        self.adx_threshold = adx_threshold
        self.sl_atr_mult = sl_atr_mult
        self.tp_r_multiple = tp_r_multiple
        self.pullback_atr_frac = pullback_atr_frac
        self.max_consecutive_losses = max_consecutive_losses

    # ------------------------------------------------------------------
    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0),
                            index=high.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0),
                             index=high.index)
        atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False,
                                      min_periods=period).mean() / atr
        minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False,
                                        min_periods=period).mean() / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _h4_regime(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Causal H4 EMA50/200 regime columns aligned onto the H1 index."""
        h4 = (
            ohlcv.set_index("timestamp")
            .resample("4h")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        close = h4["close"]
        ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
        # availability = H4 open + 4h (bar must have CLOSED)
        avail = pd.DataFrame({
            "available_at": h4["timestamp"] + pd.Timedelta("4h"),
            "h4_ema_dist_50": close / ema50 - 1.0,
            "h4_ema_dist_200": close / ema200 - 1.0,
        }).sort_values("available_at")

        aligned = pd.merge_asof(
            ohlcv[["timestamp"]].sort_values("timestamp"),
            avail,
            left_on="timestamp",
            right_on="available_at",
            direction="backward",
        )
        return aligned[["h4_ema_dist_50", "h4_ema_dist_200"]].set_index(ohlcv.index)

    # ------------------------------------------------------------------
    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        n = len(ohlcv)
        actions = [Action.HOLD] * n
        if n < 250:
            return actions  # H4 EMA200 needs 200 resampled bars

        close = ohlcv["close"].astype("float64")
        high = ohlcv["high"].astype("float64")
        low = ohlcv["low"].astype("float64")
        open_ = ohlcv["open"].astype("float64")

        ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        atr = (high - low).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        adx = self._adx(high, low, close)
        regime = self._h4_regime(ohlcv)
        body = close - open_

        # session VWAP (UTC daily reset) - the article's H1 directional filter
        tp = (high + low + close) / 3.0
        session = ohlcv["timestamp"].dt.normalize()
        cum_tp_v = (tp * ohlcv["volume"]).groupby(session).cumsum()
        cum_v = ohlcv["volume"].groupby(session).cumsum().replace(0.0, np.nan)
        vwap = cum_tp_v / cum_v

        # per-bar state
        pos = 0            # 0 flat, +1 long, -1 short
        entry_est = 0.0    # estimated entry (decision close; fill is next open)
        stop = np.nan
        consecutive_losses = 0
        day = None
        paused_until_next_day = False

        for t in range(n):
            ts = ohlcv["timestamp"].iloc[t]
            today = ts.date()
            if today != day:
                day = today
                paused_until_next_day = False

            c = close.iloc[t]
            a = atr.iloc[t]
            e50 = ema50.iloc[t]
            adx_t = adx.iloc[t]
            h4_50 = regime["h4_ema_dist_50"].iloc[t]
            h4_200 = regime["h4_ema_dist_200"].iloc[t]
            warm = any(pd.isna(x) for x in (a, e50, adx_t, h4_50, h4_200))

            # ---- manage open position first (exits fill next open) ----
            if pos != 0:
                stop_dist = self.sl_atr_mult * a
                moved = (c - entry_est) * pos
                r_multiple = moved / stop_dist if stop_dist > 0 else 0.0
                if r_multiple >= 1.0:
                    # breakeven trail once +1R reached
                    stop = max(stop, entry_est) if pos > 0 else min(stop, entry_est)
                tp_level = entry_est + pos * self.tp_r_multiple * stop_dist
                exit = False
                if pos > 0:
                    exit = (c <= stop
                            or (r_multiple >= 1.0 and c < e50)  # EMA trail
                            or c >= tp_level)                   # 3R target
                else:
                    exit = (c >= stop
                            or (r_multiple >= 1.0 and c > e50)
                            or c <= tp_level)
                if exit:
                    actions[t] = Action.CLOSE
                    was_loss = (c - entry_est) * pos < 0
                    consecutive_losses = consecutive_losses + 1 if was_loss else 0
                    if consecutive_losses >= self.max_consecutive_losses:
                        paused_until_next_day = True
                    pos = 0
                    stop = np.nan
                continue

            if paused_until_next_day or warm or pd.isna(c) or a <= 0:
                continue

            if adx_t < self.adx_threshold:
                continue

            h4_bull = h4_50 > h4_200   # H4 EMA50 above EMA200: bull regime
            h4_bear = h4_50 < h4_200
            vwap_ok_long = c > vwap.iloc[t] if not pd.isna(vwap.iloc[t]) else False
            vwap_ok_short = c < vwap.iloc[t] if not pd.isna(vwap.iloc[t]) else False
            # pullback: price within pullback_atr_frac * ATR of EMA50
            near_ema50 = abs(c - e50) <= self.pullback_atr_frac * a
            rng = max(high.iloc[t] - low.iloc[t], 1e-12)
            # rejection: close back in direction of the trend, wick into zone
            rejection_bull = (body.iloc[t] > 0
                              and low.iloc[t] <= e50 + 0.2 * a
                              and (close.iloc[t] - low.iloc[t]) >= 0.5 * rng)
            rejection_bear = (body.iloc[t] < 0
                              and high.iloc[t] >= e50 - 0.2 * a
                              and (high.iloc[t] - close.iloc[t]) >= 0.5 * rng)

            stop_dist = self.sl_atr_mult * a
            if h4_bull and vwap_ok_long and near_ema50 and rejection_bull:
                actions[t] = Action.LONG
                pos = 1
                entry_est = c
                stop = c - stop_dist
            elif h4_bear and vwap_ok_short and near_ema50 and rejection_bear:
                actions[t] = Action.SHORT
                pos = -1
                entry_est = c
                stop = c + stop_dist

        return actions


class VwapFlipStrategy:
    """Buy above session VWAP, sell below. Nothing else."""

    name = "vwap_flip"

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        n = len(ohlcv)
        actions = [Action.HOLD] * n
        if n < 2:
            return actions

        high = ohlcv["high"].astype("float64")
        low = ohlcv["low"].astype("float64")
        close = ohlcv["close"].astype("float64")
        volume = ohlcv["volume"].astype("float64")
        tp = (high + low + close) / 3.0
        session = ohlcv["timestamp"].dt.normalize()

        cum_pv = (tp * volume).groupby(session).cumsum()
        cum_v = volume.groupby(session).cumsum().replace(0.0, np.nan)
        vwap = cum_pv / cum_v

        for t in range(1, n):
            if np.isnan(vwap.iloc[t]):
                continue
            if close.iloc[t] > vwap.iloc[t]:
                actions[t] = Action.LONG
            elif close.iloc[t] < vwap.iloc[t]:
                actions[t] = Action.SHORT
        return actions


def default_benchmarks(seed: int = 42) -> list[Strategy]:
    return [
        BuyAndHoldStrategy(),
        AlwaysShortStrategy(),
        RandomTraderStrategy(seed=seed),
        MovingAverageCrossStrategy(),
        RSIMeanReversionStrategy(),
        VWAPMeanReversionStrategy(),
    ]
