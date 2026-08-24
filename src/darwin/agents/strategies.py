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


class T3Strategy:
    """Tillson T3 moving average strategies.

    T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3, where e1..e6 are cascaded EMAs
    and the coefficients are determined by the volume factor (default 0.7).

    Three modes:
    - slope : LONG when T3 rising + price above, SHORT when falling + below
    - cross : fast T3 crossing slow T3
    - bounce: price pulls back to T3 in trend direction then bounces
    """

    def __init__(
        self,
        mode: str = "slope",
        period: int = 14,
        volume_factor: float = 0.7,
        fast_period: int = 8,
        slow_period: int = 21,
        slope_lookback: int = 3,
        min_dist_pct: float = 0.0,
    ) -> None:
        self.mode = mode
        self.period = period
        self.volume_factor = volume_factor
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.slope_lookback = slope_lookback
        self.min_dist_pct = min_dist_pct
        self.name = f"t3_{mode}"

    @staticmethod
    def t3(close: pd.Series, period: int, volume_factor: float) -> pd.Series:
        e1 = close.ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        e3 = e2.ewm(span=period, adjust=False).mean()
        e4 = e3.ewm(span=period, adjust=False).mean()
        e5 = e4.ewm(span=period, adjust=False).mean()
        e6 = e5.ewm(span=period, adjust=False).mean()
        a = volume_factor
        c1 = -(a ** 3)
        c2 = 3 * a ** 2 + a ** 3
        c3 = -3 * a - 3 * a ** 2
        c4 = 1 + 3 * a + a ** 3
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        n = len(ohlcv)
        actions = [Action.HOLD] * n
        if n < max(self.slow_period, self.period * 3, 50):
            return actions

        close = ohlcv["close"].astype("float64")
        t3_fast = self.t3(close, self.period, self.volume_factor)
        t3_slow = self.t3(close, self.slow_period, self.volume_factor) \
            if self.mode == "cross" else None

        for t in range(1, n):
            c = close.iloc[t]
            if self.mode == "slope":
                if t < self.slope_lookback:
                    continue
                t3_now = t3_fast.iloc[t]
                t3_prev = t3_fast.iloc[t - self.slope_lookback]
                rising = t3_now > t3_prev
                falling = t3_now < t3_prev
                dist_ok = (self.min_dist_pct == 0
                           or abs(c / t3_now - 1) >= self.min_dist_pct)
                if rising and c > t3_now and dist_ok:
                    actions[t] = Action.LONG
                elif falling and c < t3_now and dist_ok:
                    actions[t] = Action.SHORT

            elif self.mode == "cross":
                if t < self.slow_period:
                    continue
                f_now = t3_fast.iloc[t]
                s_now = t3_slow.iloc[t]
                f_prev = t3_fast.iloc[t - 1]
                s_prev = t3_slow.iloc[t - 1]
                if f_prev <= s_prev and f_now > s_now:
                    actions[t] = Action.LONG
                elif f_prev >= s_prev and f_now < s_now:
                    actions[t] = Action.SHORT

            elif self.mode == "bounce":
                if t < self.period * 2:
                    continue
                t3_now = t3_fast.iloc[t]
                prev_c = close.iloc[t - 1]
                prev_t3 = t3_fast.iloc[t - 1]
                # touched T3 then bounced back in trend direction
                if (prev_c <= prev_t3 and c > t3_now
                        and c > prev_c and t3_now > prev_t3):
                    actions[t] = Action.LONG
                elif (prev_c >= prev_t3 and c < t3_now
                      and c < prev_c and t3_now < prev_t3):
                    actions[t] = Action.SHORT

        return actions


class T3SqueezeMomentumStrategy:
    """T3 regime + Squeeze Momentum timing.

    T3 answers: which direction is the trend?
    Squeeze answers: when to enter? (volatility compression -> expansion)

    LONG: T3 rising + squeeze releases (BB exits KC) + momentum positive
    SHORT: T3 falling + squeeze releases + momentum negative
    Exit: momentum crosses against the position, or price crosses T3
    """

    name = "t3_squeeze"

    def __init__(
        self,
        t3_period: int = 14,
        volume_factor: float = 0.7,
        squeeze_length: int = 20,
        bb_mult: float = 2.0,
        kc_mult: float = 1.5,
        min_squeeze_bars: int = 6,
        momentum_lookback: int = 20,
    ) -> None:
        self.t3_period = t3_period
        self.volume_factor = volume_factor
        self.squeeze_length = squeeze_length
        self.bb_mult = bb_mult
        self.kc_mult = kc_mult
        self.min_squeeze_bars = min_squeeze_bars
        self.momentum_lookback = momentum_lookback

    @staticmethod
    def t3(close: pd.Series, period: int, volume_factor: float) -> pd.Series:
        e1 = close.ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        e3 = e2.ewm(span=period, adjust=False).mean()
        e4 = e3.ewm(span=period, adjust=False).mean()
        e5 = e4.ewm(span=period, adjust=False).mean()
        e6 = e5.ewm(span=period, adjust=False).mean()
        a = volume_factor
        return (-(a**3) * e6 + (3 * a**2 + a**3) * e5
                + (-3 * a - 3 * a**2) * e4 + (1 + 3 * a + a**3) * e3)

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[Action]:
        n = len(ohlcv)
        actions = [Action.HOLD] * n
        if n < self.squeeze_length * 3:
            return actions

        high = ohlcv["high"].astype("float64")
        low = ohlcv["low"].astype("float64")
        close = ohlcv["close"].astype("float64")
        L = self.squeeze_length

        # --- T3 trend ---
        t3 = self.t3(close, self.t3_period, self.volume_factor)
        t3_slope = t3.diff(self.momentum_lookback)

        # --- Bollinger Bands ---
        bb_basis = close.rolling(L).mean()
        bb_dev = close.rolling(L).std()
        bb_lower = bb_basis - self.bb_mult * bb_dev
        bb_upper = bb_basis + self.bb_mult * bb_dev

        # --- Keltner Channels ---
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        kc_basis = close.ewm(span=L, adjust=False).mean()
        kc_range = tr.ewm(span=L, adjust=False).mean()
        kc_lower = kc_basis - self.kc_mult * kc_range
        kc_upper = kc_basis + self.kc_mult * kc_range

        # --- Squeeze: BB inside KC ---
        squeeze_on = (bb_lower >= kc_lower) & (bb_upper <= kc_upper)

        # --- Momentum (LazyBear simplified) ---
        highest = high.rolling(L).max()
        lowest = low.rolling(L).min()
        avg_hl = (highest + lowest) / 2.0
        avg_sma = close.rolling(L).mean()
        momentum_raw = close - (avg_hl + avg_sma) / 2.0
        # smooth with a short EMA to reduce noise
        momentum = momentum_raw.ewm(span=5, adjust=False).mean()

        # --- Squeeze duration counter ---
        squeeze_duration = np.zeros(n)
        for t in range(1, n):
            if squeeze_on.iloc[t]:
                squeeze_duration[t] = squeeze_duration[t - 1] + 1
            else:
                squeeze_duration[t] = 0

        # --- State machine ---
        pos = 0
        for t in range(1, n):
            if pos != 0:
                # exit: momentum crosses against, or price crosses T3 against
                if pos > 0 and (momentum.iloc[t] < 0 or close.iloc[t] < t3.iloc[t]):
                    actions[t] = Action.CLOSE
                    pos = 0
                elif pos < 0 and (momentum.iloc[t] > 0 or close.iloc[t] > t3.iloc[t]):
                    actions[t] = Action.CLOSE
                    pos = 0
                continue

            if t < L * 2 or pd.isna(t3_slope.iloc[t]) or pd.isna(momentum.iloc[t]):
                continue

            # squeeze release: was on for min bars, now off
            just_released = (
                not squeeze_on.iloc[t]
                and squeeze_duration[t - 1] >= self.min_squeeze_bars
            )
            if not just_released:
                continue

            t3_up = t3_slope.iloc[t] > 0
            t3_down = t3_slope.iloc[t] < 0
            mom_up = momentum.iloc[t] > 0
            mom_down = momentum.iloc[t] < 0

            if t3_up and mom_up:
                actions[t] = Action.LONG
                pos = 1
            elif t3_down and mom_down:
                actions[t] = Action.SHORT
                pos = -1

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
