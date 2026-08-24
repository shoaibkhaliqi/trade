"""Asian-session liquidity sweep strategy (community-validated mechanism).

Hypothesis (corroborated by three independent sources, 2026 search):
gold forms a range during the Asian session (00:00-06:00 UTC); during
London (07:00-11:00 UTC) price frequently SWEEPS one side of that range
(stop-loss liquidity) and then reverses. The mechanical rule:

  LONG : bar low pierces Asia low, then a later bar CLOSES back above the
         Asia low (reclaim) within the London window -> buy the reclaim
  SHORT: mirror with Asia high

Risk model (Auralis baseline-best variant): hard stop at the sweep extreme
minus/plus a small ATR buffer, fixed 2R target, first valid signal per day,
one trade at a time. Exits evaluated on closes (documented M6 limitation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from darwin.actions import Action
from darwin.environment.simulator import Action as SimAction

ASIA_START, ASIA_END = 0, 6      # UTC hours
LONDON_START, LONDON_END = 7, 11


class AsiaSweepStrategy:
    """Liquidity-sweep-and-reclaim off the Asian session range."""

    name = "asia_sweep"

    def __init__(
        self,
        tp_r_multiple: float = 2.0,
        atr_buffer_frac: float = 0.1,
        atr_period: int = 14,
    ) -> None:
        self.tp_r_multiple = tp_r_multiple
        self.atr_buffer_frac = atr_buffer_frac
        self.atr_period = atr_period

    def generate_actions(self, ohlcv: pd.DataFrame) -> list[SimAction]:
        n = len(ohlcv)
        actions: list[SimAction] = [Action.HOLD] * n
        if n < 100:
            return actions

        ts = ohlcv["timestamp"]
        hours = ts.dt.hour.to_numpy()
        close = ohlcv["close"].astype("float64").to_numpy()
        low = ohlcv["low"].astype("float64").to_numpy()
        high = ohlcv["high"].astype("float64").to_numpy()
        dates = ts.dt.date.to_numpy()

        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        atr = np.full(n, np.nan)
        atr[1:] = pd.Series(tr).ewm(alpha=1 / self.atr_period, adjust=False,
                                    min_periods=self.atr_period).mean().to_numpy()

        pos = 0
        entry_est = stop = target = np.nan
        swept_low = swept_high = np.nan
        asia_high = asia_low = np.nan
        current_day = None
        traded_today = False

        for t in range(n):
            day = dates[t]
            if day != current_day:
                current_day = day
                traded_today = False
                swept_low = swept_high = np.nan
                if hours[t] == ASIA_START:
                    asia_high = asia_low = np.nan

            in_asia = ASIA_START <= hours[t] < ASIA_END
            in_london = LONDON_START <= hours[t] < LONDON_END

            # manage open position: stop/target on closes
            if pos != 0:
                hit_stop = close[t] <= stop if pos > 0 else close[t] >= stop
                hit_tp = close[t] >= target if pos > 0 else close[t] <= target
                day_changed = day != dates[t - 1] if t > 0 else False
                if hit_stop or hit_tp or day_changed:
                    actions[t] = SimAction.CLOSE
                    pos = 0
                    entry_est = stop = target = np.nan
                continue

            # track the Asian range as it forms
            if in_asia:
                if np.isnan(asia_high):
                    asia_high, asia_low = high[t], low[t]
                else:
                    asia_high = max(asia_high, high[t])
                    asia_low = min(asia_low, low[t])
                continue

            if in_london and not traded_today and not np.isnan(asia_low):
                a = atr[t]
                if np.isnan(a) or a <= 0:
                    continue
                # sweep tracking
                if low[t] < asia_low:
                    swept_low = min(swept_low if not np.isnan(swept_low)
                                    else np.inf, low[t])
                if high[t] > asia_high:
                    swept_high = max(swept_high if not np.isnan(swept_high)
                                     else -np.inf, high[t])

                # reclaim entries
                long_signal = (not np.isnan(swept_low)
                               and close[t] > asia_low
                               and low[t] <= asia_low + a * self.atr_buffer_frac)
                short_signal = (not np.isnan(swept_high)
                                and close[t] < asia_high
                                and high[t] >= asia_high - a * self.atr_buffer_frac)

                if long_signal:
                    actions[t] = SimAction.LONG
                    pos, entry_est = 1, close[t]
                    stop = swept_low - a * self.atr_buffer_frac
                    target = entry_est + self.tp_r_multiple * (entry_est - stop)
                    traded_today = True
                elif short_signal:
                    actions[t] = SimAction.SHORT
                    pos, entry_est = -1, close[t]
                    stop = swept_high + a * self.atr_buffer_frac
                    target = entry_est - self.tp_r_multiple * (stop - entry_est)
                    traded_today = True

        return actions
