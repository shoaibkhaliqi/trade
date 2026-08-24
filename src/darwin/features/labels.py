"""Triple-barrier labels for supervised price-direction models.

For each bar t (with volatility sigma_t known AT t):
  upper barrier = close_t * exp(+tp_sigma * sigma_t)
  lower barrier = close_t * exp(-sl_sigma * sigma_t)
  scan bars t+1 .. t+horizon:
    first close above upper  -> label 1
    first close below lower  -> label 0
    no breach                -> label = 1 if close_{t+H} > close_t else 0
  last ``horizon`` bars have no complete future -> NaN

Causality: sigma_t is a trailing rolling statistic, so every barrier is
knowable at decision time. Labels LOOK forward by construction - they are
training targets, never features, and the supervised hunt only ever trains
on rows whose labels completed BEFORE its test window (walk-forward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 16            # bars until the time barrier
    tp_sigma: float = 2.0        # upper barrier in units of rolling vol
    sl_sigma: float = 2.0        # lower barrier
    vol_window: int = 96         # trailing window for sigma

    def __post_init__(self) -> None:
        if self.horizon < 1:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        if self.tp_sigma <= 0 or self.sl_sigma <= 0:
            msg = "tp_sigma/sl_sigma must be positive"
            raise ValueError(msg)
        if self.vol_window < 2:
            msg = "vol_window must be >= 2"
            raise ValueError(msg)


def triple_barrier_labels(
    close: pd.Series,
    cfg: LabelConfig | None = None,
) -> pd.Series:
    """Binary first-barrier labels: 1 = upper first, 0 = lower/timeout-down.

    Index-aligned with ``close``; NaN where the label cannot resolve.
    """
    cfg = cfg or LabelConfig()
    c = close.astype("float64")
    log_ret = np.log(c / c.shift(1))
    sigma = log_ret.rolling(cfg.vol_window).std()
    values = c.to_numpy()
    sig = sigma.to_numpy()
    n = len(values)
    labels = np.full(n, np.nan)

    for t in range(n - cfg.horizon):
        s = sig[t]
        if np.isnan(s) or s <= 0:
            continue
        upper = values[t] * np.exp(cfg.tp_sigma * s)
        lower = values[t] * np.exp(-cfg.sl_sigma * s)
        window = values[t + 1 : t + 1 + cfg.horizon]
        up_hit = np.nonzero(window > upper)[0]
        down_hit = np.nonzero(window < lower)[0]
        if up_hit.size and (not down_hit.size or up_hit[0] < down_hit[0]):
            labels[t] = 1.0
        elif down_hit.size:
            labels[t] = 0.0
        else:
            labels[t] = 1.0 if window[-1] > values[t] else 0.0

    return pd.Series(labels, index=close.index, name="label")
