"""Feature registry: names, windows, and version for reproducibility.

Every experiment records FEATURE_VERSION alongside data lineage so a result
can always be traced back to the exact feature definition that produced it.
Changing any window or formula here MUST bump FEATURE_VERSION.
"""

from __future__ import annotations

FEATURE_VERSION = "1"

EMA_SPANS: tuple[int, ...] = (5, 9, 20, 50, 100, 200)
VWMA_WINDOW = 20
RSI_PERIOD = 14
ATR_PERIOD = 14
RV_WINDOW = 20
VOL_MA_WINDOW = 20

PRICE_FEATURES = [
    "ret_1",
    "log_ret_1",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_pct",
]
TREND_FEATURES = [f"ema_{n}" for n in EMA_SPANS] + [
    f"ema_dist_{n}" for n in EMA_SPANS
]
SESSION_VWAP_FEATURES = ["vwap_dist"]
VWMA_FEATURES = ["vwma_dist_20"]
MOMENTUM_FEATURES = ["rsi_14"]
VOLATILITY_FEATURES = ["atr_14", "atr_pct", "rv_20"]
VOLUME_FEATURES = ["vol_change", "vol_ma_20", "rel_vol_20"]

ALL_FEATURES: tuple[str, ...] = tuple(
    PRICE_FEATURES
    + TREND_FEATURES
    + SESSION_VWAP_FEATURES
    + VWMA_FEATURES
    + MOMENTUM_FEATURES
    + VOLATILITY_FEATURES
    + VOLUME_FEATURES
)
