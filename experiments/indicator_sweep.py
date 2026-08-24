"""EH-v7: Systematic sweep of popular TradingView indicators.

Each indicator is implemented as a compact strategy using its most
selective variant (not raw crosses). All run through the same simulator
arena on PAXG/SOL/BTC 1h. Results reported in one comparison table.
"""
# ruff: noqa: E702
import numpy as np
import pandas as pd

from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport, format_row

# ====================================================================
# indicator strategies
# ====================================================================

def supertrend_strategy(ohlcv, period=10, mult=3.0):
    """Supertrend: trailing stop that flips on breach. Selective by design."""
    n = len(ohlcv)
    actions = [Action.HOLD] * n
    high = ohlcv["high"].astype("float64").to_numpy()
    low = ohlcv["low"].astype("float64").to_numpy()
    close = ohlcv["close"].astype("float64").to_numpy()
    hl2 = (high + low) / 2.0

    atr = np.full(n, np.nan)
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    tr[0] = high[0] - low[0]
    atr[period:] = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().to_numpy()[period:]

    direction = np.zeros(n)  # 1=up, -1=down
    st = np.full(n, np.nan)
    upper = np.full(n, np.nan); lower = np.full(n, np.nan)
    for t in range(period, n):
        if np.isnan(atr[t]):
            continue
        bu = hl2[t] + mult * atr[t]
        bl = hl2[t] - mult * atr[t]
        if t == period or np.isnan(st[t-1]):
            upper[t] = bu; lower[t] = bl; direction[t] = 1; st[t] = bl
            continue
        upper[t] = bu if (bu < upper[t-1] or close[t-1] > upper[t-1]) else upper[t-1]
        lower[t] = bl if (bl > lower[t-1] or close[t-1] < lower[t-1]) else lower[t-1]
        if direction[t-1] == 1:
            direction[t] = -1 if close[t] < lower[t] else 1
        else:
            direction[t] = 1 if close[t] > upper[t] else -1
        st[t] = lower[t] if direction[t] == 1 else upper[t]

    pos = 0
    for t in range(period, n):
        if np.isnan(direction[t]):
            continue
        if pos == 0 and direction[t] != 0 and direction[t] != direction[t-1]:
            actions[t] = Action.LONG if direction[t] == 1 else Action.SHORT
            pos = int(direction[t])
        elif pos != 0 and direction[t] != pos:
            actions[t] = Action.CLOSE
            pos = 0
            if direction[t] != 0:
                actions[t] = Action.CLOSE  # just close, re-enter next bar
    return actions


def macd_divergence_strategy(ohlcv, fast=12, slow=26, signal=9):
    """MACD histogram at extreme + reversal (not zero-cross)."""
    n = len(ohlcv)
    actions = [Action.HOLD] * n
    close = ohlcv["close"].astype("float64")
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    hist_prev = hist.shift(1)

    pos = 0
    for t in range(slow + signal, n):
        if pos != 0:
            # exit when histogram crosses zero against the position
            if pos > 0 and hist.iloc[t] < 0:
                actions[t] = Action.CLOSE; pos = 0
            elif pos < 0 and hist.iloc[t] > 0:
                actions[t] = Action.CLOSE; pos = 0
            continue
        # histogram turning from negative to less negative = bullish reversal
        if hist_prev.iloc[t] < hist_prev.iloc[t-1] and hist.iloc[t] > hist_prev.iloc[t] \
                and hist.iloc[t] < 0:
            actions[t] = Action.LONG; pos = 1
        elif hist_prev.iloc[t] > hist_prev.iloc[t-1] and hist.iloc[t] < hist_prev.iloc[t] \
                and hist.iloc[t] > 0:
            actions[t] = Action.SHORT; pos = -1
    return actions


def donchian_breakout_strategy(ohlcv, period=20, confirm_bars=2):
    """Donchian channel breakout with close confirmation."""
    n = len(ohlcv)
    actions = [Action.HOLD] * n
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")
    dc_upper = high.rolling(period).max()
    dc_lower = low.rolling(period).min()

    pos = 0
    bars_above = 0; bars_below = 0
    for t in range(period, n):
        if pos != 0:
            if pos > 0 and close.iloc[t] < dc_lower.iloc[t]:
                actions[t] = Action.CLOSE; pos = 0; bars_above = bars_below = 0
            elif pos < 0 and close.iloc[t] > dc_upper.iloc[t]:
                actions[t] = Action.CLOSE; pos = 0; bars_above = bars_below = 0
            continue
        if close.iloc[t] > dc_upper.iloc[t - 1]:
            bars_above += 1; bars_below = 0
        elif close.iloc[t] < dc_lower.iloc[t - 1]:
            bars_below += 1; bars_above = 0
        else:
            bars_above = bars_below = 0
        if bars_above >= confirm_bars and pos == 0:
            actions[t] = Action.LONG; pos = 1
        elif bars_below >= confirm_bars and pos == 0:
            actions[t] = Action.SHORT; pos = -1
    return actions


def stochastic_extreme_strategy(ohlcv, k_period=14, d_period=3,
                                 oversold=15, overbought=85):
    """Stochastic extreme reversal: only trade at extreme readings."""
    n = len(ohlcv)
    actions = [Action.HOLD] * n
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()

    pos = 0
    for t in range(k_period + d_period, n):
        if pos != 0:
            if pos > 0 and k.iloc[t] > 50:
                actions[t] = Action.CLOSE; pos = 0
            elif pos < 0 and k.iloc[t] < 50:
                actions[t] = Action.CLOSE; pos = 0
            continue
        # extreme oversold + K crosses above D
        if k.iloc[t-1] < oversold and k.iloc[t] > d.iloc[t] and k.iloc[t] < 30:
            actions[t] = Action.LONG; pos = 1
        elif k.iloc[t-1] > overbought and k.iloc[t] < d.iloc[t] and k.iloc[t] > 70:
            actions[t] = Action.SHORT; pos = -1
    return actions


def ichimoku_strategy(ohlcv, tenkan=9, kijun=26, senkou=52):
    """Ichimoku: price above cloud + tenkan > kijun = LONG, mirror for SHORT."""
    n = len(ohlcv)
    actions = [Action.HOLD] * n
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    close = ohlcv["close"].astype("float64")

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)

    pos = 0
    for t in range(senkou + kijun, n):
        above_cloud = close.iloc[t] > max(senkou_a.iloc[t], senkou_b.iloc[t])
        below_cloud = close.iloc[t] < min(senkou_a.iloc[t], senkou_b.iloc[t])
        tk_cross = tenkan_sen.iloc[t] > kijun_sen.iloc[t]

        if pos != 0:
            if pos > 0 and below_cloud:
                actions[t] = Action.CLOSE; pos = 0
            elif pos < 0 and above_cloud:
                actions[t] = Action.CLOSE; pos = 0
            continue
        if above_cloud and tk_cross:
            actions[t] = Action.LONG; pos = 1
        elif below_cloud and not tk_cross:
            actions[t] = Action.SHORT; pos = -1
    return actions


def hma_cross_strategy(ohlcv, fast=10, slow=30):
    """Hull MA cross: smoother than EMA, less lag than SMA."""
    n = len(ohlcv)
    actions = [Action.HOLD] * n
    close = ohlcv["close"].astype("float64")

    def hma(series, period):
        half = int(period / 2)
        wma_half = series.rolling(half).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
            raw=True)
        wma_full = series.rolling(period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
            raw=True)
        raw = 2 * wma_half - wma_full
        return raw.rolling(int(np.sqrt(period))).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
            raw=True)

    hma_f = hma(close, fast)
    hma_s = hma(close, slow)
    pos = 0
    for t in range(slow + int(np.sqrt(slow)), n):
        if pos != 0:
            if pos > 0 and hma_f.iloc[t] < hma_s.iloc[t]:
                actions[t] = Action.CLOSE; pos = 0
            elif pos < 0 and hma_f.iloc[t] > hma_s.iloc[t]:
                actions[t] = Action.CLOSE; pos = 0
            continue
        if hma_f.iloc[t-1] <= hma_s.iloc[t-1] and hma_f.iloc[t] > hma_s.iloc[t]:
            actions[t] = Action.LONG; pos = 1
        elif hma_f.iloc[t-1] >= hma_s.iloc[t-1] and hma_f.iloc[t] < hma_s.iloc[t]:
            actions[t] = Action.SHORT; pos = -1
    return actions


# ====================================================================
# batch runner
# ====================================================================

STRATEGIES = {
    "supertrend": supertrend_strategy,
    "macd_divergence": macd_divergence_strategy,
    "donchian_breakout": donchian_breakout_strategy,
    "stoch_extreme": stochastic_extreme_strategy,
    "ichimoku": ichimoku_strategy,
    "hma_cross": hma_cross_strategy,
}


def main():
    cfg = load_config("development")
    sim_cfg = SimulatorConfig(
        initial_capital=1000.0, taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
        slippage_pct=cfg["simulator"]["slippage_pct"],
        position_size_pct=cfg["simulator"]["fixed_position_size_pct"])
    src = DataStorage("data/processed")

    datasets = {}
    for sym in ("PAXGUSDT", "SOLUSDT", "BTCUSDT"):
        datasets[sym] = src.load(sym, "1h")

    results = []
    for sym, ohlcv in datasets.items():
        print(f"\n{'='*50}")
        print(f"  {sym} 1h")
        print(f"{'='*50}")
        print(format_row("strategy", MetricsReport.from_parts(
            pd.DataFrame({"equity": [1000.0], "position_qty": [0.0],
                          "fees_cum": [0.0]}, index=[0]), (), "1h")))

        for name, fn in STRATEGIES.items():
            try:
                actions = fn(ohlcv)
                res = TradingSimulator(sim_cfg).run(ohlcv, actions)
                rep = MetricsReport.from_result(res, "1h")
                results.append((sym, name, rep))
                marker = " ***" if rep.total_return > 0 else ""
                print(format_row(name, rep) +
                      f"  t={rep.n_trades} WR={rep.win_rate:.0%}{marker}")
            except Exception as e:
                print(f"  {name}: ERROR {e}")

    # summary
    print(f"\n{'='*60}")
    print("SUMMARY (sorted by return)")
    print(f"{'symbol':<10}{'strategy':<22}{'return':>9}{'DD':>8}{'PF':>7}{'trades':>8}")
    for sym, name, rep in sorted(results, key=lambda x: x[2].total_return, reverse=True):
        marker = " <<<" if rep.total_return > 0 and rep.profit_factor > 1.2 else ""
        print(f"{sym:<10}{name:<22}{rep.total_return:>9.2%}"
              f"{rep.max_drawdown:>8.2%}{rep.profit_factor:>7.2f}"
              f"{rep.n_trades:>8d}{marker}")


if __name__ == "__main__":
    main()
