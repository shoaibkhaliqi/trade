"""Hardened fitness baselines: beat the best PASSIVE directional strategy.

M17b's verification battery caught a fake edge: an always-SHORT agent was
crowned best-in-history by a falling selection window. The fix is at the
compass: the baseline return is max(buy&hold, naive-short) on the identical
window and sizing - positive fitness now requires beating the BEST passive
way to express ANY directional view, not merely one of them.
"""

from __future__ import annotations

import pandas as pd

from darwin.agents.strategies import AlwaysShortStrategy, BuyAndHoldStrategy
from darwin.environment.simulator import SimulatorConfig, TradingSimulator


def directional_baseline(
    window: pd.DataFrame,
    sim_cfg: SimulatorConfig,
) -> tuple[float, str]:
    """Best passive directional return on this window: max(long, short).

    Returns (return, which) where ``which`` names the winning passive side.
    Both naive strategies run through the same simulator/costs as agents.
    """
    results = {}
    for strategy in (BuyAndHoldStrategy(), AlwaysShortStrategy()):
        result = TradingSimulator(sim_cfg).run(
            window, strategy.generate_actions(window)
        )
        equity = result.equity_curve["equity"]
        results[strategy.name] = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    best_name = max(results, key=results.get)
    return results[best_name], best_name
