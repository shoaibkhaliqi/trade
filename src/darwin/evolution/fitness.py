"""Fitness functions: the compass that decides which agents deserve to breed.

Design laws:
1. NO single metric may dominate: every component is clipped to [-1, 1]
   before weighting, so fitness lives in a bounded band.
2. OPPORTUNITY COST IS MANDATORY: the return term is BASELINE-RELATIVE
   (agent return minus buy&hold on the identical window). Sitting flat while
   gold rises must score negative - otherwise evolution crowns paralysis
   (M9 measured: 5 of 8 agents collapsed to never-trade at low budgets).
3. Full transparency: every score returns a breakdown of components so any
   ranking can be audited component by component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(max(x, lo), hi))


@dataclass(frozen=True)
class FitnessConfig:
    """Weights and normalizers. Weights are non-negative; terms 4-6 subtract."""

    w_return: float = 1.0
    w_sharpe: float = 0.5
    w_sortino: float = 0.25
    w_drawdown: float = 0.75
    w_fees: float = 0.25
    w_overtrade: float = 0.25

    baseline_return: float = 0.0   # buy&hold on the SAME scoring window
    target_return: float = 0.05    # excess return that maps to +1.0
    dd_tolerance: float = 0.10     # drawdown that maps to -1.0
    fee_tolerance: float = 0.05    # fees as fraction of capital mapping to -1.0
    trade_cap: int = 200           # trades per scoring window before penalty

    def __post_init__(self) -> None:
        for name in (
            "w_return", "w_sharpe", "w_sortino",
            "w_drawdown", "w_fees", "w_overtrade",
        ):
            if getattr(self, name) < 0:
                msg = f"{name} must be non-negative"
                raise ValueError(msg)
        if self.target_return <= 0 or self.dd_tolerance <= 0 or self.fee_tolerance <= 0:
            msg = "tolerances must be positive"
            raise ValueError(msg)
        if self.trade_cap < 1:
            msg = "trade_cap must be >= 1"
            raise ValueError(msg)


def preset(name: str, **overrides: Any) -> FitnessConfig:
    """Named fitness compasses used by experiments and comparisons."""
    presets: dict[str, dict[str, Any]] = {
        # the spec's six-term formula
        "spec": {},
        # single-metric strawmen, kept to DEMONSTRATE their failure modes
        "pure_return": {
            "w_sharpe": 0.0, "w_sortino": 0.0,
            "w_drawdown": 0.0, "w_fees": 0.0, "w_overtrade": 0.0,
        },
        # risk-adjusted-only: famously crowns paralysis - see tests
        "risk_parity": {
            "w_return": 0.0, "w_sortino": 0.0,
            "w_drawdown": 0.0, "w_fees": 0.0, "w_overtrade": 0.0,
        },
        # capital preservation first
        "conservative": {
            "w_return": 0.5, "w_sharpe": 0.5, "w_sortino": 0.25,
            "w_drawdown": 2.0, "w_fees": 0.5, "w_overtrade": 0.5,
        },
    }
    if name not in presets:
        known = ", ".join(sorted(presets))
        msg = f"unknown fitness preset '{name}'. Known: {known}"
        raise ValueError(msg)
    return FitnessConfig(**{**presets[name], **overrides})


@dataclass(frozen=True)
class FitnessBreakdown:
    total: float
    components: dict[str, float] = field(default_factory=dict)


def compute_fitness(
    metrics: dict[str, Any],
    cfg: FitnessConfig,
) -> FitnessBreakdown:
    """Score one agent's MetricsReport-shaped dict under the given compass."""
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    sortino = float(metrics.get("sortino", 0.0) or 0.0)
    max_dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    fees = float(metrics.get("fees_paid", 0.0) or 0.0)
    n_trades = int(metrics.get("n_trades", 0) or 0)
    capital = float(metrics.get("initial_capital_proxy", 1000.0))

    c_return = _clip((total_return - cfg.baseline_return) / cfg.target_return)
    c_sharpe = _clip(sharpe / 3.0)
    c_sortino = _clip(sortino / 3.0)
    c_drawdown = _clip(max_dd / cfg.dd_tolerance, 0.0, 1.0)
    c_fees = _clip((fees / capital) / cfg.fee_tolerance, 0.0, 1.0)
    c_overtrade = _clip(max(0, n_trades - cfg.trade_cap) / cfg.trade_cap, 0.0, 1.0)

    total = (
        cfg.w_return * c_return
        + cfg.w_sharpe * c_sharpe
        + cfg.w_sortino * c_sortino
        - cfg.w_drawdown * c_drawdown
        - cfg.w_fees * c_fees
        - cfg.w_overtrade * c_overtrade
    )
    return FitnessBreakdown(
        total=float(total),
        components={
            "return": c_return,
            "sharpe": c_sharpe,
            "sortino": c_sortino,
            "drawdown_pen": c_drawdown,
            "fees_pen": c_fees,
            "overtrade_pen": c_overtrade,
        },
    )
