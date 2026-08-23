"""Survival criteria: ALIVE / WEAK / DEAD verdicts for evaluated agents.

Policy (all thresholds are protocol constants, recorded with experiments):
- DEAD checks run first - catastrophe dominates nuance:
    * max_drawdown >= dead_drawdown      (capital catastrophe)
    * fitness      <= dead_fitness       (hopeless under the compass)
    * never traded AND paralysis_is_death  (starvation, opt-in harshness)
- WEAK: below the soft fitness floor or beyond the comfort drawdown.
- ALIVE: everything else.
- Every verdict carries explicit machine-readable reasons; an unexplained
  death teaches nothing.
- Death is a STATUS, never a deletion: the agents row keeps its full history
  and a deaths-table certificate records why it ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STATUS_ALIVE = "alive"
STATUS_WEAK = "weak"
STATUS_DEAD = "dead"


@dataclass(frozen=True)
class SurvivalConfig:
    dead_fitness: float = -2.0
    dead_drawdown: float = 0.20      # |max_drawdown| fraction
    weak_fitness: float = -0.5
    weak_drawdown: float = 0.10
    min_trades: int = 1              # activity reference for paralysis
    paralysis_is_death: bool = False

    def __post_init__(self) -> None:
        if self.dead_fitness >= self.weak_fitness:
            msg = "dead_fitness must be < weak_fitness"
            raise ValueError(msg)
        if self.dead_drawdown <= self.weak_drawdown:
            msg = "dead_drawdown must be > weak_drawdown"
            raise ValueError(msg)
        if self.min_trades < 0:
            msg = "min_trades must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class Verdict:
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate_survival(
    metrics: dict,
    fitness_total: float,
    cfg: SurvivalConfig,
) -> Verdict:
    """Classify one evaluated agent. Ordered checks; reasons accumulate."""
    dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    n_trades = int(metrics.get("n_trades", 0) or 0)

    dead_reasons: list[str] = []
    if dd >= cfg.dead_drawdown:
        dead_reasons.append(f"drawdown {dd:.2%} >= death threshold {cfg.dead_drawdown:.0%}")
    if fitness_total <= cfg.dead_fitness:
        dead_reasons.append(
            f"fitness {fitness_total:+.3f} <= death floor {cfg.dead_fitness:+.3f}"
        )
    if cfg.paralysis_is_death and n_trades < cfg.min_trades:
        dead_reasons.append(f"paralysis: {n_trades} trades < {cfg.min_trades}")
    if dead_reasons:
        return Verdict(STATUS_DEAD, tuple(dead_reasons))

    weak_reasons: list[str] = []
    if dd >= cfg.weak_drawdown:
        weak_reasons.append(f"drawdown {dd:.2%} >= weak threshold {cfg.weak_drawdown:.0%}")
    if fitness_total <= cfg.weak_fitness:
        weak_reasons.append(
            f"fitness {fitness_total:+.3f} <= weak floor {cfg.weak_fitness:+.3f}"
        )
    if weak_reasons:
        return Verdict(STATUS_WEAK, tuple(weak_reasons))

    return Verdict(STATUS_ALIVE, ())
