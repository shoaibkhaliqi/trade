"""Risk engine: the non-negotiable layer between any actor and execution.

Design contracts:
- Actors propose; the risk manager DECIDES. It may veto an entry entirely
  (returning HOLD) or shrink it (returning an allowed size percentage).
- Auto-exits outrank proposals: stop-loss / take-profit / max-drawdown can
  escalate a HOLD into CLOSE. Triggers evaluate on candle CLOSE; the actual
  exit rides the normal next-open queue - one execution path for everything.
- The max-drawdown limit is a LATCHED emergency stop: once tripped it
  flattens and refuses every entry until the episode ends. Daily-loss and
  trade-count limits reset at UTC midnight. Cooldown decays with bars.
- CLOSE is NEVER vetoed: de-risking must always remain possible.

The manager is deliberately stateless about accounting - it reads a
RiskContext snapshot each decision and keeps only policy state (day buckets,
latch, position-transition memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from darwin.environment.simulator import Action


@dataclass(frozen=True)
class RiskConfig:
    max_position_size_pct: float = 50.0        # hard cap on entry notional (% equity)
    max_leverage: float = 1.0                  # notional/equity ceiling at entry
    max_risk_per_trade_pct: float | None = None  # max loss at stop-loss distance
    stop_loss_pct: float | None = None         # exit if close crosses entry*(1-s)
    take_profit_pct: float | None = None       # exit if close crosses entry*(1+t)
    max_daily_loss_pct: float | None = None    # block entries after -X% day pnl
    max_drawdown_pct: float | None = None      # LATCHED kill-switch
    cooldown_bars: int = 0                     # bars to wait after an exit
    max_trades_per_day: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.max_position_size_pct <= 100:
            msg = "max_position_size_pct must be in (0, 100]"
            raise ValueError(msg)
        if self.max_leverage <= 0:
            msg = "max_leverage must be positive"
            raise ValueError(msg)
        for name in (
            "max_risk_per_trade_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                msg = f"{name} must be positive when enabled"
                raise ValueError(msg)
        if self.cooldown_bars < 0:
            msg = "cooldown_bars must be >= 0"
            raise ValueError(msg)
        if self.max_trades_per_day is not None and self.max_trades_per_day < 1:
            msg = "max_trades_per_day must be >= 1"
            raise ValueError(msg)


@dataclass(frozen=True)
class RiskContext:
    """Immutable snapshot of everything the risk engine may look at."""

    timestamp: pd.Timestamp
    bar_index: int
    mark: float
    equity: float
    peak_equity: float
    position_qty: float
    entry_price: float
    total_trades: int            # completed round trips so far
    bars_since_exit: int | None  # None while a position is open


@dataclass
class RiskStats:
    entries_vetoed: int = 0
    entries_shrunk: int = 0
    auto_exits: dict[str, int] = field(default_factory=dict)
    kill_switch_tripped_at: pd.Timestamp | None = None


class RiskManager:
    """Applies RiskConfig to a stream of proposed actions."""

    def __init__(self, config: RiskConfig) -> None:
        self.cfg = config
        self.stats = RiskStats()
        self._killed = False
        self._day: date | None = None
        self._day_start_equity: float | None = None
        self._trades_at_day_start = 0

    # ------------------------------------------------------------------
    def _rollover_if_new_day(self, ctx: RiskContext) -> None:
        today = ctx.timestamp.date()
        if today != self._day:
            self._day = today
            self._day_start_equity = ctx.equity
            self._trades_at_day_start = ctx.total_trades

    def _daily_pnl_pct(self, ctx: RiskContext) -> float:
        assert self._day_start_equity is not None
        if self._day_start_equity <= 0:
            return 0.0
        return ctx.equity / self._day_start_equity - 1.0

    def _auto_exit(self, ctx: RiskContext) -> tuple[Action, str] | None:
        """Highest priority: forced exits. Returns (CLOSE, reason) or None."""
        cfg = self.cfg
        if not ctx.position_qty:
            return None
        dd = ctx.equity / ctx.peak_equity - 1.0
        if (
            cfg.max_drawdown_pct is not None
            and dd <= -cfg.max_drawdown_pct / 100.0
        ):
            return Action.CLOSE, "max_drawdown"
        long_pos = ctx.position_qty > 0
        if cfg.stop_loss_pct is not None:
            sl_px = ctx.entry_price * (
                1 - cfg.stop_loss_pct / 100.0
                if long_pos
                else 1 + cfg.stop_loss_pct / 100.0
            )
            breached = ctx.mark <= sl_px if long_pos else ctx.mark >= sl_px
            if breached:
                return Action.CLOSE, "stop_loss"
        if cfg.take_profit_pct is not None:
            tp_px = ctx.entry_price * (
                1 + cfg.take_profit_pct / 100.0
                if long_pos
                else 1 - cfg.take_profit_pct / 100.0
            )
            reached = ctx.mark >= tp_px if long_pos else ctx.mark <= tp_px
            if reached:
                return Action.CLOSE, "take_profit"
        return None

    def _allowed_entry_size_pct(
        self, base_size_pct: float, ctx: RiskContext, stop_distance_pct: float | None
    ) -> float:
        cap = min(base_size_pct, self.cfg.max_position_size_pct, self.cfg.max_leverage * 100.0)
        if (
            self.cfg.max_risk_per_trade_pct is not None
            and stop_distance_pct is not None
            and stop_distance_pct > 0
        ):
            # units: cfg is in PERCENT, stop_distance is a FRACTION - convert
            # first; the resulting size scales with the caller's base
            distance_pct = stop_distance_pct * 100.0
            cap = min(
                cap,
                self.cfg.max_risk_per_trade_pct / distance_pct * base_size_pct,
            )
        return max(cap, 0.0)

    # ------------------------------------------------------------------
    def apply(
        self,
        proposed: Action,
        ctx: RiskContext,
        *,
        base_size_pct: float = 100.0,
        stop_distance_pct: float | None = None,
    ) -> tuple[Action, float | None]:
        """Filter one decision. Returns (final_action, allowed_size_pct|None).

        ``base_size_pct`` is the caller's intended sizing (defaults to
        unclamped); it only matters for entry proposals.
        """
        self._rollover_if_new_day(ctx)

        # 1) forced exits always win
        forced = self._auto_exit(ctx)
        if forced is not None:
            action, reason = forced
            self.stats.auto_exits[reason] = self.stats.auto_exits.get(reason, 0) + 1
            if reason == "max_drawdown" and not self._killed:
                self._killed = True
                self.stats.kill_switch_tripped_at = ctx.timestamp
            return action, None

        # 2) CLOSE is never blocked (de-risking must stay possible)
        if proposed == Action.CLOSE:
            return Action.CLOSE, None

        # 3) kill-switch latch: no new exposure, ever
        if self._killed:
            if proposed in (Action.LONG, Action.SHORT):
                self.stats.entries_vetoed += 1
                return Action.HOLD, None
            return proposed, None

        # 4) entry throttles and reshaping
        if proposed in (Action.LONG, Action.SHORT):
            if (
                self.cfg.max_daily_loss_pct is not None
                and self._daily_pnl_pct(ctx) <= -self.cfg.max_daily_loss_pct / 100.0
            ):
                self.stats.entries_vetoed += 1
                return Action.HOLD, None
            if (
                self.cfg.cooldown_bars > 0
                and ctx.bars_since_exit is not None
                and ctx.bars_since_exit < self.cfg.cooldown_bars
            ):
                self.stats.entries_vetoed += 1
                return Action.HOLD, None
            if (
                self.cfg.max_trades_per_day is not None
                and ctx.total_trades - self._trades_at_day_start
                >= self.cfg.max_trades_per_day
            ):
                self.stats.entries_vetoed += 1
                return Action.HOLD, None

            size_pct = self._allowed_entry_size_pct(base_size_pct, ctx, stop_distance_pct)
            if size_pct < base_size_pct:
                self.stats.entries_shrunk += 1
            return proposed, size_pct

        return proposed, None

    @property
    def killed(self) -> bool:
        return self._killed
