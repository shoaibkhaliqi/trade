"""Event-driven trading simulator with next-open execution.

Execution contract (pairs with M2's feature contract):
- An action decided AFTER candle t closes is filled at candle t+1's OPEN,
  adversely adjusted by slippage, charged taker fee on fill notional.
- The queued-action design structurally forbids filling inside the deciding
  candle - look-ahead execution is impossible by construction, not by care.
- Accounting follows linear-USDT-perp semantics: opening moves only the fee;
  PnL settles against cash at close. Equity = cash + unrealized, identically,
  at every instant.
- Deterministic: no RNG anywhere. Identical inputs -> bit-identical outputs.

Two ways to drive it:
- ``run(candles, actions)`` - batch mode used by benchmarks/tests.
- ``prepare / submit / step / result`` - interactive stepping used by the
  Gym environment (M5+). Identical accounting, identical code paths.

Documented simplifications (revisited by later milestones):
- Equity is marked once per candle at its CLOSE (no intrabar path yet).
- Fills are permitted during zero-volume candles (open carries prior price);
  a policy layer may forbid this later - it is a modeling choice, made
  explicitly, never silently.
- Funding rates arrive with the risk engine (M6+).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from darwin.data.schema import OHLCV_COLUMNS, TIMESTAMP_COL

MIN_QTY = 1e-8


class Action(StrEnum):
    HOLD = "hold"
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"


@dataclass(frozen=True)
class SimulatorConfig:
    initial_capital: float = 1000.0
    taker_fee_pct: float = 0.055      # percent of fill notional, per side
    slippage_pct: float = 0.02        # percent adverse, per side
    position_size_pct: float = 25.0   # percent of decision-time equity
    qty_decimals: int = 8             # deterministic size rounding
    close_at_end: bool = True         # liquidate at final close for fair comparison

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            msg = "initial_capital must be positive"
            raise ValueError(msg)
        if self.taker_fee_pct < 0 or self.slippage_pct < 0:
            msg = "taker_fee_pct/slippage_pct must be non-negative"
            raise ValueError(msg)
        if not 0 < self.position_size_pct <= 100:
            msg = "position_size_pct must be in (0, 100]"
            raise ValueError(msg)


@dataclass(frozen=True)
class TradeRecord:
    trade_id: int
    direction: str  # "long" | "short"
    qty: float
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    gross_pnl: float
    fees_paid: float
    net_pnl: float
    bars_held: int


class Wallet:
    """Pure accounting state: cash plus one signed position."""

    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.qty = 0.0       # signed: >0 long, <0 short (units of base asset)
        self.entry = np.nan  # average entry price of the open position
        self.fees_paid = 0.0

    @property
    def has_position(self) -> bool:
        return self.qty != 0.0

    def unrealized(self, mark: float) -> float:
        return self.qty * (mark - self.entry) if self.has_position else 0.0

    def equity(self, mark: float) -> float:
        return self.cash + self.unrealized(mark)

    def open(self, direction: int, qty: float, price: float, fee_rate: float) -> float:
        """Open a fresh position from flat; returns the fee charged."""
        if self.has_position:
            msg = "cannot open: wallet already holds a position"
            raise RuntimeError(msg)
        if qty <= 0:
            msg = "quantity must be positive"
            raise ValueError(msg)
        fee = qty * price * fee_rate
        self.cash -= fee
        self.fees_paid += fee
        self.qty = direction * qty
        self.entry = price
        return fee

    def close(self, price: float, fee_rate: float) -> tuple[float, float]:
        """Flatten the position; returns (gross_pnl, fee)."""
        if not self.has_position:
            msg = "cannot close: wallet is flat"
            raise RuntimeError(msg)
        gross = self.qty * (price - self.entry)
        fee = abs(self.qty) * price * fee_rate
        self.cash += gross - fee
        self.fees_paid += fee
        self.qty = 0.0
        self.entry = np.nan
        return gross, fee


@dataclass
class SimResult:
    equity_curve: pd.DataFrame
    trades: tuple[TradeRecord, ...]
    n_unfilled_actions: int   # decisions on the final candle have no next open
    n_skipped_fills: int      # size rounded below MIN_QTY
    closed_at_end: bool
    config: SimulatorConfig

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve["equity"].iloc[-1])


def _buy_fill(open_px: float, slip: float) -> float:
    return open_px * (1.0 + slip)


def _sell_fill(open_px: float, slip: float) -> float:
    return open_px * (1.0 - slip)


class TradingSimulator:
    """Turns an OHLCV series plus an action stream into honest accounting."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.cfg = config or SimulatorConfig()

    # ------------------------------------------------------------------
    # batch API
    # ------------------------------------------------------------------
    def run(
        self,
        candles: pd.DataFrame,
        actions: Sequence[Action | str | None],
    ) -> SimResult:
        acts = self._prepare(candles, actions)
        n = len(candles)

        self._advance()  # candle 0: no decision could exist before it
        for i in range(1, n):
            self._pending = acts[i - 1]
            self._advance()
        if acts[n - 1] not in (Action.HOLD, None):
            self._n_unfilled += 1

        return self._finalize()

    # ------------------------------------------------------------------
    # stepping API (interactive consumers: Gym env, future live loop)
    # ------------------------------------------------------------------
    def prepare(self, candles: pd.DataFrame) -> dict[str, object]:
        """Validate data, reset internal state, process candle 0."""
        self._prepare(candles, None)
        return self._advance()

    def submit(self, action: Action | str | None) -> None:
        """Queue a decision made on the just-closed candle; fills next step."""
        self._pending = Action(action) if action is not None else Action.HOLD

    def step(self) -> dict[str, object]:
        """Advance one candle: fill any pending action at its open."""
        return self._advance()

    def result(self) -> SimResult:
        """Finalize the episode (applies close_at_end) and return artifacts."""
        return self._finalize()

    # ------------------------------------------------------------------
    # shared internals
    # ------------------------------------------------------------------
    def _prepare(
        self,
        candles: pd.DataFrame,
        actions: Sequence[Action | str | None] | None,
    ) -> list[Action]:
        missing = [c for c in OHLCV_COLUMNS if c not in candles.columns]
        if missing:
            msg = f"candles frame missing columns: {missing}"
            raise ValueError(msg)
        ts = candles[TIMESTAMP_COL]
        if not ts.is_monotonic_increasing:
            msg = "candles must be sorted by timestamp"
            raise ValueError(msg)

        self._scripted: list[Action] | None = (
            None
            if actions is None
            else [
                Action(a) if a is not None else Action.HOLD for a in actions
            ]
        )
        if self._scripted is not None and len(self._scripted) != len(candles):
            msg = (
                f"need one decision per candle: {len(self._scripted)} actions "
                f"for {len(candles)} candles"
            )
            raise ValueError(msg)

        cfg = self.cfg
        self._fee_rate = cfg.taker_fee_pct / 100.0
        self._slip = cfg.slippage_pct / 100.0
        self.wallet = Wallet(cfg.initial_capital)
        self._opens = candles["open"].to_numpy(dtype="float64")
        self._closes = candles["close"].to_numpy(dtype="float64")
        self._timestamps = ts.to_numpy()
        self._n = len(candles)

        self._trades: list[TradeRecord] = []
        self._curve_rows: list[dict[str, object]] = []
        self._pos_meta: dict[str, object] | None = None
        self._pending: Action | None = None
        self._decision_equity = cfg.initial_capital
        self._realized_cum = 0.0
        self._n_unfilled = 0
        self._n_skipped = 0
        self._next_trade_id = 0
        self._i = -1
        return self._scripted if self._scripted is not None else []

    def _advance(self) -> dict[str, object]:
        self._i += 1
        i = self._i
        if self._pending is not None and self._pending != Action.HOLD:
            self._execute(self._pending, self._opens[i], i)
        self._pending = None

        mark = self._closes[i]
        unreal = self.wallet.unrealized(mark)
        self._curve_rows.append(
            {
                TIMESTAMP_COL: self._timestamps[i],
                "cash": self.wallet.cash,
                "position_qty": self.wallet.qty,
                "entry_price": self.wallet.entry,
                "mark": mark,
                "unrealized": unreal,
                "realized_cum": self._realized_cum,
                "fees_cum": self.wallet.fees_paid,
                "equity": self.wallet.cash + unreal,
            }
        )
        equity = self.wallet.cash + unreal
        self._decision_equity = equity  # info available at close t
        return {
            "index": i,
            "timestamp": self._timestamps[i],
            "open": self._opens[i],
            "close": mark,
            "equity": equity,
            "cash": self.wallet.cash,
            "position_qty": self.wallet.qty,
        }

    def _execute(self, action: Action, raw_open: float, idx: int) -> None:
        if action == Action.CLOSE:
            if self.wallet.has_position:
                px = (
                    _sell_fill(raw_open, self._slip)
                    if self.wallet.qty > 0
                    else _buy_fill(raw_open, self._slip)
                )
                self._flatten(px, idx)
            return
        if action == Action.LONG:
            if self.wallet.qty < 0:  # flip: buy back short at the ask side
                px = _buy_fill(raw_open, self._slip)
                self._flatten(px, idx)
            if self.wallet.qty == 0:
                px = _buy_fill(raw_open, self._slip)
                notional = self._decision_equity * self.cfg.position_size_pct / 100.0
                qty = round(notional / px, self.cfg.qty_decimals)
                if qty < MIN_QTY:
                    self._n_skipped += 1
                    return
                entry_fee = self.wallet.open(direction=+1, qty=qty, price=px,
                                             fee_rate=self._fee_rate)
                self._pos_meta = {
                    "direction": "long", "qty": qty, "price": px,
                    "ts": self._timestamps[idx], "idx": idx, "entry_fee": entry_fee,
                }
            return
        if action == Action.SHORT:
            if self.wallet.qty > 0:  # flip: sell long at the bid side
                px = _sell_fill(raw_open, self._slip)
                self._flatten(px, idx)
            if self.wallet.qty == 0:
                px = _sell_fill(raw_open, self._slip)
                notional = self._decision_equity * self.cfg.position_size_pct / 100.0
                qty = round(notional / px, self.cfg.qty_decimals)
                if qty < MIN_QTY:
                    self._n_skipped += 1
                    return
                entry_fee = self.wallet.open(direction=-1, qty=qty, price=px,
                                             fee_rate=self._fee_rate)
                self._pos_meta = {
                    "direction": "short", "qty": qty, "price": px,
                    "ts": self._timestamps[idx], "idx": idx, "entry_fee": entry_fee,
                }
            return
        # Action.HOLD: nothing to do

    def _flatten(self, price: float, exit_idx: int) -> None:
        gross, fee = self.wallet.close(price, self._fee_rate)
        meta = self._pos_meta
        if meta is not None:
            total_fees = float(meta["entry_fee"]) + fee  # both legs belong to the trade
            self._trades.append(
                TradeRecord(
                    trade_id=len(self._trades),
                    direction=str(meta["direction"]),
                    qty=float(meta["qty"]),
                    entry_ts=meta["ts"],
                    entry_price=float(meta["price"]),
                    exit_ts=self._timestamps[exit_idx],
                    exit_price=price,
                    gross_pnl=gross,
                    fees_paid=total_fees,
                    net_pnl=gross - total_fees,
                    bars_held=exit_idx - int(meta["idx"]),
                )
            )

    def _finalize(self) -> SimResult:
        cfg = self.cfg
        closed_at_end = False
        if cfg.close_at_end and self.wallet.has_position and self._pos_meta is not None:
            px = (
                _sell_fill(self._closes[-1], self._slip)
                if self.wallet.qty > 0
                else _buy_fill(self._closes[-1], self._slip)
            )
            self._flatten(px, self._n - 1)
            self._pos_meta = None
            closed_at_end = True
            row = self._curve_rows[-1]
            row["cash"] = self.wallet.cash
            row["position_qty"] = 0.0
            row["entry_price"] = np.nan
            row["unrealized"] = 0.0
            row["equity"] = self.wallet.cash

        curve = pd.DataFrame(self._curve_rows)
        peak = curve["equity"].cummax()
        curve["drawdown"] = curve["equity"] / peak - 1.0

        return SimResult(
            equity_curve=curve,
            trades=tuple(self._trades),
            n_unfilled_actions=self._n_unfilled,
            n_skipped_fills=self._n_skipped,
            closed_at_end=closed_at_end,
            config=cfg,
        )
