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

    def run(
        self,
        candles: pd.DataFrame,
        actions: Sequence[Action | str | None],
    ) -> SimResult:
        missing = [c for c in OHLCV_COLUMNS if c not in candles.columns]
        if missing:
            msg = f"candles frame missing columns: {missing}"
            raise ValueError(msg)
        ts = candles[TIMESTAMP_COL]
        if not ts.is_monotonic_increasing:
            msg = "candles must be sorted by timestamp"
            raise ValueError(msg)
        if len(actions) != len(candles):
            msg = (
                f"need one decision per candle: {len(actions)} actions "
                f"for {len(candles)} candles"
            )
            raise ValueError(msg)
        acts = [
            Action(a) if a is not None else Action.HOLD
            for a in actions
        ]

        cfg = self.cfg
        fee_rate = cfg.taker_fee_pct / 100.0
        slip = cfg.slippage_pct / 100.0
        wallet = Wallet(cfg.initial_capital)

        opens = candles["open"].to_numpy(dtype="float64")
        closes = candles["close"].to_numpy(dtype="float64")
        timestamps = ts.to_numpy()
        n = len(candles)

        trades: list[TradeRecord] = []
        curve_rows: list[dict[str, float]] = []
        pos_meta: dict[str, object] | None = None  # tracks open leg for records
        pending: Action | None = None
        decision_equity = cfg.initial_capital  # equity known when deciding at t
        realized_cum = 0.0
        n_unfilled = 0
        n_skipped = 0
        next_trade_id = 0

        def _execute(action: Action, raw_open: float, idx: int) -> None:
            nonlocal next_trade_id, n_skipped, pos_meta
            if action == Action.CLOSE:
                if wallet.has_position:
                    px = (
                        _sell_fill(raw_open, slip)
                        if wallet.qty > 0
                        else _buy_fill(raw_open, slip)
                    )
                    self._flatten(wallet, px, fee_rate, pos_meta, trades, idx, timestamps)
                    pos_meta = None
                return
            if action == Action.LONG:
                if wallet.qty < 0:  # flip: buy back short at the ask side
                    px = _buy_fill(raw_open, slip)
                    self._flatten(wallet, px, fee_rate, pos_meta, trades, idx, timestamps)
                    pos_meta = None
                if wallet.qty == 0:
                    px = _buy_fill(raw_open, slip)
                    notional = decision_equity * cfg.position_size_pct / 100.0
                    qty = round(notional / px, cfg.qty_decimals)
                    if qty < MIN_QTY:
                        n_skipped += 1
                        return
                    entry_fee = wallet.open(direction=+1, qty=qty, price=px, fee_rate=fee_rate)
                    pos_meta = {
                        "direction": "long", "qty": qty, "price": px,
                        "ts": timestamps[idx], "idx": idx, "entry_fee": entry_fee,
                    }
                return
            if action == Action.SHORT:
                if wallet.qty > 0:  # flip: sell long at the bid side
                    px = _sell_fill(raw_open, slip)
                    self._flatten(wallet, px, fee_rate, pos_meta, trades, idx, timestamps)
                    pos_meta = None
                if wallet.qty == 0:
                    px = _sell_fill(raw_open, slip)
                    notional = decision_equity * cfg.position_size_pct / 100.0
                    qty = round(notional / px, cfg.qty_decimals)
                    if qty < MIN_QTY:
                        n_skipped += 1
                        return
                    entry_fee = wallet.open(direction=-1, qty=qty, price=px, fee_rate=fee_rate)
                    pos_meta = {
                        "direction": "short", "qty": qty, "price": px,
                        "ts": timestamps[idx], "idx": idx, "entry_fee": entry_fee,
                    }
                return
            # Action.HOLD: nothing

        closed_at_end = False
        for i in range(n):
            if pending is not None and pending != Action.HOLD:
                _execute(pending, opens[i], i)
            pending = None

            mark = closes[i]
            unreal = wallet.unrealized(mark)
            curve_rows.append(
                {
                    TIMESTAMP_COL: timestamps[i],
                    "cash": wallet.cash,
                    "position_qty": wallet.qty,
                    "entry_price": wallet.entry,
                    "mark": mark,
                    "unrealized": unreal,
                    "realized_cum": realized_cum,
                    "fees_cum": wallet.fees_paid,
                    "equity": wallet.cash + unreal,
                }
            )
            decision_equity = wallet.cash + unreal  # info available at close t

            if i < n - 1:
                pending = acts[i]
            elif acts[i] not in (Action.HOLD, None):
                n_unfilled += 1

        if (
            cfg.close_at_end
            and wallet.has_position
            and pos_meta is not None
        ):
            px = (
                _sell_fill(closes[-1], slip)
                if wallet.qty > 0
                else _buy_fill(closes[-1], slip)
            )
            self._flatten(wallet, px, fee_rate, pos_meta, trades, n - 1, timestamps)
            pos_meta = None
            closed_at_end = True
            curve_rows[-1]["cash"] = wallet.cash
            curve_rows[-1]["position_qty"] = 0.0
            curve_rows[-1]["entry_price"] = np.nan
            curve_rows[-1]["unrealized"] = 0.0
            curve_rows[-1]["realized_cum"] = realized_cum
            curve_rows[-1]["fees_cum"] = wallet.fees_paid
            curve_rows[-1]["equity"] = wallet.cash

        curve = pd.DataFrame(curve_rows)
        peak = curve["equity"].cummax()
        curve["drawdown"] = curve["equity"] / peak - 1.0

        return SimResult(
            equity_curve=curve,
            trades=tuple(trades),
            n_unfilled_actions=n_unfilled,
            n_skipped_fills=n_skipped,
            closed_at_end=closed_at_end,
            config=cfg,
        )

    @staticmethod
    def _flatten(
        wallet: Wallet,
        price: float,
        fee_rate: float,
        pos_meta: dict[str, object] | None,
        trades: list[TradeRecord],
        exit_idx: int,
        timestamps: np.ndarray,
    ) -> None:
        gross, fee = wallet.close(price, fee_rate)
        if pos_meta is not None:
            total_fees = float(pos_meta["entry_fee"]) + fee  # both legs belong to the trade
            trades.append(
                TradeRecord(
                    trade_id=len(trades),
                    direction=str(pos_meta["direction"]),
                    qty=float(pos_meta["qty"]),  # type: ignore[arg-type]
                    entry_ts=pos_meta["ts"],  # type: ignore[arg-type]
                    entry_price=float(pos_meta["price"]),  # type: ignore[arg-type]
                    exit_ts=timestamps[exit_idx],
                    exit_price=price,
                    gross_pnl=gross,
                    fees_paid=total_fees,
                    net_pnl=gross - total_fees,
                    bars_held=exit_idx - int(pos_meta["idx"]),  # type: ignore[arg-type]
                )
            )


def actions_from_labels(labels: Sequence[str]) -> list[Action]:
    """Convenience: ['hold','long',...] -> [Action.HOLD, ...]."""
    return [Action(lbl) for lbl in labels]
