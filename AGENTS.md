# AGENTS.md — Darwin Trading Lab

Instructions for AI agents (and humans) working in this repository.

## What This Is

An **experimental research system** evolving populations of trading agents
(PAXGUSDT tokenized-gold perpetual first) through selection/reproduction/mutation.
The user is learning while building: every milestone must be explained, tested,
and confirmed before moving on.

**Status:** M4 complete (benchmarks + metrics). Next milestone: M5 (single AI agent,
requires installing the `[ml]` extra with CUDA torch).
Full roadmap lives in the project spec; milestone table in `README.md`.

## Hard Rules

1. **One milestone at a time.** Never skip ahead. End each milestone with a
   summary block (WHAT WE BUILT / WHAT I LEARNED / TEST RESULTS / IMPORTANT
   FILES / WHAT COULD GO WRONG / NEXT MILESTONE) and wait for confirmation.
2. **No look-ahead bias.** Features at time *t* may use only data with
   timestamp ≤ *t*. Every new feature needs a shift/lag review.
3. **Never touch the test set for training or hyperparameter tuning.**
4. **Determinism.** Given the same seed + config + data, results must
   reproduce exactly.
5. **The risk manager is not bypassable** by any model or strategy.
6. **No real-money code paths** until the user explicitly approves M20.
   Paper trading (`configs/paper.yaml`) must keep `live_trading: false`.
7. **Never commit**: datasets (`data/**`), secrets/API keys, experiment DBs.
8. Tests and lint must be green before ending a session:
   ```powershell
   .venv\Scripts\python.exe -m pytest
   .venv\Scripts\python.exe -m ruff check .
   ```
9. Explain failures when they happen: diagnosis → cause → fix → rerun.
   Never silently patch.
10. New dependencies require justification and belong in `pyproject.toml`.

## Environment

- Python 3.12 venv at `.venv` (created via uv). Always use
  `.venv\Scripts\python.exe` — never the global interpreter.
- GPU: GTX 1660 Ti 6GB. ML stack (torch/gymnasium/SB3/optuna) is deferred to
  M5 via the `[ml]` extra; install then with CUDA-enabled torch if benchmarks
  justify it.
- Windows / PowerShell environment. Use `py -V:3.12` only for bootstrapping.

## Data Layer Facts (M1)

- Canonical schema: columns `timestamp,open,high,low,close,volume`; timestamp =
  candle OPEN time, tz-aware UTC ns; prices/volume float64; sorted unique.
  Normalize via `schema.to_canonical_timestamps()` — pandas 3 defaults to `us`
  resolution and Parquet round-trips at `us`, so never construct datetimes ad hoc.
- Primary pair: PAXGUSDT linear-perp (tokenized gold, 1 token = 1 troy oz),
  listed 2022-03-15; FULL history stored: 1m 2.33M / 5m 467k / 15m 156k /
  1h 38.9k candles. Its Bybit SPOT pair died 2025-03 - perp is the only feed.
- Legacy datasets also present: XAUUSDT perp (from 2026-03-09), SOLUSDT spot.
- Features: `data/features/{SYMBOL}_{tf}.parquet` via scripts/build_features.py;
  FeatureEngine v1 = 27 features (schema.py registry). Contract: row t uses
  only candles <= t (closed-candle convention); execution happens next-open
  from M3; warmup rows are NaN by design; guards: truncation-invariance +
  future-perturbation tests in test_features.py - new features MUST pass both.
- Zero-volume candles are REAL market facts (PAXG 1m: 41.6% of history,
  mostly 2022-24): they produce NaN vol_change/vwap_dist/rel_vol_20.
  Never fabricate fills; masking/dropping is an explicit downstream decision.
- Re-download: `.venv\Scripts\python.exe scripts\download_data.py`
- Explore: `.venv\Scripts\python.exe scripts\explore_data.py --timeframes 1h`
- Gaps are warnings, never silently filled; cleaning only drops impossible rows.

## Simulator Facts (M3)

- API: `TradingSimulator(cfg).run(ohlcv_df, actions) -> SimResult`
  (`equity_curve` df, `trades`, unfilled/skipped counters, config echo).
- Contract: `actions[i]` is decided AFTER candle i closes -> fills at candle
  i+1 OPEN; buys fill at open*(1+slip), sells at open*(1-slip); taker fee on
  fill notional, BOTH legs recorded per trade. Queue design makes look-ahead
  fills structurally impossible.
- Accounting: linear-USDT-perp style; opening moves only the fee;
  `equity == cash + unrealized` holds every row (tested).
- Sizing: fixed % of DECISION-time equity, qty rounded 8dp, fills below
  MIN_QTY=1e-8 skipped+counted. LONG/SHORT while positioned same-way = no-op;
  opposite = flip (close+open same fill event). close_at_end=True default
  liquidates at last CLOSE for fair comparison.
- KNOWN BEHAVIOR: fixed-at-entry sizing lets exposure drift with PnL
  (demo: 25% at entry became 49% at the Jan-2026 peak; -29.7% gold correction
  then cost -14.6% equity). M6 risk engine MUST bound drifting exposure.
- Zero-volume candles: fills currently allowed at carried-forward opens -
  documented assumption, revisit before paper trading.

## Benchmark Facts (M4)

- Strategies (`agents/strategies.py`) ALL call FeatureEngine for indicators -
  no duplicated math. Interface: `generate_actions(ohlcv) -> list[Action]`,
  one decision per candle, filled at next open by the simulator.
- `default_benchmarks()`: buy_and_hold, random(seed), ema_cross(5,20),
  rsi_reversion(30/70), vwap_reversion(0.004). Deterministic per parameters.
- MetricsReport (`evaluation/metrics.py`): total_return, max_drawdown, sharpe,
  sortino, profit_factor, win_rate, n_trades, fees_paid, avg_trade_net,
  exposure. Sharpe/Sortino report 0.0 when their denominator is zero
  (documented convention); profit_factor is NaN with no trades.
- EMPIRICAL BASELINE (PAXGUSDT 1h full history, development config): ONLY
  buy&hold is positive (+34%, one trade); every active strategy loses - fee
  drag dominates naive signals (random: -$672 fees on $1k capital). A future
  agent must beat buy_and_hold AFTER costs or it is noise.
- Comparison runner: scripts/run_benchmarks.py --timeframe X [--bars N]

## Conventions

- Layout: src-layout (`src/darwin/...`), package editable-installed.
- Configs are YAML in `configs/`; load via `darwin.config.load_config(name)`.
- Public functions get short docstrings; module docstrings state purpose.
- Type hints required on all new code; ruff (E,W,F,I,B,UP) enforces style.
- Experiments get unique IDs + seed + git commit recorded in SQLite (from M7).

## Definition of Done (per milestone)

- [ ] Code implemented for exactly one milestone scope
- [ ] New behavior covered by pytest tests
- [ ] `pytest` and `ruff` green
- [ ] Teaching summary delivered; user confirmed understanding
