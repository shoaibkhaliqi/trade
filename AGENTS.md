# AGENTS.md — Darwin Trading Lab

Instructions for AI agents (and humans) working in this repository.

## What This Is

An **experimental research system** evolving populations of trading agents
(PAXGUSDT tokenized-gold perpetual first) through selection/reproduction/mutation.
The user is learning while building: every milestone must be explained, tested,
and confirmed before moving on.

**Status:** M14 complete (generational loop automation, history table,
evolution charts). Next milestone: M15 (lineage tools).

## Generational Loop Facts (M14)

- `evolution/generation.py`: cohort_rows / cohort_stats (per-generation
  cohort semantics; even-count median averages the middle pair) +
  run_generation (evaluate pending via INJECTED evaluator -> persist ->
  stats -> diversity -> record -> select FROM COHORT -> breed g+1).
  Evaluator contract: (report, metrics, verdict); the ORCHESTRATOR owns
  persistence - fakes stay pure, production's internal write is an
  idempotent repeat.
- STALENESS GOTCHA (test-caught): re-fetch cohort rows after evaluation -
  the pre-eval snapshot carries metrics=None.
- Selection breeds from the just-evaluated cohort (generational replacement);
  the roster keeps every ancestor; hall-of-fame = roster query by fitness.
- `generations` table: best/median/worst fitness, best_return, mean_dd,
  alive/weak/dead counts, diversity mean/min, immigrants.
- scripts/evolve.py: backfill_history (pre-table cohorts), N-cycle loop
  targeting the oldest pending cohort, charts via
  visualization/evolution.py (fitness curves + diversity track with the
  0.05 convergence line), HOF print, experiment kind='evolve'.
- First live run (gen 2 -> 5, 16 agents, ~5min): machinery stable; gens 3-4
  collapsed toward the flat cluster (median=-0.719 signature) while
  diversity stayed healthy (0.30-0.40); HOF unchanged (gen-1 +0.296).
  5k-step budgets + noisy fitness = convergence pressure is REAL; longer
  budgets and/or stronger immigration are the levers.


## Diversity Facts (M13)

- `evolution/diversity.py`: genome_distance = mean per-gene |delta|/span
  (normalized so every gene counts equally); population_diversity ->
  DiversityReport(n_unique, mean/min/max pairwise, per-gene normalized std).
  converging iff n_unique<=1 or mean<0.05; formatter flags near-converged
  genes (std<0.02).
- ReproductionConfig policy knobs: intensity_decay (per-generation multiplier
  floored by min_intensity via effective_intensity) +
  immigrants_per_generation (random genomes, parent=None, persisted like
  siblings, counted in reproduce() output).
- MATH GOTCHA (test-pinned): mean-pairwise distance is NOT monotonic under
  additions (centroid dilution); coverage (max pairwise) is - assert the
  guaranteed property, not the hoped-for one.
- Live roster: 12 genomes mean 0.309 all unique; gen-2 breed demo produced
  18 genomes unique=17 with min_pairwise=0.000 - the monitor caught a
  duplicate (zero-mutation child) on its first run. That tripwire is the
  deliverable.
- scripts/diversity_report.py: CLI over the stored roster.


## Reproduction Facts (M12)

- `evolution/reproduction.py`: select_parents (DEAD excluded from the gene
  pool; alive/weak ranked by stored fitness, ties stable, unevaluated
  ineligible) + reproduce (offspring per rank tier from
  ReproductionConfig.offspring_per_rank; fresh agent ids/seeds; generation+1;
  child genomes persisted BEFORE training). Config validates tiers/rates.
- Children flow through THE SAME funnel as founders:
  `Population.evaluate_agent` (train -> score -> fitness -> survival ->
  persist). Selection never compares apples to oranges.
- Weight inheritance ("parent knowledge") is opt-in: `--inherit-weights` /
  `training.init_from_model_path` loads the parent's trained PPO, overrides
  the child's mutated learning genes, continues training. Default OFF for
  generation-1 exploration honesty.
- Legacy-roster gotcha: founders predate fitness storage; `fitness_report
  --apply` backfills fitness via `tracker.merge_agent_metrics`. Selection
  reads the ROSTER - computed-on-the-fly scores do not count.
- FIRST EVOLUTION STEP (gen0 -> gen1, 3 parents, 4 children, 5k steps):
  best child fitness +0.296 (ALIVE) vs best founder -0.417 - first
  positive-compass agent in project history. Anecdote, not proof; the
  machinery is the deliverable. Children train ~14s each on CPU.
- scripts/breed_generation.py: prints parents, per-mutation birth records,
  child evaluations, generation leaderboard; records experiment kind='breed'.


## Survival Facts (M11)

- `evolution/survival.py`: `evaluate_survival(metrics, fitness_total,
  SurvivalConfig)` -> Verdict(status, reasons). Ordered checks: DEAD
  (dd>=0.20 | fitness<=-2.0 | opt-in paralysis) > WEAK (dd>=0.10 |
  fitness<=-0.5) > ALIVE. Thresholds inclusive; reasons machine-readable.
- Death is a STATUS, never a deletion: the agents row keeps its full metrics
  history; `deaths` table adds a certificate (reason, fitness, dd, timestamp)
  via `tracker.mark_agent_status`. `get_agents(status=...)` filters the
  active roster; `get_deaths()` lists certificates.
- Default policy: paralysis (0 trades) is WEAK, not DEAD - the fitness
  compass already taxes it via baseline-relative return;
  `paralysis_is_death=True` escalates for harsh regimes (visible config).
- run_population prints per-agent verdicts (+/~/x) and persists them;
  fitness_report --apply re-audits stored rosters and writes verdicts.
- First real audit (M9 8-agent roster): 1 alive / 6 weak / 1 dead
  (2026-002: fitness -2.07 <= death floor -2.0).


## Fitness Facts (M10)

- `evolution/fitness.py`: `compute_fitness(metrics_dict, FitnessConfig)` ->
  FitnessBreakdown(total, components). Every component clipped to [-1,1];
  fitness bounded by sum of weights. Full component breakdown always returned
  (auditable rankings).
- THE ANTI-PARALYSIS LAW: return term is BASELINE-RELATIVE (agent return -
  buy&hold on identical score window). M9 measured 5-of-8 agents collapsing
  to never-trade; risk-adjusted-only scoring would crown paralysis
  (test-pinned: risk_parity strawman ranks flat above mild losers; the
  conservative preset let the flat cluster sweep the podium in the live demo).
- Presets: spec (default), pure_return + risk_parity (strawmen kept to
  demonstrate failure modes), conservative (dd-heavy). `--fitness` flag on
  run_population; fitness stored in agent metrics_json; leaderboard sorts by it.
- Live demo (fitness_report.py, 8-agent roster, window b&h=+3.59%): spec
  ranks the only real trader first; conservative ranks the FLAT CLUSTER
  1st-5th and the profitable agent 6th. Fitness IS the objective - choose it
  like one.
- Baseline is computed per-run on the exact score window (window b&h can
  differ a lot from full-test b&h: +3.59% vs +0.84% on M9 data).


## Population Facts (M9)

- `Population(size, db).initialize(master_seed)`: deterministic roster of
  (genome, seed) pairs; genomes+agents persisted BEFORE evaluation; roster
  ids reproducible from master_seed; `record_agent` is idempotent
  (ON CONFLICT DO NOTHING) so re-runs RESUME - run_population skips
  already-evaluated agents.
- `training.train_and_evaluate(..., score_window_bars=N)` scores TEST on the
  first N test rows (population-scale); None = full slice. Same arena for
  every agent: identical slices/costs/risk-baseline/timesteps.
- FIRST RUN (8 agents, 5k steps, 5k-bar score window, ~39s/agent):
  5 of 8 collapsed to NEVER-TRADE (0 trades, exactly 0.00%); traders spread
  [+0.93%, -2.82%]. The flat-policy attractor is real at low training budgets
  - M10 fitness MUST handle it (a naive risk-adjusted ranking would crown
  paralysis; minimum-activity or baseline-relative scoring needed).
- scripts/run_population.py: progress prints, DB-read leaderboard (roster is
  truth, not memory), dispersion summary, experiment row kind='population'.


## Genome Facts (M8)

- 8 consequential genes in `evolution/genome.py` GENE_SPECS: behavioral
  (position_size_pct 5-50, stop_loss_pct 0.5-8, take_profit_pct 0.75-15,
  cooldown_bars 0-24 int, max_trades_per_day 2-80 int) + learning
  (learning_rate 1e-4..3e-3 LOG-SCALE, ent_coef 0..0.03, gamma 0.90..0.999).
  Every gene binds real behavior - no cosmetic genes allowed.
- Mutation: per-gene Bernoulli(rate) then gaussian; sigma = intensity*span
  (log-space sigma for learning_rate => multiplicative moves); clamped to
  bounds; integer genes round. Child carries MutationRecord(gene, old, new,
  sigma) tuple + parent_id + generation.
- max_drawdown_pct (kill-switch) is NOT a gene - protocol constant; evolution
  must never breed its way out of the emergency brake (test-pinned).
- Persistence: `genomes` table + `lineage_of()` parent walk (tracker.py);
  `training.risk_config_from_genome` overlays genes onto yaml RiskConfig;
  `ppo_kwargs_from_genome` wires learning genes into PPO.
- Import-cycle lesson: `Action` moved to leaf module `darwin/actions.py`;
  execution and environment both import it - cycle structurally impossible.


## Validation Facts (M7)

- `experiments/splits.py`: chronological_split + walk_forward_splits
  (rolling origin, EMBARGO gap between train end and test start; with
  step==test_bars the test windows tile the tail contiguously - geometry
  is hand-computed in test_splits.py).
- Walk-forward protocol (scripts/walk_forward.py): strategy sees
  train+embargo+test slice so indicators warm on pre-test data only;
  decisions before the last pre-test bar are forced HOLD; metrics scored on
  the out-of-sample equity portion only. Action lists are TARGET-STATE series
  - BuyAndHold emits persistent LONG (fold sampling reads intent at any bar).
- Seed variance (scripts/seed_sweep.py, reusable core in
  experiments/training.py): identical configs across seeds spread
  [-1.44%, +1.22%] TEST return at 20k steps - single-seed results are noise.
  Seeds 7 and 123 converged to IDENTICAL policies (same attractor).
- Empirical walk-forward baseline (PAXGUSDT 15m, 25 folds x 5000 bars):
  b&h +0.92%/fold std 1.89% (sharpe_mu 1.36); random -30.9%; ema_cross
  -16.4%; rsi/vwap ~-1.6%. CSV per-fold details under experiments/.
- scripts/list_experiments.py prints metadata.sqlite history (handles both
  train_agent and aggregate sweep rows).


## Risk Engine Facts (M6)

- `RiskManager(RiskConfig).apply(proposed, ctx, base_size_pct=100.0,
  stop_distance_pct=None) -> (Action, size_pct|None)`. Wired INSIDE
  `TradingEnv.step`, so no policy can submit an unfiltered action; benchmark
  strategies intentionally run raw (they are reference traders, not models).
- Priority: auto-exits (max_drawdown LATCHED kill-switch > stop_loss >
  take_profit; triggers read CLOSE, fills ride next-open queue) > CLOSE always
  allowed > latch vetoes entries forever > daily-loss / cooldown /
  max-trades-per-day veto > size/leverage/risk-per-trade clamps shrink via
  `simulator.submit(action, size_pct=...)` (effective = min(cfg, allowed)).
- UNITS GOTCHA (load-bearing test pins it): config fields are PERCENT;
  `stop_distance_pct` is a FRACTION. A wrong-unit bug here fails OPEN
  (permissive), which is the dangerous direction - test asserts 1% risk @2%
  stop => 50% size.
- Config keys under yaml `risk:` map 1:1 to RiskConfig fields across
  development/backtest/paper profiles.
- Real-data proof (M5 seed42 agent on TEST): unguarded -0.92%/dd -7.41% vs
  guarded -1.44%/dd -6.49%, 10 stop-loss exits, 22 vetoed entries. Protection
  costs fees and buys bounded tails - insurance, not alpha.


## RL Environment Facts (M5)

- `TradingEnv(candles, features, config, start_idx, end_idx)` wraps the
  stepping simulator. Obs = 27 features (NaN->0 post-warmup) +
  [position_sign, unrealized/equity, drawdown] float32. Action Discrete(4)
  -> Action via `action_to_signal`. Reward = log(equity ratio); final step
  includes close_at_end liquidation costs; equity<=0 => -10 guard.
- Simulator has public stepping API (`prepare/submit/step/result`) -
  batch `run()` is a thin wrapper over identical internals (102 tests
  stayed green through that refactor).
- Protocol: chronological 70/15/15 TRAIN/VAL/TEST. Training sees TRAIN only;
  EvalCallback uses a recent-VAL proxy window (~3000 rows); TEST touched ONCE
  at the end by agent AND benchmarks under identical costs.
- Training script: scripts/train_agent.py -> saves SB3 zip under
  experiments/runs/, logs experiment id to experiments/metadata.sqlite.
- FIRST RESULT (seed 42, 40960 steps): agent = "hold something long" clone of
  buy&hold, slightly worse (-0.92% vs +0.84% on TEST). Verdict per our bar:
  NOISE, not skill. Also noted: SB3 warns MLP-PPO trains FASTER on CPU
  (~175 fps here); revisit device choice per experiment.

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
- GPU: GTX 1660 Ti 6GB. ML stack INSTALLED: torch 2.13.0+cu130 (local wheel,
  sha256-verified), gymnasium 1.3, stable-baselines3 2.9, optuna 4.9.
  CUDA confirmed working (device visible, matmul OK).
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
