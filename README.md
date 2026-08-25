# Darwin Trading Lab

An **experimental research system** that evolves populations of cryptocurrency
trading agents through selection, reproduction, and mutation — then verifies
every candidate with a hardened, fake-edge-immune validation battery.

> Research software. Not financial advice. Never connected to real money until
> every validation gate (M0–M18) has been passed explicitly.

## What Was Found

After testing **15+ strategy families** across **10 assets**, **3 timeframes**,
and **2 ML paradigms** (RL + supervised), the lab identified:

| | SOL | ETH |
|---|---|---|
| **Return** | +130% | +53% |
| **Max DD** | −10.7% | −15.1% |
| **Profit factor** | 2.17 | 2.14 |
| **Sharpe** | 1.25 | 0.88 |
| **Stress tested** | 3× fees + 5× slippage → +102% | → +43% |
| **Monte Carlo P(loss)** | 0.2% | 3.3% |

**Strategy**: Ichimoku (9,26,52) trend signals filtered by a LightGBM
meta-model trained on 5 cross-asset-stable price-action features
(wick anatomy, volume dynamics, regime context).

**Key findings**: PAXG (tokenized gold) is untradeable with technical analysis.
Meta-labeling improves 7/9 assets. Parameter fragility is real. The tail test
remains unsolved. Forward validation is the next step.

## Architecture

```text
Market Data → Feature Engine → Simulator ← Agent → Risk Manager → Executor → Metrics
                                    ▲                                        │
                                    └────────── Evolution Engine ◀───────────┘
```

## Milestones

| #   | Milestone              | Status |
|-----|------------------------|--------|
| M0  | Project setup          | ✅     |
| M1  | Market data            | ✅     |
| M2  | Feature engine         | ✅     |
| M3  | Trading simulator      | ✅     |
| M4  | Benchmark strategies   | ✅     |
| M5  | Single AI agent        | ✅     |
| M6  | Risk engine            | ✅     |
| M7  | Proper validation      | ✅     |
| M8  | Agent genome           | ✅     |
| M9  | Population             | ✅     |
| M10 | Fitness                | ✅     |
| M11 | Death                  | ✅     |
| M12 | Reproduction           | ✅     |
| M13 | Mutation policy        | ✅     |
| M14 | Generations loop       | ✅     |
| M15 | Lineage tools          | ✅     |
| M16 | Market regimes         | ✅     |
| M17 | Stress testing         | ✅     |
| M18 | Paper trading          | ✅     |
| M19 | Dashboard              | ✅     |
| M20 | Live trading           | 🔒 requires explicit approval |

## Edge Hunt Program

| Phase | What was tested | Result |
|---|---|---|
| EH-v2 | Verification battery + hardened compass | Fake edge exposed and prevented |
| EH-v3 | LSTM, multi-TF features, SOL | No edge beyond SOL/ETH |
| EH-v4 | Supervised: LightGBM + triple-barrier | No edge after costs |
| EH-v5 | Meta-labeling, community strategies | Filter mechanism proven |
| EH-v6 | M5 data, logistic, feature selection, multi-asset | Feature selection promising |
| EH-v7 | 6 popular TradingView indicators | Ichimoku best on SOL |
| EH-v8 | Bybit Top 10 assets | 7/9 positive, 7/9 improved |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
uv pip install --python .venv\Scripts\python.exe -e ".[ml]"
```

## Common Commands

```powershell
# run all 259 tests
.venv\Scripts\python.exe -m pytest

# lint
.venv\Scripts\python.exe -m ruff check .

# launch dashboard (M19)
.venv\Scripts\python.exe -m streamlit run dashboard\app.py

# generate live paper-trading signal
.venv\Scripts\python.exe scripts\forward_tracker.py --symbol SOLUSDT

# run forward tracker for all assets (leave running)
.venv\Scripts\python.exe -u scripts\forward_runner.py

# download market data
.venv\Scripts\python.exe scripts\download_data.py
.venv\Scripts\python.exe scripts\download_derivatives.py

# run supervised hunt (60s per experiment)
.venv\Scripts\python.exe scripts\supervised_hunt.py --model lightgbm

# run meta-labeling hunt
.venv\Scripts\python.exe scripts\meta_hunt.py

# run RL population
.venv\Scripts\python.exe scripts\run_population.py --size 8 --timesteps 20000

# walk-forward benchmarks
.venv\Scripts\python.exe scripts\walk_forward.py --timeframe 15m

# indicator sweep
.venv\Scripts\python.exe experiments\indicator_sweep.py
```

## Repository Layout

| Path | Purpose |
|------|---------|
| `configs/` | YAML profiles: development, backtest, paper |
| `data/` | Market data (never committed) |
| `src/darwin/` | Core library (data, features, environment, agents, evolution, evaluation, execution, experiments, visualization) |
| `src/darwin/actions.py` | Shared Action enum (leaf module, no imports) |
| `src/darwin/features/labels.py` | Triple-barrier labeling |
| `src/darwin/features/multi_timeframe.py` | Causal HTF merge |
| `src/darwin/features/derivatives.py` | Funding/OI features |
| `src/darwin/evolution/` | Genome, fitness, survival, reproduction, diversity, lineage, behavior |
| `src/darwin/execution/risk.py` | Non-bypassable risk engine |
| `dashboard/app.py` | Streamlit dashboard (M19) |
| `scripts/` | CLI entry points (download, hunt, evolve, track) |
| `tests/` | 259 tests |
| `experiments/` | Research scripts + metadata DB |

## Non-Negotiable Rules

1. No look-ahead bias: features use only information available at time *t*.
2. Train/validation/test separation is sacred; never optimize on test.
3. Every experiment is logged with seed, config, and git commit.
4. The risk manager cannot be bypassed by any model.
5. Paper trading before live; live only after explicit human approval.
6. The fitness baseline is max(buy-and-hold, naive-short) — directional bias is not edge.
7. New features must pass truncation-invariance and future-perturbation tests.
