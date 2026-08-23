# Darwin Trading Lab

An **experimental research system** that evolves populations of cryptocurrency
trading agents through selection, reproduction, and mutation.

> Research software. Not financial advice. Never connected to real money until
> every validation gate (M0–M18) has been passed explicitly.

## Core Idea

A population of autonomous agents receives market features and outputs trading
actions (`HOLD / LONG / SHORT / CLOSE`). Each agent is scored by a fitness
function emphasizing risk-adjusted return, drawdown, and consistency — never raw
profit alone. Top agents reproduce with small controlled mutations; weak agents
are archived. Over generations, robustness is selected for, not luck.

## Architecture

```text
Market Data → Feature Engine → Simulator ← Agent → Risk Manager → Executor → Metrics
                                    ▲                                        │
                                    └────────── Evolution Engine ◀───────────┘
```

## Milestones

| #  | Milestone              | Status |
|----|------------------------|--------|
| M0 | Project setup          | ✅     |
| M1 | Market data            | ✅     |
| M2 | Feature engine         | ✅     |
| M3 | Trading simulator      | ✅     |
| M4 | Benchmark strategies   | ✅     |
| M5 | Single AI agent        | ✅     |
| M6 | Risk engine            | ✅     |
| M7 | Proper validation      | ✅     |
| M8 | Agent genome           | ✅     |
| M9 | Population             | ✅     |
| M10 | Fitness               | ✅     |
| M11 | Death                 | ✅     |
| M12 | Reproduction          | ✅     |
| M13 | Mutation policy       | ⬜     |
| M14 | Generations loop      | ⬜     |
| M15 | Lineage tools         | ⬜     |
| M16+ | Regimes → live       | ⬜     |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"    # core + test tools
uv pip install --python .venv\Scripts\python.exe -e ".[ml]"     # torch stack (from M5)
```

## Common Commands

```powershell
.venv\Scripts\python.exe -m pytest                  # run tests
.venv\Scripts\python.exe -m ruff check .            # lint
.venv\Scripts\python.exe scripts\check_setup.py     # environment report
```

## Repository Layout

| Path                | Purpose                                              |
|---------------------|------------------------------------------------------|
| `configs/`          | YAML profiles: development, backtest, paper          |
| `data/raw`          | Downloaded exchange candles (never committed)        |
| `data/processed`    | Validated/cleaned Parquet datasets                   |
| `data/features`     | Computed feature matrices                            |
| `src/darwin/data`   | Downloader, validator, cleaner, storage (M1)         |
| `src/darwin/features` | Deterministic feature pipeline (M2)                |
| `src/darwin/environment` | Trading simulator & wallet accounting (M3)      |
| `src/darwin/agents` | Policies: benchmarks first, then neural agents       |
| `src/darwin/evolution` | Genomes, fitness, selection, mutation             |
| `src/darwin/evaluation` | Sharpe, Sortino, drawdown, profit factor          |
| `src/darwin/execution` | Risk manager + backtest/paper/live executors      |
| `src/darwin/experiments` | Experiment tracking in SQLite                   |
| `src/darwin/visualization` | Research plots                                |
| `tests/`            | pytest suite — every milestone must keep it green    |
| `experiments/`      | Run artifacts + metadata database                    |
| `scripts/`          | Utility entry points                                 |

## Non-Negotiable Rules

1. No look-ahead bias: features use only information available at time *t*.
2. Train/validation/test separation is sacred; never optimize on test.
3. Every experiment is logged with seed, config, and git commit.
4. The risk manager cannot be bypassed by any model.
5. Paper trading before live; live only after explicit human approval.
