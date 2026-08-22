# AGENTS.md — Darwin Trading Lab

Instructions for AI agents (and humans) working in this repository.

## What This Is

An **experimental research system** evolving populations of crypto trading
agents (SOLUSDT first) through selection/reproduction/mutation. The user is
learning while building: every milestone must be explained, tested, and
confirmed before moving on.

**Status:** M0 complete (project setup). Next milestone: M1 (market data).
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
