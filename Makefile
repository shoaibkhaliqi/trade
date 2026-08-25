# Linux / Copilot Workspace setup
# Windows users: see README.md for PowerShell commands

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e ".[dev,ml]"

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

dashboard:
	.venv/bin/python -m streamlit run dashboard/app.py

signal:
	.venv/bin/python scripts/forward_tracker.py --symbol SOLUSDT

download:
	.venv/bin/python scripts/download_data.py
	.venv/bin/python scripts/download_derivatives.py

hunt:
	.venv/bin/python scripts/supervised_hunt.py --model lightgbm

population:
	.venv/bin/python scripts/run_population.py --size 8 --timesteps 20000
