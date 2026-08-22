"""Configuration loading.

Configs are YAML files in <repo>/configs. Every experiment records which config
it used so runs stay reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by name (e.g. ``"development"``) and return it as a dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        msg = f"Unknown config '{name}'. Expected file: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
