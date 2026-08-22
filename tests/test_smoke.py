"""Smoke tests: verify the skeleton is importable and configs are loadable."""

from __future__ import annotations

import importlib

import pytest

import darwin


def test_package_version() -> None:
    assert darwin.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "module_name",
    [
        "darwin.config",
        "darwin.data",
        "darwin.features",
        "darwin.environment",
        "darwin.agents",
        "darwin.evolution",
        "darwin.evaluation",
        "darwin.execution",
        "darwin.experiments",
        "darwin.visualization",
    ],
)
def test_subpackage_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("config_name", ["development", "backtest", "paper"])
def test_configs_load_and_have_required_keys(config_name: str) -> None:
    from darwin.config import load_config

    cfg = load_config(config_name)
    assert isinstance(cfg, dict)
    for key in ("mode", "seed", "data", "simulator"):
        assert key in cfg, f"config '{config_name}' missing required key '{key}'"
    assert cfg["data"]["symbol"] == "SOLUSDT"


def test_paper_config_cannot_trade_live() -> None:
    """Guardrail: the paper profile must never enable live trading."""
    from darwin.config import load_config

    cfg = load_config("paper")
    assert cfg["paper_trading"] is True
    assert cfg["live_trading"] is False
