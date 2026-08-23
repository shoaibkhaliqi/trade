"""Shared domain vocabulary - dependency-free leaf module.

`Action` lives here (not inside environment) so that execution/risk and
environment/simulator can both import it without creating an import cycle
between the two packages. Everything re-exports it; this definition is law.
"""

from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    HOLD = "hold"
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
