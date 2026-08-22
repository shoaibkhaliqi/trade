"""Prints a full environment report: Python, packages, GPU, project sanity."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys


def module_version(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "NOT INSTALLED"
    mod = importlib.import_module(name)
    return getattr(mod, "__version__", "unknown")


def main() -> None:
    print("=" * 60)
    print("Darwin Trading Lab - Environment Report")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]} ({platform.system()})")
    print(f"Executable  : {sys.executable}")

    print("\n-- Core libraries --")
    for name in ("numpy", "pandas", "pyarrow", "yaml", "matplotlib"):
        print(f"{name:<12}: {module_version(name)}")

    print("\n-- ML libraries (required from M5) --")
    for name in ("torch", "gymnasium", "stable_baselines3", "optuna"):
        print(f"{name:<12}: {module_version(name)}")

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            print(f"\nGPU          : {props.name} ({vram_gb:.1f} GB VRAM)")
            print(f"CUDA version : {torch.version.cuda}")
        else:
            print("\nGPU          : none visible to PyTorch (CPU-only training)")
    except ImportError:
        print("\nGPU          : PyTorch not installed - cannot probe")

    import darwin

    print(f"\ndarwin       : v{darwin.__version__} OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
