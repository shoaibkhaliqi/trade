"""Research plots for market data exploration (matplotlib, Agg-friendly)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: must run before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from darwin.data.schema import expected_interval


def plot_overview(
    df: pd.DataFrame,
    *,
    timeframe: str,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Price + volume + returns in three aligned panels."""
    interval = expected_interval(timeframe)
    ts = df["timestamp"]
    returns = df["close"].pct_change()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1.5]},
    )

    axes[0].plot(ts, df["close"], lw=0.8, color="#1f77b4")
    axes[0].set_ylabel("close price (USDT)")
    axes[0].set_title(title)

    axes[1].bar(
        ts,
        df["volume"],
        width=interval.total_seconds() / 86_400,  # bar width in days on date axis
        color="#2ca02c",
    )
    axes[1].set_ylabel("volume (SOL)")

    axes[2].plot(ts, returns, lw=0.6, color="#d62728")
    axes[2].axhline(0.0, color="black", lw=0.5)
    axes[2].set_ylabel("simple return")
    axes[2].set_xlabel("time (UTC)")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    return fig


def plot_returns_histogram(
    df: pd.DataFrame,
    *,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Distribution of per-candle simple returns with summary stats annotated."""
    returns = df["close"].pct_change().dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(returns, bins=100, color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("per-candle return")
    ax.set_ylabel("frequency")
    stats = (
        f"mean={returns.mean():.6f}\nstd={returns.std():.4f}\n"
        f"skew={returns.skew():.2f}\nkurtosis={returns.kurt():.1f}"
    )
    ax.annotate(
        stats,
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        family="monospace",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    return fig
