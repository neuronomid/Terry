"""PNG chart generation for the isolated backtest API."""

from __future__ import annotations

import os
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_backtest_charts(equity_curve: list[dict], trades: list[dict],
                             output_root: str = "storage/backtest-charts") -> tuple[str, str]:
    session_id = str(uuid.uuid4())
    folder = os.path.abspath(os.path.join(output_root, session_id))
    os.makedirs(folder, exist_ok=True)
    values = np.asarray([point["value"] for point in equity_curve], dtype=float)
    times = pd.to_datetime([point["time"] for point in equity_curve], unit="ms", utc=True)
    if len(values) == 0:
        return session_id, folder
    peaks = np.maximum.accumulate(values)
    drawdown = np.divide(values - peaks, peaks, out=np.zeros_like(values), where=peaks != 0) * 100
    returns = pd.Series(values, index=times).resample("1D").last().pct_change().dropna()

    _line(times, values, "Portfolio Equity", "Equity", os.path.join(folder, "equity_curve.png"))
    _line(times, drawdown, "Drawdown", "Percent", os.path.join(folder, "drawdown.png"))
    _area(times, drawdown, "Underwater", os.path.join(folder, "underwater.png"))

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    _bars(monthly.index, monthly.values, "Monthly Returns", "Percent",
          os.path.join(folder, "monthly_returns_heatmap.png"))
    _hist(monthly.values, "Monthly Return Distribution", "Percent",
          os.path.join(folder, "monthly_distribution.png"))
    pnls = [trade.get("PNL", 0) for trade in trades]
    _hist(pnls, "Trade P&L Distribution", "P&L",
          os.path.join(folder, "trade_pnl.png"))
    return session_id, folder


def _line(x, y, title, ylabel, path):
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(x, y, linewidth=1.5)
    axis.set(title=title, ylabel=ylabel)
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _area(x, y, title, path):
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.fill_between(x, y, 0, alpha=.55, color="#d9534f")
    axis.set(title=title, ylabel="Percent")
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _bars(x, y, title, ylabel, path):
    fig, axis = plt.subplots(figsize=(10, 4.5))
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in y]
    axis.bar(x, y, width=20, color=colors)
    axis.set(title=title, ylabel=ylabel)
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _hist(values, title, xlabel, path):
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.hist(values if len(values) else [0], bins=min(30, max(5, len(values) // 2 or 5)), alpha=.8)
    axis.set(title=title, xlabel=xlabel, ylabel="Count")
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
