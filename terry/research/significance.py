"""
Rule Significance Testing — bootstrap test for whether an entry rule has a genuine edge.

Method (ported from Jesse's research.rule_significance_testing):
  1. Signal-only backtest: record +1/-1/0 entry signal and close price at each bar.
  2. Log returns r_t = ln(P_t / P_{t-1}); detrend by subtracting the mean.
  3. rule_returns = signal * detrended  (neutral bars contribute 0).
  4. observed_mean = mean(rule_returns).
  5. Bootstrap: resample the zero-centred rule_returns N times, take each mean → null dist.
  6. p_value = fraction of simulated means >= observed_mean.  p<0.05 => significant edge.
"""
from datetime import datetime
import os

import numpy as np

from .backtest import backtest
from ._workers import parallel_results, resolve_workers

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 30


def rule_significance_test(config, routes, data_routes, candles, warmup_candles=None,
                           hyperparameters=None, n_simulations=2000, random_seed=None,
                           progress_bar=False, cpu_cores=None, progress_callback=None, *,
                           strategies_dir=None, strategy_classes=None, strategy_sources=None,
                           should_cancel=None):
    if len(routes) != 1:
        raise ValueError("rule_significance_test() requires exactly one trading route.")

    res = backtest(config, routes, data_routes or [], candles or {}, warmup_candles=warmup_candles,
                   hyperparameters=hyperparameters, strategies_dir=strategies_dir,
                   strategy_classes=strategy_classes, strategy_sources=strategy_sources,
                   signal_only=True, should_cancel=should_cancel)
    signal_log = res["signals"]
    if len(signal_log) < 3:
        raise ValueError("Not enough bars to run a significance test.")

    close_prices = np.array([s[1] for s in signal_log], dtype=float)
    signals = np.array([s[2] for s in signal_log], dtype=float)

    # drop non-finite
    mask = np.isfinite(close_prices) & np.isfinite(signals)
    close_prices, signals = close_prices[mask], signals[mask]

    log_returns = np.log(close_prices[1:] / close_prices[:-1])
    signals = signals[1:]
    valid = np.isfinite(log_returns)
    log_returns, signals = log_returns[valid], signals[valid]

    n_obs = len(log_returns)
    detrended = log_returns - log_returns.mean()
    rule_returns = signals * detrended
    observed_mean = float(rule_returns.mean()) if n_obs else 0.0

    resolved_seed = 42 if random_seed is None else random_seed
    sim_means = _bootstrap(rule_returns, observed_mean, n_simulations, resolved_seed,
                           progress_bar=progress_bar, progress_callback=progress_callback,
                           should_cancel=should_cancel, cpu_cores=cpu_cores)
    p_value = float(np.mean(sim_means >= observed_mean)) if len(sim_means) else 1.0

    return {
        "observed_mean": observed_mean,
        "annualized_return": observed_mean * TRADING_DAYS_PER_YEAR,
        "simulated_means": sim_means,
        "p_value": p_value,
        "n_simulations": int(len(sim_means)),
        "n_observations": int(n_obs),
        "significant": bool(p_value < 0.05),
        "verdict": _verdict(p_value),
        "min_observations_recommended": MIN_OBSERVATIONS,
        "enough_observations": bool(n_obs >= MIN_OBSERVATIONS),
    }


def _bootstrap(rule_returns, observed_mean, n_simulations, random_seed,
               progress_bar=False, progress_callback=None, should_cancel=None,
               cpu_cores=None):
    if len(rule_returns) == 0:
        return np.array([])
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")
    centered = rule_returns - observed_mean
    n = len(centered)
    # Bound each worker's index matrix and keep fixed chunk boundaries so a seed
    # produces identical simulations regardless of the requested worker count.
    chunk_size = max(1, min(n_simulations, 1024, 2_000_000 // n))
    starts = list(range(0, n_simulations, chunk_size))
    workers = resolve_workers(cpu_cores, len(starts))
    means = np.empty(n_simulations, dtype=float)
    progress = None
    if progress_bar:
        from tqdm import tqdm
        progress = tqdm(total=n_simulations, desc="Simulations (bootstrap)")
    try:
        def simulate(start):
            finish = min(start + chunk_size, n_simulations)
            rng = np.random.default_rng(
                np.random.SeedSequence([int(random_seed), start]))
            idx = rng.integers(0, n, size=(finish - start, n))
            return centered[idx].mean(axis=1)

        completed = 0
        for start, batch in parallel_results(
                simulate, starts, workers, should_cancel):
            finish = start + len(batch)
            means[start:finish] = batch
            completed += len(batch)
            if progress_callback:
                progress_callback(completed, n_simulations)
            if progress:
                progress.update(len(batch))
    finally:
        if progress:
            progress.close()
    return means


def _verdict(p):
    if p < 0.05:
        return "significant"       # genuine edge — proceed
    if p <= 0.10:
        return "borderline"        # inconclusive — flag to user
    return "not_significant"       # indistinguishable from random — hard stop


def plot_significance_test(result: dict, charts_folder: str = None,
                           theme: str = "light", dpi: int = 150,
                           show_title: bool = True) -> str:
    """Save Jesse's bootstrap-distribution chart and return its absolute path."""
    themes = {
        "light": {
            "figure": "white", "axes": "#f8f8f8", "text": "black",
            "grid": "#cccccc", "hist": "steelblue", "reject": "tomato",
            "line": "darkred", "box": "lightyellow",
        },
        "dark": {
            "figure": "#333333", "axes": "#2a2a2a", "text": "#e5e7eb",
            "grid": "#4a4a4a", "hist": "#3b82f6", "reject": "#ef4444",
            "line": "#fca5a5", "box": "#444444",
        },
    }
    colors = themes.get(theme, themes["light"])
    simulated = np.asarray(result["simulated_means"], dtype=float)
    if simulated.size == 0:
        raise ValueError("simulated_means cannot be empty")
    observed = float(result["observed_mean"])

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    figure.patch.set_facecolor(colors["figure"])
    axis.set_facecolor(colors["axes"])
    bins = min(50, max(20, len(simulated) // 10))
    _, edges, patches = axis.hist(
        simulated, bins=bins, color=colors["hist"], edgecolor=colors["figure"],
        linewidth=0.4, alpha=0.85, label="Simulated means (H₀)",
    )
    for patch, left_edge in zip(patches, edges[:-1]):
        if left_edge >= observed:
            patch.set_facecolor(colors["reject"])
            patch.set_alpha(0.9)
    axis.axvline(observed, color=colors["line"], linewidth=1.8, linestyle="--",
                 label=f"Observed mean = {observed:.6f}")
    info = (
        f"p-value = {float(result['p_value']):.4f}\n"
        f"Annualised return = {float(result['annualized_return']) * 100:.4f} %\n"
        f"Observations = {result['n_observations']} bars   |   "
        f"Simulations = {result['n_simulations']}"
    )
    axis.text(
        0.02, 0.97, info, transform=axis.transAxes, verticalalignment="top",
        fontsize=9, family="monospace", color=colors["text"],
        bbox={"boxstyle": "round,pad=0.4", "facecolor": colors["box"],
              "alpha": 0.85, "edgecolor": colors["grid"]},
    )
    if show_title:
        axis.set_title("Rule Significance Test — Bootstrap", color=colors["text"])
    axis.set_xlabel("Mean bar-level log return", color=colors["text"])
    axis.set_ylabel("Frequency", color=colors["text"])
    axis.tick_params(colors=colors["text"])
    axis.grid(True, color=colors["grid"], linewidth=0.5, alpha=0.5)
    legend = axis.legend(fontsize=9)
    for label in legend.get_texts():
        label.set_color(colors["text"])
    figure.tight_layout()

    folder = os.path.abspath(charts_folder or "charts")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"rule_significance_bootstrap_{datetime.now():%Y%m%d_%H%M%S}.png")
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=colors["figure"])
    plt.close(figure)
    print(f"Saved significance test chart to: {path}")
    return path


__all__ = ["rule_significance_test", "plot_significance_test"]
