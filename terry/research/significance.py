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
import numpy as np

from .backtest import backtest

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 30


def rule_significance_test(config, routes, data_routes=None, candles=None, warmup_candles=None,
                           hyperparameters=None, n_simulations=2000, random_seed=42,
                           strategies_dir=None, strategy_classes=None, strategy_sources=None,
                           progress_callback=None, should_cancel=None):
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

    sim_means = _bootstrap(rule_returns, observed_mean, n_simulations, random_seed,
                           progress_callback=progress_callback, should_cancel=should_cancel)
    p_value = float(np.mean(sim_means >= observed_mean)) if len(sim_means) else 1.0

    return {
        "observed_mean": observed_mean,
        "annualized_return": observed_mean * TRADING_DAYS_PER_YEAR,
        "p_value": p_value,
        "n_simulations": int(len(sim_means)),
        "n_observations": int(n_obs),
        "significant": bool(p_value < 0.05),
        "verdict": _verdict(p_value),
        "min_observations_recommended": MIN_OBSERVATIONS,
        "enough_observations": bool(n_obs >= MIN_OBSERVATIONS),
    }


def _bootstrap(rule_returns, observed_mean, n_simulations, random_seed,
               progress_callback=None, should_cancel=None):
    if len(rule_returns) == 0:
        return np.array([])
    centered = rule_returns - observed_mean
    rng = np.random.default_rng(random_seed)
    n = len(centered)
    # Bound the temporary index matrix to roughly 16 MB. A single allocation of
    # (n_simulations, n_observations) can otherwise exhaust memory on 1m tests.
    chunk_size = max(1, min(n_simulations, 2_000_000 // n))
    means = np.empty(n_simulations, dtype=float)
    for start in range(0, n_simulations, chunk_size):
        if should_cancel and should_cancel():
            raise InterruptedError("Research run canceled")
        finish = min(start + chunk_size, n_simulations)
        idx = rng.integers(0, n, size=(finish - start, n))
        means[start:finish] = centered[idx].mean(axis=1)
        if progress_callback:
            progress_callback(finish, n_simulations)
    return means


def _verdict(p):
    if p < 0.05:
        return "significant"       # genuine edge — proceed
    if p <= 0.10:
        return "borderline"        # inconclusive — flag to user
    return "not_significant"       # indistinguishable from random — hard stop
