"""
Monte Carlo robustness analysis (ported from Jesse's research.monte_carlo).

Two modes:
  - candles: block-bootstrap the 1m price path, re-run the strategy on each variant, and
    build percentile bands (best_5=95th, median=50th, worst_5=5th) on key metrics. Answers
    "is this backtest overfit / lucky?".
  - trades: reshuffle the executed trade order and recompute the drawdown path. Only
    max_drawdown / calmar carry information (total return & win rate are shuffle-invariant).
"""
import numpy as np

from .backtest import backtest

KEY_METRICS = ["sharpe_ratio", "net_profit_percentage", "max_drawdown", "calmar_ratio",
               "win_rate", "sortino_ratio"]
PERCENTILES = {"worst_5": 5, "low_quartile": 25, "median": 50, "high_quartile": 75, "best_5": 95}


def _block_bootstrap_returns(returns, rng, block=60):
    n = len(returns)
    out = np.empty(n)
    i = 0
    while i < n:
        start = rng.integers(0, max(1, n - block))
        b = returns[start:start + block]
        take = min(len(b), n - i)
        out[i:i + take] = b[:take]
        i += take
    return out


def _resample_candles(candles_1m, rng, block=60):
    """Block-bootstrap the close log-returns and rebuild a candle array preserving per-bar range."""
    c = np.array(candles_1m, dtype=float)
    close = c[:, 2]
    log_ret = np.diff(np.log(close))
    new_ret = _block_bootstrap_returns(log_ret, rng, block)
    new_close = np.empty_like(close)
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.exp(np.cumsum(new_ret))
    out = c.copy()
    # preserve each original candle's high/low/open ratios relative to its close
    with np.errstate(divide="ignore", invalid="ignore"):
        hi_ratio = np.where(close != 0, c[:, 3] / close, 1.0)
        lo_ratio = np.where(close != 0, c[:, 4] / close, 1.0)
    out[:, 2] = new_close
    out[:, 1] = np.concatenate([[new_close[0]], new_close[:-1]])  # open = prev close
    out[:, 3] = np.maximum(new_close * hi_ratio, np.maximum(out[:, 1], new_close))
    out[:, 4] = np.minimum(new_close * lo_ratio, np.minimum(out[:, 1], new_close))
    return out


def monte_carlo_candles(config, routes, data_routes=None, candles=None, warmup_candles=None,
                        hyperparameters=None, num_scenarios=200, random_seed=42, block=60,
                        strategies_dir=None, strategy_classes=None, strategy_sources=None,
                        progress_callback=None, should_cancel=None):
    data_routes = data_routes or []
    candles = candles or {}

    def _run(cndls):
        return backtest(config, routes, data_routes, cndls, warmup_candles=warmup_candles,
                        hyperparameters=hyperparameters, strategies_dir=strategies_dir,
                        strategy_classes=strategy_classes, strategy_sources=strategy_sources,
                        should_cancel=should_cancel)["metrics"]

    original = _run(candles)

    rng = np.random.default_rng(random_seed)
    collected = {k: [] for k in KEY_METRICS}
    completed = 0
    for s in range(num_scenarios):
        if should_cancel and should_cancel():
            raise InterruptedError("Research run canceled")
        resampled = {}
        for key, v in candles.items():
            resampled[key] = {**v, "candles": _resample_candles(v["candles"], rng, block)}
        try:
            m = _run(resampled)
        except Exception:
            continue
        for k in KEY_METRICS:
            val = m.get(k)
            if val is not None and np.isfinite(val):
                collected[k].append(val)
        completed += 1
        if progress_callback:
            progress_callback(completed, num_scenarios)

    summary = {}
    for k in KEY_METRICS:
        vals = np.array(collected[k], dtype=float)
        entry = {"original": _num(original.get(k))}
        if len(vals):
            for name, p in PERCENTILES.items():
                entry[name] = float(np.percentile(vals, p))
            entry["mean"] = float(vals.mean())
        summary[k] = entry

    return {
        "mode": "candles",
        "num_scenarios": completed,
        "original_metrics": {k: _num(original.get(k)) for k in KEY_METRICS},
        "summary_metrics": summary,
        "overfit_verdict": _overfit_verdict(summary.get("sharpe_ratio", {})),
    }


def monte_carlo_trades(trades, num_scenarios=1000, random_seed=42, starting_balance=10000.0,
                       should_cancel=None):
    """Shuffle trade order and report the distribution of max drawdown (path-dependent)."""
    pnls = np.array([t["PNL"] for t in trades], dtype=float)
    if len(pnls) < 2:
        return {"mode": "trades", "num_scenarios": 0, "note": "not enough trades"}

    def _max_dd(order):
        equity = starting_balance + np.cumsum(pnls[order])
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(dd.min() * 100)

    original_dd = _max_dd(np.arange(len(pnls)))
    rng = np.random.default_rng(random_seed)
    dds = []
    for _ in range(num_scenarios):
        if should_cancel and should_cancel():
            raise InterruptedError("Research run canceled")
        dds.append(_max_dd(rng.permutation(len(pnls))))
    dds = np.asarray(dds)
    summary = {"original": original_dd}
    for name, p in PERCENTILES.items():
        summary[name] = float(np.percentile(dds, p))
    return {
        "mode": "trades",
        "num_scenarios": num_scenarios,
        "max_drawdown": summary,
        "note": "Only max_drawdown/calmar are informative under trade shuffling.",
    }


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _overfit_verdict(sharpe_summary):
    o = sharpe_summary.get("original")
    med = sharpe_summary.get("median")
    best = sharpe_summary.get("best_5")
    if o is None or med is None or best is None:
        return "unknown"
    if o > best:
        return "overfit_suspect"       # real result beats 95% of resampled paths
    if o > med:
        return "borderline"
    return "robust"                    # original at/below median → not overfit
