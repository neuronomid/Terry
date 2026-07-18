"""Jesse-compatible Monte Carlo robustness research.

The public call signatures and primary result keys mirror Jesse 2.5 while Terry keeps
its compact percentile summaries for the MCP dashboard and existing callers.
"""

from __future__ import annotations

from datetime import datetime
import inspect
import os
import numpy as np

from .backtest import backtest


KEY_METRICS = [
    "sharpe_ratio",
    "net_profit_percentage",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "sortino_ratio",
]
PERCENTILES = {
    "worst_5": 5,
    "low_quartile": 25,
    "median": 50,
    "high_quartile": 75,
    "best_5": 95,
}


def _block_bootstrap_returns(returns, rng, block=60):
    n = len(returns)
    out = np.empty(n)
    i = 0
    while i < n:
        block_size = min(block, n)
        start = int(rng.integers(0, max(1, n - block_size + 1)))
        values = returns[start:start + block_size]
        take = min(len(values), n - i)
        out[i:i + take] = values[:take]
        i += take
    return out


def _resample_candles(candles_1m, rng, block=60):
    """Block-bootstrap close returns while retaining valid OHLC ranges."""
    source = np.asarray(candles_1m, dtype=float)
    if len(source) < 2:
        return source.copy()
    close = source[:, 2]
    if np.any(close <= 0):
        raise ValueError("Monte Carlo candle prices must be positive.")
    log_returns = np.diff(np.log(close))
    new_returns = _block_bootstrap_returns(log_returns, rng, block)
    new_close = np.empty_like(close)
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.exp(np.cumsum(new_returns))
    out = source.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        high_ratio = np.where(close != 0, source[:, 3] / close, 1.0)
        low_ratio = np.where(close != 0, source[:, 4] / close, 1.0)
    out[:, 2] = new_close
    out[:, 1] = np.concatenate([[new_close[0]], new_close[:-1]])
    out[:, 3] = np.maximum(new_close * high_ratio, np.maximum(out[:, 1], new_close))
    out[:, 4] = np.minimum(new_close * low_ratio, np.minimum(out[:, 1], new_close))
    return out


def monte_carlo_candles(
    config: dict,
    routes: list[dict[str, str]],
    data_routes: list[dict[str, str]],
    candles: dict,
    warmup_candles: dict | None = None,
    hyperparameters: dict | None = None,
    fast_mode: bool = True,
    num_scenarios: int = 1000,
    progress_bar: bool = False,
    candles_pipeline_class=None,
    candles_pipeline_kwargs: dict | None = None,
    cpu_cores: int | None = None,
    progress_callback=None,
    result_callback=None,
    *,
    random_seed: int = 42,
    block: int = 60,
    strategies_dir=None,
    strategy_classes=None,
    strategy_sources=None,
    should_cancel=None,
    max_equity_points: int | None = None,
) -> dict:
    """Run Jesse-shaped market-path Monte Carlo simulations.

    ``cpu_cores`` is accepted for source compatibility. Terry deliberately uses its
    deterministic in-process engine; callers receive the same streamed callback and
    result contracts without requiring Ray.
    """
    del cpu_cores
    if num_scenarios < 1:
        raise ValueError("num_scenarios must be at least 1")
    if block < 1:
        raise ValueError("block must be at least 1")
    data_routes = data_routes or []
    candles = candles or {}
    pipeline_kwargs = candles_pipeline_kwargs or {}

    def run_scenario(scenario_candles, pipeline_class=None):
        return backtest(
            config,
            routes,
            data_routes,
            scenario_candles,
            warmup_candles=warmup_candles,
            generate_equity_curve=True,
            hyperparameters=hyperparameters,
            fast_mode=fast_mode,
            candles_pipeline_class=pipeline_class,
            candles_pipeline_kwargs=pipeline_kwargs,
            strategies_dir=strategies_dir,
            strategy_classes=strategy_classes,
            strategy_sources=strategy_sources,
            should_cancel=should_cancel,
        )

    progress = _Progress(progress_bar, num_scenarios, "Monte Carlo Candles Scenarios")
    try:
        original = _scenario_result(run_scenario(candles), 0, max_equity_points)
        _emit_result(result_callback, original)
        _emit_progress(progress_callback, 1, num_scenarios)
        progress.update()

        rng = np.random.default_rng(random_seed)
        scenarios = []
        for scenario_index in range(1, num_scenarios):
            _check_canceled(should_cancel)
            if candles_pipeline_class is None:
                scenario_candles = {
                    key: {
                        **value,
                        "candles": _resample_candles(value["candles"], rng, block),
                    }
                    for key, value in candles.items()
                }
                raw_result = run_scenario(scenario_candles)
            else:
                raw_result = run_scenario(candles, candles_pipeline_class)
            scenario = _scenario_result(raw_result, scenario_index, max_equity_points)
            scenarios.append(scenario)
            _emit_result(result_callback, scenario)
            _emit_progress(progress_callback, scenario_index + 1, num_scenarios)
            progress.update()
    finally:
        progress.close()

    original_metrics = original.get("metrics", {})
    scenario_metrics = [scenario.get("metrics", {}) for scenario in scenarios]
    summary = _compact_summary(original_metrics, scenario_metrics, KEY_METRICS)
    return {
        # Jesse 2.5 contract
        "original": original,
        "scenarios": scenarios,
        "confidence_analysis": _confidence_analysis(
            original_metrics, scenario_metrics, KEY_METRICS
        ),
        "num_scenarios": len(scenarios),
        "total_requested": num_scenarios,
        # Terry dashboard/backward-compatible contract
        "mode": "candles",
        "original_metrics": {key: _num(original_metrics.get(key)) for key in KEY_METRICS},
        "summary_metrics": summary,
        "overfit_verdict": _overfit_verdict(summary.get("sharpe_ratio", {})),
    }


def monte_carlo_trades(
    config: dict | list,
    routes: list[dict[str, str]] | None = None,
    data_routes: list[dict[str, str]] | None = None,
    candles: dict | None = None,
    warmup_candles: dict | None = None,
    benchmark: bool = False,
    hyperparameters: dict | None = None,
    fast_mode: bool = True,
    num_scenarios: int = 1000,
    progress_bar: bool = False,
    cpu_cores: int | None = None,
    progress_callback=None,
    result_callback=None,
    *,
    random_seed: int = 42,
    starting_balance: float | None = None,
    strategies_dir=None,
    strategy_classes=None,
    strategy_sources=None,
    should_cancel=None,
    max_equity_points: int | None = None,
) -> dict:
    """Shuffle trades using Jesse's research signature or Terry's legacy trade list.

    Passing a configuration dict runs the original backtest, matching Jesse. Passing a
    list of closed trades remains supported for Terry 0.1 callers and session storage.
    """
    del cpu_cores
    if num_scenarios < 1:
        raise ValueError("num_scenarios must be at least 1")

    if isinstance(config, dict):
        if routes is None or candles is None:
            raise TypeError("routes, data_routes, and candles are required with a config dict")
        balance = float(
            config.get("starting_balance", 10_000)
            if starting_balance is None else starting_balance
        )
        raw_original = backtest(
            config,
            routes,
            data_routes or [],
            candles,
            warmup_candles=warmup_candles,
            generate_equity_curve=True,
            benchmark=benchmark,
            hyperparameters=hyperparameters,
            fast_mode=fast_mode,
            strategies_dir=strategies_dir,
            strategy_classes=strategy_classes,
            strategy_sources=strategy_sources,
            should_cancel=should_cancel,
        )
        original = _scenario_result(raw_original, 0, max_equity_points)
        trades = list(raw_original.get("trades") or [])
        if not trades:
            raise ValueError(
                "No trades found in original backtest. Cannot perform trade-shuffling "
                "Monte Carlo."
            )
        # A shuffled-trade equity path changes only when a trade closes. Keeping one
        # point per trade is mathematically sufficient and avoids duplicating a
        # minute-level curve across hundreds of scenarios.
        original_points = _equity_from_trades(trades, balance)
    else:
        trades = list(config)
        balance = float(10_000 if starting_balance is None else starting_balance)
        if not trades:
            return _empty_trade_result(num_scenarios)
        original_points = _equity_from_trades(trades, balance)
        original_scenario_metrics = _trade_metrics(original_points, balance)
        original = {
            "scenario_index": 0,
            "metrics": {
                "net_profit_percentage": original_scenario_metrics["total_return"],
                "max_drawdown": original_scenario_metrics["max_drawdown"],
                "sharpe_ratio": original_scenario_metrics["sharpe_ratio"],
                "calmar_ratio": original_scenario_metrics["calmar_ratio"],
            },
            "trades": trades,
            "equity_curve": _series_equity(original_points),
        }

    rng = np.random.default_rng(random_seed)
    scenarios = []
    progress = _Progress(progress_bar, num_scenarios, "Monte Carlo Scenarios")
    try:
        for scenario_index in range(num_scenarios):
            _check_canceled(should_cancel)
            shuffled = [trades[index] for index in rng.permutation(len(trades))]
            points = _reconstruct_equity_curve(shuffled, original_points, balance)
            scenario = {
                **_trade_metrics(points, balance),
                "scenario_index": scenario_index,
                "trades": shuffled,
                "equity_curve": _series_equity(points),
            }
            scenarios.append(scenario)
            _emit_result(result_callback, scenario)
            _emit_progress(progress_callback, scenario_index + 1, num_scenarios)
            progress.update()
    finally:
        progress.close()

    confidence = _trade_confidence(original, scenarios)
    drawdowns = np.asarray([scenario["max_drawdown"] for scenario in scenarios])
    drawdown_summary = {"original": _num(original.get("metrics", {}).get("max_drawdown"))}
    for name, percentile in PERCENTILES.items():
        drawdown_summary[name] = float(np.percentile(drawdowns, percentile))
    return {
        # Jesse 2.5 contract
        "original": original,
        "scenarios": scenarios,
        "confidence_analysis": confidence,
        "num_scenarios": len(scenarios),
        "total_requested": num_scenarios,
        # Terry legacy contract
        "mode": "trades",
        "max_drawdown": drawdown_summary,
        "note": "Only path-dependent metrics are informative under trade shuffling.",
    }


def _scenario_result(result: dict, scenario_index: int,
                     max_equity_points: int | None = None) -> dict:
    normalized = dict(result)
    normalized["scenario_index"] = scenario_index
    if "equity_curve" in normalized:
        points = _equity_points(normalized["equity_curve"])
        normalized["equity_curve"] = _series_equity(
            _sample_points(points, max_equity_points)
        )
    return normalized


def _equity_points(equity_curve) -> list[dict]:
    if not equity_curve:
        return []
    if isinstance(equity_curve, list) and equity_curve and "data" in equity_curve[0]:
        return list(equity_curve[0].get("data") or [])
    return list(equity_curve)


def _series_equity(points) -> list[dict]:
    return [{"name": "Portfolio", "data": list(points)}]


def _sample_points(points, maximum):
    if maximum is None or maximum < 1 or len(points) <= maximum:
        return list(points)
    step = max(1, int(np.ceil(len(points) / maximum)))
    sampled = list(points[::step])
    if sampled and sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _equity_from_trades(trades, starting_balance) -> list[dict]:
    balance = starting_balance
    points = [{"time": 0, "value": balance}]
    for index, trade in enumerate(trades, start=1):
        balance += _trade_pnl(trade)
        timestamp = trade.get("exit_date", trade.get("closed_at", index))
        try:
            timestamp = int(timestamp)
        except (TypeError, ValueError):
            timestamp = index
        points.append({"time": timestamp, "value": balance})
    return points


def _reconstruct_equity_curve(trades, original_points, starting_balance) -> list[dict]:
    if not original_points:
        return _equity_from_trades(trades, starting_balance)
    time_points = [point.get("time", point.get("timestamp", index))
                   for index, point in enumerate(original_points)]
    current_balance = starting_balance
    trade_index = 0
    total_trades = len(trades)
    total_points = len(time_points)
    trades_per_point = total_trades / total_points if total_points else 1
    output = []
    for index, timestamp in enumerate(time_points):
        target = int((index + 1) * trades_per_point)
        while trade_index < min(target, total_trades):
            current_balance += _trade_pnl(trades[trade_index])
            trade_index += 1
        output.append({"time": timestamp, "value": current_balance})
    while trade_index < total_trades:
        current_balance += _trade_pnl(trades[trade_index])
        trade_index += 1
    if output and trade_index == total_trades:
        output[-1]["value"] = current_balance
    return output


def _trade_pnl(trade) -> float:
    return float(trade.get("PNL", trade.get("pnl", 0)) or 0)


def _trade_metrics(points, starting_balance) -> dict:
    values = np.asarray([point["value"] for point in points], dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot calculate Monte Carlo metrics without an equity curve.")
    final_value = float(values[-1])
    total_return = ((final_value - starting_balance) / starting_balance) * 100
    peaks = np.maximum.accumulate(values)
    drawdowns = np.divide(
        values - peaks, peaks, out=np.zeros_like(values), where=peaks != 0
    )
    max_drawdown = float(drawdowns.min() * 100)
    returns = np.divide(
        np.diff(values), values[:-1], out=np.zeros(max(len(values) - 1, 0)),
        where=values[:-1] != 0,
    )
    volatility = float(np.std(returns) * np.sqrt(365)) if len(returns) else 0.0
    annualized_return = float(np.mean(returns) * 365) if len(returns) else 0.0
    sharpe = annualized_return / volatility if volatility > 0 else 0.0
    calmar = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return {
        "total_return": float(total_return),
        "final_value": final_value,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe),
        "calmar_ratio": float(calmar),
        "starting_balance": float(starting_balance),
    }


def _compact_summary(original_metrics, scenario_metrics, keys):
    summary = {}
    for key in keys:
        values = np.asarray([
            value for metrics in scenario_metrics
            if (value := _finite_num(metrics.get(key))) is not None
        ], dtype=float)
        entry = {"original": _num(original_metrics.get(key))}
        if len(values):
            for name, percentile in PERCENTILES.items():
                entry[name] = float(np.percentile(values, percentile))
            entry["mean"] = float(values.mean())
        summary[key] = entry
    return summary


def _confidence_analysis(original_metrics, scenario_metrics, keys):
    if not scenario_metrics:
        return {"error": "No simulation results to analyze"}
    analysis = {}
    for key in keys:
        values = np.asarray([
            value for metrics in scenario_metrics
            if (value := _finite_num(metrics.get(key))) is not None
        ], dtype=float)
        original = _finite_num(original_metrics.get(key))
        if original is None or len(values) == 0:
            continue
        percentiles = {
            f"{percentile}th" if percentile != 50 else "50th":
                float(np.percentile(values, percentile))
            for percentile in (5, 25, 50, 75, 95)
        }
        # Terry/Jesse drawdown values are signed percentages (for example -12.4),
        # so a larger value is better for every metric in this analysis.
        p_value = float(np.mean(values >= original))
        analysis[key] = {
            "original": original,
            "simulations": {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": int(len(values)),
            },
            "percentiles": percentiles,
            "confidence_intervals": {
                "90%": {
                    "lower": float(np.percentile(values, 5)),
                    "upper": float(np.percentile(values, 95)),
                },
                "95%": {
                    "lower": float(np.percentile(values, 2.5)),
                    "upper": float(np.percentile(values, 97.5)),
                },
            },
            "p_value": p_value,
            "is_significant_5pct": p_value < 0.05,
            "is_significant_1pct": p_value < 0.01,
        }
    interpretations = [_interpret_metric(name, value) for name, value in analysis.items()]
    return {
        "summary": {
            "num_simulations": len(scenario_metrics),
            "significant_metrics_5pct": sum(
                item["is_significant_5pct"] for item in analysis.values()
            ),
            "significant_metrics_1pct": sum(
                item["is_significant_1pct"] for item in analysis.values()
            ),
            "total_metrics": len(analysis),
        },
        "metrics": analysis,
        "interpretation": {
            "detailed": interpretations,
            "overall": (
                f"{sum(item['is_significant_5pct'] for item in analysis.values())} "
                f"of {len(interpretations)} metrics are statistically significant."
            ),
        },
    }


def _trade_confidence(original, scenarios):
    original_metrics = original.get("metrics", {})
    normalized_original = {
        "total_return": original_metrics.get("net_profit_percentage", 0),
        "max_drawdown": original_metrics.get("max_drawdown", 0),
        "sharpe_ratio": original_metrics.get("sharpe_ratio", 0),
        "calmar_ratio": original_metrics.get("calmar_ratio", 0),
    }
    return _confidence_analysis(
        normalized_original,
        scenarios,
        ["total_return", "max_drawdown", "sharpe_ratio", "calmar_ratio"],
    )


def _interpret_metric(name, analysis):
    original = analysis["original"]
    percentile = analysis["percentiles"]
    rank = _rank(original, percentile)
    p_value = analysis["p_value"]
    if p_value < 0.01:
        significance = "highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    return {
        "metric": name,
        "significance": significance,
        "rank": rank,
        "p_value": p_value,
        "message": f"{name}: {significance}, original result in {rank} of simulations",
    }


def _rank(original, percentiles):
    if original >= percentiles["95th"]:
        return "top 5%"
    if original >= percentiles["75th"]:
        return "top 25%"
    if original >= percentiles["50th"]:
        return "above median"
    if original >= percentiles["25th"]:
        return "below median"
    return "bottom 25%"


def _empty_trade_result(total_requested):
    return {
        "original": {},
        "scenarios": [],
        "confidence_analysis": {"error": "No simulation results to analyze"},
        "num_scenarios": 0,
        "total_requested": total_requested,
        "mode": "trades",
        "note": "not enough trades",
    }


def print_monte_carlo_candles_summary(results: dict) -> None:
    _print_confidence_summary(results, "MONTE CARLO CANDLES")


def print_monte_carlo_trades_summary(results: dict) -> None:
    _print_confidence_summary(results, "MONTE CARLO TRADES")


def _print_confidence_summary(results, title):
    confidence = results.get("confidence_analysis", {})
    metrics = confidence.get("metrics", {})
    if not metrics:
        print("No confidence analysis available")
        return
    print(f"\n{title}")
    print("Metric | Original | Worst 5% | Median | Best 5%")
    for name, analysis in metrics.items():
        percentiles = analysis["percentiles"]
        print(
            f"{name} | {analysis['original']:.4f} | {percentiles['5th']:.4f} | "
            f"{percentiles['50th']:.4f} | {percentiles['95th']:.4f}"
        )


def plot_monte_carlo_candles_chart(results: dict, charts_folder: str = None) -> str | None:
    return _plot_monte_carlo(results, charts_folder, "monte_carlo_candles_chart.png",
                             "Monte Carlo Candles - Equity Curve")


def plot_monte_carlo_trades_chart(results: dict, charts_folder: str = None) -> str | None:
    return _plot_monte_carlo(results, charts_folder, "monte_carlo_trades_chart.png",
                             "Monte Carlo Trades - Equity Curve")


def _plot_monte_carlo(results, charts_folder, filename, title):
    scenarios = results.get("scenarios") or []
    if not scenarios:
        print("No simulation results to plot")
        return None
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 8))
    for scenario in scenarios[:50]:
        points = _equity_points(scenario.get("equity_curve"))
        if points:
            axis.plot([point["value"] for point in points], color="cornflowerblue",
                      alpha=0.5, linewidth=0.8)
    original_points = _equity_points(results.get("original", {}).get("equity_curve"))
    if original_points:
        axis.plot([point["value"] for point in original_points], color="green",
                  linewidth=2, label="Original Strategy")
        axis.legend()
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel("Portfolio Value")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    folder = os.path.abspath(charts_folder or "charts")
    os.makedirs(folder, exist_ok=True)
    stem, extension = os.path.splitext(filename)
    path = os.path.join(folder, f"{stem}_{datetime.now():%Y%m%d_%H%M%S}{extension}")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved Monte Carlo chart to: {path}")
    return path


class _Progress:
    def __init__(self, enabled, total, description):
        self.bar = None
        if enabled:
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=description)

    def update(self):
        if self.bar is not None:
            self.bar.update(1)

    def close(self):
        if self.bar is not None:
            self.bar.close()


def _emit_progress(callback, done, total):
    if callback is None:
        return
    try:
        signature = inspect.signature(callback)
        try:
            signature.bind(done, total)
        except TypeError:
            callback(done)
        else:
            callback(done, total)
    except (TypeError, ValueError):
        callback(done, total)


def _emit_result(callback, result):
    if callback is not None:
        try:
            callback(result)
        except Exception:
            # Jesse treats streaming as best-effort and does not fail research when
            # a UI/result consumer disconnects or raises.
            pass


def _check_canceled(callback):
    if callback and callback():
        raise InterruptedError("Research run canceled")


def _finite_num(value):
    number = _num(value)
    return number if number is not None and np.isfinite(number) else None


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _overfit_verdict(sharpe_summary):
    original = sharpe_summary.get("original")
    median = sharpe_summary.get("median")
    best = sharpe_summary.get("best_5")
    if original is None or median is None or best is None:
        return "unknown"
    if original > best:
        return "overfit_suspect"
    if original > median:
        return "borderline"
    return "robust"


__all__ = [
    "monte_carlo_trades",
    "monte_carlo_candles",
    "print_monte_carlo_trades_summary",
    "plot_monte_carlo_trades_chart",
    "print_monte_carlo_candles_summary",
    "plot_monte_carlo_candles_chart",
]
