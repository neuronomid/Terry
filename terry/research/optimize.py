"""Optuna-backed hyperparameter optimization with out-of-sample validation."""

from __future__ import annotations

import base64
import json

import numpy as np
import optuna

from .backtest import backtest

_OBJECTIVES = {
    "sharpe": "sharpe_ratio", "sharpe_ratio": "sharpe_ratio",
    "calmar": "calmar_ratio", "calmar_ratio": "calmar_ratio",
    "sortino": "sortino_ratio", "sortino_ratio": "sortino_ratio",
    "omega": "omega_ratio", "omega_ratio": "omega_ratio",
    "serenity": "serenity_index", "serenity_index": "serenity_index",
    "smart sharpe": "sharpe_ratio", "smart_sharpe": "sharpe_ratio",
    "smart sortino": "sortino_ratio", "smart_sortino": "sortino_ratio",
    "net_profit_percentage": "net_profit_percentage",
}


def optimize(config: dict, routes: list[dict], data_routes: list[dict] | None = None,
             training_candles: dict | None = None,
             training_warmup_candles: dict | None = None,
             testing_candles: dict | None = None,
             testing_warmup_candles: dict | None = None,
             optimal_total: int = 200, fast_mode: bool = True,
             cpu_cores: int | None = None, trials: int = 200,
             objective_function: str = "sharpe", best_candidates_count: int = 20,
             progress_bar: bool = True, *,
             # Terry 0.1 compatibility aliases used by the dashboard/MCP layer.
             candles: dict | None = None, warmup_candles: dict | None = None,
             hp_space: list[dict] | None = None, objective: str | None = None,
             n_trials: int | None = None, random_seed: int = 42,
             train_test_split: float = 0.75, min_trades: int = 5,
             strategies_dir=None, strategy_classes=None, strategy_sources=None,
             progress_callback=None, should_cancel=None) -> dict:
    """Tune on training candles and validate ranked candidates on unseen candles.

    The leading parameters mirror Jesse 2.5's research API. ``candles``/``n_trials``
    remain accepted for Terry's single-window dashboard and are converted to a
    chronological train/test split.
    """
    del optimal_total, fast_mode, cpu_cores, progress_bar
    if not routes:
        raise ValueError("At least one route is required.")
    data_routes = data_routes or []
    flat_config, routes, data_routes = _normalize_config_and_routes(config, routes, data_routes)
    training_candles = training_candles or candles or {}
    training_warmup_candles = training_warmup_candles or warmup_candles
    if not training_candles:
        raise ValueError("training_candles cannot be empty")
    if objective is not None:
        objective_function = objective
    try:
        metric_key = _OBJECTIVES[objective_function]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported objective_function '{objective_function}'. "
            f"Choose one of: {', '.join(sorted(_OBJECTIVES))}."
        ) from exc

    if hp_space is None:
        hp_space = _discover_hp(routes, strategies_dir, strategy_classes, strategy_sources)
    if not hp_space:
        raise ValueError("The strategy defines no hyperparameters() to optimize.")

    if testing_candles is None:
        (training_candles, testing_candles,
         testing_warmup_candles) = _split_candles(
            training_candles, train_test_split, routes, flat_config)
    total_trials = int(n_trials if n_trials is not None else trials * len(hp_space))
    if total_trials < 1:
        raise ValueError("trials must be at least 1")
    best_candidates_count = max(1, int(best_candidates_count))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))
    completed = []
    failures = []

    for index in range(total_trials):
        if should_cancel and should_cancel():
            raise InterruptedError("Research run canceled")
        trial = study.ask()
        parameters = {item["name"]: _suggest(trial, item) for item in hp_space}
        try:
            training_metrics = backtest(
                flat_config, routes, data_routes, training_candles,
                warmup_candles=training_warmup_candles,
                hyperparameters=parameters, strategies_dir=strategies_dir,
                strategy_classes=strategy_classes, strategy_sources=strategy_sources,
                should_cancel=should_cancel,
            )["metrics"]
            score = _score(training_metrics, metric_key, min_trades)
            study.tell(trial, score)
            completed.append({
                "trial": index, "params": parameters, "fitness": score,
                "training_metrics": training_metrics,
            })
        except InterruptedError:
            raise
        except Exception as exc:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            failures.append(f"{type(exc).__name__}: {exc}")
        if progress_callback:
            progress_callback(index + 1, total_trials)

    if not completed:
        detail = failures[0] if failures else "no trial result was produced"
        raise RuntimeError(f"All optimization trials failed. First error: {detail}")
    completed.sort(key=lambda item: item["fitness"], reverse=True)
    selected = completed[:min(best_candidates_count, len(completed))]
    for rank, candidate in enumerate(selected, 1):
        if should_cancel and should_cancel():
            raise InterruptedError("Research run canceled")
        try:
            testing_metrics = backtest(
                flat_config, routes, data_routes, testing_candles,
                warmup_candles=testing_warmup_candles,
                hyperparameters=candidate["params"], strategies_dir=strategies_dir,
                strategy_classes=strategy_classes, strategy_sources=strategy_sources,
                should_cancel=should_cancel,
            )["metrics"]
        except InterruptedError:
            raise
        except Exception as exc:
            testing_metrics = None
            candidate["testing_error"] = f"{type(exc).__name__}: {exc}"
        candidate.update({
            "rank": rank,
            "dna": _encode_dna(candidate["params"]),
            "testing_metrics": testing_metrics,
        })
        # Backward-compatible fields used by Terry's current dashboard/report.
        candidate.update({
            "hp": candidate["params"], "train_score": candidate["fitness"],
            "train_metrics": _slim(candidate["training_metrics"]),
            "test_score": (_score(testing_metrics, metric_key, 0)
                           if testing_metrics is not None else None),
            "test_metrics": _slim(testing_metrics) if testing_metrics is not None else None,
        })

    return {
        "best_trials": selected,
        "total_trials": total_trials,
        "completed_trials": len(completed),
        "failed_trials": len(failures),
        "trial_errors": failures[:10],
        "objective_function": objective_function,
        "objective": metric_key,
        "n_trials": len(completed),
        "best": selected[0] if selected else None,
        "candidates": selected,
    }


def print_optimize_summary(result: dict, show_params: bool = False) -> None:
    """Print the ranked train/test objective scores from :func:`optimize`."""
    print(f"Objective: {result.get('objective_function')} · "
          f"{result.get('completed_trials', 0)}/{result.get('total_trials', 0)} trials")
    for candidate in result.get("best_trials", []):
        representation = candidate["params"] if show_params else candidate["dna"]
        print(
            f"#{candidate['rank']} {representation} · "
            f"train={candidate['train_score']:.4f} · test={candidate.get('test_score')}"
        )


def _normalize_config_and_routes(config, routes, data_routes):
    if isinstance(config.get("exchange"), dict):
        exchange = config["exchange"]
        flat = {
            "exchange": exchange["name"],
            "starting_balance": exchange["balance"],
            "fee": exchange["fee"],
            "type": exchange["type"],
            "futures_leverage": exchange.get("futures_leverage", 1),
            "futures_leverage_mode": exchange.get("futures_leverage_mode", "cross"),
            "quote_asset": exchange.get("quote_asset", "USDT"),
            "warm_up_candles": config.get("warm_up_candles", 0),
        }
    else:
        flat = dict(config)
    exchange_name = flat["exchange"]
    normalized_routes = [{**route, "exchange": route.get("exchange", exchange_name)}
                         for route in routes]
    normalized_data_routes = [{**route, "exchange": route.get("exchange", exchange_name)}
                              for route in data_routes]
    return flat, normalized_routes, normalized_data_routes


def _suggest(trial, parameter):
    name = str(parameter["name"])
    kind = parameter["type"]
    kind = kind.__name__ if isinstance(kind, type) else str(kind).strip("'\"")
    if kind == "int":
        return trial.suggest_int(name, int(parameter["min"]), int(parameter["max"]),
                                 step=int(parameter.get("step") or 1))
    if kind == "float":
        step = parameter.get("step")
        return trial.suggest_float(name, float(parameter["min"]), float(parameter["max"]),
                                   step=float(step) if step is not None else None)
    if kind == "categorical":
        return trial.suggest_categorical(name, parameter["options"])
    raise ValueError(f"Unsupported hyperparameter type: {kind}")


def _score(metrics, objective, min_trades):
    if metrics is None or metrics.get("total", 0) < min_trades:
        return -1e18
    value = metrics.get(objective)
    return float(value) if value is not None and np.isfinite(value) else -1e18


def _discover_hp(routes, strategies_dir, strategy_classes, strategy_sources):
    from ..loader import load_strategy_class, load_strategy_from_source
    name = routes[0]["strategy"]
    if strategy_classes and name in strategy_classes:
        cls = strategy_classes[name]
    elif strategy_sources and name in strategy_sources:
        cls = load_strategy_from_source(name, strategy_sources[name])
    else:
        cls = load_strategy_class(name, strategies_dir)
    return cls().hyperparameters()


def _split_candles(candles, ratio, routes, config):
    if not 0.1 < ratio < 0.9:
        raise ValueError("train_test_split must be greater than 0.1 and less than 0.9")
    training, testing, testing_warmup = {}, {}, {}
    timeframe_minutes = 1
    if routes:
        from .. import helpers as jh
        timeframe_minutes = jh.timeframe_to_one_minutes(routes[0]["timeframe"])
    warmup_rows = int(config.get("warm_up_candles", 0) or 0) * timeframe_minutes
    for key, value in candles.items():
        array = np.asarray(value["candles"])
        split = int(len(array) * ratio)
        if split < 2 or len(array) - split < 2:
            raise ValueError("train_test_split produces an insufficient train or test window")
        training[key] = {**value, "candles": array[:split]}
        testing[key] = {**value, "candles": array[split:]}
        if warmup_rows:
            testing_warmup[key] = {
                **value, "candles": array[max(0, split - warmup_rows):split],
            }
    return training, testing, testing_warmup or None


def _encode_dna(parameters):
    payload = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _slim(metrics):
    keys = ["total", "win_rate", "net_profit_percentage", "sharpe_ratio",
            "calmar_ratio", "max_drawdown", "sortino_ratio", "omega_ratio"]
    return {key: metrics.get(key) for key in keys}
