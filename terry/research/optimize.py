"""
Hyperparameter optimization via random search with a train/test split.

Jesse uses Optuna + a genetic loop; Terry uses a dependency-free random search that samples
the strategy's hyperparameters() space, scores each trial on the training window by a chosen
objective, and validates the best on the out-of-sample test window. Returns ranked candidates.
"""
import json

import numpy as np

from .backtest import backtest


def _sample(param, rng):
    ptype = param["type"]
    if ptype in (int, "int"):
        step = param.get("step", 1)
        lo, hi = param["min"], param["max"]
        choices = list(range(lo, hi + 1, step))
        return int(rng.choice(choices))
    if ptype in (float, "float"):
        step = param.get("step")
        lo, hi = param["min"], param["max"]
        if step:
            n = int(round((hi - lo) / step))
            return float(lo + rng.integers(0, n + 1) * step)
        return float(rng.uniform(lo, hi))
    if ptype == "categorical":
        return param["options"][int(rng.integers(0, len(param["options"])))]
    raise ValueError(f"Unknown hyperparameter type: {ptype}")


def _score(metrics, objective):
    v = metrics.get(objective)
    if v is None or not np.isfinite(v):
        return -1e18
    return float(v)


def optimize(config, routes, data_routes=None, candles=None, warmup_candles=None,
             hp_space=None, objective="sharpe_ratio", n_trials=100, random_seed=42,
             train_test_split=0.75, min_trades=5,
             strategies_dir=None, strategy_classes=None, strategy_sources=None,
             progress_callback=None, should_cancel=None):
    """
    hp_space: the strategy's hyperparameters() list. If None, it is read from the strategy.
    Splits the candle series into train/test by time; optimizes on train, validates on test.
    """
    data_routes = data_routes or []
    candles = candles or {}

    # discover hp space from the strategy if not provided
    if hp_space is None:
        hp_space = _discover_hp(routes, strategies_dir, strategy_classes, strategy_sources)
    if not hp_space:
        raise ValueError("The strategy defines no hyperparameters() to optimize.")

    # time split the candles
    train_c, test_c = _split_candles(candles, train_test_split)

    rng = np.random.default_rng(random_seed)
    trials = []
    for i in range(n_trials):
        if should_cancel and should_cancel():
            raise InterruptedError("Research run canceled")
        hp = {p["name"]: _sample(p, rng) for p in hp_space}
        try:
            train_m = backtest(config, routes, data_routes, train_c, warmup_candles=warmup_candles,
                               hyperparameters=hp, strategies_dir=strategies_dir,
                               strategy_classes=strategy_classes, strategy_sources=strategy_sources,
                               should_cancel=should_cancel)["metrics"]
        except Exception as e:
            continue
        if train_m.get("total", 0) < min_trades:
            score = -1e17
        else:
            score = _score(train_m, objective)
        trials.append({"hp": hp, "train_score": score,
                       "train_metrics": _slim(train_m), "dna": _encode_dna(hp)})
        if progress_callback:
            progress_callback(i + 1, n_trials)

    trials.sort(key=lambda t: t["train_score"], reverse=True)
    top = trials[:min(10, len(trials))]

    # out-of-sample validation of the top candidates
    for t in top:
        if should_cancel and should_cancel():
            raise InterruptedError("Research run canceled")
        try:
            test_m = backtest(config, routes, data_routes, test_c, warmup_candles=warmup_candles,
                              hyperparameters=t["hp"], strategies_dir=strategies_dir,
                              strategy_classes=strategy_classes, strategy_sources=strategy_sources,
                              should_cancel=should_cancel)["metrics"]
            t["test_score"] = _score(test_m, objective)
            t["test_metrics"] = _slim(test_m)
        except Exception:
            t["test_score"] = None
            t["test_metrics"] = None

    return {
        "objective": objective,
        "n_trials": len(trials),
        "best": top[0] if top else None,
        "candidates": top,
    }


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


def _split_candles(candles, ratio):
    train, test = {}, {}
    for k, v in candles.items():
        arr = np.asarray(v["candles"])
        cut = int(len(arr) * ratio)
        train[k] = {**v, "candles": arr[:cut]}
        test[k] = {**v, "candles": arr[cut:]}
    return train, test


def _encode_dna(hp):
    return json.dumps(hp, sort_keys=True, default=str)


def _slim(m):
    keys = ["total", "win_rate", "net_profit_percentage", "sharpe_ratio",
            "calmar_ratio", "max_drawdown", "sortino_ratio"]
    return {k: m.get(k) for k in keys}
