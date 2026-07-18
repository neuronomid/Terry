"""
Isolated backtest() — a pure function that assembles the engine, runs the simulation, and
returns metrics/trades/equity. Mirrors jesse.research.backtest()'s contract.

config example:
    {'starting_balance': 10_000, 'fee': 0.001, 'type': 'futures',
     'futures_leverage': 2, 'futures_leverage_mode': 'cross',
     'exchange': 'Binance Perpetual Futures', 'warm_up_candles': 0}
routes:      [{'exchange','symbol','timeframe','strategy'}]
data_routes: [{'exchange','symbol','timeframe'}]
candles:     {'exchange-symbol': {'exchange','symbol','candles': np.ndarray(1m)}}
"""
import numpy as np

from .. import helpers as jh
from ..engine.store import Store
from ..engine.exchange import Exchange
from ..engine.simulator import Simulator
from ..engine.metrics import trades_metrics
from ..models import Position, Route
from ..loader import load_strategy_class, load_strategy_from_source
from ..exceptions import InvalidRoutes


def _make_exchange(config):
    return Exchange(
        name=config["exchange"],
        starting_balance=config["starting_balance"],
        fee_rate=config["fee"],
        exchange_type=config["type"],
        futures_leverage=config.get("futures_leverage", 1),
        futures_leverage_mode=config.get("futures_leverage_mode", "cross"),
        quote_asset=config.get("quote_asset", "USDT"),
    )


def backtest(config, routes, data_routes=None, candles=None, warmup_candles=None,
             generate_equity_curve=False, hyperparameters=None,
             strategies_dir=None, strategy_classes=None, strategy_sources=None,
             signal_only=False, should_cancel=None):
    """
    Run a single backtest. Strategy classes are resolved (in priority order) from
    `strategy_classes` (name->class), `strategy_sources` (name->source), or the
    on-disk `strategies_dir`.
    """
    data_routes = data_routes or []
    candles = candles or {}

    store = Store()
    exchange = _make_exchange(config)
    store.add_exchange(exchange)
    store.app.trading_mode = "backtest"

    # load candles
    store.candles.init_from_dict(candles)

    # validate 1m
    for k, v in candles.items():
        arr = np.asarray(v["candles"])
        if len(arr) >= 2 and int(arr[1][0] - arr[0][0]) != 60_000:
            raise ValueError("Candles passed to backtest() must be 1m candles.")

    # warmup injection
    warmup_1m = 0
    if warmup_candles:
        for k, v in warmup_candles.items():
            store.candles.inject_warmup(v["exchange"], v["symbol"], v["candles"])
        first = warmup_candles[next(iter(warmup_candles))]
        warmup_1m = len(np.asarray(first["candles"]))
    else:
        wc = int(config.get("warm_up_candles", 0) or 0)
        if wc:
            tf = jh.timeframe_to_one_minutes(routes[0]["timeframe"])
            warmup_1m = wc * tf

    # build routes, positions, strategies
    route_objs = []
    for r in routes:
        pos = Position(exchange, r["symbol"])
        store.positions.storage[r["symbol"]] = pos

        if strategy_classes and r["strategy"] in strategy_classes:
            cls = strategy_classes[r["strategy"]]
        elif strategy_sources and r["strategy"] in strategy_sources:
            cls = load_strategy_from_source(r["strategy"], strategy_sources[r["strategy"]])
        elif strategies_dir:
            cls = load_strategy_class(r["strategy"], strategies_dir)
        else:
            raise InvalidRoutes("No strategy source provided (strategies_dir/classes/sources).")

        strat = cls()
        strat.name = r["strategy"]
        strat.symbol = r["symbol"]
        strat.exchange = r["exchange"]
        strat.timeframe = r["timeframe"]
        strat.position = pos
        strat.store = store
        strat.hp = _resolve_hp(strat, hyperparameters)
        pos.strategy = strat

        route = Route(r["exchange"], r["symbol"], r["timeframe"], r["strategy"])
        route.strategy = strat
        route_objs.append(route)

    data_route_objs = [Route(r["exchange"], r["symbol"], r["timeframe"]) for r in data_routes]

    sim = Simulator(store, route_objs, data_route_objs, run_silently=True)
    sim.warmup_1m = warmup_1m
    sim.signal_only = signal_only
    for r in route_objs:
        r.strategy.simulator = sim
        r.strategy.store_routes = route_objs

    run_out = sim.run(generate_equity_curve=generate_equity_curve, should_cancel=should_cancel)

    metrics = trades_metrics(
        store.closed_trades, store.app.daily_balance,
        exchange.starting_balance, exchange.balance,
        store.app.starting_time, store.app.ending_time,
        store.app.total_open_trades, store.app.total_open_pl,
    )

    result = {
        "metrics": metrics,
        "trades": [t.to_dict() for t in store.closed_trades],
        "logs": sim.logs,
    }
    if generate_equity_curve:
        result["equity_curve"] = run_out["equity_curve"]
    if signal_only:
        result["signals"] = sim.signal_log
    return result


def _resolve_hp(strategy, overrides):
    """Build the hp dict from the strategy's hyperparameters() defaults + any overrides."""
    hp = {}
    for p in strategy.hyperparameters():
        hp[p["name"]] = p.get("default")
    if overrides:
        hp.update({k: v for k, v in overrides.items() if k in hp})
    return hp
