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
from ..store import set_current_store


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
             generate_tradingview=False, generate_hyperparameters=False,
             generate_equity_curve=False, benchmark=False, generate_csv=False,
             generate_json=False, generate_logs=False, hyperparameters=None,
             fast_mode=False, candles_pipeline_class=None,
             candles_pipeline_kwargs=None, generate_charts=False,
             charts_output_root=None,
             strategies_dir=None, strategy_classes=None, strategy_sources=None,
             signal_only=False, should_cancel=None, cashflows=None):
    """
    Run a single backtest. Strategy classes are resolved (in priority order) from
    `strategy_classes` (name->class), `strategy_sources` (name->source), or the
    on-disk `strategies_dir`.
    """
    data_routes = data_routes or []
    candles = candles or {}

    store = Store()
    # Make Jesse's process-global-looking ``store`` import resolve to this
    # backtest's isolated state for strategy constructors and lifecycle hooks.
    set_current_store(store)
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

    _apply_candle_pipelines(
        store, route_objs, data_route_objs,
        candles_pipeline_class, candles_pipeline_kwargs or {},
    )

    sim = Simulator(store, route_objs, data_route_objs, run_silently=True)
    sim.warmup_1m = warmup_1m
    sim.signal_only = signal_only
    sim.cashflows = cashflows
    for r in route_objs:
        r.strategy.simulator = sim
        r.strategy.broker = sim
        r.strategy.store_routes = route_objs

    del fast_mode  # Terry's single engine already uses the streamlined execution path.
    store.app.is_active = True
    try:
        run_out = sim.run(
            generate_equity_curve=generate_equity_curve or benchmark or generate_charts,
            should_cancel=should_cancel,
        )
    finally:
        # Keep the historical store inspectable through the Jesse facade while
        # preventing later session/config timestamps from reusing candle time.
        store.app.is_active = False

    metrics = trades_metrics(
        store.closed_trades, store.app.daily_balance,
        exchange.starting_balance, exchange.balance,
        store.app.starting_time, store.app.ending_time,
        store.app.total_open_trades, store.app.total_open_pl,
    )

    trades = [t.to_dict() for t in store.closed_trades]
    # Per-route indicator overlays captured while the strategy ran, keyed the same way
    # the dashboard requests candles (exchange-symbol-timeframe).
    chart_data = {}
    for route in route_objs:
        overlays = route.strategy._chart_overlays()
        if overlays["candle_lines"] or overlays["candle_hlines"] or overlays["extra_charts"]:
            chart_data[jh.key(route.exchange, route.symbol, route.timeframe)] = overlays
    result = {
        "metrics": metrics,
        "trades": trades,
        "daily_balance": list(store.app.daily_balance),
        "chart_data": chart_data or None,
        "logs": ([*sim.logs, *store.logs.info, *store.logs.errors]
                 if generate_logs else None),
        "ml_data": [point for route in route_objs
                    for point in route.strategy._ml_data_points],
    }
    if generate_equity_curve:
        result["equity_curve"] = run_out["equity_curve"]
    if signal_only:
        result["signals"] = sim.signal_log
    if generate_csv:
        from .exports import trades_csv
        result["csv"] = trades_csv(trades)
    if generate_json:
        from .exports import trades_json
        result["json"] = trades_json(trades)
    if generate_tradingview:
        from .exports import tradingview_pine
        result["tradingview"] = tradingview_pine(trades)
    if generate_hyperparameters:
        result["hyperparameters"] = dict(route_objs[0].strategy.hp) if route_objs else {}
    if benchmark:
        raw = store.candles.raw_1m[jh.key(routes[0]["exchange"], routes[0]["symbol"])]
        trading_raw = raw[warmup_1m:] if warmup_1m < len(raw) else raw[-1:]
        first_price, last_price = float(trading_raw[0, 2]), float(trading_raw[-1, 2])
        result["benchmark"] = {
            "starting_price": first_price, "finishing_price": last_price,
            "return_percentage": ((last_price / first_price) - 1) * 100 if first_price else 0,
        }
    if generate_charts and trades:
        from .charts import generate_backtest_charts
        chart_id, chart_folder = generate_backtest_charts(
            run_out["equity_curve"], trades,
            output_root=charts_output_root or "storage/backtest-charts")
        result["charts_session_id"] = chart_id
        result["charts_folder"] = chart_folder
    return result


def _apply_candle_pipelines(store, routes, data_routes, pipeline_class, pipeline_kwargs):
    """Apply explicit or strategy-provided pipelines once per exchange-symbol pair."""
    transformed = set()
    for route in routes:
        key = jh.key(route.exchange, route.symbol)
        if key in transformed:
            continue
        pipeline = (pipeline_class(**pipeline_kwargs) if pipeline_class is not None
                    else route.strategy.candles_pipeline())
        if pipeline is not None:
            if not hasattr(pipeline, "transform"):
                raise TypeError("candles_pipeline() must return a BaseCandlesPipeline instance")
            store.candles.raw_1m[key] = pipeline.transform(store.candles.raw_1m[key])
            transformed.add(key)
    if pipeline_class is not None:
        for route in data_routes:
            key = jh.key(route.exchange, route.symbol)
            if key in transformed:
                continue
            pipeline = pipeline_class(**pipeline_kwargs)
            store.candles.raw_1m[key] = pipeline.transform(store.candles.raw_1m[key])
            transformed.add(key)
    if transformed:
        store.candles._agg_cache = {}


def _resolve_hp(strategy, overrides):
    """Build the hp dict from the strategy's hyperparameters() defaults + any overrides."""
    definitions = strategy.hyperparameters()
    hp = {}
    for p in definitions:
        hp[p["name"]] = p.get("default")
    dna = strategy.dna()
    if dna:
        hp.update(jh.dna_to_hp(definitions, dna))
    if overrides:
        hp.update({k: v for k, v in overrides.items() if k in hp})
    return hp
