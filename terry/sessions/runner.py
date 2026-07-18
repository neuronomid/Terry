"""
Background runner for sessions. Loads candles from the SQLite store, runs the appropriate
research function in a thread, streams progress into the session, and writes an HTML report.
"""
import threading

import numpy as np

from .. import helpers as jh
from ..research import (backtest, rule_significance_test,
                        monte_carlo_candles, monte_carlo_trades, optimize)


def _candle_pipeline(state):
    name = state.get("pipeline_type")
    if not name:
        return None, None
    from ..candle_pipelines import (
        GaussianNoiseCandlesPipeline, GaussianResamplerCandlesPipeline,
        MovingBlockBootstrapCandlesPipeline,
    )
    pipelines = {
        "moving_block_bootstrap": MovingBlockBootstrapCandlesPipeline,
        "gaussian_noise": GaussianNoiseCandlesPipeline,
        "gaussian_resampler": GaussianResamplerCandlesPipeline,
    }
    try:
        pipeline = pipelines[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown pipeline_type '{name}'. Choose one of: {', '.join(pipelines)}."
        ) from exc
    params = dict(state.get("pipeline_params") or {})
    params.setdefault("batch_size", 10_080)
    if name == "gaussian_noise":
        params.setdefault("close_sigma", 0.001)
        params.setdefault("high_sigma", 0.0001)
        params.setdefault("low_sigma", 0.0001)
    return pipeline, params


class MissingCandles(Exception):
    pass


class Runner:
    def __init__(self, ctx):
        self.ctx = ctx
        self._threads = {}
        self._canceled = set()

    # ------------------------------------------------------------------ public
    def run(self, sid):
        session = self.ctx.sessions.get(sid)
        if session is None:
            raise KeyError(sid)
        worker = self._threads.get(sid)
        if worker is not None and worker.is_alive():
            return {"error": "worker_active", "status": session["status"], "session_id": sid}
        if session["status"] == "running":
            return {"status": "running", "session_id": sid}
        self._canceled.discard(sid)
        self.ctx.sessions.set_status(sid, "running")
        self.ctx.sessions.set_progress(sid, 0)
        t = threading.Thread(target=self._dispatch, args=(sid,), daemon=True)
        self._threads[sid] = t
        t.start()
        return {"status": "started", "session_id": sid}

    def cancel(self, sid, status="canceled"):
        """Mark a run canceled so a worker cannot overwrite the terminal state on completion."""
        session = self.ctx.sessions.get(sid)
        if session is None:
            return {"error": "not_found", "session_id": sid}
        if session["status"] != "running":
            return {"error": "not_running", "status": session["status"], "session_id": sid}
        self._canceled.add(sid)
        self.ctx.sessions.set_status(sid, status)
        return {"status": status, "session_id": sid}

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, sid):
        session = self.ctx.sessions.get(sid)
        kind = session["kind"]
        state = session["state"]
        try:
            if kind == "backtest":
                results = self._run_backtest(sid, state)
            elif kind == "significance_test":
                results = self._run_significance(sid, state)
            elif kind == "monte_carlo":
                results = self._run_monte_carlo(sid, state)
            elif kind == "optimization":
                results = self._run_optimization(sid, state)
            else:
                raise ValueError(f"Unknown kind {kind}")
            if sid in self._canceled:
                return
            results["dashboard_url"] = self.ctx.write_report(sid, kind, state, results)
            self.ctx.sessions.set_results(sid, results, status="finished")
        except MissingCandles as e:
            if sid in self._canceled:
                return
            self.ctx.sessions.set_results(
                sid, {"error": "missing_candles", "message": str(e)}, status="stopped")
        except Exception as e:
            if sid in self._canceled:
                return
            self.ctx.sessions.set_results(
                sid, {"error": type(e).__name__, "message": str(e)}, status="stopped")
        finally:
            self._threads.pop(sid, None)
            self._canceled.discard(sid)

    # ------------------------------------------------------------------ candles
    def _prepare(self, state, start_date=None, finish_date=None):
        from ..data.binance import EXCHANGES
        exchange = state["exchange"]
        config = self.ctx.config.engine_config(state.get("config"))
        # the route's exchange drives the engine exchange (name must match candle keys),
        # and its market type (spot/futures) unless explicitly overridden in state config.
        overrides = state.get("config") or {}
        config["exchange"] = exchange
        if "type" not in overrides and exchange in EXCHANGES:
            config["type"] = EXCHANGES[exchange][2]
            if config["type"] == "spot":
                config["futures_leverage"] = 1
        routes = state.get("routes") or [{
            "exchange": exchange, "symbol": state["symbol"],
            "timeframe": state["timeframe"], "strategy": state["strategy"],
        }]
        data_routes = state.get("data_routes") or []
        if any(route.get("exchange", exchange) != exchange
               for route in routes + data_routes):
            raise ValueError("All routes in one research run must use the selected exchange.")
        route_pairs = [(route["exchange"], route["symbol"]) for route in routes]
        if len(route_pairs) != len(set(route_pairs)):
            raise ValueError("Two trading routes cannot use the same exchange-symbol pair.")

        warm = int(config.get("warm_up_candles", 0) or 0)
        all_routes = routes + data_routes
        max_tf = max(jh.timeframe_to_one_minutes(route["timeframe"])
                     for route in all_routes)
        warmup_ms = warm * max_tf * 60_000

        requested_start = start_date or state["start_date"]
        requested_finish = finish_date or state.get("finish_date")
        start_ts = jh.date_to_timestamp(requested_start)
        finish_ts = (jh.date_to_timestamp(requested_finish)
                     if requested_finish else jh.today_to_timestamp())
        if finish_ts <= start_ts:
            raise ValueError("finish_date must be after start_date")
        candles, warmup_candles = {}, {}
        unique_feeds = {(route["exchange"], route["symbol"]): route
                        for route in all_routes}
        for (feed_exchange, symbol), route in unique_feeds.items():
            rows = self.ctx.candle_db.get(
                feed_exchange, symbol, start_ts - warmup_ms, finish_ts)
            if len(rows) == 0:
                raise MissingCandles(
                    f"No candle data for {feed_exchange} {symbol} between "
                    f"{jh.timestamp_to_date(start_ts)} and {jh.timestamp_to_date(finish_ts)}. "
                    f"Import candles starting ~2 months before {requested_start} first."
                )
            trading = rows[rows[:, 0] >= start_ts]
            warmup = rows[rows[:, 0] < start_ts]
            tf = jh.timeframe_to_one_minutes(route["timeframe"])
            if len(trading) < tf * 2:
                raise MissingCandles(
                    f"Insufficient candle data for {feed_exchange} {symbol} in the "
                    "requested window."
                )
            key = jh.key(feed_exchange, symbol)
            candles[key] = {"exchange": feed_exchange, "symbol": symbol,
                            "candles": trading}
            if len(warmup) >= tf:
                warmup_candles[key] = {
                    "exchange": feed_exchange, "symbol": symbol, "candles": warmup,
                }
        # if we already have warmup candles, don't double-reserve inside the engine
        engine_config = dict(config)
        if warmup_candles:
            engine_config["warm_up_candles"] = 0
        return engine_config, routes, data_routes, candles, warmup_candles or None

    def _progress_cb(self, sid):
        def cb(done, total):
            if sid in self._canceled:
                raise InterruptedError("Research run canceled")
            self.ctx.sessions.set_progress(sid, int(min(done / max(total, 1), 1.0) * 100))
        return cb

    def _should_cancel(self, sid):
        return lambda: sid in self._canceled

    # ------------------------------------------------------------------ runners
    def _run_backtest(self, sid, state):
        config, routes, data_routes, candles, warmup = self._prepare(state)
        res = backtest(config, routes, data_routes, candles, warmup_candles=warmup,
                       generate_equity_curve=True,
                       generate_tradingview=state.get("export_tradingview", False),
                       generate_csv=state.get("export_csv", False),
                       generate_json=state.get("export_json", False),
                       generate_logs=state.get("debug_mode", False),
                       generate_charts=state.get("export_chart", False),
                       charts_output_root=f"{self.ctx.storage_dir}/backtest-charts",
                       benchmark=state.get("benchmark", False),
                       fast_mode=state.get("fast_mode", True),
                       hyperparameters=state.get("hyperparameters"),
                       strategies_dir=self.ctx.strategies_dir,
                       should_cancel=self._should_cancel(sid))
        self.ctx.sessions.set_progress(sid, 100)
        output = {
            "metrics": res["metrics"],
            "num_trades": len(res["trades"]),
            "trades": res["trades"][:500],
            "equity_curve": res.get("equity_curve", [])[::max(1, len(res.get("equity_curve", [])) // 500 or 1)],
        }
        for key in ("csv", "json", "tradingview", "benchmark", "charts_session_id",
                    "charts_folder", "logs"):
            if res.get(key) is not None:
                output[key] = res[key]
        return output

    def _run_significance(self, sid, state):
        config, routes, data_routes, candles, warmup = self._prepare(state)
        n_sims = int(state.get("n_simulations", 2000))
        res = rule_significance_test(config, routes, data_routes, candles, warmup_candles=warmup,
                                     hyperparameters=state.get("hyperparameters"),
                                     n_simulations=n_sims, strategies_dir=self.ctx.strategies_dir,
                                     random_seed=state.get("random_seed"),
                                     cpu_cores=state.get("cpu_cores"),
                                     progress_callback=self._progress_cb(sid),
                                     should_cancel=self._should_cancel(sid))
        # Jesse's pure research function returns the full ndarray for plotting. MCP
        # sessions retain the compact, JSON-safe statistics instead of thousands of
        # individual samples.
        res.pop("simulated_means", None)
        self.ctx.sessions.set_progress(sid, 100)
        return {"results": res}

    def _run_monte_carlo(self, sid, state):
        config, routes, data_routes, candles, warmup = self._prepare(state)
        num = int(state.get("num_scenarios", 200))
        run_candles = state.get("run_candles", True)
        run_trades = state.get("run_trades", False)
        pipeline_class, pipeline_params = _candle_pipeline(state)
        out = {}
        if run_candles:
            candles_result = monte_carlo_candles(
                config, routes, data_routes, candles, warmup_candles=warmup,
                hyperparameters=state.get("hyperparameters"), num_scenarios=num,
                cpu_cores=state.get("cpu_cores"),
                fast_mode=state.get("fast_mode", True),
                candles_pipeline_class=pipeline_class,
                candles_pipeline_kwargs=pipeline_params,
                strategies_dir=self.ctx.strategies_dir, progress_callback=self._progress_cb(sid),
                should_cancel=self._should_cancel(sid), max_equity_points=500)
            out["candles"] = _compact_monte_carlo(candles_result)
            out.setdefault("equity_curves", {})["candles"] = _monte_carlo_curves(
                candles_result)
        if run_trades:
            trades_result = monte_carlo_trades(
                config, routes, data_routes, candles, warmup_candles=warmup,
                hyperparameters=state.get("hyperparameters"),
                num_scenarios=num, cpu_cores=state.get("cpu_cores"),
                fast_mode=state.get("fast_mode", True),
                strategies_dir=self.ctx.strategies_dir,
                progress_callback=self._progress_cb(sid),
                should_cancel=self._should_cancel(sid), max_equity_points=500)
            out["trades"] = _compact_monte_carlo(trades_result)
            out.setdefault("equity_curves", {})["trades"] = _monte_carlo_curves(
                trades_result)
        self.ctx.sessions.set_progress(sid, 100)
        return out

    def _run_optimization(self, sid, state):
        explicit = state.get("training_start_date") is not None
        if explicit:
            config, routes, data_routes, training, training_warmup = self._prepare(
                state, state["training_start_date"], state["training_finish_date"])
            _, testing_routes, testing_data_routes, testing, testing_warmup = self._prepare(
                state, state["testing_start_date"], state["testing_finish_date"])
            if routes != testing_routes or data_routes != testing_data_routes:
                raise ValueError("Training and testing routes must match.")
            kwargs = {
                "training_candles": training,
                "training_warmup_candles": training_warmup,
                "testing_candles": testing,
                "testing_warmup_candles": testing_warmup,
            }
        else:
            config, routes, data_routes, training, training_warmup = self._prepare(state)
            kwargs = {"candles": training, "warmup_candles": training_warmup,
                      "train_test_split": state.get("train_test_split", 0.75)}
        if "n_trials" in state:
            kwargs["n_trials"] = int(state["n_trials"])
        else:
            kwargs["trials"] = int(state.get("trials", 200))
        res = optimize(
            config, routes, data_routes,
            objective_function=state.get("objective_function", state.get("objective", "sharpe")),
            optimal_total=state.get("optimal_total", 200),
            best_candidates_count=state.get("best_candidates_count", 20),
            fast_mode=state.get("fast_mode", True), cpu_cores=state.get("cpu_cores"),
            strategies_dir=self.ctx.strategies_dir,
            progress_callback=self._progress_cb(sid),
            should_cancel=self._should_cancel(sid), **kwargs)
        self.ctx.sessions.set_progress(sid, 100)
        return res


def _compact_monte_carlo(result):
    """Keep session polling light; curves remain available through their dedicated tool."""
    return {key: value for key, value in result.items()
            if key not in {"original", "scenarios"}}


def _monte_carlo_curves(result, max_points=500):
    return {
        "original": _scenario_curve(result.get("original"), max_points),
        "scenarios": [
            _scenario_curve(scenario, max_points)
            for scenario in result.get("scenarios", [])
        ],
    }


def _scenario_curve(scenario, max_points):
    if not scenario:
        return None
    curve = scenario.get("equity_curve") or []
    if curve and isinstance(curve[0], dict) and "data" in curve[0]:
        points = curve[0].get("data") or []
    else:
        points = curve
    step = max(1, int(np.ceil(len(points) / max_points)))
    sampled = points[::step]
    if points and sampled[-1] != points[-1]:
        sampled = [*sampled, points[-1]]
    metrics = scenario.get("metrics")
    if metrics is None:
        metrics = {key: scenario.get(key) for key in (
            "total_return", "final_value", "max_drawdown", "volatility",
            "sharpe_ratio", "calmar_ratio", "starting_balance",
        ) if scenario.get(key) is not None}
    return {
        "scenario_index": scenario.get("scenario_index"),
        "metrics": metrics,
        "equity_curve": [{"name": "Portfolio", "data": sampled}],
    }
