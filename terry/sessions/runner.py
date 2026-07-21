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
            if kind == "demo":
                # Live paper trading manages its own status (stays "running" and
                # streams updated results until stopped), so it returns here.
                self._run_demo_live(sid, state)
                return
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
    def _prepare(self, state, start_date=None, finish_date=None,
                 start_ts=None, finish_ts=None):
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
        start_ts = (int(start_ts) if start_ts is not None
                    else jh.date_to_timestamp(requested_start))
        finish_ts = (int(finish_ts) if finish_ts is not None else
                     (jh.date_to_timestamp(requested_finish)
                      if requested_finish else jh.today_to_timestamp()))
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
                    f"Import candles starting ~2 months before "
                    f"{jh.timestamp_to_date(start_ts)} first."
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
        # Keep every feed's 1m timeline the same length so the simulator's shared
        # index stays aligned across routes (feeds list at different dates, so BTC
        # may carry far more warm-up history than a newer symbol like ETH).
        if len(candles) > 1:
            min_trading = min(len(v["candles"]) for v in candles.values())
            for v in candles.values():
                v["candles"] = v["candles"][:min_trading]
        if len(warmup_candles) > 1:
            min_warmup = min(len(v["candles"]) for v in warmup_candles.values())
            for v in warmup_candles.values():
                v["candles"] = v["candles"][-min_warmup:]
        # Warm-up must be present for every feed or none, otherwise injection shifts
        # only some routes and desynchronises the shared timeline.
        if warmup_candles and len(warmup_candles) != len(candles):
            warmup_candles = {}
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
        daily_balance = res.get("daily_balance", []) or []
        output = {
            "metrics": res["metrics"],
            "num_trades": len(res["trades"]),
            # Retain the newest rows so the dashboard's default newest-first table always
            # surfaces newly completed trades even in very active sessions.
            "trades": res["trades"][-500:],
            "equity_curve": res.get("equity_curve", [])[::max(1, len(res.get("equity_curve", [])) // 500 or 1)],
            "daily_balance": daily_balance[::max(1, len(daily_balance) // 1000 or 1)],
            "monthly_returns": _compute_monthly_returns(daily_balance, state.get("start_date")),
        }
        for key in ("csv", "json", "tradingview", "benchmark", "charts_session_id",
                    "charts_folder", "logs", "chart_data"):
            if res.get(key) is not None:
                output[key] = res[key]
        if state.get("benchmark") and daily_balance:
            curve = self._benchmark_curve(state, res["metrics"].get("starting_balance"))
            if curve:
                output["benchmark_curve"] = curve
        return output

    # ------------------------------------------------------------------ live demo
    def _run_demo_live(self, sid, state):
        """Live paper trading: fetch fresh market candles, replay the strategy over a
        rolling [now - lookback, now] window with paper money, and stream the full
        backtest report. Loops until the user stops it. Because the engine is
        deterministic and look-ahead free, replaying the window each tick is
        equivalent to trading each new candle as it closes."""
        import time as _time
        exchange = state["exchange"]
        symbol = state["symbol"]
        timeframe = state["timeframe"]
        lookback_days = int(state.get("lookback_days", 14))
        config = self.ctx.config.engine_config(state.get("config"))
        warm = int(config.get("warm_up_candles", 0) or 0)
        warmup_ms = warm * jh.timeframe_to_one_minutes(timeframe) * 60_000
        # Market OHLC refreshes independently of strategy decisions. The current candle is
        # refreshed every second, while the heavier deterministic replay only runs when
        # another one-minute candle has closed.
        poll_seconds = max(1, int(state.get("poll_seconds", 1)))
        produced = False
        last_strategy_finish = None
        last_strategy_update = None
        live_tick = 0
        while sid not in self._canceled:
            try:
                now_ts = jh.now_to_timestamp(force_fresh=True)
                current_minute = now_ts // 60_000 * 60_000
                timeframe_ms = jh.timeframe_to_one_minutes(timeframe) * 60_000
                current_bucket = current_minute // timeframe_ms * timeframe_ms
                # Starting on a selected-timeframe boundary keeps the simulator's candle-close
                # decisions aligned with the same buckets shown on the live chart.
                start_ts = current_bucket - lookback_days * 86_400_000
                self._refresh_live_candles(exchange, symbol, start_ts - warmup_ms, now_ts, sid)
                if sid in self._canceled:
                    break
                live_candle = self._demo_live_candle(
                    exchange, symbol, timeframe, current_bucket, current_minute)
                if live_candle is not None:
                    # Candle ``time`` is its selected-timeframe bucket. ``tick_time`` tracks
                    # the current minute independently so an open-trade connector can advance
                    # without changing the candlestick's stable bucket identity.
                    live_candle["tick_time"] = int(current_minute / 1000)
                if not produced or current_minute != last_strategy_finish:
                    # Exclude the still-forming 1m candle. Stops/limits and selected-timeframe
                    # entry rules therefore only see completed market data.
                    output = self._demo_backtest_window(
                        sid, state, start_ts, current_minute)
                    last_strategy_finish = current_minute
                    last_strategy_update = now_ts
                else:
                    session = self.ctx.sessions.get(sid) or {}
                    output = dict(session.get("results") or {})
                output.pop("error", None)
                output.pop("message", None)
                live_tick += 1
                output["live"] = {
                    "is_live": True, "updated_at": now_ts, "poll_seconds": poll_seconds,
                    "tick": live_tick,
                    "lookback_days": lookback_days,
                    "window_start": jh.timestamp_to_date(start_ts),
                    "window_end": jh.timestamp_to_date(now_ts),
                    "window_start_ts": start_ts, "window_end_ts": current_minute,
                    "strategy_updated_at": last_strategy_update,
                    "price": live_candle.get("close") if live_candle else None,
                    "candle": live_candle,
                }
                self.ctx.sessions.set_results(sid, output, status="running")
                produced = True
            except InterruptedError:
                break
            except Exception as exc:  # keep the live session alive across transient errors
                message = f"{type(exc).__name__}: {exc}"
                # Never leave the UI looking silently frozen. Preserve the most recent valid
                # candle/report, advance the feed revision, and expose the delayed-feed error.
                live_tick += 1
                session = self.ctx.sessions.get(sid) or {}
                output = dict(session.get("results") or {})
                previous_live = dict(output.get("live") or {})
                output.update({"error": "live_error", "message": message})
                output["live"] = {
                    **previous_live, "is_live": True, "error": message,
                    "updated_at": jh.now_to_timestamp(force_fresh=True),
                    "poll_seconds": poll_seconds, "tick": live_tick,
                }
                self.ctx.sessions.set_results(sid, output, status="running")
            for _ in range(max(1, poll_seconds)):
                if sid in self._canceled:
                    break
                _time.sleep(1)
        self._finalize_demo(sid, state)

    def _finalize_demo(self, sid, state):
        session = self.ctx.sessions.get(sid)
        final = dict((session or {}).get("results") or {})
        if isinstance(final.get("live"), dict):
            final["live"] = {**final["live"], "is_live": False,
                             "stopped_at": jh.now_to_timestamp(force_fresh=True)}
        if final.get("metrics"):
            try:
                final["dashboard_url"] = self.ctx.write_report(sid, "demo", state, final)
            except Exception:
                pass
        self.ctx.sessions.set_results(sid, final, status="finished")

    def _refresh_live_candles(self, exchange, symbol, start_ts, finish_ts, sid):
        """Refresh the live tail of the local 1m store through ``finish_ts``.

        The newest exchange candle is still forming, so it must be fetched again instead of
        advancing past its timestamp. ``CandleDB.upsert`` replaces that row's evolving OHLCV.
        """
        from ..data.binance import fetch_1m_range
        existing = self.ctx.candle_db.get(exchange, symbol, start_ts, finish_ts)
        fetch_from = start_ts
        if len(existing):
            fetch_from = max(start_ts, int(existing[-1][0]))
        if fetch_from >= finish_ts:
            return
        chunk = fetch_1m_range(exchange, symbol, fetch_from, finish_ts,
                               should_stop=lambda: sid in self._canceled)
        if len(chunk):
            self.ctx.candle_db.upsert(exchange, symbol, chunk)

    def _demo_live_candle(self, exchange, symbol, timeframe, bucket_start, current_minute):
        """Build the selected timeframe's still-forming candle from refreshed 1m rows."""
        rows = self.ctx.candle_db.get(
            exchange, symbol, bucket_start, current_minute + 60_000)
        if not len(rows):
            return None
        return {
            "time": int(bucket_start / 1000),
            "open": float(rows[0, 1]), "close": float(rows[-1, 2]),
            "high": float(rows[:, 3].max()), "low": float(rows[:, 4].min()),
            "volume": float(rows[:, 5].sum()), "timeframe": timeframe,
        }

    def _demo_backtest_window(self, sid, state, start_ts, finish_ts):
        start_date = jh.timestamp_to_date(start_ts)
        finish_date = jh.timestamp_to_date(finish_ts)
        config, routes, data_routes, candles, warmup = self._prepare(
            state, start_date, finish_date, start_ts=start_ts, finish_ts=finish_ts)
        res = backtest(config, routes, data_routes, candles, warmup_candles=warmup,
                       generate_equity_curve=True, benchmark=state.get("benchmark", True),
                       fast_mode=state.get("fast_mode", True),
                       hyperparameters=state.get("hyperparameters"),
                       strategies_dir=self.ctx.strategies_dir,
                       should_cancel=self._should_cancel(sid))
        daily_balance = res.get("daily_balance", []) or []
        equity = res.get("equity_curve", []) or []
        output = {
            "metrics": res["metrics"],
            "num_trades": len(res["trades"]),
            "trades": res["trades"][-500:],
            "equity_curve": equity[::max(1, len(equity) // 500 or 1)],
            "daily_balance": daily_balance[::max(1, len(daily_balance) // 1000 or 1)],
            "monthly_returns": _compute_monthly_returns(daily_balance, start_date),
            "paper_account": _paper_account_summary(state, config, res["metrics"]),
        }
        if res.get("chart_data") is not None:
            output["chart_data"] = res["chart_data"]
        if daily_balance:
            curve = self._benchmark_curve(
                {**state, "start_date": start_date, "finish_date": finish_date},
                res["metrics"].get("starting_balance"))
            if curve:
                output["benchmark_curve"] = curve
        return output

    def _benchmark_curve(self, state, starting_balance):
        """Daily buy-and-hold equity for the primary route, for the equity-chart overlay."""
        try:
            routes = state.get("routes")
            exchange = routes[0]["exchange"] if routes else state["exchange"]
            symbol = routes[0]["symbol"] if routes else state["symbol"]
            start_ts = jh.date_to_timestamp(state["start_date"])
            finish_ts = jh.date_to_timestamp(state["finish_date"])
            raw = self.ctx.candle_db.get(exchange, symbol, start_ts, finish_ts)
            if len(raw) < 2:
                return None
            from ..engine.candle_store import aggregate_candles
            daily = aggregate_candles(raw, "1D")
            base = float(daily[0, 2]) or 1.0
            balance = float(starting_balance or daily[0, 2])
            return [{"time": int(c[0] / 1000), "value": balance * float(c[2]) / base}
                    for c in daily]
        except Exception:
            return None

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


def _paper_account_summary(state, config, metrics):
    """Paper-account P&L that excludes any deposits/withdrawals from trading profit."""
    transfers = state.get("transfers") or []
    deposits = sum(float(t.get("amount") or 0) for t in transfers if t.get("type") == "deposit")
    withdrawals = sum(float(t.get("amount") or 0) for t in transfers if t.get("type") == "withdraw")
    starting = float(config.get("starting_balance", 0) or 0)
    net_funded = starting + deposits - withdrawals
    final_equity = float(metrics.get("finishing_balance") or 0)
    net_pnl = final_equity - net_funded
    return {
        "starting_balance": starting, "deposits": deposits, "withdrawals": withdrawals,
        "net_funded": net_funded, "final_equity": final_equity, "net_pnl": net_pnl,
        "net_pnl_percentage": (net_pnl / net_funded * 100) if net_funded else 0.0,
        "transfers": transfers,
    }


def _compute_monthly_returns(daily_balance, start_date):
    """Monthly percentage returns grouped into a year x month grid (mirrors Jesse).

    Returns ``{"years": [...], "rows": [{"year", "months": [12 values|None], "total"}]}``
    where each monthly value is ``(last_balance / first_balance - 1) * 100`` for that
    calendar month and ``total`` is the compounded annual return.
    """
    if not daily_balance or len(daily_balance) < 2 or not start_date:
        return None
    from datetime import timedelta
    try:
        anchor = jh.timestamp_to_arrow(jh.date_to_timestamp(start_date)).datetime
    except Exception:
        return None
    buckets = {}
    for offset, balance in enumerate(daily_balance):
        day = anchor + timedelta(days=offset)
        key = (day.year, day.month)
        if key not in buckets:
            buckets[key] = {"first": balance, "last": balance}
        else:
            buckets[key]["last"] = balance
    monthly = {}
    for key in sorted(buckets):
        first, last = buckets[key]["first"], buckets[key]["last"]
        monthly[key] = ((last / first - 1) * 100) if first else 0.0
    years = sorted({year for year, _ in monthly})
    rows = []
    for year in years:
        months = [monthly.get((year, month)) for month in range(1, 13)]
        compound = 1.0
        seen = False
        for value in months:
            if value is not None:
                compound *= (1 + value / 100)
                seen = True
        rows.append({"year": year, "months": months,
                     "total": (compound - 1) * 100 if seen else None})
    return {"years": years, "rows": rows}


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
