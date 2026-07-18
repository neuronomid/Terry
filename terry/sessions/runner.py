"""
Background runner for sessions. Loads candles from the SQLite store, runs the appropriate
research function in a thread, streams progress into the session, and writes an HTML report.
"""
import threading

import numpy as np

from .. import helpers as jh
from ..research import (backtest, rule_significance_test,
                        monte_carlo_candles, monte_carlo_trades, optimize)


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
    def _prepare(self, state):
        from ..data.binance import EXCHANGES
        exchange = state["exchange"]
        symbol = state["symbol"]
        timeframe = state["timeframe"]
        config = self.ctx.config.engine_config(state.get("config"))
        # the route's exchange drives the engine exchange (name must match candle keys),
        # and its market type (spot/futures) unless explicitly overridden in state config.
        overrides = state.get("config") or {}
        config["exchange"] = exchange
        if "type" not in overrides and exchange in EXCHANGES:
            config["type"] = EXCHANGES[exchange][2]
            if config["type"] == "spot":
                config["futures_leverage"] = 1
        warm = int(config.get("warm_up_candles", 0) or 0)
        tf = jh.timeframe_to_one_minutes(timeframe)
        warmup_ms = warm * tf * 60_000

        start_ts = jh.date_to_timestamp(state["start_date"])
        finish_ts = (jh.date_to_timestamp(state["finish_date"])
                     if state.get("finish_date") else jh.today_to_timestamp())
        if finish_ts <= start_ts:
            raise ValueError("finish_date must be after start_date")

        rows = self.ctx.candle_db.get(exchange, symbol, start_ts - warmup_ms, finish_ts)
        if len(rows) == 0:
            raise MissingCandles(
                f"No candle data for {exchange} {symbol} between "
                f"{jh.timestamp_to_date(start_ts)} and {jh.timestamp_to_date(finish_ts)}. "
                f"Import candles starting ~2 months before {state['start_date']} first."
            )
        trading = rows[rows[:, 0] >= start_ts]
        warmup = rows[rows[:, 0] < start_ts]
        if len(trading) < tf * 2:
            raise MissingCandles(
                f"Insufficient candle data for {exchange} {symbol} in the requested window."
            )

        key = jh.key(exchange, symbol)
        candles = {key: {"exchange": exchange, "symbol": symbol, "candles": trading}}
        warmup_candles = ({key: {"exchange": exchange, "symbol": symbol, "candles": warmup}}
                          if len(warmup) >= tf else None)
        # if we already have warmup candles, don't double-reserve inside the engine
        engine_config = dict(config)
        if warmup_candles is not None:
            engine_config["warm_up_candles"] = 0
        routes = [{"exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                   "strategy": state["strategy"]}]
        return engine_config, routes, candles, warmup_candles

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
        config, routes, candles, warmup = self._prepare(state)
        res = backtest(config, routes, [], candles, warmup_candles=warmup,
                       generate_equity_curve=True, hyperparameters=state.get("hyperparameters"),
                       strategies_dir=self.ctx.strategies_dir,
                       should_cancel=self._should_cancel(sid))
        self.ctx.sessions.set_progress(sid, 100)
        return {
            "metrics": res["metrics"],
            "num_trades": len(res["trades"]),
            "trades": res["trades"][:500],
            "equity_curve": res.get("equity_curve", [])[::max(1, len(res.get("equity_curve", [])) // 500 or 1)],
        }

    def _run_significance(self, sid, state):
        config, routes, candles, warmup = self._prepare(state)
        n_sims = int(state.get("n_simulations", 2000))
        res = rule_significance_test(config, routes, [], candles, warmup_candles=warmup,
                                     hyperparameters=state.get("hyperparameters"),
                                     n_simulations=n_sims, strategies_dir=self.ctx.strategies_dir,
                                     progress_callback=self._progress_cb(sid),
                                     should_cancel=self._should_cancel(sid))
        self.ctx.sessions.set_progress(sid, 100)
        return {"results": res}

    def _run_monte_carlo(self, sid, state):
        config, routes, candles, warmup = self._prepare(state)
        num = int(state.get("num_scenarios", 200))
        run_candles = state.get("run_candles", True)
        run_trades = state.get("run_trades", False)
        out = {}
        if run_candles:
            out["candles"] = monte_carlo_candles(
                config, routes, [], candles, warmup_candles=warmup,
                hyperparameters=state.get("hyperparameters"), num_scenarios=num,
                strategies_dir=self.ctx.strategies_dir, progress_callback=self._progress_cb(sid),
                should_cancel=self._should_cancel(sid))
        if run_trades:
            base = backtest(config, routes, [], candles, warmup_candles=warmup,
                            hyperparameters=state.get("hyperparameters"),
                            strategies_dir=self.ctx.strategies_dir,
                            should_cancel=self._should_cancel(sid))
            out["trades"] = monte_carlo_trades(
                base["trades"], num_scenarios=max(500, num * 5),
                starting_balance=config["starting_balance"],
                should_cancel=self._should_cancel(sid))
        self.ctx.sessions.set_progress(sid, 100)
        return out

    def _run_optimization(self, sid, state):
        config, routes, candles, warmup = self._prepare(state)
        res = optimize(config, routes, [], candles, warmup_candles=warmup,
                       objective=state.get("objective", "sharpe_ratio"),
                       n_trials=int(state.get("n_trials", 100)),
                       train_test_split=state.get("train_test_split", 0.75),
                       strategies_dir=self.ctx.strategies_dir,
                       progress_callback=self._progress_cb(sid),
                       should_cancel=self._should_cancel(sid))
        self.ctx.sessions.set_progress(sid, 100)
        return res
