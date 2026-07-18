"""Optimization tools (draft → run → poll), with explicit out-of-sample validation."""
from . import _common as c
from ...context import get_context

_EXAMPLE_ROUTES = ('[{"exchange": "Binance Perpetual Futures", '
                   '"strategy": "ExampleStrategy", "symbol": "BTC-USDT", '
                   '"timeframe": "4h"}]')


def register_optimization_tools(mcp):
    @mcp.tool()
    def create_optimization_draft(
            exchange: str = "Binance Perpetual Futures", routes: str = _EXAMPLE_ROUTES,
            data_routes: str = "[]", training_start_date: str = "2021-01-01",
            training_finish_date: str = "2022-06-01",
            testing_start_date: str = "2022-06-01",
            testing_finish_date: str = "2023-01-01", optimal_total: int = 50,
            objective_function: str = "sharpe", trials: int = 200,
            best_candidates_count: int = 20, warm_up_candles: int = 210,
            fast_mode: bool = True, cpu_cores: int = None, title: str = None,
            description: str = None, strategy_summary: str = None,
            hypothesis: str = None, rationale: str = None, *,
            strategy: str = None, symbol: str = None, timeframe: str = None,
            start_date: str = None, finish_date: str = None,
            objective: str = None, n_trials: int = None,
            train_test_split: float = 0.75, config: str = None) -> dict:
        """Create an optimization draft. The strategy must define hyperparameters().

        Jesse-style callers may provide JSON routes and separate training/testing date windows.
        Terry's shorthand start/finish + n_trials form remains supported.
        """
        terry_dates = start_date is not None or finish_date is not None
        explicit_windows = not terry_dates
        if terry_dates and not all((start_date, finish_date)):
            return c._error(
                "invalid_config", "start_date and finish_date are required together.")
        if explicit_windows and not all((training_start_date, training_finish_date,
                                         testing_start_date, testing_finish_date)):
            return c._error(
                "invalid_config", "All four training/testing date fields are required together.")
        base_start = start_date if terry_dates else training_start_date
        base_finish = finish_date if terry_dates else training_finish_date
        route_input = None if strategy is not None and routes == _EXAMPLE_ROUTES else routes
        state, err = c.build_routes_state(
            strategy, symbol, timeframe, exchange, base_start, base_finish, config,
            route_input, data_routes)
        if err:
            return c._error("invalid_config", err)
        if int(optimal_total) <= 1 or int(best_candidates_count) < 1:
            return c._error(
                "invalid_config",
                "optimal_total must be greater than 1 and best_candidates_count at least 1.")
        if ((n_trials is not None and int(n_trials) < 1)
                or (n_trials is None and int(trials) < 1)):
            return c._error("invalid_config", "trials must be at least 1.")
        if not 0.1 < float(train_test_split) < 0.9:
            return c._error(
                "invalid_config",
                "train_test_split must be greater than 0.1 and less than 0.9.")
        if cpu_cores is not None and int(cpu_cores) < 1:
            return c._error(
                "invalid_config", "cpu_cores must be an integer greater than 0.")
        resolved_objective = (objective or objective_function or "sharpe").lower()
        if resolved_objective not in {
                "sharpe", "sharpe_ratio", "calmar", "sortino", "omega",
                "serenity", "smart sharpe", "smart sortino"}:
            return c._error(
                "invalid_config",
                f'Unsupported objective_function "{resolved_objective}".')
        state.update({
            "objective_function": resolved_objective,
            "train_test_split": float(train_test_split),
            "optimal_total": int(optimal_total),
            "best_candidates_count": int(best_candidates_count),
            "fast_mode": bool(fast_mode),
            "cpu_cores": (int(cpu_cores) if cpu_cores is not None
                          else c.default_cpu_cores()),
        })
        if n_trials is not None:
            state["n_trials"] = int(n_trials)
        else:
            state["trials"] = int(trials)
        if explicit_windows:
            state.update({
                "training_start_date": training_start_date,
                "training_finish_date": training_finish_date,
                "testing_start_date": testing_start_date,
                "testing_finish_date": testing_finish_date,
            })
        state.setdefault("config", {})["warm_up_candles"] = int(warm_up_candles)
        notes = "\n\n".join(filter(None, [description, strategy_summary,
                                           hypothesis, rationale]))
        return c.create_draft(
            "optimization", state, notes=notes, title=title, description=description)

    @mcp.tool()
    def update_optimization_draft(session_id: str, state: str) -> dict:
        """Update an optimization draft's state (JSON string)."""
        return c.update_draft("optimization", session_id, state)

    @mcp.tool()
    def update_optimization_notes(session_id: str, title: str = None,
                                  description: str = None,
                                  strategy_codes: str = None, *,
                                  notes: str = None) -> dict:
        """Update Jesse-compatible note metadata and captured strategy source."""
        return c.update_notes(session_id, title, description, strategy_codes, notes)

    @mcp.tool()
    def get_optimization_session(session_id: str) -> dict:
        """Get an optimization session's status and (when finished) best hp + candidates."""
        return c.get_session(session_id)

    @mcp.tool()
    def get_optimization_sessions(limit: int = 50, offset: int = 0,
                                  title_search: str = None,
                                  status_filter: str = None,
                                  date_filter: str = None) -> dict:
        """List optimization sessions with Jesse-compatible pagination and filters."""
        return c.list_sessions(
            "optimization", limit, offset, title_search, status_filter, date_filter)

    @mcp.tool()
    def get_optimization_logs(session_id: str) -> dict:
        """Return diagnostic info for an optimization session (status + any error)."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"status": "error", "error": "not_found",
                    "message": f"Log file for session {session_id} not found"}
        result = s.get("results") or {}
        error = result.get("message") or result.get("error") or ""
        return {"status": "success", "session_id": session_id,
                "session_status": s["status"], "logs": str(error),
                "results": s.get("results"),
                "message": "Optimization diagnostics retrieved successfully"}

    @mcp.tool()
    def run_optimization(session_id: str) -> dict:
        """Run an optimization draft (returns immediately). Poll until terminal."""
        return c.run_session(session_id, "optimization")

    @mcp.tool()
    def rerun_optimization(session_id: str) -> dict:
        """Re-run a finished/stopped optimization session."""
        return c.run_session(session_id, "optimization")

    @mcp.tool()
    def cancel_optimization(session_id: str) -> dict:
        """Cancel a running optimization session."""
        return c.cancel_session(session_id, "optimization")

    @mcp.tool()
    def terminate_optimization(session_id: str) -> dict:
        """Force-terminate an optimization session."""
        return c.cancel_session(session_id, "optimization", new_status="terminated")

    @mcp.tool()
    def purge_optimization_sessions(days_old: int = None) -> dict:
        """Delete optimization sessions (older than days_old, or all if omitted)."""
        return c.purge_sessions("optimization", days_old)
