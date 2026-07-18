"""Optimization tools (draft → run → poll), with explicit out-of-sample validation."""
from . import _common as c
from ...context import get_context


def register_optimization_tools(mcp):
    @mcp.tool()
    def create_optimization_draft(strategy: str = None, symbol: str = None, timeframe: str = None,
                                  exchange: str = None, start_date: str = None,
                                  finish_date: str = None, objective: str = "sharpe_ratio",
                                  n_trials: int = None, train_test_split: float = 0.75,
                                  config: str = None, routes: str = None,
                                  data_routes: str = "[]", training_start_date: str = None,
                                  training_finish_date: str = None,
                                  testing_start_date: str = None,
                                  testing_finish_date: str = None,
                                  optimal_total: int = 50,
                                  objective_function: str = None, trials: int = 200,
                                  best_candidates_count: int = 20,
                                  warm_up_candles: int = None, fast_mode: bool = True,
                                  cpu_cores: int = None, title: str = None,
                                  description: str = None, strategy_summary: str = None,
                                  hypothesis: str = None, rationale: str = None) -> dict:
        """Create an optimization draft. The strategy must define hyperparameters().

        Jesse-style callers may provide JSON routes and separate training/testing date windows.
        Terry's shorthand start/finish + n_trials form remains supported.
        """
        explicit_windows = any((training_start_date, training_finish_date,
                                testing_start_date, testing_finish_date))
        if explicit_windows and not all((training_start_date, training_finish_date,
                                         testing_start_date, testing_finish_date)):
            return {"error": "invalid_config", "message":
                    "All four training/testing date fields are required together."}
        base_start = training_start_date or start_date
        base_finish = training_finish_date or finish_date
        state, err = c.build_routes_state(
            strategy, symbol, timeframe, exchange, base_start, base_finish, config,
            routes, data_routes)
        if err:
            return {"error": "invalid_config", "message": err}
        state.update({
            "objective_function": objective_function or objective,
            "train_test_split": float(train_test_split),
            "optimal_total": int(optimal_total),
            "best_candidates_count": int(best_candidates_count),
            "fast_mode": bool(fast_mode), "cpu_cores": cpu_cores,
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
        if warm_up_candles is not None:
            state.setdefault("config", {})["warm_up_candles"] = int(warm_up_candles)
        notes = "\n\n".join(filter(None, [title, description, strategy_summary,
                                           hypothesis, rationale]))
        return c.create_draft("optimization", state, notes=notes)

    @mcp.tool()
    def update_optimization_draft(session_id: str, state: str) -> dict:
        """Update an optimization draft's state (JSON string)."""
        return c.update_draft("optimization", session_id, state)

    @mcp.tool()
    def update_optimization_notes(session_id: str, notes: str) -> dict:
        """Attach or update notes on an optimization session."""
        return c.update_notes(session_id, notes)

    @mcp.tool()
    def get_optimization_session(session_id: str) -> dict:
        """Get an optimization session's status and (when finished) best hp + candidates."""
        return c.get_session(session_id)

    @mcp.tool()
    def get_optimization_sessions(limit: int = 20) -> dict:
        """List recent optimization sessions."""
        return c.list_sessions("optimization", limit)

    @mcp.tool()
    def get_optimization_logs(session_id: str) -> dict:
        """Return diagnostic info for an optimization session (status + any error)."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"error": "not_found"}
        return {"status": s["status"], "results": s.get("results")}

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
