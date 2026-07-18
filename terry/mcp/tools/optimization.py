"""Optimization tools (draft → run → poll), random search with train/test split."""
from . import _common as c
from ...context import get_context


def register_optimization_tools(mcp):
    @mcp.tool()
    def create_optimization_draft(strategy: str, symbol: str = None, timeframe: str = None,
                                  exchange: str = None, start_date: str = None,
                                  finish_date: str = None, objective: str = "sharpe_ratio",
                                  n_trials: int = 100, train_test_split: float = 0.75,
                                  config: str = None) -> dict:
        """Create an optimization draft. The strategy must define hyperparameters().

        Optimizes on the training window and validates the best candidates out-of-sample on the
        test window. objective is a metric key (sharpe_ratio, net_profit_percentage, calmar_ratio…).
        Then call run_optimization(session_id).
        """
        state, err = c.build_base_state(strategy, symbol, timeframe, exchange,
                                        start_date, finish_date, config)
        if err:
            return {"error": "invalid_config", "message": err}
        state.update({"objective": objective, "n_trials": int(n_trials),
                      "train_test_split": float(train_test_split)})
        return c.create_draft("optimization", state)

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
