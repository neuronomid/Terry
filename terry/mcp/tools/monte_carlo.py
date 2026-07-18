"""Monte Carlo tools (draft → run → poll), candles + trades modes."""
from . import _common as c
from ...context import get_context


def register_monte_carlo_tools(mcp):
    @mcp.tool()
    def create_monte_carlo_draft(strategy: str, symbol: str = None, timeframe: str = None,
                                 exchange: str = None, start_date: str = None,
                                 finish_date: str = None, num_scenarios: int = 200,
                                 run_candles: bool = True, run_trades: bool = False,
                                 config: str = None) -> dict:
        """Create a Monte Carlo robustness draft.

        Default = candles mode (`run_candles=True, run_trades=False`), 200 scenarios — answers
        "is this backtest overfit/lucky?". Only enable run_trades on explicit request.
        Then call run_monte_carlo(session_id).
        """
        state, err = c.build_base_state(strategy, symbol, timeframe, exchange,
                                        start_date, finish_date, config)
        if err:
            return {"error": "invalid_config", "message": err}
        state.update({"num_scenarios": int(num_scenarios),
                      "run_candles": bool(run_candles), "run_trades": bool(run_trades)})
        return c.create_draft("monte_carlo", state)

    @mcp.tool()
    def update_monte_carlo_draft(session_id: str, state: str) -> dict:
        """Update a Monte Carlo draft's state (JSON string)."""
        return c.update_draft("monte_carlo", session_id, state)

    @mcp.tool()
    def update_monte_carlo_notes(session_id: str, notes: str) -> dict:
        """Attach or update notes on a Monte Carlo session."""
        return c.update_notes(session_id, notes)

    @mcp.tool()
    def get_monte_carlo_session(session_id: str) -> dict:
        """Get a Monte Carlo session's status and (when finished) summary_metrics + verdict."""
        payload = c.get_session(session_id)
        if isinstance(payload.get("results"), dict):
            payload["results"].pop("equity_curves", None)
        return payload

    @mcp.tool()
    def get_monte_carlo_sessions(limit: int = 20) -> dict:
        """List recent Monte Carlo sessions."""
        return c.list_sessions("monte_carlo", limit)

    @mcp.tool()
    def get_monte_carlo_equity_curves(session_id: str) -> dict:
        """Return per-scenario Portfolio equity curves for custom analysis or charts."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"status": "error", "error": "not_found",
                    "message": f"Session {session_id} not found"}
        curves = (s.get("results") or {}).get("equity_curves")
        if not curves:
            return {"status": "error", "error": "no_equity_curves",
                    "message": "The session has no Monte Carlo equity curves."}
        return {"status": "success", "session_id": session_id,
                "trades": curves.get("trades"), "candles": curves.get("candles"),
                "message": "Equity curves retrieved successfully"}

    @mcp.tool()
    def get_monte_carlo_logs(session_id: str) -> dict:
        """Return diagnostic info for a Monte Carlo session (status + any error)."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"status": "error", "error": "not_found",
                    "message": f"Session {session_id} not found"}
        result = s.get("results") or {}
        error = result.get("message") or result.get("error") or ""
        return {"status": "success", "session_id": session_id,
                "logs": str(error), "session_status": s["status"],
                "message": "Monte Carlo diagnostics retrieved successfully"}

    @mcp.tool()
    def run_monte_carlo(session_id: str) -> dict:
        """Run a Monte Carlo draft (returns immediately). Poll until terminal (may take a while)."""
        return c.run_session(session_id, "monte_carlo")

    @mcp.tool()
    def resume_monte_carlo(session_id: str) -> dict:
        """Resume a stopped/terminated Monte Carlo session by re-running it."""
        return c.run_session(session_id, "monte_carlo")

    @mcp.tool()
    def cancel_monte_carlo(session_id: str) -> dict:
        """Cancel a running Monte Carlo session."""
        return c.cancel_session(session_id, "monte_carlo")

    @mcp.tool()
    def terminate_monte_carlo(session_id: str) -> dict:
        """Force-terminate a Monte Carlo session."""
        return c.cancel_session(session_id, "monte_carlo", new_status="terminated")

    @mcp.tool()
    def purge_monte_carlo_sessions(days_old: int = None) -> dict:
        """Delete Monte Carlo sessions (older than days_old, or all if omitted)."""
        return c.purge_sessions("monte_carlo", days_old)
