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
        return c.get_session(session_id)

    @mcp.tool()
    def get_monte_carlo_sessions(limit: int = 20) -> dict:
        """List recent Monte Carlo sessions."""
        return c.list_sessions("monte_carlo", limit)

    @mcp.tool()
    def get_monte_carlo_equity_curves(session_id: str) -> dict:
        """Return per-scenario summary stats for a finished candles Monte Carlo session."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"error": "not_found"}
        res = (s.get("results") or {}).get("candles")
        if not res:
            return {"error": "no_candles_results", "message": "Run candles Monte Carlo first."}
        return {"summary_metrics": res.get("summary_metrics"), "num_scenarios": res.get("num_scenarios")}

    @mcp.tool()
    def get_monte_carlo_logs(session_id: str) -> dict:
        """Return diagnostic info for a Monte Carlo session (status + any error)."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"error": "not_found"}
        return {"status": s["status"], "results": s.get("results")}

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
