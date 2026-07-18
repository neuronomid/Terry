"""Rule Significance Test tools (draft → run → poll)."""
from . import _common as c


def register_significance_test_tools(mcp):
    @mcp.tool()
    def create_significance_test_draft(strategy: str, symbol: str = None, timeframe: str = None,
                                       exchange: str = None, start_date: str = None,
                                       finish_date: str = None, n_simulations: int = 2000,
                                       hypothesis: str = "", rationale: str = "",
                                       config: str = None) -> dict:
        """Create a Rule Significance Test draft for an entry signal (exactly one route).

        Validates whether should_long/should_short have a genuine edge vs random via a bootstrap
        p-value. Use n_simulations >= 2000. Then call run_significance_test(session_id).
        """
        state, err = c.build_base_state(strategy, symbol, timeframe, exchange,
                                        start_date, finish_date, config)
        if err:
            return {"error": "invalid_config", "message": err}
        state["n_simulations"] = int(n_simulations)
        state["hypothesis"] = hypothesis
        state["rationale"] = rationale
        return c.create_draft("significance_test", state)

    @mcp.tool()
    def update_significance_test_draft(session_id: str, state: str) -> dict:
        """Update a significance-test draft's state (JSON string)."""
        return c.update_draft("significance_test", session_id, state)

    @mcp.tool()
    def update_significance_test_notes(session_id: str, notes: str) -> dict:
        """Attach or update notes on a significance-test session."""
        return c.update_notes(session_id, notes)

    @mcp.tool()
    def get_significance_test_session(session_id: str) -> dict:
        """Get a significance-test session's status and (when finished) p_value + interpretation."""
        return c.get_session(session_id)

    @mcp.tool()
    def get_significance_test_sessions(limit: int = 20) -> dict:
        """List recent significance-test sessions."""
        return c.list_sessions("significance_test", limit)

    @mcp.tool()
    def run_significance_test(session_id: str) -> dict:
        """Run a significance-test draft (returns immediately). Poll until terminal."""
        return c.run_session(session_id, "significance_test")

    @mcp.tool()
    def cancel_significance_test(session_id: str) -> dict:
        """Cancel a running significance-test session."""
        return c.cancel_session(session_id, "significance_test")

    @mcp.tool()
    def purge_significance_test_sessions(days_old: int = None) -> dict:
        """Delete significance-test sessions (older than days_old, or all if omitted)."""
        return c.purge_sessions("significance_test", days_old)
