"""Rule Significance Test tools (draft → run → poll)."""
from . import _common as c

_EXAMPLE_ROUTES = ('[{"exchange": "Binance Perpetual Futures", '
                   '"strategy": "ExampleStrategy", "symbol": "BTC-USDT", '
                   '"timeframe": "4h"}]')


def register_significance_test_tools(mcp):
    @mcp.tool()
    def create_significance_test_draft(
            exchange: str = "Binance Perpetual Futures", routes: str = _EXAMPLE_ROUTES,
            data_routes: str = "[]", start_date: str = "2021-01-01",
            finish_date: str = "2022-01-01", n_simulations: int = 2000,
            random_seed: int = None, title: str = None,
            description: str = None, strategy_summary: str = None,
            hypothesis: str = None, rationale: str = None, *,
            strategy: str = None, symbol: str = None, timeframe: str = None,
            config: str = None, cpu_cores: int = None) -> dict:
        """Create a Rule Significance Test draft for an entry signal (exactly one route).

        Validates whether should_long/should_short have a genuine edge vs random via a bootstrap
        p-value. Use n_simulations >= 2000. Then call run_significance_test(session_id).
        """
        route_input = None if strategy is not None and routes == _EXAMPLE_ROUTES else routes
        state, err = c.build_routes_state(
            strategy, symbol, timeframe, exchange, start_date, finish_date,
            config, route_input, data_routes)
        if err:
            return {"error": "invalid_config", "message": err}
        if int(n_simulations) < 2_000:
            return {"error": "invalid_config", "message":
                    "n_simulations must be at least 2000."}
        if cpu_cores is not None and int(cpu_cores) < 1:
            return {"error": "invalid_config", "message":
                    "cpu_cores must be an integer greater than 0."}
        if len(state.get("routes") or [state]) != 1:
            return {"error": "invalid_config", "message":
                    "Rule Significance Test requires exactly one trading route."}
        state["n_simulations"] = int(n_simulations)
        state["random_seed"] = random_seed
        state["cpu_cores"] = cpu_cores
        state["hypothesis"] = hypothesis or ""
        state["rationale"] = rationale or ""
        notes = "\n\n".join(filter(None, [title, description, strategy_summary,
                                           hypothesis, rationale]))
        return c.create_draft("significance_test", state, notes=notes)

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
