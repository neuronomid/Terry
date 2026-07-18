"""Backtest tools (draft → run → poll)."""
from . import _common as c


def register_backtest_tools(mcp):
    @mcp.tool()
    def create_backtest_draft(strategy: str, symbol: str = None, timeframe: str = None,
                              exchange: str = None, start_date: str = None,
                              finish_date: str = None, config: str = None) -> dict:
        """Create a backtest draft. All params except `strategy` default from saved config.

        finish_date defaults to yesterday; start_date defaults to ~1 year earlier.
        `config` is an optional JSON string of engine overrides (starting_balance, fee, type,
        futures_leverage, warm_up_candles). Then call run_backtest(session_id).
        """
        state, err = c.build_base_state(strategy, symbol, timeframe, exchange,
                                        start_date, finish_date, config)
        if err:
            return {"error": "invalid_config", "message": err}
        return c.create_draft("backtest", state)

    @mcp.tool()
    def update_backtest_draft(backtest_id: str, state: str) -> dict:
        """Update a backtest draft's state (JSON string) before running it."""
        return c.update_draft("backtest", backtest_id, state)

    @mcp.tool()
    def update_backtest_notes(session_id: str, notes: str) -> dict:
        """Attach or update free-text notes on a backtest session."""
        return c.update_notes(session_id, notes)

    @mcp.tool()
    def get_backtest_session(session_id: str) -> dict:
        """Get a backtest session's status, progress, and (when finished) metrics + dashboard_url."""
        return c.get_session(session_id)

    @mcp.tool()
    def get_backtest_sessions(limit: int = 20) -> dict:
        """List recent backtest sessions."""
        return c.list_sessions("backtest", limit)

    @mcp.tool()
    def run_backtest(session_id: str) -> dict:
        """Run a backtest draft (returns immediately). Poll get_backtest_session until terminal."""
        return c.run_session(session_id, "backtest")

    @mcp.tool()
    def cancel_backtest(session_id: str) -> dict:
        """Cancel a running backtest session."""
        return c.cancel_session(session_id, "backtest")

    @mcp.tool()
    def purge_backtest_sessions(days_old: int = None) -> dict:
        """Delete backtest sessions (older than days_old, or all if omitted)."""
        return c.purge_sessions("backtest", days_old)
