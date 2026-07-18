"""Backtest tools (draft → run → poll)."""
from . import _common as c

_EXAMPLE_ROUTES = ('[{"exchange": "Binance Perpetual Futures", '
                   '"strategy": "ExampleStrategy", "symbol": "BTC-USDT", '
                   '"timeframe": "4h"}]')


def register_backtest_tools(mcp):
    @mcp.tool()
    def create_backtest_draft(
            exchange: str = "Binance Perpetual Futures", routes: str = _EXAMPLE_ROUTES,
            data_routes: str = "[]", start_date: str = "2024-01-01",
            finish_date: str = "2024-03-01", debug_mode: bool = False,
            export_csv: bool = False, export_json: bool = False,
            export_chart: bool = True, export_tradingview: bool = False,
            fast_mode: bool = True, benchmark: bool = True, title: str = None,
            description: str = None, strategy_summary: str = None,
            change_summary: str = None, rationale: str = None, *,
            strategy: str = None, symbol: str = None, timeframe: str = None,
            config: str = None) -> dict:
        """Create a backtest draft using shorthand fields or Jesse-compatible JSON routes.

        The leading defaults mirror Jesse's MCP contract. Terry's shorthand fields remain
        available as keyword-only extensions.
        `config` is an optional JSON string of engine overrides (starting_balance, fee, type,
        futures_leverage, warm_up_candles). Then call run_backtest(session_id).
        """
        route_input = None if strategy is not None and routes == _EXAMPLE_ROUTES else routes
        state, err = c.build_routes_state(
            strategy, symbol, timeframe, exchange, start_date, finish_date, config,
            route_input, data_routes)
        if err:
            return c._error("invalid_config", err)
        state.update({
            "debug_mode": debug_mode, "export_csv": export_csv, "export_json": export_json,
            "export_chart": export_chart, "export_tradingview": export_tradingview,
            "fast_mode": fast_mode, "benchmark": benchmark,
        })
        notes = "\n\n".join(filter(None, [description, strategy_summary,
                                           change_summary, rationale]))
        return c.create_draft(
            "backtest", state, notes=notes, title=title, description=description)

    @mcp.tool()
    def update_backtest_draft(backtest_id: str, state: str) -> dict:
        """Update a backtest draft's state (JSON string) before running it."""
        return c.update_draft("backtest", backtest_id, state)

    @mcp.tool()
    def update_backtest_notes(session_id: str, title: str = None,
                              description: str = None,
                              strategy_codes: str = None, *, notes: str = None) -> dict:
        """Update Jesse-compatible note metadata and captured strategy source."""
        return c.update_notes(session_id, title, description, strategy_codes, notes)

    @mcp.tool()
    def get_backtest_session(session_id: str) -> dict:
        """Get a backtest session's status, progress, and (when finished) metrics + dashboard_url."""
        return c.get_session(session_id)

    @mcp.tool()
    def get_backtest_sessions(limit: int = 50, offset: int = 0,
                              title_search: str = None, status_filter: str = None,
                              date_filter: str = None) -> dict:
        """List backtest sessions with Jesse-compatible pagination and filters."""
        return c.list_sessions(
            "backtest", limit, offset, title_search, status_filter, date_filter)

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
