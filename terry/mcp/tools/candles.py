"""Candle data tools: import / status / inspect / delete."""
from ... import helpers as jh
from ...context import get_context
from ...data.binance import EXCHANGES
from ...engine.candle_store import aggregate_candles


def register_candles_tools(mcp):
    @mcp.tool()
    def import_candles(exchange: str, symbol: str, start_date: str,
                       finish_date: str = None) -> dict:
        """Start importing 1m candles from a free public exchange API (returns immediately).

        Poll get_candle_import_status(import_id) until status == "finished". Data already in
        the store is skipped automatically, so re-running from the same start_date is safe.

        Args:
            exchange: e.g. "Binance Perpetual Futures" or "Binance Spot".
            symbol:   e.g. "BTC-USDT".
            start_date: "YYYY-MM-DD".
            finish_date: optional "YYYY-MM-DD" (defaults to today).
        """
        ctx = get_context()
        if exchange not in EXCHANGES:
            return {"error": "unknown_exchange", "supported": list(EXCHANGES)}
        try:
            import_id = ctx.importer.start_import(exchange, symbol, start_date, finish_date)
        except ValueError as e:
            return {"error": "invalid_input", "message": str(e)}
        return {"status": "started", "import_id": import_id,
                "message": f"Importing {symbol} 1m candles from {exchange}. "
                           f"Check get_candle_import_status('{import_id}')."}

    @mcp.tool()
    def cancel_candle_import(import_id: str) -> dict:
        """Cancel an in-progress candle import."""
        ok = get_context().importer.cancel(import_id)
        return {"status": "canceled" if ok else "not_found", "import_id": import_id}

    @mcp.tool()
    def get_candle_import_status(import_id: str) -> dict:
        """Get the live status/progress of a candle import (fast lookup)."""
        return get_context().importer.get_status(import_id)

    @mcp.tool()
    def clear_candle_cache() -> dict:
        """Clear any in-memory candle cache. (Terry reads candles directly from SQLite.)"""
        return {"status": "ok", "message": "No in-memory cache to clear."}

    @mcp.tool()
    def get_candles(exchange: str, symbol: str, timeframe: str) -> dict:
        """Return coverage info and the last few candles for a given exchange/symbol/timeframe."""
        ctx = get_context()
        cov = ctx.candle_db.coverage(exchange, symbol)
        if cov is None:
            return {"error": "no_data", "message": f"No candles for {exchange} {symbol}."}
        arr = ctx.candle_db.get(exchange, symbol)
        agg = aggregate_candles(arr, timeframe)
        last = agg[-5:].tolist() if len(agg) else []
        return {"exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                "coverage": cov, "count": len(agg),
                "last_candles": [
                    {"time": jh.timestamp_to_time(int(c[0])), "open": c[1], "close": c[2],
                     "high": c[3], "low": c[4], "volume": c[5]} for c in last]}

    @mcp.tool()
    def get_existing_candles() -> dict:
        """List all exchange/symbol pairs with stored candle data and their date ranges."""
        return {"existing": get_context().candle_db.existing()}

    @mcp.tool()
    def delete_candles(exchange: str, symbol: str) -> dict:
        """Delete all stored candles for an exchange/symbol pair."""
        get_context().candle_db.delete(exchange, symbol)
        return {"status": "deleted", "exchange": exchange, "symbol": symbol}
