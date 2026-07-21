"""Candle data tools: import / status / inspect / delete."""
from ... import helpers as jh
from ...context import get_context
from ...data.binance import EXCHANGES
from ...engine.candle_store import aggregate_candles


def register_candles_tools(mcp):
    @mcp.tool()
    def import_candles(exchange: str, symbol: str, start_date: str,
                       import_id: str = None, *, finish_date: str = None) -> dict:
        """Start importing 1m candles from a free public source (returns immediately).

        Poll get_candle_import_status(import_id) until status == "finished". Data already in
        the store is skipped automatically, so re-running from the same start_date is safe.

        Crypto exchanges (Binance/Bybit/…) provide their own symbols. The "Dukascopy"
        source provides non-crypto assets (Forex, metals, energy, indices, stock CFDs) as
        BASE-QUOTE symbols, e.g. "EUR-USD", "XAU-USD", "WTI-USD", "US500-USD", "AAPL-USD".

        Args:
            exchange: e.g. "Binance Perpetual Futures", "Binance Spot", or "Dukascopy".
            symbol:   e.g. "BTC-USDT" (crypto) or "EUR-USD" (Dukascopy).
            start_date: "YYYY-MM-DD".
            finish_date: optional "YYYY-MM-DD" (defaults to today).
        """
        ctx = get_context()
        if exchange not in EXCHANGES:
            return {"status": "error", "action": "candle_import_failed",
                    "error": "unknown_exchange", "error_type": "validation_error",
                    "exchange": exchange, "symbol": symbol,
                    "supported": list(EXCHANGES),
                    "message": f"Unknown exchange: {exchange}"}
        try:
            import_id = ctx.importer.start_import(
                exchange, symbol, start_date, finish_date, import_id=import_id)
        except ValueError as e:
            return {"status": "error", "action": "candle_import_failed",
                    "error": "invalid_input", "error_type": "validation_error",
                    "exchange": exchange, "symbol": symbol, "message": str(e)}
        return {"status": "started", "action": "candle_import_started",
                "import_id": import_id, "exchange": exchange, "symbol": symbol,
                "start_date": start_date,
                "message": f"Importing {symbol} 1m candles from {exchange}. "
                           f"Check get_candle_import_status('{import_id}')."}

    @mcp.tool()
    def cancel_candle_import(import_id: str) -> dict:
        """Cancel an in-progress candle import."""
        ok = get_context().importer.cancel(import_id)
        if not ok:
            return {"status": "error", "action": "cancel_failed",
                    "error": "not_found", "error_type": "not_found",
                    "import_id": import_id,
                    "message": f"Running candle import {import_id} not found"}
        return {"status": "success", "action": "candle_import_cancelled",
                "session_status": "canceled", "import_id": import_id,
                "message": f"Candle import process {import_id} has been requested for termination"}

    @mcp.tool()
    def get_candle_import_status(import_id: str) -> dict:
        """Get the live status/progress of a candle import (fast lookup)."""
        result = get_context().importer.get_status(import_id)
        result.setdefault("message", f"Import {import_id} is {result['status']}.")
        if result["status"] == "not_found":
            result.update({"status": "error", "error": "not_found",
                           "error_type": "not_found"})
        return result

    @mcp.tool()
    def clear_candle_cache() -> dict:
        """Clear any in-memory candle cache. (Terry reads candles directly from SQLite.)"""
        return {"status": "success", "action": "cache_cleared",
                "message": "No in-memory cache to clear; Terry reads directly from SQLite."}

    @mcp.tool()
    def get_candles(exchange: str, symbol: str, timeframe: str) -> dict:
        """Return coverage info and the last few candles for a given exchange/symbol/timeframe."""
        ctx = get_context()
        cov = ctx.candle_db.coverage(exchange, symbol)
        if cov is None:
            return {"status": "error", "action": "candles_retrieval_failed",
                    "error": "no_data", "error_type": "not_found",
                    "exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                    "message": f"No candles for {exchange} {symbol}."}
        arr = ctx.candle_db.get(exchange, symbol)
        try:
            agg = aggregate_candles(arr, timeframe)
        except ValueError as exc:
            return {"status": "error", "action": "candles_retrieval_failed",
                    "error": "invalid_timeframe", "error_type": "validation_error",
                    "exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                    "message": str(exc)}
        last = agg[-5:].tolist() if len(agg) else []
        return {"status": "success", "action": "candles_retrieved",
                "exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                "coverage": cov, "count": len(agg), "candle_count": len(agg),
                "candles": agg.tolist(),
                "last_candles": [
                    {"time": jh.timestamp_to_time(int(c[0])), "open": c[1], "close": c[2],
                     "high": c[3], "low": c[4], "volume": c[5]} for c in last],
                "message": f"Retrieved {len(agg)} candles for {symbol} on {exchange} ({timeframe})"}

    @mcp.tool()
    def get_existing_candles() -> dict:
        """List all exchange/symbol pairs with stored candle data and their date ranges."""
        rows = get_context().candle_db.existing()
        return {"status": "success", "action": "existing_candles_retrieved",
                "candle_sets_count": len(rows), "candle_sets": rows,
                "existing": rows,
                "message": f"Found {len(rows)} candle datasets in database"}

    @mcp.tool()
    def delete_candles(exchange: str, symbol: str) -> dict:
        """Delete all stored candles for an exchange/symbol pair."""
        get_context().candle_db.delete(exchange, symbol)
        return {"status": "success", "action": "candles_deleted",
                "session_status": "deleted", "exchange": exchange, "symbol": symbol,
                "message": f"Candles for {symbol} on {exchange} deleted successfully"}
