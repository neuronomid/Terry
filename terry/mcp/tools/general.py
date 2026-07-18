"""General/status tools."""
from ...version import __version__
from ...context import get_context
from ... import indicators as ta
from ...data.binance import EXCHANGES


def register_general_tools(mcp):
    @mcp.tool()
    def get_terry_status() -> dict:
        """Return Terry's status: version, project paths, config summary, and data counts."""
        ctx = get_context()
        cfg = ctx.config.get()
        existing = ctx.candle_db.existing()
        return {
            "name": "Terry", "version": __version__, "status": "running",
            "health_status": "ok", "message": "Terry is running",
            "project_root": ctx.project_root,
            "strategies_dir": ctx.strategies_dir,
            "config": {k: cfg[k] for k in ("exchange", "starting_balance", "fee", "type",
                                           "futures_leverage")},
            "indicators_available": len(ta.__all__),
            "supported_exchanges": list(EXCHANGES),
            "datasets": len(existing),
            "sessions": {
                "backtest": len(ctx.sessions.list("backtest", 1000)),
                "significance_test": len(ctx.sessions.list("significance_test", 1000)),
                "monte_carlo": len(ctx.sessions.list("monte_carlo", 1000)),
                "optimization": len(ctx.sessions.list("optimization", 1000)),
            },
        }

    @mcp.tool()
    def greet_user(name: str) -> dict:
        """Greet the user by name."""
        return {"status": "success", "action": "greeting", "user_name": name,
                "message": f"Hello, {name}! Terry is ready to research, backtest, and "
                           f"stress-test trading strategies."}
