"""Configuration tools."""
import json

from ...context import get_context


def register_config_tools(mcp):
    @mcp.tool()
    def get_config() -> dict:
        """Return Terry's full saved configuration (balance, fee, leverage, exchange, defaults)."""
        return get_context().config.get()

    @mcp.tool()
    def update_config(config: str) -> dict:
        """Recursive-merge a partial config (JSON string) into the saved settings.

        Only use for user-driven changes (e.g. "set balance to 50000"), never to work around
        a tool error. Example: '{"starting_balance": 50000, "fee": 0.0006}'.
        """
        try:
            partial = json.loads(config) if isinstance(config, str) else config
        except json.JSONDecodeError as e:
            return {"error": "invalid_json", "message": str(e)}
        return {"status": "updated", "config": get_context().config.update(partial)}

    @mcp.tool()
    def get_backtest_config() -> dict:
        """Return the config used for backtests (engine-shaped)."""
        return get_context().config.backtest_config()

    @mcp.tool()
    def get_live_config() -> dict:
        """Return the live-trading config (note: live trading is not implemented in Terry)."""
        return get_context().config.live_config()

    @mcp.tool()
    def get_optimization_config() -> dict:
        """Return the optimization config (objective, trials, train/test split)."""
        return get_context().config.optimization_config()
