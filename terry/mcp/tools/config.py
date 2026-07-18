"""Configuration tools."""
import json

from ...context import get_context


def register_config_tools(mcp):
    @mcp.tool()
    def get_config() -> dict:
        """Return Terry's full saved configuration (balance, fee, leverage, exchange, defaults)."""
        config = get_context().config.get()
        return {"status": "success", "config": config,
                "message": "Configuration loaded successfully", **config}

    @mcp.tool()
    def update_config(config: str) -> dict:
        """Recursive-merge a partial config (JSON string) into the saved settings.

        Only use for user-driven changes (e.g. "set balance to 50000"), never to work around
        a tool error. Example: '{"starting_balance": 50000, "fee": 0.0006}'.
        """
        try:
            partial = json.loads(config) if isinstance(config, str) else config
        except json.JSONDecodeError as e:
            return {"status": "error", "error": "Invalid JSON format",
                    "error_code": "invalid_json", "details": str(e),
                    "message": "Failed to parse configuration JSON"}
        if not isinstance(partial, dict):
            return {"status": "error", "error": "Invalid configuration",
                    "error_code": "invalid_config",
                    "message": "Configuration must be a JSON object"}
        try:
            updated = get_context().config.update(partial)
        except (TypeError, ValueError) as exc:
            return {"status": "error", "error": "Invalid configuration",
                    "error_code": "invalid_config", "details": str(exc),
                    "message": "Failed to update configuration"}
        return {"status": "success", "session_status": "updated",
                "config": updated, "message": "Configuration updated successfully"}

    @mcp.tool()
    def get_backtest_config() -> dict:
        """Return the config used for backtests (engine-shaped)."""
        config = get_context().config.backtest_config()
        return {"status": "success", "section": "backtest", "config": config,
                "message": 'Configuration section "backtest" loaded successfully', **config}

    @mcp.tool()
    def get_live_config() -> dict:
        """Return the live-trading config (note: live trading is not implemented in Terry)."""
        config = get_context().config.live_config()
        return {"status": "success", "section": "live", "config": config,
                "message": 'Configuration section "live" loaded successfully', **config}

    @mcp.tool()
    def get_optimization_config() -> dict:
        """Return the optimization config (objective, trials, train/test split)."""
        config = get_context().config.optimization_config()
        return {"status": "success", "section": "optimization", "config": config,
                "message": 'Configuration section "optimization" loaded successfully', **config}
