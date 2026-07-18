"""Register all Terry MCP tools (same tool names/semantics as Jesse's MCP server)."""
from .strategy import register_strategy_tools
from .backtest import register_backtest_tools
from .config import register_config_tools
from .candles import register_candles_tools
from .indicator import register_indicator_tools
from .significance_test import register_significance_test_tools
from .monte_carlo import register_monte_carlo_tools
from .optimization import register_optimization_tools
from .general import register_general_tools


def register_tools(mcp):
    register_general_tools(mcp)
    register_strategy_tools(mcp)
    register_backtest_tools(mcp)
    register_config_tools(mcp)
    register_indicator_tools(mcp)
    register_candles_tools(mcp)
    register_significance_test_tools(mcp)
    register_monte_carlo_tools(mcp)
    register_optimization_tools(mcp)
