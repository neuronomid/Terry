"""Terry — a local, self-contained MCP-server clone of the Jesse crypto trading framework."""
from .version import __version__
from .strategy import Strategy, cached
from . import indicators
from . import utils
from . import helpers

# Jesse-compatible import aliases so strategy source is drop-in:
#   from terry.strategies import Strategy
#   import terry.indicators as ta
#   from terry import utils
import sys as _sys
import types as _types

_strategies_shim = _types.ModuleType("terry.strategies")
_strategies_shim.Strategy = Strategy
_strategies_shim.cached = cached
_sys.modules.setdefault("terry.strategies", _strategies_shim)

__all__ = ["Strategy", "cached", "indicators", "utils", "helpers", "__version__"]
