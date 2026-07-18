import sys

from . import order as _order_module
from . import position as _position_module
from . import route as _route_module
from . import trade as _trade_module
from .order import Order
from .position import Position
from .trade import ClosedTrade
from .route import Route

# Jesse uses capitalized model module paths. Pre-register aliases to the loaded
# lowercase modules before importlib can replace these package-level class
# exports with module objects (for example after importing ``models.Order``).
sys.modules.setdefault(f"{__name__}.Order", _order_module)
sys.modules.setdefault(f"{__name__}.Position", _position_module)
sys.modules.setdefault(f"{__name__}.ClosedTrade", _trade_module)
sys.modules.setdefault(f"{__name__}.Route", _route_module)

__all__ = ["Order", "Position", "ClosedTrade", "Route"]
