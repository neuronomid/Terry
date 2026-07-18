"""Exceptions mirroring Jesse's exception surface."""


class TerryException(Exception):
    pass


class EmptyPosition(TerryException):
    pass


class OpenPositionError(TerryException):
    pass


class OrderNotAllowed(TerryException):
    pass


class InvalidStrategy(TerryException):
    pass


class ConflictingRules(TerryException):
    pass


class InvalidRoutes(TerryException):
    pass


class RouteNotFound(TerryException):
    def __init__(self, symbol, timeframe):
        super().__init__(
            f"Data route is required but missing: symbol='{symbol}', "
            f"timeframe='{timeframe}'")


class InvalidConfig(TerryException):
    pass


class InvalidDateRange(TerryException):
    pass


class InvalidTimeframe(TerryException):
    pass


class CandleNotFoundInDatabase(TerryException):
    pass


class CandleNotFoundInExchange(TerryException):
    pass


class CandlesNotFound(TerryException):
    pass


class SymbolNotFound(TerryException):
    pass


class InvalidSymbol(TerryException):
    pass


class ExchangeInMaintenance(TerryException):
    pass


class ExchangeNotResponding(TerryException):
    pass


class ExchangeRejectedOrder(TerryException):
    pass


class ExchangeRejectedLeverageNumber(TerryException):
    pass


class ExchangeOrderNotFound(TerryException):
    pass


class InvalidShape(TerryException):
    pass


class NegativeBalance(TerryException):
    pass


class InsufficientMargin(TerryException):
    pass


class InsufficientBalance(TerryException):
    pass


class InvalidShortSellOnSpot(InvalidStrategy):
    pass


class InvalidExchangeApiKeys(TerryException):
    pass


class ExchangeError(TerryException):
    pass


class NotSupportedError(TerryException):
    pass


class Termination(TerryException):
    """Raised to gracefully abort a running session."""
    pass
