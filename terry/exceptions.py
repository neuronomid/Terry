"""Exceptions mirroring Jesse's exception surface."""


class TerryException(Exception):
    pass


class InvalidStrategy(TerryException):
    pass


class InvalidRoutes(TerryException):
    pass


class InvalidConfig(TerryException):
    pass


class InvalidDateRange(TerryException):
    pass


class CandleNotFoundInDatabase(TerryException):
    pass


class InsufficientMargin(TerryException):
    pass


class InvalidShortSellOnSpot(TerryException):
    pass


class ExchangeError(TerryException):
    pass


class Termination(TerryException):
    """Raised to gracefully abort a running session."""
    pass
