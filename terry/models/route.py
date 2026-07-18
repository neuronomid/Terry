class Route:
    """A trading or data route: exchange + symbol + timeframe (+ strategy for trading routes)."""

    def __init__(self, exchange, symbol, timeframe, strategy_name=None):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_name = strategy_name
        self.strategy = None  # the instantiated Strategy object (set by the engine)

    def __repr__(self):
        return (
            f"Route({self.exchange}, {self.symbol}, {self.timeframe}, "
            f"{self.strategy_name})"
        )
