"""Enumerations mirroring Jesse's constants (timeframes, sides, order types/status)."""


class timeframes:
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    MINUTE_45 = "45m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_3 = "3h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1D"
    DAY_3 = "3D"
    WEEK_1 = "1W"


# Ordered list of supported timeframes (must divide evenly from 1m where possible)
ALL_TIMEFRAMES = [
    "1m", "3m", "5m", "15m", "30m", "45m",
    "1h", "2h", "3h", "4h", "6h", "8h", "12h",
    "1D", "3D", "1W",
]


class sides:
    BUY = "buy"
    SELL = "sell"


class trade_types:
    LONG = "long"
    SHORT = "short"


class order_types:
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class order_statuses:
    ACTIVE = "ACTIVE"       # submitted, waiting to fill
    EXECUTED = "EXECUTED"   # filled
    CANCELED = "CANCELED"
    QUEUED = "QUEUED"


class order_roles:
    OPEN_POSITION = "OPEN POSITION"
    CLOSE_POSITION = "CLOSE POSITION"
    INCREASE_POSITION = "INCREASE POSITION"
    REDUCE_POSITION = "REDUCE POSITION"


class exchange_types:
    SPOT = "spot"
    FUTURES = "futures"
