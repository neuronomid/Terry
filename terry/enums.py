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
    MONTH_1 = "1M"


# Ordered list of supported timeframes (must divide evenly from 1m where possible)
ALL_TIMEFRAMES = [
    "1m", "3m", "5m", "15m", "30m", "45m",
    "1h", "2h", "3h", "4h", "6h", "8h", "12h",
    "1D", "3D", "1W", "1M",
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
    FOK = "FOK"
    STOP_LIMIT = "STOP LIMIT"


class order_statuses:
    ACTIVE = "ACTIVE"       # submitted, waiting to fill
    EXECUTED = "EXECUTED"   # filled
    CANCELED = "CANCELED"
    QUEUED = "QUEUED"
    PARTIALLY_FILLED = "PARTIALLY FILLED"
    LIQUIDATED = "LIQUIDATED"
    REJECTED = "REJECTED"


class order_roles:
    OPEN_POSITION = "OPEN POSITION"
    CLOSE_POSITION = "CLOSE POSITION"
    INCREASE_POSITION = "INCREASE POSITION"
    REDUCE_POSITION = "REDUCE POSITION"


class exchange_types:
    SPOT = "spot"
    FUTURES = "futures"


class colors:
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    MAGENTA = "magenta"
    BLACK = "black"


class exchanges:
    SANDBOX = "Sandbox"
    COINBASE_SPOT = "Coinbase Spot"
    BITFINEX_SPOT = "Bitfinex Spot"
    BINANCE_SPOT = "Binance Spot"
    BINANCE_US_SPOT = "Binance US Spot"
    BINANCE_PERPETUAL_FUTURES = "Binance Perpetual Futures"
    BINANCE_PERPETUAL_FUTURES_TESTNET = "Binance Perpetual Futures Testnet"
    BYBIT_USDT_PERPETUAL = "Bybit USDT Perpetual"
    BYBIT_USDC_PERPETUAL = "Bybit USDC Perpetual"
    BYBIT_USDT_PERPETUAL_TESTNET = "Bybit USDT Perpetual Testnet"
    BYBIT_USDC_PERPETUAL_TESTNET = "Bybit USDC Perpetual Testnet"
    BYBIT_SPOT = "Bybit Spot"
    BYBIT_SPOT_TESTNET = "Bybit Spot Testnet"
    FTX_PERPETUAL_FUTURES = "FTX Perpetual Futures"
    FTX_SPOT = "FTX Spot"
    FTX_US_SPOT = "FTX US Spot"
    BITGET_SPOT = "Bitget Spot"
    BITGET_USDT_PERPETUAL = "Bitget USDT Perpetual"
    BITGET_USDT_PERPETUAL_TESTNET = "Bitget USDT Perpetual Testnet"
    DYDX_PERPETUAL = "Dydx Perpetual"
    DYDX_PERPETUAL_TESTNET = "Dydx Perpetual Testnet"
    APEX_OMNI_PERPETUAL_TESTNET = "Apex Omni Perpetual Testnet"
    APEX_OMNI_PERPETUAL = "Apex Omni Perpetual"
    GATE_USDT_PERPETUAL = "Gate USDT Perpetual"
    GATE_SPOT = "Gate Spot"
    HYPERLIQUID_PERPETUAL = "Hyperliquid Perpetual"
    HYPERLIQUID_PERPETUAL_TESTNET = "Hyperliquid Perpetual Testnet"
    LIGHTER_PERPETUAL = "Lighter Perpetual"
    LIGHTER_PERPETUAL_TESTNET = "Lighter Perpetual Testnet"
    KRAKEN_SPOT = "Kraken Pro Spot"
    KRAKEN_PERPETUAL = "Kraken Pro Futures"
    KRAKEN_PERPETUAL_TESTNET = "Kraken Pro Futures Testnet"


class migration_actions:
    ADD = "add"
    DROP = "drop"
    RENAME = "rename"
    MODIFY_TYPE = "modify_type"
    ALLOW_NULL = "allow_null"
    DENY_NULL = "deny_null"
    ADD_INDEX = "add_index"
    DROP_INDEX = "drop_index"


class order_submitted_via:
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"


class live_session_statuses:
    DRAFT = "draft"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"


class live_session_modes:
    LIVETRADE = "livetrade"
    PAPERTRADE = "papertrade"
