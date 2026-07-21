"""Non-crypto instrument registry.

Terry symbols are dashed ``BASE-QUOTE`` (for example ``EUR-USD``, ``XAU-USD``,
``US500-USD``, ``AAPL-USD``). Each non-crypto symbol resolves to:

* a **Dukascopy** datafeed instrument code plus the integer price ``scale`` used
  to decode its ``.bi5`` tick files (historical 1-minute backfill), and
* a **Yahoo Finance** ticker used for Demo Mode's live chart / forming candle.

For stock **indices** the Yahoo ticker is deliberately a futures or ETF proxy
(``ES=F``/``NQ=F``/``YM=F`` …) because Yahoo delays the raw index level ~15
minutes, whereas the matching future streams in near real time.

Price scale reference (verified against the live Dukascopy feed):
  * FX majors (non-JPY quote): 100000   (EUR-USD ``108443`` -> 1.08443)
  * FX with a JPY quote:       1000     (USD-JPY ``156615`` -> 156.615)
  * metals / energy / indices / stock CFDs: 1000
"""
from __future__ import annotations

# --- asset classes -----------------------------------------------------------
CRYPTO = "Cryptocurrency"
FOREX = "Forex"
METALS = "Metals"
ENERGY = "Energy"
INDICES = "Indices"
STOCKS = "Stocks"

# Ordered for the dashboard "Assets" selector. Crypto keeps its existing exchanges;
# every other class is served historically by the Dukascopy driver.
ASSET_ORDER = [CRYPTO, FOREX, METALS, ENERGY, INDICES, STOCKS]

# The single historical source powering every non-crypto asset class.
DUKASCOPY_EXCHANGE = "Dukascopy"


def _entry(symbol, asset, dukascopy, scale, yahoo, description):
    return {
        "symbol": symbol, "asset": asset, "dukascopy": dukascopy,
        "scale": scale, "yahoo": yahoo, "description": description,
    }


# Keyed by the dashless, upper-cased symbol (``EUR-USD`` -> ``EURUSD``).
_ENTRIES = [
    # ---- Forex majors / crosses -------------------------------------------------
    _entry("EUR-USD", FOREX, "EURUSD", 100000, "EURUSD=X", "Euro / US Dollar"),
    _entry("GBP-USD", FOREX, "GBPUSD", 100000, "GBPUSD=X", "British Pound / US Dollar"),
    _entry("USD-JPY", FOREX, "USDJPY", 1000, "USDJPY=X", "US Dollar / Japanese Yen"),
    _entry("AUD-USD", FOREX, "AUDUSD", 100000, "AUDUSD=X", "Australian Dollar / US Dollar"),
    _entry("USD-CHF", FOREX, "USDCHF", 100000, "USDCHF=X", "US Dollar / Swiss Franc"),
    _entry("USD-CAD", FOREX, "USDCAD", 100000, "USDCAD=X", "US Dollar / Canadian Dollar"),
    _entry("NZD-USD", FOREX, "NZDUSD", 100000, "NZDUSD=X", "New Zealand Dollar / US Dollar"),
    _entry("EUR-JPY", FOREX, "EURJPY", 1000, "EURJPY=X", "Euro / Japanese Yen"),
    _entry("GBP-JPY", FOREX, "GBPJPY", 1000, "GBPJPY=X", "British Pound / Japanese Yen"),
    _entry("EUR-GBP", FOREX, "EURGBP", 100000, "EURGBP=X", "Euro / British Pound"),
    # ---- Metals -----------------------------------------------------------------
    _entry("XAU-USD", METALS, "XAUUSD", 1000, "GC=F", "Spot Gold (Yahoo: gold future)"),
    _entry("XAG-USD", METALS, "XAGUSD", 1000, "SI=F", "Spot Silver (Yahoo: silver future)"),
    # ---- Energy -----------------------------------------------------------------
    _entry("WTI-USD", ENERGY, "LIGHTCMDUSD", 1000, "CL=F", "WTI Crude Oil"),
    _entry("BRENT-USD", ENERGY, "BRENTCMDUSD", 1000, "BZ=F", "Brent Crude Oil"),
    # ---- Indices (Yahoo ticker is a real-time futures/ETF proxy) -----------------
    _entry("US500-USD", INDICES, "USA500IDXUSD", 1000, "ES=F", "S&P 500 (Yahoo: E-mini future)"),
    _entry("US100-USD", INDICES, "USATECHIDXUSD", 1000, "NQ=F", "Nasdaq 100 (Yahoo: E-mini future)"),
    _entry("US30-USD", INDICES, "USA30IDXUSD", 1000, "YM=F", "Dow Jones 30 (Yahoo: E-mini future)"),
    _entry("DE40-EUR", INDICES, "DEUIDXEUR", 1000, "^GDAXI", "DAX 40"),
    _entry("UK100-GBP", INDICES, "GBRIDXGBP", 1000, "^FTSE", "FTSE 100"),
    _entry("JP225-JPY", INDICES, "JPNIDXJPY", 1000, "^N225", "Nikkei 225"),
    # ---- US stock CFDs ----------------------------------------------------------
    _entry("AAPL-USD", STOCKS, "AAPLUSUSD", 1000, "AAPL", "Apple Inc."),
    _entry("MSFT-USD", STOCKS, "MSFTUSUSD", 1000, "MSFT", "Microsoft Corp."),
    _entry("AMZN-USD", STOCKS, "AMZNUSUSD", 1000, "AMZN", "Amazon.com Inc."),
    _entry("TSLA-USD", STOCKS, "TSLAUSUSD", 1000, "TSLA", "Tesla Inc."),
    _entry("NVDA-USD", STOCKS, "NVDAUSUSD", 1000, "NVDA", "NVIDIA Corp."),
    _entry("GOOG-USD", STOCKS, "GOOGUSUSD", 1000, "GOOG", "Alphabet Inc. (Class C)"),
]

def _dashless(symbol: str) -> str:
    return symbol.replace("-", "").upper()


# Keyed by the dashless, upper-cased symbol (``EUR-USD`` -> ``EURUSD``).
INSTRUMENTS = {_dashless(entry["symbol"]): entry for entry in _ENTRIES}


def resolve(symbol: str) -> dict | None:
    """Return the registry entry for a Terry symbol, or ``None`` if unknown."""
    if not symbol:
        return None
    return INSTRUMENTS.get(_dashless(symbol))


def is_registered(symbol: str) -> bool:
    return resolve(symbol) is not None


def dukascopy_instrument(symbol: str):
    """Return ``(dukascopy_code, price_scale)`` for a Terry symbol.

    Raises ``ValueError`` with the list of supported symbols when unknown, since the
    ``.bi5`` feed cannot be decoded without the correct instrument code and price scale.
    """
    entry = resolve(symbol)
    if entry is None:
        supported = ", ".join(e["symbol"] for e in _ENTRIES)
        raise ValueError(
            f"'{symbol}' is not a supported Dukascopy instrument. "
            f"Supported symbols: {supported}."
        )
    return entry["dukascopy"], entry["scale"]


def yahoo_ticker(symbol: str) -> str | None:
    """Yahoo Finance ticker for Demo Mode, or ``None`` when unregistered."""
    entry = resolve(symbol)
    return entry["yahoo"] if entry else None


def symbols_for_asset(asset: str) -> list[str]:
    return [e["symbol"] for e in _ENTRIES if e["asset"] == asset]


def dashboard_asset_classes(crypto_exchanges) -> list[dict]:
    """Metadata for the dashboard "Assets" selector.

    Crypto maps to its existing exchange list; every other class maps to the
    Dukascopy source together with its supported example symbols.
    """
    classes = [{
        "name": CRYPTO, "exchanges": list(crypto_exchanges),
        "example": "BTC-USDT", "symbols": [],
    }]
    for asset in ASSET_ORDER:
        if asset == CRYPTO:
            continue
        symbols = symbols_for_asset(asset)
        if not symbols:
            continue
        classes.append({
            "name": asset, "exchanges": [DUKASCOPY_EXCHANGE],
            "example": symbols[0], "symbols": symbols,
        })
    return classes
