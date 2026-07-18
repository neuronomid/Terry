"""Helper functions — a focused analog of jesse.helpers (jh)."""
import base64
from bisect import bisect_left
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
import platform
from pprint import pprint
import random
import string
import sys
import uuid
from datetime import datetime
from multiprocessing import cpu_count
from typing import Any

import arrow
import click
import numpy as np

# ---------------------------------------------------------------------------
# Timeframe math
# ---------------------------------------------------------------------------
_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "45m": 45,
    "1h": 60, "2h": 120, "3h": 180, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1D": 1440, "1d": 1440, "3D": 4320, "3d": 4320,
    "1W": 10080, "1w": 10080, "1M": 43_200,
}


def timeframe_to_one_minutes(timeframe: str) -> int:
    try:
        return _TF_MINUTES[timeframe]
    except KeyError:
        raise ValueError(
            f"Timeframe '{timeframe}' is invalid. Supported: {list(_TF_MINUTES)}"
        )


def max_timeframe(timeframes_list) -> str:
    order = list(_TF_MINUTES.keys())
    best = order[0]
    for tf in timeframes_list:
        if order.index(tf) > order.index(best):
            best = tf
    return best


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
def dashless_symbol(symbol: str) -> str:
    return symbol.replace("-", "")


def dashy_symbol(symbol: str) -> str:
    if "-" in symbol:
        return symbol
    # try common quote assets
    for quote in ("USDT", "USDC", "BUSD", "FDUSD", "USD", "UST", "EUT",
                  "EUR", "GBP", "JPY", "MIM", "TRY", "BTC", "ETH"):
        if symbol.endswith(quote):
            return f"{symbol[:-len(quote)]}-{quote}"
    return symbol


def base_asset(symbol: str) -> str:
    return symbol.split("-")[0]


def quote_asset(symbol: str) -> str:
    try:
        return symbol.split("-")[1]
    except IndexError as exc:
        from .exceptions import InvalidRoutes
        raise InvalidRoutes(
            "The symbol format is incorrect. Correct example: 'BTC-USDT'. "
            f"Yours is '{symbol}'") from exc


def key(exchange: str, symbol: str, timeframe: str = None) -> str:
    if timeframe is None:
        return f"{exchange}-{symbol}"
    return f"{exchange}-{symbol}-{timeframe}"


# ---------------------------------------------------------------------------
# Time / dates  (all timestamps are milliseconds since epoch, UTC)
# ---------------------------------------------------------------------------
def now_to_timestamp(force_fresh=False) -> int:
    if not force_fresh:
        try:
            from .store import get_current_store
            current = get_current_store()
            if current.app.is_active and current.app.time is not None:
                return int(current.app.time)
        except RuntimeError:
            pass
    return arrow.utcnow().int_timestamp * 1000


def today_to_timestamp() -> int:
    return arrow.utcnow().floor("day").int_timestamp * 1000


def date_to_timestamp(date: str) -> int:
    """'2021-01-01' -> ms timestamp (UTC midnight)."""
    return arrow.get(date, "YYYY-MM-DD").int_timestamp * 1000


def timestamp_to_date(timestamp: int) -> str:
    return str(arrow.get(timestamp / 1000))[:10]


def timestamp_to_iso8601(timestamp: int) -> str:
    return arrow.get(timestamp / 1000).isoformat()


def timestamp_to_time(timestamp: int) -> str:
    return str(arrow.get(timestamp / 1000))


def date_diff_in_days(date1: arrow.Arrow, date2: arrow.Arrow) -> int:
    if type(date1) is not arrow.Arrow or type(date2) is not arrow.Arrow:
        raise TypeError("dates must be Arrow instances")
    return abs((date2 - date1).days)


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------
def generate_unique_id() -> str:
    return str(uuid.uuid4())


def is_valid_uuid(uuid_to_test: str, version: int = 4) -> bool:
    """Return whether a string is a canonical UUID of the requested version."""
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
    except (AttributeError, TypeError, ValueError):
        return False
    return str(uuid_obj) == uuid_to_test


def random_str(num_characters: int = 8) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(num_characters))


def string_after_character(s: str, character: str) -> str:
    try:
        return s.split(character, 1)[1]
    except IndexError:
        return None


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def convert_number(old_max: float, old_min: float, new_max: float,
                   new_min: float, old_value: float) -> float:
    if old_value > old_max or old_value < old_min:
        raise ValueError(
            f"old_value:{old_value} must be within the range. {old_min}-{old_max}")
    return (((old_value - old_min) * (new_max - new_min)) /
            (old_max - old_min)) + new_min


def dna_to_hp(strategy_hp, dna: str):
    """Decode Jesse's JSON/Base64 or legacy printable-character DNA format."""
    try:
        decoded = json.loads(base64.b64decode(dna).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("DNA payload must be a JSON object")
        return decoded
    except (ValueError, UnicodeDecodeError, base64.binascii.Error,
            json.JSONDecodeError):
        hp = {}
        for gene, item in zip(dna, strategy_hp):
            value = convert_number(
                119, 40, item["max"], item["min"], ord(gene))
            if item["type"] is int:
                value = int(round(value))
            elif item["type"] is not float:
                raise TypeError("Only int and float types are implemented")
            hp[item["name"]] = value
        return hp


def round_price_for_live_mode(price, precision: int):
    return np.round(price, precision)


def floor_with_precision(num: float, precision: int = 0) -> float:
    temp = 10 ** precision
    return __import__("math").floor(num * temp) / temp


def prepare_qty(qty, side: str) -> float:
    if side.lower() in ("sell", "short"):
        return -abs(qty)
    if side.lower() in ("buy", "long"):
        return abs(qty)
    if side.lower() == "close":
        return 0.0
    raise ValueError(f"{side} is not a valid input")


def np_shift(arr: np.ndarray, num: int, fill_value=0) -> np.ndarray:
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def np_ffill(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [
        np.arange(k)[
            tuple(slice(None) if dim == i else np.newaxis for dim in range(len(arr.shape)))
        ]
        for i, k in enumerate(arr.shape)
    ]
    slc[axis] = idx
    return arr[tuple(slc)]


def same_length(bigger: np.ndarray, shorter: np.ndarray) -> np.ndarray:
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))


# ---------------------------------------------------------------------------
# Candle source selection (used by the indicator library)
# ---------------------------------------------------------------------------
CANDLE_SOURCE_MAPPING = {
    "open":   lambda c: c[:, 1],
    "close":  lambda c: c[:, 2],
    "high":   lambda c: c[:, 3],
    "low":    lambda c: c[:, 4],
    "volume": lambda c: c[:, 5],
    "hl2":    lambda c: (c[:, 3] + c[:, 4]) / 2,
    "hlc3":   lambda c: (c[:, 3] + c[:, 4] + c[:, 2]) / 3,
    "ohlc4":  lambda c: (c[:, 1] + c[:, 3] + c[:, 4] + c[:, 2]) / 4,
}

# number of trailing candles kept when computing a non-sequential indicator value
WARMUP_CANDLES_NUM = 240


def get_candle_source(candles: np.ndarray, source_type: str = "close") -> np.ndarray:
    """Return the price series for the requested source type."""
    try:
        return CANDLE_SOURCE_MAPPING[source_type](candles)
    except KeyError:
        raise ValueError(f"Source type '{source_type}' not recognised")


def slice_candles(candles: np.ndarray, sequential: bool) -> np.ndarray:
    """For non-sequential calls, trim to the last WARMUP_CANDLES_NUM candles (matches Jesse)."""
    if not sequential and candles.shape[0] > WARMUP_CANDLES_NUM:
        candles = candles[-WARMUP_CANDLES_NUM:]
    return candles


# ---------------------------------------------------------------------------
# Jesse public helper compatibility
# ---------------------------------------------------------------------------
CACHED_CONFIG = {}
SUPPORTED_COLORS = {"green", "yellow", "red", "magenta", "black", "blue", "cyan", "white"}


def _current_store_or_none():
    try:
        from .store import get_current_store
        return get_current_store()
    except RuntimeError:
        return None


def app_currency() -> str:
    current = _current_store_or_none()
    exchange = current.exchanges.trading_exchange if current is not None else None
    return exchange.quote_asset if exchange is not None else "USDT"


def app_mode() -> str:
    current = _current_store_or_none()
    return current.app.trading_mode if current is not None else "backtest"


def arrow_to_timestamp(arrow_time: arrow.Arrow) -> int:
    return arrow_time.int_timestamp * 1000


def timestamp_to_arrow(timestamp: int) -> arrow.Arrow:
    return arrow.get(timestamp / 1000)


def get_arrow(timestamp: int) -> arrow.Arrow:
    return timestamp_to_arrow(timestamp)


def iso8601_to_timestamp(iso8601: str) -> int:
    return int(arrow.get(iso8601).datetime.timestamp()) * 1000


def binary_search(arr: list, item) -> int:
    index = bisect_left(arr, item)
    return index if index != len(arr) and arr[index] == item else -1


def class_iter(Class):
    return (value for variable, value in vars(Class).items()
            if not callable(getattr(Class, variable)) and
            not variable.startswith("__"))


def clean_orderbook_list(arr):
    return [[float(item[0]), float(item[1])] for item in arr]


def color(msg_text: str, msg_color: str) -> str:
    if not msg_text:
        return ""
    if msg_color in SUPPORTED_COLORS:
        return click.style(msg_text, fg=msg_color)
    if msg_color == "gray":
        return click.style(msg_text, fg="white")
    raise ValueError("unsupported color")


def style(msg_text: str, msg_style: str) -> str:
    if msg_style is None:
        return msg_text
    if msg_style.lower() in ("bold", "b"):
        return click.style(msg_text, bold=True)
    if msg_style.lower() in ("underline", "u"):
        return click.style(msg_text, underline=True)
    raise ValueError("unsupported style")


def underline_to_dashy_symbol(symbol: str) -> str:
    return symbol.replace("_", "-")


def dashy_to_underline(symbol: str) -> str:
    return symbol.replace("-", "_")


def get_base_asset(symbol: str) -> str:
    return base_asset(symbol)


def get_quote_asset(symbol: str) -> str:
    return quote_asset(symbol)


def dump_exception() -> None:
    import traceback
    print(traceback.format_exc())
    raise SystemExit(1)


def estimate_average_price(order_qty: float, order_price: float, current_qty: float,
                           current_entry_price: float) -> float:
    return ((abs(order_qty) * order_price +
             abs(current_qty) * current_entry_price) /
            (abs(order_qty) + abs(current_qty)))


def estimate_PNL(qty: float, entry_price: float, exit_price: float,
                 trade_type: str, trading_fee: float = 0) -> float:
    qty = abs(qty)
    profit = qty * (exit_price - entry_price)
    if trade_type == "short":
        profit *= -1
    return profit - trading_fee * qty * (entry_price + exit_price)


def estimate_PNL_percentage(qty: float, entry_price: float, exit_price: float,
                            trade_type: str) -> float:
    qty = abs(qty)
    profit = qty * (exit_price - entry_price)
    if trade_type == "short":
        profit *= -1
    return (profit / (qty * entry_price)) * 100


def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def clear_file(path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("")


def make_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_currency(num: float) -> str:
    return f"{num:,}"


def format_price(price: float) -> str:
    if price is None:
        return ""
    if price == 0:
        return "0.00"
    sign = "-" if price < 0 else ""
    absolute = abs(price)
    price_str = f"{absolute:.20f}"
    if absolute >= 1:
        integer, decimal = price_str.split(".")
        return f"{sign}{integer}.{decimal[:2]}"
    decimal = price_str.split(".")[1]
    first = next((index for index, digit in enumerate(decimal)
                  if digit != "0"), -1)
    if first == -1:
        return "0.00"
    return f"{sign}0.{decimal[:first + 5]}"


def generate_short_unique_id() -> str:
    return str(uuid.uuid4())[:22]


def get_config(keys: str, default: Any = None) -> Any:
    if not keys:
        raise ValueError("keys string cannot be empty")
    env_name = convert_to_env_name(keys.replace(".", " "))
    if env_name in os.environ:
        return os.environ[env_name]

    current = _current_store_or_none()
    synthetic = {
        "app": {
            "trading_mode": current.app.trading_mode if current else "backtest",
            "debug_mode": False,
        },
        "env": {"data": {"warmup_candles_num": WARMUP_CANDLES_NUM}},
    }
    try:
        from .context import _CTX
        project_config = _CTX.config.get() if _CTX is not None else {}
    except (AttributeError, RuntimeError):
        project_config = {}
    data = merge_dicts(synthetic, project_config)
    value = data
    for part in keys.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def get_store():
    from .store import store
    return store


def get_strategy_class(strategy_name: str):
    try:
        from .context import get_context
        from .loader import load_strategy_class
        return load_strategy_class(strategy_name, get_context().strategies_dir)
    except Exception:
        return None


def insecure_hash(msg: str) -> str:
    return hashlib.md5(msg.encode()).hexdigest()


def secure_hash(msg: str) -> str:
    return hashlib.sha256(msg.encode()).hexdigest()


def insert_list(index: int, item, arr: list) -> list:
    return arr + [item] if index == -1 else arr[:index] + [item] + arr[index:]


def is_backtesting() -> bool:
    return app_mode() == "backtest"


def is_significance_testing() -> bool:
    return app_mode() == "significance_test"


def is_debuggable(debug_item) -> bool:
    return is_debugging() and normalize_bool(
        get_config(f"env.logging.{debug_item}", True))


def is_debugging() -> bool:
    return normalize_bool(get_config("app.debug_mode", False))


def is_importing_candles() -> bool:
    return app_mode() == "candles"


def is_live() -> bool:
    return is_livetrading() or is_paper_trading()


def is_livetrading() -> bool:
    return app_mode() == "livetrade"


def is_optimizing() -> bool:
    return app_mode() == "optimize"


def is_paper_trading() -> bool:
    return app_mode() == "papertrade"


def is_unit_testing() -> bool:
    return ("pytest" in sys.modules or
            os.path.basename(sys.argv[0]) in ("pytest", "py.test") or
            bool(os.environ.get("PYTEST_CURRENT_TEST")))


def normalize(x: float, x_min: float, x_max: float) -> float:
    return (x - x_min) / (x_max - x_min)


def now(force_fresh=False) -> int:
    return int(now_to_timestamp(force_fresh))


def now_to_datetime():
    return arrow.utcnow().datetime


def current_1m_candle_timestamp():
    return arrow.utcnow().floor("minute").int_timestamp * 1000


@lru_cache
def opposite_side(s: str) -> str:
    from .enums import sides
    if s == sides.BUY:
        return sides.SELL
    if s == sides.SELL:
        return sides.BUY
    raise ValueError(f"{s} is not a valid input for side")


@lru_cache
def opposite_type(t: str) -> str:
    from .enums import trade_types
    if t == trade_types.LONG:
        return trade_types.SHORT
    if t == trade_types.SHORT:
        return trade_types.LONG
    raise ValueError("unsupported type")


def orderbook_insertion_index_search(arr, target: int, ascending: bool = True):
    target = target[0]
    prices = [row[0] for row in arr]
    if not ascending:
        prices = [-price for price in prices]
        target = -target
    index = bisect_left(prices, target)
    return index < len(prices) and prices[index] == target, index


def orderbook_trim_price(p: float, ascending: bool, unit: float) -> float:
    if ascending:
        trimmed = np.ceil(p / unit) * unit
        if math.log10(unit) < 0:
            trimmed = round(trimmed, abs(int(math.log10(unit))))
        return p if trimmed == p + unit else trimmed
    trimmed = np.ceil(p / unit) * unit - unit
    if math.log10(unit) < 0:
        trimmed = round(trimmed, abs(int(math.log10(unit))))
    return p if trimmed == p - unit else trimmed


def python_version() -> tuple:
    return sys.version_info[:2]


def readable_duration(seconds: int, granularity: int = 2) -> str:
    intervals = (("weeks", 604800), ("days", 86400), ("hours", 3600),
                 ("minutes", 60), ("seconds", 1))
    result = []
    seconds = int(seconds)
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            result.append(f"{value} {name.rstrip('s') if value == 1 else name}")
    return ", ".join(result[:granularity])


def relative_to_absolute(path: str) -> str:
    return os.path.abspath(path)


def round_or_none(x, digits: int = 0):
    return None if x is None else round(x, digits)


def round_decimals_down(number, decimals: int = 2):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    if decimals == 0:
        return np.floor(number)
    if decimals > 0:
        factor = 10 ** decimals
        return np.floor(number * factor) / factor
    factor = 10 ** (-decimals)
    return np.floor(number / factor) * factor


def round_qty_for_live_mode(roundable_qty, precision: int):
    input_type = type(roundable_qty)
    values = (roundable_qty if isinstance(roundable_qty, np.ndarray)
              else np.array([roundable_qty], dtype=float))
    rounded = round_decimals_down(values, precision)
    for index, qty in enumerate(rounded):
        if qty == 0:
            if precision < 0:
                raise ValueError("qty is too small")
            rounded[index] = 1 / 10 ** precision
    return float(rounded[0]) if input_type in (float, np.float64) else rounded


def is_almost_equal(a: float, b: float, tolerance: float = 1e-8) -> bool:
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    if abs(a) < tolerance and abs(b) < tolerance:
        return abs(a - b) <= tolerance
    return abs((a - b) / max(abs(a), abs(b))) <= tolerance


def should_execute_silently() -> bool:
    return is_optimizing() or is_unit_testing()


@lru_cache
def side_to_type(s: str) -> str:
    from .enums import sides, trade_types
    s = s.lower()
    if s == sides.BUY:
        return trade_types.LONG
    if s == sides.SELL:
        return trade_types.SHORT
    raise ValueError


@lru_cache
def type_to_side(t: str) -> str:
    from .enums import sides, trade_types
    if t == trade_types.LONG:
        return sides.BUY
    if t == trade_types.SHORT:
        return sides.SELL
    raise ValueError(
        f'unsupported type: "{t}". Only "long" and "short" are supported.')


def unique_list(arr) -> list:
    seen = set()
    return [item for item in arr if not (item in seen or seen.add(item))]


def closing_side(position_type: str) -> str:
    if position_type.lower() == "long":
        return "sell"
    if position_type.lower() == "short":
        return "buy"
    raise ValueError(
        f"Value entered for position_type ({position_type}) is not valid")


def merge_dicts(d1: dict, d2: dict) -> dict:
    result = dict(d1)
    for key, value in d2.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def computer_name():
    return platform.node()


def validate_response(response):
    if response.status_code != 200:
        try:
            message = response.json()["message"]
        except (KeyError, TypeError, ValueError):
            message = getattr(response, "text", "Unexpected response")
        raise ConnectionError(f"[{response.status_code}]: {message}")


def get_session_id():
    current = _current_store_or_none()
    if current is None:
        return generate_unique_id()
    if not current.app.session_id:
        current.app.session_id = generate_unique_id()
    return current.app.session_id


def get_pid():
    return os.getpid()


def is_jesse_project():
    return os.path.isdir("strategies") and os.path.isdir("storage")


def dd(item):
    dump(item)
    raise SystemExit(1)


def dump(*item):
    value = item[0] if len(item) == 1 else item
    print(color("\n========= DEBUGGING VALUE ==========", "yellow"))
    pprint(value)
    print(color("====================================\n", "yellow"))


def debug(*item):
    message = (f"==> {item[0]}" if len(item) == 1 else
               f"==> {', '.join(str(value) for value in item)}")
    dump(message)
    from .services import logger
    logger.info(message)


def terminal_debug(*item):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = item[0] if len(item) == 1 else ", ".join(map(str, item))
    dump(f"[{timestamp}] ==> {message}")


def float_or_none(item):
    return None if item is None or item == "" else float(item)


def str_or_none(item, encoding="utf-8"):
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, bytes):
        return item.decode(encoding)
    return str(item)


def cpu_cores_count():
    return cpu_count()


def convert_to_env_name(name: str) -> str:
    return name.replace(" ", "_").upper()


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


def get_os() -> str:
    name = platform.system()
    if name == "Darwin":
        return "mac"
    if name == "Linux":
        return "linux"
    if name == "Windows":
        return "windows"
    raise NotImplementedError(f'Unsupported OS: "{name}"')


def is_docker() -> bool:
    return os.path.exists("/.dockerenv")


def clear_output():
    if is_notebook():
        from IPython.display import clear_output as notebook_clear
        notebook_clear(wait=True)
    else:
        click.clear()


def clean_nan_values(obj):
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(value) for value in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def clean_infinite_values(obj):
    if isinstance(obj, dict):
        return {key: clean_infinite_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_infinite_values(value) for value in obj]
    if isinstance(obj, (float, np.floating)) and math.isinf(float(obj)):
        return None
    return obj


def get_class_name(cls):
    return cls if isinstance(cls, str) else cls.__name__


def next_candle_timestamp(candle: np.ndarray, timeframe: str) -> int:
    return int(candle[0] + timeframe_to_one_minutes(timeframe) * 60_000)


def get_candle_start_timestamp_based_on_timeframe(
        timeframe: str, num_candles_to_fetch: int) -> int:
    return (now(force_fresh=True) - num_candles_to_fetch *
            timeframe_to_one_minutes(timeframe) * 60_000)


def is_price_near(order_price, price_to_compare,
                  percentage_threshold=0.00015):
    return abs(1 - order_price / price_to_compare) <= percentage_threshold


def gzip_compress(data):
    return gzip.compress(json.dumps(data).encode("utf-8"))


def compressed_response(content: str) -> dict:
    return {
        "is_compressed": True,
        "data": base64.b64encode(gzip_compress(content)).decode("utf-8"),
    }


def validate_cwd() -> None:
    if not is_jesse_project():
        raise SystemExit(
            "Current directory is not a Terry/Jesse strategy project.")


def has_live_trade_plugin() -> bool:
    return False


def normalize_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v == 1
    if isinstance(v, str):
        normalized = v.strip().lower()
        if normalized in ("1", "true"):
            return True
        if normalized in ("0", "false"):
            return False
    return bool(v)


def terminate_app() -> None:
    raise SystemExit(1)


def error(msg: str, force_print: bool = False) -> None:
    del force_print
    print("\n" + color("========== CRITICAL ERROR ==========", "red"))
    print(color(str(msg), "red"))
    print(color("====================================", "red"))
