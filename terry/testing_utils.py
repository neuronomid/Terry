"""Deterministic strategy-engine helpers compatible with ``jesse.testing_utils``.

These helpers intentionally use Terry's pure historical engine.  They are for tests,
examples, and local research only; no exchange connection or live order path exists.
"""
from __future__ import annotations

import os
from typing import Mapping, Type

from . import helpers as jh
from .factories import candles_from_close_prices
from .research.backtest import backtest
from .strategy import Strategy

SANDBOX = "Sandbox"


def get_btc_and_eth_candles() -> dict:
    """Return deterministic BTC uptrend and ETH uptrend datasets."""
    return {
        jh.key(SANDBOX, "BTC-USDT"): {
            "exchange": SANDBOX,
            "symbol": "BTC-USDT",
            "candles": candles_from_close_prices(range(101, 200)),
        },
        jh.key(SANDBOX, "ETH-USDT"): {
            "exchange": SANDBOX,
            "symbol": "ETH-USDT",
            "candles": candles_from_close_prices(range(1, 100)),
        },
    }


def get_btc_candles(candles_count: int = 100) -> dict:
    """Return the Jesse test fixture's ascending close-price sequence."""
    return {
        jh.key(SANDBOX, "BTC-USDT"): {
            "exchange": SANDBOX,
            "symbol": "BTC-USDT",
            "candles": candles_from_close_prices(range(1, candles_count)),
        }
    }


def get_downtrend_candles(candles_count: int = 100) -> dict:
    """Return the Jesse test fixture's descending close-price sequence."""
    return {
        jh.key(SANDBOX, "BTC-USDT"): {
            "exchange": SANDBOX,
            "symbol": "BTC-USDT",
            "candles": candles_from_close_prices(range(candles_count, 10, -1)),
        }
    }


def set_up(is_futures_trading: bool = True, leverage: float = 1,
           leverage_mode: str = "cross", fee: float = 0) -> dict:
    """Build the isolated engine config used by the strategy test helpers."""
    if leverage_mode not in {"cross", "isolated"}:
        raise ValueError("leverage_mode must be 'cross' or 'isolated'")
    if float(leverage) < 1:
        raise ValueError("leverage must be at least 1")
    if float(fee) < 0:
        raise ValueError("fee cannot be negative")
    return {
        "exchange": SANDBOX,
        "starting_balance": 10_000,
        "fee": float(fee),
        "type": "futures" if is_futures_trading else "spot",
        "futures_leverage": float(leverage) if is_futures_trading else 1,
        "futures_leverage_mode": leverage_mode,
        "warm_up_candles": 0,
    }


def _source_options(strategy_classes: Mapping[str, Type[Strategy]] | None,
                    strategies_dir: str | os.PathLike | None) -> dict:
    if strategy_classes is None and strategies_dir is None:
        strategies_dir = os.path.join(os.getcwd(), "strategies")
    return {
        "strategy_classes": dict(strategy_classes) if strategy_classes else None,
        "strategies_dir": os.fspath(strategies_dir) if strategies_dir is not None else None,
    }


def single_route_backtest(
        strategy_name: str, is_futures_trading: bool = True,
        leverage: float = 1, leverage_mode: str = "cross", trend: str = "up",
        fee: float = 0, candles_count: int = 100, timeframe: str = "1m", *,
        strategy_classes: Mapping[str, Type[Strategy]] | None = None,
        strategies_dir: str | os.PathLike | None = None,
) -> dict:
    """Run one deterministic route through Terry's real historical engine."""
    config = set_up(is_futures_trading, leverage, leverage_mode, fee)
    routes = [{"exchange": SANDBOX, "symbol": "BTC-USDT",
               "timeframe": timeframe, "strategy": strategy_name}]
    if trend == "up":
        candles = get_btc_candles(candles_count)
    elif trend == "down":
        candles = get_downtrend_candles(candles_count)
    else:
        raise ValueError("trend must be 'up' or 'down'")
    return backtest(config, routes, [], candles,
                    **_source_options(strategy_classes, strategies_dir))


def two_routes_backtest(
        strategy_name1: str, strategy_name2: str,
        is_futures_trading: bool = True, leverage: float = 1,
        leverage_mode: str = "cross", trend: str = "up", *,
        strategy_classes: Mapping[str, Type[Strategy]] | None = None,
        strategies_dir: str | os.PathLike | None = None,
) -> dict:
    """Run BTC and ETH trading routes in the same isolated backtest."""
    config = set_up(is_futures_trading, leverage, leverage_mode)
    routes = [
        {"exchange": SANDBOX, "symbol": "BTC-USDT", "timeframe": "1m",
         "strategy": strategy_name1},
        {"exchange": SANDBOX, "symbol": "ETH-USDT", "timeframe": "1m",
         "strategy": strategy_name2},
    ]
    return backtest(config, routes, [], get_btc_and_eth_candles(),
                    **_source_options(strategy_classes, strategies_dir))


def two_data_routes_backtest(
        strategy_name1: str, strategy_name2: str,
        is_futures_trading: bool = True, leverage: float = 1,
        leverage_mode: str = "cross", trend: str = "up", *,
        strategy_classes: Mapping[str, Type[Strategy]] | None = None,
        strategies_dir: str | os.PathLike | None = None,
) -> dict:
    """Run two trading routes with Jesse-shaped additional data routes."""
    config = set_up(is_futures_trading, leverage, leverage_mode)
    routes = [
        {"exchange": SANDBOX, "symbol": "BTC-USDT", "timeframe": "1m",
         "strategy": strategy_name1},
        {"exchange": SANDBOX, "symbol": "ETH-USDT", "timeframe": "5m",
         "strategy": strategy_name2},
    ]
    data_routes = [
        {"exchange": SANDBOX, "symbol": "BTC-USDT", "timeframe": "5m"},
        {"exchange": SANDBOX, "symbol": "ETH-USDT", "timeframe": "15m"},
    ]
    return backtest(config, routes, data_routes, get_btc_and_eth_candles(),
                    **_source_options(strategy_classes, strategies_dir))


__all__ = [
    "SANDBOX", "get_btc_and_eth_candles", "get_btc_candles",
    "get_downtrend_candles", "set_up", "single_route_backtest",
    "two_routes_backtest", "two_data_routes_backtest",
]
