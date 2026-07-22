"""Unit coverage for the non-crypto data sources: Dukascopy (historical .bi5),
Yahoo Finance (Demo Mode live), the instrument registry, Forex market sessions and
the 1m gap-fill that keeps session-market data engine-compatible. No network access —
every driver's transport is faked."""
from __future__ import annotations

import lzma
import struct

import numpy as np
import pytest

import terry.helpers as jh
from terry.data import binance, dukascopy, instruments, market_hours, yahoo
from terry.engine.candle_store import aggregate_candles, fill_1m_gaps


# --------------------------------------------------------------------------- registry
def test_instrument_registry_resolves_codes_scales_and_proxies():
    assert instruments.dukascopy_instrument("EUR-USD") == ("EURUSD", 100000)
    assert instruments.dukascopy_instrument("usd-jpy") == ("USDJPY", 1000)  # JPY quote
    assert instruments.dukascopy_instrument("XAU-USD") == ("XAUUSD", 1000)
    assert instruments.dukascopy_instrument("WTI-USD") == ("LIGHTCMDUSD", 1000)
    # stock indices resolve to a real-time futures proxy for Yahoo, not the raw index
    assert instruments.yahoo_ticker("US500-USD") == "ES=F"
    assert instruments.yahoo_ticker("US100-USD") == "NQ=F"
    assert instruments.yahoo_ticker("US30-USD") == "YM=F"
    assert instruments.yahoo_ticker("EUR-USD") == "EURUSD=X"


def test_unknown_dukascopy_symbol_raises_with_supported_list():
    with pytest.raises(ValueError) as excinfo:
        instruments.dukascopy_instrument("FOO-BAR")
    assert "EUR-USD" in str(excinfo.value)


def test_dashboard_asset_classes_group_sources():
    classes = instruments.dashboard_asset_classes(["Binance Spot", "Bybit Spot"])
    by_name = {c["name"]: c for c in classes}
    assert by_name["Cryptocurrency"]["exchanges"] == ["Binance Spot", "Bybit Spot"]
    assert by_name["Forex"]["exchanges"] == ["Dukascopy"]
    assert by_name["Forex"]["example"] == "EUR-USD"
    assert "XAU-USD" in by_name["Metals"]["symbols"]
    assert by_name["Indices"]["exchanges"] == ["Dukascopy"]


# --------------------------------------------------------------------------- Dukascopy
def _make_bi5(ticks):
    """Compress ``(ms, ask, bid, ask_vol, bid_vol)`` records like a real .bi5 file."""
    raw = b"".join(struct.pack(">3I2f", *t) for t in ticks)
    return lzma.compress(raw)


class _FakeResp:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


def test_dukascopy_hour_url_uses_zero_indexed_month():
    # 2024-06-03 13:00 UTC -> month component must be 05 (June is index 5).
    ts = jh.date_to_timestamp("2024-06-03") + 13 * 3_600_000
    url = dukascopy._hour_url("EURUSD", ts)
    assert url.endswith("/EURUSD/2024/05/03/13h_ticks.bi5")


def test_fetch_hour_ticks_decodes_bid_prices_with_scale():
    hour = jh.date_to_timestamp("2024-06-03") + 13 * 3_600_000
    blob = _make_bi5([(1000, 108445, 108443, 1.8, 3.6),
                      (2000, 108448, 108446, 1.0, 6.3)])

    class _Session:
        def get(self, url, timeout=None):
            return _FakeResp(200, blob)

    ticks = dukascopy.fetch_hour_ticks("EURUSD", 100000, hour, session=_Session())
    assert len(ticks) == 2
    ts0, price0, vol0 = ticks[0]
    assert ts0 == hour + 1000
    assert price0 == pytest.approx(1.08443)   # bid / 100000
    assert vol0 == pytest.approx(3.6)          # bid volume


def test_fetch_hour_ticks_treats_404_as_closed_market():
    class _Session:
        def get(self, url, timeout=None):
            return _FakeResp(404)

    assert dukascopy.fetch_hour_ticks("EURUSD", 100000, 0, session=_Session()) == []


def test_aggregate_ticks_to_1m_builds_ohlcv():
    m0 = 1_700_000_040_000
    ticks = [
        (m0 + 1_000, 1.10, 1.0),
        (m0 + 2_000, 1.15, 2.0),   # high
        (m0 + 3_000, 1.05, 3.0),   # low
        (m0 + 4_000, 1.12, 4.0),   # close
        (m0 + 60_000, 2.00, 5.0),  # next minute
    ]
    rows = dukascopy.aggregate_ticks_to_1m(ticks)
    assert rows.shape == (2, 6)
    ts, o, c, h, l, v = rows[0]
    assert (o, c, h, l, v) == (1.10, 1.12, 1.15, 1.05, 10.0)
    assert rows[1][1] == 2.00


def test_dukascopy_fetch_1m_range_filters_to_window(monkeypatch):
    start = 1_700_000_000_000 // 3_600_000 * 3_600_000
    finish = start + 3_600_000

    def fake_hour(instrument, scale, hour, session):
        # one tick per minute for this hour
        return [(hour + i * 60_000 + 500, 1.10 + i * 0.001, 1.0) for i in range(60)]

    monkeypatch.setattr(dukascopy, "fetch_hour_ticks", fake_hour)
    rows = dukascopy.fetch_1m_range("EUR-USD", start + 10 * 60_000,
                                    start + 20 * 60_000, rate_limit_sleep=0)
    assert len(rows) == 10
    assert rows[0, 0] == start + 10 * 60_000
    assert rows[-1, 0] == start + 19 * 60_000


# --------------------------------------------------------------------------- Yahoo
def test_yahoo_ticker_uses_proxy_then_falls_back():
    assert yahoo._ticker("US500-USD") == "ES=F"       # index -> future proxy
    assert yahoo._ticker("EUR-USD") == "EURUSD=X"
    assert yahoo._ticker("ZZZ-ZZZ") == "ZZZZZZ"       # unknown -> dashless


def test_yahoo_fetch_1m_range_parses_and_drops_nulls(monkeypatch):
    base = 1_700_000_040  # minute-aligned, like real Yahoo 1m timestamps
    result = {
        "timestamp": [base, base + 60, base + 120, base + 180],
        "indicators": {"quote": [{
            "open":  [1.10, None, 1.12, 1.13],
            "high":  [1.11, None, 1.13, 1.14],
            "low":   [1.09, None, 1.11, 1.12],
            "close": [1.105, None, 1.125, 1.135],  # second bar null -> dropped
            "volume": [10, None, 20, 30],
        }]},
    }
    monkeypatch.setattr(yahoo, "_fetch_chart", lambda *a, **k: result)
    rows = yahoo.fetch_1m_range("EUR-USD", base * 1000, (base + 240) * 1000)
    assert len(rows) == 3                     # the null-close bar is skipped
    assert [round(c, 3) for c in rows[:, 2]] == [1.105, 1.125, 1.135]
    # dropping the null bar leaves a real one-minute hole (gap-fill happens later)
    assert rows[1, 0] - rows[0, 0] == 120_000


def test_yahoo_fetch_live_price_reads_meta_then_none(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"chart": {"result": [{"meta": {"regularMarketPrice": 1.2345}}]}}

    class _Session:
        def get(self, url, params=None, timeout=None):
            return _Resp()

    monkeypatch.setattr(yahoo, "_session", lambda: _Session())
    assert yahoo.fetch_live_price("EUR-USD") == 1.2345

    def _boom():
        raise RuntimeError("down")

    monkeypatch.setattr(yahoo, "_session", _boom)
    assert yahoo.fetch_live_price("EUR-USD") is None


def test_yahoo_live_price_prefers_precise_1m_close_over_rounded_meta(monkeypatch):
    """Yahoo rounds meta.regularMarketPrice (hiding sub-pip FX moves); the fresher, full-
    precision 1m close must win so the live forming candle keeps moving tick by tick."""
    class _Resp:
        status_code = 200

        def json(self):
            return {"chart": {"result": [{
                "meta": {"regularMarketPrice": 1.141, "regularMarketTime": 1000},
                "timestamp": [940, 1000, 1060],
                "indicators": {"quote": [{"close": [1.1408, 1.14103, None]}]},
            }]}}

    class _Session:
        def get(self, url, params=None, timeout=None):
            return _Resp()

    monkeypatch.setattr(yahoo, "_session", lambda: _Session())
    # Newest non-null close is 1.14103 at t=1000 (ties the meta time) — precise value wins.
    assert yahoo.fetch_live_price("EUR-USD") == 1.14103


# --------------------------------------------------------------------------- sessions
def test_forex_market_open_and_close_boundaries():
    # Monday 13:00 UTC — open, London + New York overlap.
    mon = jh.date_to_timestamp("2024-06-03") + 13 * 3_600_000
    assert market_hours.is_forex_open(mon)
    assert set(market_hours.active_sessions(mon)) == {"London", "New York"}
    assert "overlap" in market_hours.session_label(mon)

    # Saturday — always closed.
    sat = jh.date_to_timestamp("2024-06-08") + 13 * 3_600_000
    assert not market_hours.is_forex_open(sat)
    assert market_hours.session_label(sat) == "Closed"

    # Friday 22:00 UTC — after the weekly close.
    fri_late = jh.date_to_timestamp("2024-06-07") + 22 * 3_600_000
    assert not market_hours.is_forex_open(fri_late)

    # Sunday 22:00 UTC — market has reopened (Tokyo pre-open / Sydney).
    sun_late = jh.date_to_timestamp("2024-06-09") + 22 * 3_600_000
    assert market_hours.is_forex_open(sun_late)

    info = market_hours.session_info(mon)
    assert info["is_open"] and info["label"] == market_hours.session_label(mon)


# --------------------------------------------------------------------------- gap-fill
def test_fill_1m_gaps_makes_data_contiguous_and_is_noop_when_dense():
    t0 = 1_700_000_040_000
    gapped = np.array([
        [t0, 1, 2, 3, 1, 10],
        [t0 + 120_000, 5, 6, 7, 4, 20],   # one-minute hole at t0+60_000
    ], dtype=float)
    filled = fill_1m_gaps(gapped)
    assert filled.shape == (3, 6)
    assert bool(np.all(np.diff(filled[:, 0]) == 60_000))
    # the inserted candle is flat at the previous close (2) with zero volume
    assert filled[1].tolist() == [t0 + 60_000, 2, 2, 2, 2, 0]

    dense = np.array([[t0, 1, 1, 1, 1, 5], [t0 + 60_000, 1, 1, 1, 1, 5]], dtype=float)
    assert np.array_equal(fill_1m_gaps(dense), dense)          # crypto path unchanged
    assert len(fill_1m_gaps(np.empty((0, 6)))) == 0            # degenerate inputs safe

    # aggregation over filled data still lines up on real clock buckets
    agg = aggregate_candles(filled, "3m")
    assert agg.shape == (1, 6)


# --------------------------------------------------------------------------- dispatch
def test_binance_layer_registers_and_routes_dukascopy(monkeypatch):
    assert "Dukascopy" in binance.EXCHANGES
    assert binance.is_session_market("Dukascopy")
    assert not binance.is_session_market("Binance Spot")
    # Dukascopy has no real-time feed of its own (Demo Mode uses Yahoo via the runner).
    assert binance.fetch_live_price("Dukascopy", "EUR-USD") is None

    sentinel = np.array([[0, 1, 1, 1, 1, 1]], dtype=float)
    captured = {}

    def fake_range(symbol, start_ts, finish_ts, on_progress=None, should_stop=None):
        captured.update(symbol=symbol, start=start_ts, finish=finish_ts)
        return sentinel

    monkeypatch.setattr(dukascopy, "fetch_1m_range", fake_range)
    out = binance.fetch_1m_range("Dukascopy", "EUR-USD", 100, 200)
    assert np.array_equal(out, sentinel)
    assert captured == {"symbol": "EUR-USD", "start": 100, "finish": 200}
