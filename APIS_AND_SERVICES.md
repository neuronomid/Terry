# APIs & Services Terry Needs

**Short version: Terry needs essentially nothing paid.** It runs fully locally and gets market data
from a free public API with no key. This document lists what's used, what's optional, and the
cheap/free choices — because you asked to keep costs at zero where possible.

---

## 1. What's required (all free)

| Need | What Terry uses | Cost | API key? |
|---|---|---|---|
| **Price history (candles)** | Public REST APIs from Binance, Bitfinex, Coinbase, Bybit, Gate.io, and Kraken | **Free** | **No key** |
| **Database** | **SQLite** (bundled with Python) | Free | — |
| **Runtime** | **Python 3.10–3.13** + the packages pinned in `requirements.txt` | Free | — |

That's the entire hard requirement. No account, no billing, no cloud.

### About the market-data endpoints
They are unauthenticated OHLCV endpoints. Terry never calls account or order endpoints. Pagination
and throttling are adjusted per provider. Providers can impose geo-restrictions, rolling history
limits, maintenance, and symbol-specific availability; use another supported market when needed.

---

## 2. What Jesse needs that Terry deliberately dropped (so you save money/effort)

The real Jesse requires more infrastructure. Terry replaced each with a zero-config local option:

| Jesse requires | Why | Terry's replacement |
|---|---|---|
| **PostgreSQL** | stores candles & sessions | **SQLite** file in `storage/` |
| **Redis** | pub/sub + caching between its two processes | not needed (Terry is a single process) |
| **Jesse license / credits** | the `run_*` MCP tools are metered on Jesse's paid plan | **removed** — everything is free & unlimited in Terry |
| **Ray** | distributed optimization/research workers | bounded local worker threads + Optuna TPE |

So compared to Jesse you save: a Postgres server, a Redis server, and any subscription. If you ever
*want* Postgres you can add it, but it is not necessary.

---

## 3. Optional upgrades (only if you want them later)

These are **not needed** for anything Terry does today, but if you decide to extend it:

| If you want… | Option | Cost |
|---|---|---|
| More exchanges beyond the ten Jesse backtest markets | Their public REST kline endpoints (mostly keyless) | Free |
| Higher-quality/tick data or very old history | [Tardis.dev](https://tardis.dev), [Kaiko], CryptoDataDownload CSVs | Free CSVs → paid tiers |
| An LLM to *drive* Terry autonomously | You already use Claude Code / an MCP-capable agent | Your existing plan |
| A hosted, always-on server | A small VPS (e.g. Hetzner/DigitalOcean) | ~$4–6/month (optional) |
| **Live/paper trading** (NOT built) | Would require **your exchange API keys** + careful risk controls | Out of scope — Terry never trades real money |

### ⚠️ Important about live trading
Terry intentionally does **not** implement live or paper trading. That is the one place real API
keys (and real financial risk) would come in. It was left out on purpose for safety. If you ever add
it, you'd create **read-only or trade-scoped API keys** on your exchange and store them locally —
never commit them to git, and start with tiny sizes on a testnet.

---

## 4. Cost summary

| Scenario | Monthly cost |
|---|---|
| Terry as delivered (local public market data) | **$0** |
| Terry on a small always-on VPS (optional) | ~$4–6 |
| Terry + paid historical data vendor (optional) | vendor-dependent |
| Live trading | not provided |

**Bottom line:** run it as-is and you pay nothing. The only external calls are free, keyless
market-data endpoints. Everything else lives on your machine.
