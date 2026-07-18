# Terry — Build Plan

**Terry** is a local, self-contained clone of the [Jesse](https://jesse.trade) crypto
trading framework, exposed as an **MCP server** so AI agents (Claude Code, Cursor, etc.)
can research, build, backtest, and stress-test trading strategies end-to-end on this machine.

This document is the engineering plan I (the agent) follow to build and verify Terry. It
records what Jesse is, the design decisions, and the concrete module-by-module task list.

---

## 1. What Jesse is (research summary)

Jesse (v2.5.0, MIT-licensed core) is a Python framework for **backtesting, optimizing, and
live-trading crypto strategies**. Its standout features:

- A **candle-by-candle backtesting engine** with no look-ahead bias. Input candles are always
  1-minute; the requested route timeframe (1h, 4h, 1D…) is aggregated on the fly.
- A **Strategy base class** with a clean lifecycle (`should_long`/`go_long`/`update_position`/…)
  and a **smart-order mechanism** (you set `self.buy = qty, price` and it infers market/limit/stop).
- A big **technical-indicator library** and utility/sizing helpers.
- **Research tools**: Rule Significance Testing (bootstrap p-value on entry signals),
  Monte Carlo robustness analysis, and genetic/Optuna hyperparameter optimization.
- A **web dashboard** (FastAPI + Vue) on port 9000, and a **built-in MCP server** (FastMCP,
  streamable-http) on port 9002 exposing ~56 tools to LLM agents. The compute-heavy `run_*`
  tools are credit-gated (Jesse's paid tier).
- Requires **PostgreSQL + Redis**; live trading is a separate paid plugin.

The MCP agent workflow (from Jesse's canonical `agent_rules.md`): the agent imports candles,
writes a minimal strategy, validates the entry rule with a significance test, backtests,
iterates with small changes, runs Monte Carlo, and writes a Markdown report — all via MCP tools.

## 2. Design decisions for Terry

| Concern | Jesse | Terry |
|---|---|---|
| Nature | Framework + SaaS | **Independent, self-contained clone** you own |
| Storage | PostgreSQL + Redis | **SQLite** (zero-config, local, free) |
| Process model | Dashboard :9000 + MCP :9002 (HTTP) | **Single in-process MCP server**; tools call the engine directly |
| Transport | FastMCP streamable-http :9002 | **Same** (FastMCP streamable-http, default port 9021) |
| Tool surface | ~56 tools, `run_*` credit-gated | **Same tool names/semantics, ungated (free/unlimited)** |
| Strategy API | `Strategy` base class | **Same API** (drop-in compatible source) |
| Metrics | 44-key dict | **Same keys/definitions** (validated vs Jesse) |
| Data source | Binance/Bybit/… drivers | **Binance public REST** (no API key) + pluggable |
| "Dashboard URL" | live Vue dashboard | **self-contained HTML report** per session (file URL) |
| Live trading | paid plugin | **out of scope** (documented as future; safety) |
| Optimization | Optuna + Ray | **Optuna** (random/TPE), in-process multiprocessing |

**Fidelity target:** an agent using Jesse's workflow should be able to drive Terry with the
same tool calls and get results with the same shape and meaning. The engine is validated
against ground-truth captured from a real `jesse.research.backtest()` run.

## 3. Architecture / module map

```
terry/
  config.py            defaults + user config (SQLite-backed)
  helpers.py           timeframe math, date parsing, keys, ids  (jesse.helpers analog)
  enums.py             timeframes, sides, order types/status
  exceptions.py
  utils.py             size_to_qty, risk_to_qty, crossed, z_score, cointegration…
  indicators/          numpy indicator library (Jesse-compatible signatures)
  models/              Candle, Order, Position, ClosedTrade, Route
  engine/
    store.py           global state (candles, positions, orders, trades, app clock)
    exchange.py        simulated exchange: balance, margin, fees, leverage, spot/futures
    broker.py          order placement + smart-order routing
    candle_store.py    1m → timeframe aggregation, warmup injection
    simulator.py       the candle-by-candle backtest loop
    metrics.py         metrics computation (44 keys, matches Jesse)
  strategy.py          Strategy base class (the developer API)
  loader.py            load strategy class from strategies/<Name>/__init__.py
  data/
    binance.py         Binance public /klines fetcher
    storage.py         SQLite candle store (dedup, coverage queries)
    importer.py        import orchestration + progress tracking
  research/
    backtest.py        backtest() pure function
    significance.py    rule_significance_test() (bootstrap)
    monte_carlo.py     monte_carlo candles + trades
    optimize.py        optimize() via Optuna
  sessions/
    db.py              SQLite session store: drafts/sessions/results/notes
    runner.py          background thread runner for backtest/mc/rst/opt
  report/
    html.py            self-contained HTML "dashboard" report per session
  mcp/
    server.py          FastMCP streamable-http entry point
    resources.py       terry:// doc resources (strategy, indicators, metrics…)
    tools/             strategy, backtest, config, candles, indicator,
                       significance_test, monte_carlo, optimization, general
strategies/            user strategies (created via MCP tools)
storage/               candles.db, sessions.db, reports/
AGENTS.md              canonical agent prompt (adapted from Jesse's)
terry (CLI)            `terry serve` starts the MCP server; `terry init`, `terry doctor`
tests/                 unit + end-to-end MCP tests
```

## 4. Task checklist

1. **Scaffold**: package, venv, requirements, CLI, config.
2. **Core models**: Candle/Order/Position/ClosedTrade/Route + helpers/enums/exceptions.
3. **Indicators**: ~40 indicators with `sequential`/`source_type` and named-tuple multiline.
4. **Engine**: store → exchange → broker → candle_store → simulator → metrics.
5. **Strategy base class**: full lifecycle + `self.*` API + sizing/exits.
6. **Data layer**: Binance fetch + SQLite storage + importer with progress.
7. **Research funcs**: backtest, significance, monte_carlo, optimize.
8. **Sessions**: SQLite store + background runner + HTML report.
9. **MCP server**: FastMCP streamable-http + all tools + resources + AGENTS.md.
10. **Tests**: unit (engine/indicators/metrics/sizing) + end-to-end (MCP tool flow) +
    validation against Jesse ground-truth.
11. **Docs**: `HOW_TERRY_WORKS.md` (user + agent guide), `APIS_AND_SERVICES.md` (services).

## 5. Verification strategy

- Unit tests for indicators (vs known values), sizing helpers, metrics on hand-built trades.
- Engine tests: a deterministic strategy on synthetic candles → assert trade count, PnL,
  order-type routing (market/limit/stop), stop-loss/take-profit fills, spot vs futures.
- End-to-end: start the MCP server in-process, call every tool through the MCP client,
  run the full agent workflow (import → create strategy → significance → backtest →
  monte carlo → report) and assert terminal states + result shapes.
- Cross-check: run the same synthetic scenario through Jesse (installed in scratchpad) and
  Terry; confirm metric definitions and directional agreement.

## 6. Out of scope (documented, not built)

Real live/paper trading against exchanges; the full Vue dashboard; ML pipeline;
100% of Jesse's 300+ indicators (Terry ships the common ~40, easily extended);
multi-process Ray cluster (Terry uses local multiprocessing/threads).
