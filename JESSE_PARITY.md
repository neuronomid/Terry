# Terry ↔ Jesse 2.5 Compatibility Audit

Audit date: 2026-07-18

Jesse baseline: `jesse-ai/jesse` 2.5.0, commit
`fa63531cae6c09b978711dc1892285067304e2df` (2026-07-13)

This is a capability comparison, not a claim that the two repositories have identical internals.
Terry is a zero-config local implementation built around SQLite; Jesse uses a larger service and
frontend architecture. The relevant upstream references are the
[Jesse repository](https://github.com/jesse-ai/jesse),
[getting-started guide](https://docs.jesse.trade/docs/getting-started/), and
[research documentation](https://docs.jesse.trade/docs/research/).

## Audit result

Terry now covers Jesse's open historical-research workflow: strategy authoring, public candle
imports, multi-route/data-route backtests, metrics, exports, chart packs, benchmarks, rule
significance, Monte Carlo, candle pipelines, ML gather/train/deploy, and train/test optimization.

It is **not an exact replacement for the entire Jesse product**. Terry's project rules prohibit
real live trading, and Jesse's live execution is distributed as a separate plugin/service rather
than as part of the open research core. Terry therefore does not provide exchange credentials,
multiple live accounts, real/paper order routing, live logs/notifications, or DEX execution.
Single-process optimization and frontend implementation also differ. Those boundaries must not be
described as full product parity.

## Capability matrix

| Area | Jesse 2.5 baseline | Terry 0.2 status | Evidence / difference |
|---|---|---|---|
| Strategy lifecycle and smart orders | Yes | Compatible | Every public method/property name on Jesse's `Strategy` is represented; smart market/limit/stop inference, scale-in/out, stops, targets, filters, hooks, and same-candle flips are exercised by engine tests. |
| Order, Position, ClosedTrade developer API | Yes | Compatible for research | Jesse status aliases, signed remaining quantity, ROI/cost, direction/state properties, serialization, mark/funding/liquidation fields are present. Live-only values are placeholders in historical mode. |
| Indicators | 174 public modules in audited repository | 174/174 modules | Module names match. Existing numerical cross-checks cover the exported indicator set; named-tuple and sequential behavior have regression tests. |
| Utilities | Jesse public utility functions | Public surface matched | Added sequential `crossed`, signal lines, streaks, strict trends, float math, `dd`, timeframe conversion, exact Engle-Granger cointegration, and Jesse fee/floor sizing semantics. |
| Historical engine | Spot + futures, multiple routes/timeframes | Compatible local engine | 44 metric keys, smart orders, fees, leverage modes, spot restrictions, warm-up/data routes, multiple symbols, cancellation, and exports. One research run still uses one selected exchange, matching Jesse's draft form. |
| Historical exchanges | 10 enabled markets | 10 markets | Binance Spot, Binance US Spot, Binance Perpetual Futures, Bitfinex Spot, Coinbase Spot, Bybit USDT/USDC Perpetual, Bybit Spot, Gate USDT Perpetual, Kraken Pro Futures. All ten real public response paths were checked on the audit date; offline fixtures guard normalization. |
| Research candle API | get/store/factories | Compatible | SQLite-backed `get_candles`, `store_candles`, `fake_candle`, `fake_range_candles`, and close-price factories. Terry also offers a blocking notebook import helper. |
| Backtest exports | CSV, JSON, TradingView, charts, benchmark | Implemented | Pure API and MCP runner support CSV, JSON, Pine v5, six PNG chart outputs, logs, hyperparameters, equity curve, and buy-and-hold benchmark. |
| Candle pipelines | Gaussian noise, Gaussian resampler, moving-block bootstrap | Implemented | Jesse-compatible pipeline classes, strategy hook, deterministic seed support, and OHLC-invariant tests. |
| Rule significance test | Bootstrap entry-rule test | Implemented | Jesse-compatible research signature/result samples and plotting, plus draft/run/status workflow, >=2,000 simulation default, observations, p-value, and dashboard report. |
| Monte Carlo | Candle and trade modes | Implemented | Jesse-compatible call signatures, scenario/confidence payloads, streamed callbacks, summary/plot helpers, candle block bootstrap, trade-order mode, downside/overfit verdicts, and dedicated per-scenario equity-curve retrieval. |
| ML research/deploy | Gather, sklearn train, artifacts, inference | Core API implemented | Chronological splits, binary/multiclass/regression metrics, model/scaler artifacts, feature importance, CSV loading, lazy Strategy inference. Jesse's much richer console tables and RFE/permutation diagnostic report are not reproduced exactly. |
| Optimization | Optuna + Ray, explicit train/test windows | API/results compatible, execution differs | Optuna TPE, hyperparameter types, trials-per-parameter, explicit OOS windows, legacy split, DNA, train/test metrics, best candidates. `cpu_cores` and `fast_mode` are accepted for compatibility; Terry currently runs single-process and has no Ray cluster. Smart objectives map to the corresponding historical Sharpe/Sortino metric rather than Jesse's richer smart-fitness internals. |
| MCP | 58 tools, 12 resources | 58 tools, 12 resources | Tool names match except the expected product rename `get_jesse_status` → `get_terry_status`; all resource topics exist under `terry://`, including optimization. Draft APIs accept Terry shorthand and Jesse route/date contracts. Response envelopes remain Terry-native. |
| Browser frontend | Nuxt/Vue, Monaco, research/live screens | Research workflow implemented, UI not code-identical | Local responsive dashboard has strategy editor, imports, settings, history, backtest/export controls, optimization, Monte Carlo, Rule Test, metrics, reports, auth, and accessibility controls. It is vanilla JS/FastAPI rather than Nuxt/Monaco, and it has no live account/execution screens. |
| Storage/runtime | PostgreSQL, Redis, multiple services | Deliberately different | SQLite candle/session/config files and local background threads; no Redis or Postgres required. |
| Live/paper trading | Separate plugin/product capability | Not implemented | Explicit project boundary. No credentials, account management, live orders, notifications, DEX, or multiple-account execution. |

## Changes made by this audit

1. Corrected sizing, crossover, streak/signal, timeframe, cointegration, and model compatibility
   gaps found by direct source comparison.
2. Added Jesse 2.5 Strategy fields and methods for routing, positions, live-state introspection,
   mark/funding/liquidation data, caching, candle pipelines, and ML.
3. Added research candles, ML, three candle pipelines, chart generation, CSV/JSON/Pine exports,
   benchmark results, and Optuna optimization with chronological OOS validation.
4. Expanded candle importing to all ten exchanges Jesse enables for backtesting.
5. Expanded MCP drafts for JSON routes/data routes, multi-route execution, export controls, and
   explicit optimization windows; added the missing optimization resource.
6. Updated the browser backtest controls and capability copy, added a parity regression suite,
   and corrected stale documentation that previously overstated full parity.
7. Aligned the complete `terry.research` export/signature surface, including Monte Carlo
   scenario/confidence results, equity-curve retrieval, significance plotting, and optimization
   summary options.

## Verification

- Offline suite: `python -m pytest -q` — **46 passed** (engine, dashboard/API, model/util
  parity, exchange payloads, research signatures/results, ML artifacts, candle pipelines,
  optimizer, plotting, resources).
- Dependency integrity: `python -m pip check`.
- Syntax integrity: `python -m compileall -q terry` and `git diff --check`.
- Public-surface audit: 174/174 indicator modules; no missing public `Strategy` or utility names;
  all 17 `terry.research` exports and their Jesse-leading argument order; 58 MCP tools and 12
  resources (with the expected Terry product-name substitutions).
- Network driver smoke test: one-minute BTC data returned from all ten supported historical
  markets on 2026-07-18. This verifies the current response shape, not guaranteed future provider
  availability.
- Full MCP network workflow: `python tests/test_mcp_e2e.py` imports public candles and runs the
  significance, Jesse-style backtest/export, Monte Carlo, explicit-window optimization, report,
  equity-curve retrieval, and resource workflow. The final run imported 57,600 candles and
  completed all stages successfully. Its temporary project and local HTML reports were removed by
  the test cleanup, so those test dashboard URLs are no longer available.

## Risk note

Matching Jesse's historical behavior does not make a strategy profitable. Optimization, ML, and
repeated backtest iteration can overfit the same data. Always keep untouched out-of-sample data,
inspect downside scenarios, account for exchange/data quality differences, and treat past
performance as evidence about a simulation—not a guarantee of future returns.
