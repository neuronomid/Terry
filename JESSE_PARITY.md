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
The worker runtime and frontend implementation also differ. Those boundaries must not be described
as full product parity.

## Backend algorithm parity re-audit (2026-07-20)

A source-level review of Terry's four research backends against the current
`jesse-ai/jesse` `master` (`jesse/services/metrics.py`,
`jesse/modes/optimize_mode/fitness.py`, `jesse/research/monte_carlo/*`,
`jesse/research/rule_significance_testing/*`):

- **Backtest metrics** — Sharpe/Sortino/Calmar/Omega/Serenity, max drawdown, and
  CAGR use Jesse's exact formulas and the crypto 365-period annualization
  (`ddof=1` std, same downside/drawdown definitions). Matches.
- **Optimization** — identical fitness: `total > 5` gate, `total_effect_rate =
  min(log10(total)/log10(optimal_total), 1)`, `normalize(ratio, low, high)` with
  the same per-objective ranges (sharpe −.5/5, calmar −.5/30, sortino −.5/15,
  omega −.5/5, serenity −.5/15), negative-ratio short-circuit, and
  `score = total_effect_rate · ratio_normalized`. Terry uses a seeded Optuna TPE
  study over a thread pool instead of Jesse's Ray workers (more reproducible; same
  math). The `smart sharpe/sortino` objectives fall back to the base metric because
  Jesse's own metrics dict does not expose `smart_*` keys.
- **Monte Carlo** — worst_5/median/best_5 map to the 5th/50th/95th percentiles,
  with 365-day annualization, base seed 42, and the same 2.5/5/25/75/95/97.5
  confidence intervals. Candle pipelines (moving-block bootstrap, gaussian noise,
  gaussian resampler) were ported from Jesse with the same defaults. Matches.
- **Rule significance** — signal-only pass → log returns → detrend →
  `signal · detrended` → bootstrap of zero-centred returns → `p = mean(sims ≥
  observed)`. Matches. **Fixed:** `annualized_return` annualized over 252 days;
  corrected to 365 to match Jesse's `TRADING_DAYS_PER_YEAR` (crypto trades 24/7).

## Capability matrix

| Area | Jesse 2.5 baseline | Terry 0.2 status | Evidence / difference |
|---|---|---|---|
| Strategy lifecycle and smart orders | Yes | Compatible | Every public method/property name on Jesse's `Strategy` is represented; zero-based candle index, entry/filter state, smart market/limit/stop inference, deterministic intrabar sorting, scale-in/out, tiered stops/targets, callbacks, termination hooks, and same-candle flips are exercised by engine tests. |
| Order, Position, ClosedTrade developer API | Yes | Compatible for research | Jesse status aliases, signed remaining quantity, ROI/cost, direction/state properties, serialization, trade/order IDs, timeframe/session metadata, mark/funding/liquidation fields, and capitalized model imports are present. Live-only values are placeholders in historical mode. |
| Indicators | 174 public modules in audited repository | 174/174 modules | Module names match. Existing numerical cross-checks cover the exported indicator set; named-tuple and sequential behavior have regression tests. |
| Helpers and utilities | 120 helper + 23 utility functions | Public names/signatures matched | Jesse's complete public helper and utility names and keyword-compatible signatures are present, including timestamps/Arrow, config/mode checks, symbols, order-book math, PNL, DNA, formatting, cleaning/compression, sequential `crossed`, signal lines, streaks, float math, cointegration, and fee/floor sizing semantics. Live-only helper paths report Terry's explicit no-live boundary. |
| Historical engine | Spot + futures, multiple routes/timeframes | Compatible local engine | 44 metric keys, smart orders, exact fee/balance handling, futures margin enforcement, spot reservations and exit limits, cross/isolated leverage with liquidation, warm-up/data routes, multiple symbols, cancellation, and exports. One research run still uses one selected exchange, matching Jesse's draft form. |
| Historical exchanges | 10 enabled markets | 10 markets | Binance Spot, Binance US Spot, Binance Perpetual Futures, Bitfinex Spot, Coinbase Spot, Bybit USDT/USDC Perpetual, Bybit Spot, Gate USDT Perpetual, Kraken Pro Futures. All ten real public response paths were checked on the audit date; offline fixtures guard normalization. |
| Research candle API | get/store/factories | Compatible | SQLite-backed `get_candles`, `store_candles`, `fake_candle`, `fake_range_candles`, and close-price factories. Terry also offers a blocking notebook import helper. |
| Backtest exports | CSV, JSON, TradingView, charts, benchmark | Implemented | Pure API and MCP runner support CSV, JSON, Pine v5, six PNG chart outputs, logs, hyperparameters, equity curve, and buy-and-hold benchmark. |
| Candle pipelines | Gaussian noise, Gaussian resampler, moving-block bootstrap | Implemented | Jesse-compatible pipeline classes, strategy hook, deterministic seed support, and OHLC-invariant tests. |
| Rule significance test | Bootstrap entry-rule test | Implemented | Jesse-compatible research signature/result samples and plotting, plus route/data-route MCP drafts, reproducible seeds, bounded `cpu_cores` workers, >=2,000 simulation default, observations, p-value, and dashboard report. |
| Monte Carlo | Candle and trade modes | Implemented | Jesse-compatible call signatures, concurrent isolated scenarios, route/data-route MCP drafts, pipeline controls, streamed callbacks, scenario/confidence payloads, summary/plot helpers, trade-order mode, downside/overfit verdicts, and dedicated per-scenario equity-curve retrieval. |
| ML research/deploy | Gather, sklearn train, artifacts, inference | Core API implemented | Chronological splits, binary/multiclass/regression metrics, model/scaler artifacts, Jesse-shaped RFE/F-test/correlation/CV-removal consensus diagnostics, per-feature retraining impact, calibration bins, CSV loading, and lazy Strategy inference. Jesse's verbose console presentation is not code-identical. |
| Optimization | Optuna + Ray, explicit train/test windows | API/results compatible, execution differs | Optuna TPE, hyperparameter types, trials-per-parameter, explicit OOS windows, legacy split, DNA, Jesse's normalized/trade-count-weighted fitness, train/test metrics, best candidates, and bounded `cpu_cores` workers. Terry uses local worker threads instead of a Ray cluster. Smart objectives remain Terry extensions mapped to their corresponding historical ratio because Jesse's audited metric payload does not emit separate smart-ratio keys. |
| MCP | 58 tools, 12 resources | 58 tools, 12 resources | Tool names match except the expected product rename `get_jesse_status` → `get_terry_status`; all resource topics exist under `terry://`, including optimization. Jesse-leading schemas, success/error actions, draft/session envelopes, nested dashboard state, session filters, structured notes/source snapshots, retryable candle import IDs, and candle/config/indicator/strategy results are covered by contract tests. Terry shorthand remains available as keyword-only extensions. |
| Browser frontend | Nuxt/Vue, Monaco, research/live screens | Research workflow implemented, UI not code-identical | Local responsive dashboard has an IDE-like editor with line numbers/indentation/save shortcut, multi-route and data-route inputs, worker/optimization/pipeline controls, imports, settings, editable/deletable titled sessions, notes/source snapshots, history, all research modes, metrics, reports, auth, guarded destructive actions, unsaved-change protection, and accessibility controls. It is vanilla JS/FastAPI rather than Nuxt/Monaco, and it has no live account/execution screens. |
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
8. Added bounded local workers for optimization, Monte Carlo, and significance; isolated
   `Strategy.shared_vars` per engine store; and matched Jesse's optimization fitness formula.
9. Matched Jesse's structured ML diagnostics and expanded the MCP/dashboard research forms with
   route arrays, data routes, worker controls, reproducible seeds, and candle-pipeline settings.
10. Upgraded the browser strategy editor with a line gutter, cursor status, auto-indentation,
    Tab/Shift-Tab handling, and Ctrl/Command-S saving.
11. Aligned Jesse-leading MCP schemas and response envelopes, including paginated/filtered
    sessions, structured note metadata, automatic strategy-code snapshots, and retryable candle
    import IDs, while retaining Terry's keyword-only convenience extensions.
12. Added unchanged Jesse-import strategy loading, Jesse-compatible `terry.testing_utils`, and a
    shared `terry-strategy-tests` agent skill discoverable through `.agents/skills` and
    `.claude/skills`.
13. Added session titles and research notes to the browser forms/history/results, backed by the
    same persisted note metadata and strategy snapshots exposed through MCP.
14. Replayed Jesse's unchanged strategy regression corpus and fixed lifecycle ordering, zero-based
    indexing, order priority/replacement, partial exits, trade metadata, isolated liquidation,
    futures margin, spot asset reservations, and strategy termination behavior.
15. Added Jesse-compatible capitalized core model modules, the historical global-store facade,
    logger service, and all 120 public helper functions with Jesse's keyword-compatible
    signatures; 142/143 upstream regression strategies now load unchanged (the remaining file is
    intentionally fully commented and defines no strategy class).
16. Audited the dashboard against the current Web Interface Guidelines and added session
    rename/delete controls, guarded cancellation, unsaved-note protection, cached `Intl` number
    and date formatting, restored focus visibility, and coarse-pointer touch targets.

## Verification

- Offline suite: `python -m pytest -q` — **60 passed** (engine, dashboard/API, model/helper/util
  parity, exchange payloads, research signatures/results, ML artifacts, candle pipelines,
  optimizer, plotting, resources).
- Dependency integrity: `python -m pip check`.
- Package integrity: `python -m pip wheel --no-deps .` produced
  `terry_trade-0.2.3-py3-none-any.whl` with the expected metadata.
- Syntax integrity: `python -m compileall -q terry` and `git diff --check`.
- Browser smoke test: headless Chrome rendered the strategy editor and Monte Carlo form with the
  new line gutter, status bar, worker, pipeline, and advanced-route controls.
- Public-surface audit: 174/174 indicator modules; no missing public `Strategy` names; exact
  120/120 helper and 23/23 utility function names/signature shapes; all upstream enum values and
  exception names; all 17 `terry.research` exports and their Jesse-leading argument order; 58 MCP
  tools and 12 resources (with the expected Terry product-name substitutions).
- Upstream strategy audit: 142/143 Jesse regression strategy files load unchanged; the sole
  non-loadable file contains only comments. All 120 literal upstream historical-helper calls
  match their expected outcome: 101 complete successfully and 19 raise the expected validation
  path, with zero unexpected outcomes.
- MCP contract audit: Jesse-leading parameters for all 58 tools, structured success/error
  envelopes, serialized details for 174/174 indicators, nested draft/session state, source
  snapshots, filtered pagination, and a full local create/draft/read/update/list workflow.
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
