# 03 — Deliver Auditable Metrics, History, and Explanations

**What to build:** Turn the operational Paper Account into the complete dashboard the user can inspect
each evening. Live, History, and System expose trustworthy account, risk, position, benchmark, model,
and service information; linked financial charts explain when decisions and fills occurred; every
decision has an exact Decision Record and an asynchronous, clearly approximate Model Attribution;
archived Paper Accounts remain browsable and comparable.

**Blocked by:** 02 — Make Paper Account Lifecycle and Recovery Honest.

**Status:** resolved

- [x] Live presents trading/lifecycle state, current Operating Window, data freshness, errors, next
      decision, pending fill, fitting state, recent activity, account metrics, risk metrics, positions,
      charts, and the accepted lifecycle controls in one scannable view.
- [x] History presents a filterable, durably ordered timeline of decisions, fills, Missed Executions,
      funding, reconstructed gaps, Operator Interventions, Risk Stop, policy revisions, Fitted Policy
      handoffs, and account archival/reset events.
- [x] System presents market-feed status, model and Policy Protocol identity, fitting status and errors,
      Operating Windows, notification health, database/backup status, and service diagnostics.
- [x] Account metrics include starting/current equity, net P&L, Compounded Net Return, gross trading
      P&L, Transaction Cost, funding paid or received, and turnover with a documented reconciliation to
      Marked Equity.
- [x] Risk metrics include current and Maximum Drawdown, high-water equity, the 20% Drawdown Limit,
      gross and net exposure, Cash Weight, and per-instrument concentration.
- [x] Every instrument row includes side, quantity, mark, notional, Current Weight, Target Weight,
      average entry, realized P&L, and unrealized P&L where applicable.
- [x] Average-cost accounting correctly handles same-direction adds, partial reductions, full closes,
      long-to-short reversals, short-to-long reversals, costs, and funding while Marked Equity remains
      the authoritative total.
- [x] Activity metrics distinguish all decisions, Executable Portfolio Changes, instrument fills,
      below-threshold decisions, unchanged targets, Missed Executions, and Operator Interventions.
- [x] Each Paper Account includes cash and passive equal-weight-long benchmarks beginning at the same
      Flat Start; the equal-weight benchmark starts 25% long each ticker, incurs initial cost and actual
      funding, drifts passively, and remains exposed during downtime.
- [x] The main chart shows one selected ticker at a time with candlesticks and distinct signal, fill,
      Missed Execution, funding, and Operator Intervention markers.
- [x] Current and Target Weight appear beneath price, and one synchronized crosshair aligns price with
      Paper Account equity, both benchmarks, drawdown, and exposure panels.
- [x] Current-window, 24-hour, seven-day, and full-account ranges work without introducing multiple
      incompatible price scales or dropping material event markers.
- [x] Clicking a chart marker resolves to its durable event and opens the linked Decision Record,
      execution outcome, and Model Attribution rather than chart-only metadata.
- [x] A Decision Record is persisted for every due policy decision, including executable,
      below-threshold, and effectively unchanged outcomes, before attribution is started.
- [x] Each Decision Record includes signal time, model/input identity, Current Portfolio, raw and
      constrained Target Weights, projected turnover, threshold outcome, pending and fill timing,
      actual Reference and Effective Fill, costs, applicable constraints, and final outcome.
- [x] Every instrument fill and Missed Execution links back to exactly one Decision Record; signal and
      fill are represented as separate times and graph markers.
- [x] Model Attribution uses Integrated Gradients against the neutral normalized-market baseline and
      aggregates signed influence by ticker, temporal region, momentum, volatility/range,
      flow/activity, derivatives positioning/carry, availability, and Current Portfolio.
- [x] Attribution starts only after the Decision Record and pending execution are durable, runs
      asynchronously at lower priority than trading work, exposes Attribution Pending, and cannot
      delay the scheduled fill.
- [x] Stored attribution includes the top signed influences, method parameters, input hash, and model
      hash; the full tensor is not retained, but the canonical input and immutable model identities are
      sufficient to detect or reproduce the result.
- [x] Every explanation explicitly labels Model Attribution as approximate post-hoc influence evidence,
      while the Decision Record and execution outcome are presented as exact.
- [x] Every one-minute account mark and all material events are retained indefinitely; long chart
      queries downsample only for rendering and never discard stored history or event markers.
- [x] Archived Paper Accounts are immutable, selectable, and comparable without changing the one active
      account or its controls.
- [x] All persistence and calculations use timezone-aware UTC; the UI defaults to Europe/Kiev and shows
      UTC in detailed views and tooltips, including correct daylight-saving transitions.
- [x] Financial data refreshes once per minute while clocks, freshness ages, and countdowns can update
      once per second without creating tick-level account history.
- [x] The complete UI remains build-free, uses only locally bundled assets, and works without an
      external CDN, Node toolchain, SPA framework, or second process.
- [x] The dashboard contains no prediction accuracy, arbitrary per-fill win rate, manual buy/sell
      controls, export/import feature, or language claiming causal explanation or proven profitability.
- [x] ASGI tests cover every accepted metric and event type, marker-to-detail linkage, archived account
      selection, Kyiv/UTC conversion, range/downsampling behavior, attribution scheduling, and all
      three views through externally visible responses.
- [x] Deterministic numerical tests validate Integrated Gradients completeness on small models without
      coupling high-level behavior to exact attribution values from a trained TCN.
- [x] A small real-browser smoke verifies chart startup, live refresh, navigation, marker detail, and
      control wiring without treating DOM layout or CSS classes as the behavioral contract.
- [x] The locked environment, formatting, lint, strict type checking, and complete test suite pass for
      this slice.

## Comments

- 2026-08-25: completed the auditable dashboard slice through the public ASGI seam and deterministic
  numerical attribution/accounting seams. Lock verification, Ruff formatting/lint, strict mypy, and
  the complete suite passed with 111 passed, 1 optional Playwright smoke skipped, and 14 third-party
  Torch deprecation warnings. A real `agent-browser` smoke against a temporary localhost ASGI instance
  verified chart startup, signal/fill marker detail, exact Decision Record and execution data, signed
  approximate attribution, Live/History/System navigation, Pause wiring, and the automatic one-minute
  financial refresh.
