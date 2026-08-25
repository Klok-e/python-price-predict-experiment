# 02 — Build the Direct Net-Growth Policy and Development Evidence

**Status:** resolved

This historical ticket delivered the direct portfolio-policy implementation, Binance-native market
data, causal validation, immutable artifacts, and the initial Historical Holdout. The proof-oriented
paper-operation portion was abandoned and is superseded by
`.scratch/paper-trading-dashboard/spec.md` and ADR-0005.

- [x] Implement the deep workflow and the data synchronization, validation, and Historical Holdout
      commands retained as Development Evidence.
- [x] Build causal Market State, direct Target Weights, realistic execution costs, funding, portfolio
      constraints, scheduled fitting, and terminal Risk Stop behavior.
- [x] Record validation and Historical Holdout evidence without substituting prediction metrics or
      reused holdout evidence.
- [x] Remove the proof operator, proof clock, proof gate, qualifying-change terminology, service,
      configuration, tests, and compatibility aliases during the persistent Paper Account cutover.
- [x] Keep authenticated exchange access and real-order placement outside repository scope.

## Comments

- 2026-08-20: implementation and ROCm stability fixes committed through `acb7470`; exact validation
  evidence committed in `9c5e54d`. Immutable validation result `8a5063fa74509b1c` selected
  `tcn-64-7-ensemble` at +29.026712% Compounded Net Return, 7.887010% Maximum Drawdown, and 300
  Qualifying Portfolio Changes. Both final review axes reported no findings; locked verification
  passed with 55 tests including a full 8,640-step ROCm backward regression.
- 2026-08-20: the first Forward Paper Proof attempt for protocol `9a9e48be...` was stopped after its
  bootstrap path took longer than one minute and strict gap detection rejected the next mark. The
  attempt recorded only one flat minute and is not proof evidence. A hard-cut paper-feed fix now
  collects only execution-complete mark inputs and fetches all four tickers concurrently; both final
  review axes are clean and the regression proves request overlap. Because runtime code identity is
  part of the Policy Protocol, fresh validation must finish before a new flat Proof Clock starts.
- 2026-08-20: final gap-safe protocol `7bed0db0...` passed fresh validation in immutable artifact
  `73d9227a245d1eba` at +29.026712% Compounded Net Return, 7.887010% Maximum Drawdown, and 300
  Qualifying Portfolio Changes. Fresh Forward Paper Proof started flat at `2026-08-20T01:22:00Z`
  under the enabled `netgrowth-paper-proof.service`, advanced through three consecutive minutes with
  zero restarts, and is reboot-persistent with user lingering enabled. The 60-day / 100-change /
  positive-return / no-breach gate remains open.
- 2026-08-23: retry-resilient protocol `39a5a83c...` passed fresh ROCm validation in immutable
  artifact `ca6207a2011194aa` at +29.026712% Compounded Net Return, 7.887010% Maximum Drawdown, and
  300 Qualifying Portfolio Changes. Fresh Forward Paper Proof started flat at
  `2026-08-23T17:12:00Z` under the enabled, lingering `netgrowth-paper-proof.service`, advanced
  through consecutive `17:12` and `17:13` observations from PID `339493` with zero restarts, and
  remains active. The 60-day / 100-change / positive-return / no-breach gate remains open.
- 2026-08-20 through 2026-08-23: the historical policy, validation, and holdout work completed; exact
  immutable artifact identities and metrics remain recorded in `experiment_log.md` and git history.
- 2026-08-23: the persistent Paper Account explicitly superseded proof-oriented operation. The legacy
  state was archived as immutable Development Evidence and the old service was removed.

## Answer

The reusable Trading Policy and Development Evidence workflows are complete. Ongoing operation is
owned exclusively by the persistent Paper Account; this ticket carries no active proof gate or
independent service.
