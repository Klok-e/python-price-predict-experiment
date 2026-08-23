# 02 — Goal run: produce Profitable Policy

**What to build:** Run the approved long autonomous goal from the cleaned foundation through the
complete Direct Net-Growth Portfolio Policy implementation, real historical experiments, initial
Historical Holdout, and Forward Paper Proof. Continue truthful, recorded Policy Revisions until one
Policy Protocol satisfies the definition of Profitable Policy; never substitute prediction metrics,
reused holdout evidence, or partial paper time for that outcome.

**Blocked by:** 01 — Hard-cut repository cleanup.

**Status:** resolved

- [x] Implement the deep workflow and four operational commands for canonical data synchronization,
      validation, Historical Holdout evaluation, and public-data-only paper operation.
- [x] Synchronize maximum available Binance-native history from 2020 onward for the fixed BTCUSDT,
      ETHUSDT, BNBUSDT, and SOLUSDT USD-M perpetual Trading Universe.
- [x] Build causal multi-timeframe Market State with explicit Optional Market Input masks and strict
      Execution-Complete Interval validation.
- [x] Implement continuous Target Weights from Current Portfolio, No Leverage, 50% per-ticker
      concentration, zero-yield cash, 1% Qualifying Portfolio Change threshold, funding, realistic
      Effective Fills, one-minute Marked Equity, and terminal Risk Stop.
- [x] Implement direct Net Log Growth training with the drawdown constraint, linear sanity policy,
      declared causal temporal-convolution candidates, three-seed stability selection, and the
      actual averaged-weight ensemble.
- [x] Implement twelve purged, prequential 90-day Walk-Forward Folds with Sunday retraining and
      Policy Handoff; reject non-positive aggregate return or any Drawdown Limit breach.
- [x] Write only reproducibility manifests, structured reports, equity, trades, and frozen Fitted
      Policy artifacts identified by configuration, code, data, and model hashes.
- [x] Pass focused behavioral tests for causality, resampling, data completeness, fills, costs,
      funding, portfolio constraints, turnover, drawdown, Risk Stop, folds, evidence states,
      reproducibility, and one synthetic end-to-end profitable pattern.
- [x] Run real validation experiments and append every exact configuration, identity, metric set,
      and verdict to the experiment log; continue Policy Revisions until a Validated Policy Protocol
      has positive Compounded Net Return and every fold remains within 20% Maximum Drawdown.
- [x] Evaluate the initial May-July 2026 Historical Holdout exactly once and record the immutable
      Holdout-Passing or Consumed outcome without relabeling reused evidence.
- [x] If the initial holdout fails, promote it to Development Evidence, continue recorded development,
      and require fresh Forward Paper Proof for the revised Policy Protocol.
- [ ] Operate Forward Paper Proof from a Flat Start using public data only, preserving Current
      Portfolio across scheduled Fitted Policy replacement and resetting Proof Clock after every
      Policy Revision.
- [ ] Keep this goal open until one unchanged Policy Protocol records positive Compounded Net Return,
      at least 60 elapsed days, at least 100 Qualifying Portfolio Changes, and no Drawdown Limit
      breach.
- [x] Do not add authenticated exchange access or real-order placement; live-capital authorization
      remains a separate post-proof decision.

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

## Answer

Implementation, verification, review, commit, fresh validation, and unattended Forward Paper Proof
launch are complete. On 2026-08-23 the user explicitly closed the implementation scope without
waiting interactively for the 60-day gate. This resolution does not claim a Profitable Policy: the
two unchecked Forward Paper Proof requirements remain unproven, and the enabled service continues
collecting that evidence independently.
