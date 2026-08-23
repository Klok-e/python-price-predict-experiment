# 02 — Make Paper Account Lifecycle and Recovery Honest

**What to build:** Extend the durable Paper Account through the complete operator-controlled and
failure-recovery lifecycle. The user can safely pause, flatten, resume, and reset from Live; crashes,
shutdowns, data outages, stale pending fills, funding, and Risk Stop remain visible and economically
honest; every lifecycle state survives restart and is protected by backups and actionable desktop
notifications.

**Blocked by:** 01 — Serve a Durable Paper Account.

**Status:** ready-for-agent

- [ ] Every application start and graceful or detected stop creates a durable Operating Window
      boundary without resetting cash, positions, model identity, or account history.
- [ ] Restart after downtime reconstructs objectively recoverable one-minute valuation marks and all
      published funding for retained long and short positions, labels them as reconstructed inside an
      explicit gap, and includes them in equity and Maximum Drawdown.
- [ ] Recovery never creates a retroactive policy decision, Target Weight, bid/ask fill, or Operator
      Intervention for an interval in which the service was not operating.
- [ ] A pending policy change restored no more than two minutes after its scheduled fill can execute
      only against the first fresh observed quote; an older pending change becomes one durable Missed
      Execution and changes neither cash nor positions.
- [ ] Public-data unavailability enters Data Stale, reports the error and age of the last observation,
      suspends decisions and fills, keeps the UI and read controls available, and retries in process
      with bounded backoff rather than terminating for systemd recovery.
- [ ] Data recovery backfills reconstructible marks and funding, expires overdue pending execution,
      opens a new Operating Window segment as appropriate, and continues the same Paper Account.
- [ ] Pause is an immediate, idempotent Operator Intervention that cancels unfilled policy execution,
      retains current exposure, blocks later decisions/fills, and survives application recreation.
- [ ] Resume is immediate and idempotent only from ordinary Paused, waits for the next naturally due
      Decision Bar, and cannot bypass Reset Pending, Risk Stopped, or Migration Required.
- [ ] Flatten and Pause confirms current exposure and estimated cost, cancels pending policy execution,
      fills every open position at the next fresh bid/ask without policy Decision Latency, charges the
      normal Transaction Cost, and remains durably Paused and flat.
- [ ] Manual Reset requires its separate confirmation, enters durable Reset Pending, uses the same
      immediate-next-fresh-quote flattening contract, and cannot archive the account while any position
      or required fill remains unresolved.
- [ ] Successful Manual Reset immutably archives the old Paper Account and all of its history, then
      creates exactly one new $10,000 Flat Start; it never deletes or rewrites the archived account.
- [ ] A Drawdown Limit breach schedules the established delayed terminal Risk Stop fill, rejects later
      policy decisions, survives restart, and can return to trading only through Manual Reset.
- [ ] Trading, Paused, Reset Pending, Risk Stopped, and Migration Required are durable lifecycle states;
      Data Stale and notification health are separately visible operational overlays.
- [ ] State-changing browser operations use POST semantics, same-origin and CSRF protection, durable
      idempotency, and state-version checks so duplicate clicks, stale pages, and invalid transitions
      cannot create duplicate or out-of-order interventions.
- [ ] Live shows lifecycle state, gap/data freshness, pending or missed execution, current exposure,
      control availability, Reset progress, and Risk Stop without requiring database inspection.
- [ ] One online SQLite backup is created before every schema migration and at most once per active
      day; only the seven newest daily backups are retained, and backup age or failure is visible
      without stopping trading.
- [ ] Desktop notifications are emitted and deduplicated for executed Portfolio Changes, Risk Stop,
      completed Manual Reset, data stale beyond five minutes, fitting failure, and incompatible Policy
      Revision; routine decisions and no-trade outcomes do not notify.
- [ ] Failure to reach the desktop notification bus is recorded for System visibility but never changes
      trading or lifecycle state.
- [ ] ASGI tests cover long, short, mixed, flat, stale-data, duplicate-control, partial-failure, process
      restart, and machine-equivalent restart scenarios using deterministic time and feeds.
- [ ] Pure financial tests cover average execution effects of operator and Risk Stop fills without
      coupling lifecycle tests to internal SQL, worker, or coroutine structure.
- [ ] The locked environment, formatting, lint, strict type checking, and complete test suite pass for
      this slice.
