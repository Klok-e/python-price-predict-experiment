# 04 — Complete Policy Lifecycle and Hard Cutover

**What to build:** Finish the unattended operating path and replace the legacy proof service. Weekly
GPU fitting and Policy Handoff work without interrupting trading, Policy Revisions are compatibility
gated and visible, current legacy state is imported or archived safely, the proof runtime is removed,
and an enabled lingering user service serves the verified localhost dashboard whenever the workstation
is on.

**Blocked by:** 03 — Deliver Auditable Metrics, History, and Explanations.

**Status:** ready-for-agent

- [ ] On the first Operating Window after the weekly deadline, fitting starts on ROCm in the background
      while observations, controls, marks, and ordinary decisions continue with the current Fitted
      Policy.
- [ ] Successful fitting persists an immutable checkpoint and performs one atomic Policy Handoff that
      preserves Current Portfolio and records old/new model identities and timing in History and System.
- [ ] Failed or interrupted fitting retains the current model, changes neither positions nor Paper
      Account identity, records diagnostics, and produces the accepted actionable notification.
- [ ] A compatible Policy Revision preserves the Paper Account, positions, and history, records a
      visible revision boundary, and segments performance metrics by Policy Protocol.
- [ ] Compatibility checks cover at least Trading Universe, account currency, and position, execution,
      and risk semantics; an incompatible revision enters durable Migration Required and performs no
      automated reset, flatten, conversion, or later policy decision.
- [ ] The thin CLI exposes the dashboard serve operation and retains data synchronization, validation,
      and Historical Holdout as Development Evidence workflows.
- [ ] The proof-oriented paper command, Proof Clock, pass/fail gate, duration/activity counters,
      proof-only configuration, reports, operating instructions, tests, and compatibility aliases are
      removed rather than redirected to the Paper Account.
- [ ] Remaining activity names use Executable Portfolio Change rather than proof-oriented qualifying
      terminology wherever they describe the new runtime.
- [ ] Before cutover, the exact current legacy state and service status are inspected read-only and the
      legacy user unit is stopped and disabled so old and new operators cannot consume the same market
      interval concurrently.
- [ ] Compatible legacy cash, positions, simulation accounting, model checkpoint, protocol identity,
      and auditable history import exactly once into the first Paper Account without invented events.
- [ ] Incompatible or unverifiable legacy state is preserved as immutable Development Evidence and the
      new service creates a $10,000 Flat Start; no ongoing legacy reader remains after this decision.
- [ ] The replacement is an enabled lingering user systemd service that runs without root, starts at
      boot independently of login, uses the project working directory and locked environment directly,
      exposes only `127.0.0.1:8765`, and never syncs dependencies or launches a browser.
- [ ] The unit waits for network availability, uses the ROCm operating environment, restarts only after
      process failure, and leaves provider outages to the application's in-process Data Stale recovery.
- [ ] Structural service verification passes and the obsolete proof unit can no longer be started as an
      alternate runtime after the hard cut.
- [ ] Lock verification, formatting, lint, strict type checking, the complete deterministic test suite,
      and the real-browser smoke all pass after proof removal.
- [ ] The installed user service is enabled and active, the localhost dashboard loads through that exact
      unit, its database and seven-backup policy operate under the ignored repository data boundary,
      and ROCm-backed fitting/inference is observable from System diagnostics.
- [ ] Live acceptance retains one real fully closed 15-minute Decision Record and completed Model
      Attribution from public data; the policy may legitimately choose no Executable Portfolio Change,
      so a live fill is not a delivery gate.
- [ ] Service restart after live acceptance restores the same Paper Account, durable lifecycle state,
      history, model identity, and dashboard metrics without Manual Reset.
- [ ] The experiment log records the cutover decision, legacy import/archive result, exact verification
      commands and outcomes, service identity/status, model/protocol identity, and live acceptance
      evidence without claiming real-world profitability.
- [ ] Documentation consistently describes the persistent Paper Account service and Development
      Evidence boundary and contains no active 60-day/100-change operating instructions.
- [ ] No real-money orders, exchange credentials, Binance account/testnet integration, LAN exposure,
      manual per-instrument trading, tick-level account history, export/import, or long-duration proof
      gate is introduced during cutover.
