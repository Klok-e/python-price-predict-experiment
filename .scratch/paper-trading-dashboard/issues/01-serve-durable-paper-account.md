# 01 — Serve a Durable Paper Account

**What to build:** Deliver the first complete tracer bullet of the replacement paper operator: one
manually runnable Python web service creates or restores a persistent $10,000 Paper Account, consumes
public market observations, applies the established policy execution contract, stores durable account
state in SQLite, and exposes a minimal localhost Live view. This slice establishes the single
application/test seam while preserving the existing pure financial simulation as its calculation
authority.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Starting with no operational database creates exactly one active Paper Account with $10,000
      cash, zero positions, a durable identity, and a Flat Start event.
- [ ] Operational SQLite state, recovery data, and fitted-policy checkpoints live under an explicitly
      Git-ignored generated-data boundary inside the repository.
- [ ] SQLite runs in WAL mode with one application-owned writer and atomically ordered account events
      plus a versioned recovery snapshot sufficient to restore current state.
- [ ] One paper-dashboard application owns observation, execution, persistence, API state, and the
      browser snapshot; browser requests never write account tables independently.
- [ ] The application can be constructed with an injected clock, market feed, policy backend,
      notification recorder, and temporary database for deterministic in-process ASGI tests.
- [ ] The service consumes one-minute public observations, marks equity, and uses only fully closed
      15-minute Decision Bars for policy decisions.
- [ ] Normal policy changes retain the current 60-second Decision Latency, minimum-turnover threshold,
      fresh public bid/ask Reference Price, 0.07% adverse all-in Transaction Cost, actual funding,
      exposure limits, concentration limits, and drawdown calculation.
- [ ] The service can restore and use the currently selected Fitted Policy without requiring a new
      validation or holdout run.
- [ ] A due decision can schedule an Executable Portfolio Change, receive an Effective Fill, and
      persist the resulting cash, quantities, equity, costs, funding, pending state, and model/input
      identity as one coherent account transition.
- [ ] Desired turnover below the execution threshold changes neither cash nor quantities and remains
      distinguishable from an execution failure.
- [ ] Closing and recreating the complete application over the same database restores the exact Paper
      Account, positions, pending execution, Fitted Policy identity, last observation, and event order
      instead of creating a new Flat Start.
- [ ] A minimal build-free Live view and read API show service status, Paper Account identity, equity,
      positions, data freshness, next decision, and pending-fill state from the same transactional
      snapshot.
- [ ] The HTTP listener defaults to `127.0.0.1:8765`, requires no authentication, and does not launch a
      browser or expose a LAN listener.
- [ ] Locally required frontend assets are bundled with tracked application source and no Node build,
      SPA framework, or external CDN is required.
- [ ] The high-level ASGI test proves Flat Start, one minute mark, one decision/fill path, below-threshold
      behavior, browser-visible state, and exact restoration without sleeping or asserting private
      implementation structure.
- [ ] Existing pure simulation tests continue to be the lower-level authority for fill, cost, funding,
      exposure, Marked Equity, drawdown, and Risk Stop mathematics.
- [ ] The locked environment, formatting, lint, strict type checking, and complete test suite pass for
      this slice.
- [ ] No production service cutover occurs in this ticket; the legacy operator remains the only enabled
      unit until the final hard-cut ticket prevents concurrent operation.
