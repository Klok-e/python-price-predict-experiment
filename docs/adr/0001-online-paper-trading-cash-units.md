# ADR 0001: Online Paper Trading Uses Cash/Units Accounting

## Status

Accepted

## Context

Historical paper replay can score a complete holding period because both entry and exit bars already
exist. Online paper trading cannot do that without reading future data. The old forward path reused
closed-period replay semantics, which made a single forward decision compute a fill and exit in one
call.

## Decision

Forward paper trading is now **Online Paper Trading**. It uses only a **Closed Signal Bar** to append
predictions, target weights, and a **Pending Paper Order**. The order fills only after next-open data
exists. Open positions are marked to market from current bars.

The online ledger reduces into **Cash/Units Paper State**: cash and per-ticker units are authoritative;
weights, exposures, equity, and PnL are derived from marks. Simple margin is allowed for the current
policy, including negative cash and short units, but borrow, funding, and liquidation are out of
scope.

## Consequences

- Online paper trading no longer reads future exit prices.
- The ledger can recover state after process restart from JSONL records alone.
- Historical replay remains the tool for closed-period evaluation.
- The online path is a closer simulation of live operation, but it is still paper trading and does not
  place real exchange orders.
