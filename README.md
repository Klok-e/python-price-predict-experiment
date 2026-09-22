# Persistent Net-Growth Paper Account

A causal 15-minute portfolio-policy experiment for BTCUSDT, ETHUSDT, BNBUSDT, and SOLUSDT Binance
USD-M perpetuals. The Trading Policy emits continuous long, short, or cash Target Weights and is
optimized against portfolio Net Log Growth after an all-in turnover cost and actual funding.

This repository places no real orders and uses no exchange credentials. Historical validation,
Historical Holdout results, and Paper Account history are Development Evidence; none establishes
real-world profitability.

## Install

The project uses Python 3.13, the committed `uv.lock`, and the official ROCm 7.2 Torch source.

```bash
uv sync --locked --all-groups
```

All Policy Protocol decisions live in `policy.toml`. Runtime overrides are limited to paths, the
compute device, and the localhost listener.

## Workflow

```bash
uv run netgrowth data-sync
uv run netgrowth validate --device cuda
uv run netgrowth holdout --device cuda
uv run netgrowth serve --device cuda
```

- `data-sync` retains maximum available public Binance-native history from 2020 onward.
- `validate` runs twelve purged, prequential 90-day Walk-Forward Folds and records Development
  Evidence for the Policy Protocol.
- `holdout` consumes the configured Historical Holdout exactly once after validation.
- `serve` runs one persistent local Paper Account, the autonomous market loop, background fitting,
  the control API, and the browser dashboard in one process. It binds only to
  `http://127.0.0.1:8765/` and never opens a browser.

The initial Paper Account starts flat with $10,000. Its cash, positions, pending state, lifecycle,
model identity, one-minute marks, and append-only history survive process and workstation restarts.
The dashboard exposes Live, History, and System views. Live summarizes account value, profit/loss,
performance versus hold, drawdown, and exposure alongside operating status and the latest decision.
Recent simulated trades shows the last five executed buys and sells with quantity, execution price,
and simulated cost; these remain separate from decisions that did not execute a trade.
Expand Financial details for accounting and risk breakdowns; History holds account-lifetime activity
totals and a Trades filter for executions, and System holds operational diagnostics. The hold comparison
is measured from its recorded revision boundary, while account profit/loss and return cover the account lifetime.

Pause or Resume appears beside Account actions, which contains Close all positions and pause and
Reset account. Each action requires confirmation; there are no manual per-instrument trades.

Open positions remain exposed during downtime. On recovery, the application reconstructs only
objective one-minute marks and published funding, labels the gap, and never invents decisions or
historical bid/ask fills. Fresh policy execution retains the 60-second Decision Latency, 0.07%
adverse all-in Transaction Cost, No Leverage and concentration constraints, and the 20% Drawdown
Limit.

Eligibility for an ordinary portfolio change is determined at Signal Time using the 1% turnover
threshold. Below-threshold decisions complete immediately. Qualifying decisions retain eligibility
at their delayed fresh-price fill, subject to expiry and risk controls.

For an existing account using the earlier rule, follow the
[validated revision procedure](docs/paper-policy-revision.md). This preserves the account and starts
a forward Hold Benchmark at the recorded revision boundary.

## Unattended user service

The checked-in user unit executes the already synchronized environment directly and does not launch
a browser or synchronize dependencies. Install the validated bundle at
`computed-data/paper-dashboard/active-revision.json` before starting this unit; see the
[revision procedure](docs/paper-policy-revision.md).


```bash
systemctl --user link "$PWD/systemd/netgrowth-paper-dashboard.service"
systemctl --user enable --now netgrowth-paper-dashboard.service
loginctl enable-linger "$USER"
```

The service retries provider outages inside the application while keeping dashboard reads available.
Systemd restart is reserved for process failure. Always stop and disable any older operator before
linking the replacement; two operators must never consume the same market interval concurrently.

## Data and artifacts

- Canonical public data: `computed-data/dataset/`
- Historical Development Evidence: `computed-data/evidence/`
- Operational database, checkpoints, and rotating backups: `computed-data/paper-dashboard/`

`computed-data/` is ignored by Git. The Paper Account uses SQLite WAL with a single application-owned
writer, an append-only ordered event ledger, and versioned recovery snapshots. Archived accounts are
immutable and remain browsable after Manual Reset. Model Attribution is asynchronous approximate
post-hoc influence evidence; exact Decision Records and execution outcomes remain authoritative.

## Verification

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```
