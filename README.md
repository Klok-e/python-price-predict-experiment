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
The dashboard exposes Live, History, and System views. Human actions are limited to Pause, Resume,
Flatten and Pause, and Manual Reset; there are no manual per-instrument trades.

Open positions remain exposed during downtime. On recovery, the application reconstructs only
objective one-minute marks and published funding, labels the gap, and never invents decisions or
historical bid/ask fills. Fresh policy execution retains the 60-second Decision Latency, 0.07%
adverse all-in Transaction Cost, No Leverage and concentration constraints, and the 20% Drawdown
Limit.

## Unattended user service

The checked-in user unit executes the already synchronized environment directly and does not launch
a browser or synchronize dependencies:

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
