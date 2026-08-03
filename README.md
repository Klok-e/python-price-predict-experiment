# Direct Net-Growth Portfolio Policy

A causal 15-minute portfolio-policy experiment for BTCUSDT, ETHUSDT, BNBUSDT, and SOLUSDT
Binance USD-M perpetuals. The Trading Policy emits continuous long, short, or cash Target Weights
and trains against portfolio Net Log Growth after an all-in turnover cost and actual funding.

This repository does not place real orders. A policy is not called profitable until it completes
Forward Paper Proof with positive Compounded Net Return and no Drawdown Limit breach.

## Install

The project uses the committed Python 3.13 `uv.lock` and CPU-only Torch source.

```bash
uv sync --locked
```

All Policy Protocol decisions live in `policy.toml`. CLI overrides are intentionally limited to
configuration/data/output paths and the compute device.

## Workflow

```bash
uv run netgrowth data-sync
uv run netgrowth validate --device cpu
uv run netgrowth holdout --device cpu
uv run netgrowth paper --device cpu
```

- `data-sync` retains maximum available public Binance-native history from 2020 onward.
- `validate` runs twelve purged, prequential 90-day Walk-Forward Folds and freezes an eligible
  Policy Protocol.
- `holdout` is locked until validation passes and consumes the May-July 2026 Historical Holdout
  exactly once.
- `paper` runs public-data-only Forward Paper Proof. Completion requires both 60 days and 100
  Qualifying Portfolio Changes. A Policy Revision resets its Proof Clock; a scheduled Sunday Fitted
  Policy handoff does not.

Independent evidence runs start flat with $10,000. Gross Exposure is capped at 100%, absolute
exposure to one ticker at 50%, and a 20% Drawdown Limit triggers a delayed flattening Risk Stop.

## Data and artifacts

Raw data is cached under `computed-data/dataset/`. Evidence is content-addressed under
`computed-data/evidence/`. Each run stores only:

- `manifest.json`
- `report.json`
- `equity.csv`
- `trades.csv`
- `model.pt`

The manifest identifies configuration, code, canonical data, and Fitted Policy hashes. Charts are
generated from equity and trades on demand; no HTML timeline or decision ledger is persisted.

## Verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```
