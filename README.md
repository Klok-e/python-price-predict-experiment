# Direct Net-Growth Portfolio Policy

A causal 15-minute portfolio-policy experiment for BTCUSDT, ETHUSDT, BNBUSDT, and SOLUSDT
Binance USD-M perpetuals. The Trading Policy emits continuous long, short, or cash Target Weights
and trains against portfolio Net Log Growth after an all-in turnover cost and actual funding.

This repository does not place real orders. A policy is not called profitable until it completes
Forward Paper Proof with positive Compounded Net Return and no Drawdown Limit breach.

## Install

The project uses the committed Python 3.13 `uv.lock` and the official ROCm 7.2 Torch source.

```bash
uv sync --locked
```

All Policy Protocol decisions live in `policy.toml`. CLI overrides are intentionally limited to
configuration/data/output paths and the compute device.

## Workflow

```bash
uv run netgrowth data-sync
uv run netgrowth validate --device cuda
uv run netgrowth holdout --device cuda
uv run netgrowth paper --device cuda
```

- `data-sync` retains maximum available public Binance-native history from 2020 onward.
- `validate` runs twelve purged, prequential 90-day Walk-Forward Folds and freezes an eligible
  Policy Protocol.
- `holdout` is locked until validation passes and consumes the May-July 2026 Historical Holdout
  exactly once.
- `paper` is the long-running public-data-only Forward Paper Proof operator. It samples closed
  one-minute public bars and fresh Binance best bid/ask quotes, checkpoints the virtual portfolio,
  and resumes pending fills after restart. Keep the command running continuously: missed midpoint
  snapshots are rejected rather than reconstructed, the interrupted attempt is frozen for audit,
  and operation resumes from a new Flat Start and Proof Clock. Completion requires both 60 days and
  100 Qualifying Portfolio Changes. A Policy Revision also starts flat and resets its Proof Clock;
  scheduled Sunday fitting runs alongside minute collection, and its atomic Fitted Policy handoff
  preserves Current Portfolio without resetting the Proof Clock.

For an unattended proof on this workstation, link and enable the checked-in user service, then
enable user lingering so it starts during boot rather than waiting for an interactive login:

```bash
systemctl --user link "$PWD/systemd/netgrowth-paper-proof.service"
systemctl --user enable --now netgrowth-paper-proof.service
loginctl enable-linger "$USER"
```

The service executes the already synchronized locked environment directly. Paper operation retries
public-data availability failures in-process for up to 30 attempts at two-second intervals so a
brief outage does not pay model startup latency. The workflow archives an irreconstructible gap and
restarts flat when a fresh observation is available; a gap it cannot recover exits immediately.
Persistent failures return to systemd, which restarts after 15 seconds and stops after three failures
in ten minutes rather than entering an unbounded loop.

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
While proof is active, `evidence-state.json`, `paper-session.json`, and immutable checkpoints under
`paper-models/` provide repository-anchored, atomic operational recovery independent of
artifact-run selection.
The bounded active artifact directory is frozen into a
content-addressed paper artifact when proof passes or fails.

## Verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```
