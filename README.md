# Python Price Prediction Experiment

Operational cryptocurrency price-prediction workflow for the current winner:

`computed-data/runs/f1ab217c947e/rank_signal_report.json`

The winning rank policy is the market-regime return-rank selector. It trains both `ridge` and
`hist_gradient_boosting`, then uses validation-period market-regime evidence to choose the active
model family and allocation parameters before evaluation or paper trading.

The current historical paper replay policy is `guarded-market-regime`: it starts from the
market-regime selector, then applies validation-only regime guards before entering the next
evaluation window.

## Workflow

1. Download spot OHLC data:

```bash
.venv/bin/python download_data.py \
  --tickers NEARUSDT,SOLUSDT,ETHUSDT,BNBUSDT \
  --start-date 2023-04-01
```

2. Download futures metrics:

```bash
.venv/bin/python download_futures_metrics_direct.py \
  --tickers NEARUSDT,SOLUSDT,ETHUSDT,BNBUSDT \
  --start-date 2023-04-01 \
  --end-date 2026-04-30
```

3. Download premium index klines:

```bash
.venv/bin/python download_premium_index_klines_direct.py \
  --tickers NEARUSDT,SOLUSDT,ETHUSDT,BNBUSDT \
  --start-date 2025-01-01 \
  --end-date 2026-04-30
```

4. Train and evaluate the market-regime rank selector:

```bash
.venv/bin/python run_rank_signal_experiment.py \
  --model-family both \
  --selection-mode market-regime \
  --include-futures-metrics \
  --include-premium-index
```

The primary objective in this report is the final holdout window: beat buy-and-hold over the last
`--test-days` days, with training and validation ending before that window starts. Older rolling
windows are robustness context, not the primary success target.

5. Run historical paper replay:

```bash
.venv/bin/python run_paper_replay.py \
  --selector-policy guarded-market-regime \
  --model-family both \
  --include-futures-metrics \
  --include-premium-index
```

6. Run online paper trading dry-run:

```bash
.venv/bin/python run_paper_forward.py --dry-run
```

7. Append one online paper trading cycle:

```bash
.venv/bin/python run_paper_forward.py --append
```

8. Run the online paper trading daemon:

```bash
.venv/bin/python run_paper_forward.py --daemon
```

9. Inspect online paper state:

```bash
.venv/bin/python run_paper_forward.py --state
```

Forward paper trading is a live simulation. It uses only fully closed signal bars, appends pending
paper orders, fills them when next-open data is available, and reduces state from cash and units.
Appending the same signal timestamp twice is idempotent; the second append writes zero records.

## Artifacts

- Market data cache: `computed-data/dataset/`
- Winning run: `computed-data/runs/f1ab217c947e/`
- Historical paper replay winner: `computed-data/runs/434a6f82d8ce/`
- 90-day replay window report:
  `computed-data/runs/434a6f82d8ce/paper_replay_90d_window_report.json`
- Rank reports: `computed-data/runs/<run_id>/rank_signal_report.json`
- Historical replay reports: `computed-data/runs/<run_id>/paper_replay_report.json`
- Online paper ledger: `computed-data/runs/<policy_id>/paper_ledger.jsonl`

## Tests

```bash
.venv/bin/python -m pytest -q
```
