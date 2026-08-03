# Python Price Prediction Experiment

Learned one-minute cryptocurrency day-trading experiment.

The operational workflow is a long-only intraday backtest. It scores every closed 1m bar, looks for
small after-cost opportunities on liquid majors, and uses walk-forward validation to select model
family, opportunity horizon, trade threshold, and max hold.

There is no online paper trading path and no shorting path.

## Workflow

1. Download spot OHLC data:

```bash
.venv/bin/python download_data.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start-date 2025-01-01
```

2. Download futures metrics:

```bash
.venv/bin/python download_futures_metrics_direct.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start-date 2025-01-01 \
  --end-date 2026-04-30
```

3. Download premium index klines:

```bash
.venv/bin/python download_premium_index_klines_direct.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start-date 2025-01-01 \
  --end-date 2026-04-30
```

4. Run learned intraday walk-forward backtest:

```bash
.venv/bin/python run_intraday_experiment.py
```

Use rolling backtest cross-validation when you want several recent chronological folds instead of
one walk-forward run:

```bash
.venv/bin/python run_intraday_experiment.py \
  --cv-folds 3 \
  --cv-windows-per-fold 3 \
  --min-cv-fold-trades 10
```

Defaults:

- Tickers: `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT`
- Windows: 60d train, 14d validation, 14d evaluation, 14d stride
- Cross-validation: off by default; `--cv-folds` enables rolling recent folds with continuous
  out-of-sample evaluation coverage
- Horizon/max-hold candidates: 1, 5, 15, 30, 60 minutes
- Model families: `ridge`, `ridge_shared`, `market_ridge`, `huber`, `logistic_positive`,
  `hist_gradient_boosting`, optional `ridge_alpha_<N>` and `ridge_decay_<days>` variants, and
  optional `_market_excess`, `_market_return`, `_path_mean`, `_vol_scaled`,
  `_path_mean_vol_scaled`, or `_market_excess_vol_scaled` target variants
- Selection: active positive after-cost candidates ranked by validation excess over buy-and-hold
  with validation-selected entry/exit thresholds plus optional selection trade floor/activity weight
  and optional validation-slice stability ranking
- CV proof: positive Sharpe, drawdown control, fold activity, and buy-and-hold excess-return gates
- Execution: long-only, 25% equity per active ticker by default, optional risk-unit grid, 100% max exposure
- Costs: commission plus slippage
- Diagnostics: selected windows include validation-to-evaluation drift for excess return, Sharpe,
  drawdown, and trade count

## Artifacts

- Market data cache: `computed-data/dataset/`
- Intraday reports: `computed-data/runs/<run_id>/intraday_report.json`
- Intraday positions: `computed-data/runs/<run_id>/intraday_positions.csv`
- Intraday buy-and-hold positions: `computed-data/runs/<run_id>/intraday_buy_hold_positions.csv`
- Intraday ledger: `computed-data/runs/<run_id>/intraday_ledger.jsonl`
- Intraday decision timeline: `computed-data/runs/<run_id>/intraday_decision_timeline.html`
- Intraday CV reports: `computed-data/runs/<run_id>/intraday_cv_report.json`
- Intraday CV fold artifacts: `computed-data/runs/<run_id>/cv_folds/fold_<NN>/`
- Intraday CV pooled equity: `computed-data/runs/<run_id>/intraday_cv_pooled_positions.csv`

Run ids include the intraday config and a fingerprint of the intraday runner, data builder, and
policy code. Use `--config-only-run-id` only when intentionally comparing against legacy
config-hash artifacts.

## Tests

```bash
.venv/bin/python -m pytest -q
```
