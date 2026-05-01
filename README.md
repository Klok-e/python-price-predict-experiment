# Python Price Prediction Experiment

Cryptocurrency price-prediction experiment using Binance-style OHLC bars, causal technical features, PyTorch supervised models, and backtests.

## Local Smoke Run

Run the data, feature, first-hit label, and strict split pipeline without Jupyter or network access:

```bash
python3 run_local.py --no-train
```

Run a small one-epoch MLP training pass on synthetic data:

```bash
python3 run_local.py --epochs 1
```

Run tests:

```bash
python3 -m pytest -q
```

## Real Data Runs

Populate the Binance cache explicitly:

```bash
python3 download_data.py --tickers SOLUSDT
```

Run a tail validation report from cached data:

```bash
python3 run_experiment.py --tickers SOLUSDT --epochs 1 --backtest-days 1
```

`run_experiment.py` does not download market data. It fails if the requested cached ticker data is missing.
It defaults to `--start-date 2023-04-01`, the first continuous window found in the current default ticker cache.

## Current Experiment Contract

The supervised target is a conservative barrier label: after a signal on candle `t`, enter at candle `t+1` open, then label the sample positive only if take-profit is reached before stop-loss inside the lookahead horizon. Same-candle TP/SL hits, stop-loss-first hits, and no-hit outcomes are negative.

Training uses chronological train/validation/test splits. Scalers are fitted on training data only, then reused for validation and test.
