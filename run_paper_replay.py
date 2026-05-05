from __future__ import annotations

import argparse
import os
import random

import numpy as np

from utils.experiment import stable_config_hash
from utils.experiment_runner import DEFAULT_CASH
from utils.rank_data import build_rank_datasets
from utils.paper_replay import PaperReplayConfig, run_historical_paper_replay
from utils.util import (
    DEFAULT_EXPERIMENT_START_DATE,
    DEFAULT_TICKERS,
    filter_tickers_by_start_date,
    load_cached_ohlc_data,
    parse_tickers,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay the rank-signal paper policy over historical bars.")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--computed-data-dir", default="computed-data")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--start-date", default=DEFAULT_EXPERIMENT_START_DATE)
    parser.add_argument("--end-date")
    parser.add_argument("--bar-size", default="4h")
    parser.add_argument("--prediction-horizon-bars", type=int, default=24)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--selection-cadence-days", type=int, default=30)
    parser.add_argument("--rebalance-cadence-bars", type=int, default=1)
    parser.add_argument(
        "--selector-policy",
        choices=(
            "full-validation",
            "positive-return",
            "split-validation",
            "split-positive-return",
            "market-regime",
            "guarded-market-regime",
        ),
        default="market-regime",
    )
    parser.add_argument("--model-family", choices=("ridge", "hist_gradient_boosting", "both"), default="both")
    parser.add_argument("--include-futures-metrics", action="store_true")
    parser.add_argument("--include-premium-index", action="store_true")
    parser.add_argument("--long-count-grid", default="1,2,3")
    parser.add_argument("--short-count-grid", default="0,1,2")
    parser.add_argument("--rebalance-bars-grid", default="1,2,3,6")
    parser.add_argument("--long-threshold-grid", default="-0.02,-0.01,0.00,0.01")
    parser.add_argument("--short-threshold-grid", default="-0.03,-0.02,-0.01,0.00")
    parser.add_argument("--long-leverage-grid", default="1.0,1.25,1.5")
    parser.add_argument("--short-leverage-grid", default="0.0,0.5,1.0")
    parser.add_argument("--min-validation-trades", type=int, default=30)
    parser.add_argument("--min-replay-trades", type=int, default=30)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_config(args):
    return PaperReplayConfig(
        tickers=parse_tickers(args.tickers),
        start_date=args.start_date,
        end_date=args.end_date,
        bar_size=args.bar_size,
        prediction_horizon_bars=args.prediction_horizon_bars,
        validation_days=args.validation_days,
        min_train_days=args.min_train_days,
        selection_cadence_days=args.selection_cadence_days,
        rebalance_cadence_bars=args.rebalance_cadence_bars,
        selector_policy=args.selector_policy,
        model_family=args.model_family,
        include_futures_metrics=args.include_futures_metrics,
        include_premium_index=args.include_premium_index,
        long_count_grid=args.long_count_grid,
        short_count_grid=args.short_count_grid,
        rebalance_bars_grid=args.rebalance_bars_grid,
        long_threshold_grid=args.long_threshold_grid,
        short_threshold_grid=args.short_threshold_grid,
        long_leverage_grid=args.long_leverage_grid,
        short_leverage_grid=args.short_leverage_grid,
        min_validation_trades=args.min_validation_trades,
        min_replay_trades=args.min_replay_trades,
        commission=args.commission,
        cash=args.cash,
        seed=args.seed,
    )


def main(argv=None):
    args = parse_args(argv)
    _set_seed(args.seed)
    config = build_config(args)
    run_id = stable_config_hash(config)
    run_dir = f"{args.computed_data_dir}/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    raw_tickers = filter_tickers_by_start_date(
        load_cached_ohlc_data(args.data_dir, tickers=config.tickers),
        config.start_date,
    )
    datasets = build_rank_datasets(
        raw_tickers,
        config.bar_size,
        config.prediction_horizon_bars,
        data_dir=args.data_dir,
        include_futures_metrics=config.include_futures_metrics,
        include_premium_index=config.include_premium_index,
    )
    report = run_historical_paper_replay(raw_tickers, datasets, config.tickers, config, run_dir, run_id)

    print(f"historical_paper_replay_report run_id={run_id}")
    print(f"summary={report['artifact_paths']['report']}")
    aggregate = report["replay"]["aggregate"]
    print(
        "replay: "
        f"model_return={aggregate['model']['cumulative_return']:.4f}, "
        f"buy_hold_return={aggregate['buy_and_hold']['cumulative_return']:.4f}, "
        f"trades={aggregate['model']['trades']}, "
        f"passed={report['replay']['success_gate']['passed']}"
    )


if __name__ == "__main__":
    main()
