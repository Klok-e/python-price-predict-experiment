import argparse
import json
import os
import random

import numpy as np
import torch

from models import PricePredictorModel
from train_supervised import train_supervised_model
from utils.experiment import ExperimentConfig, TradingContract, stable_config_hash
from utils.experiment_runner import (
    DEFAULT_CASH,
    backtest_buy_and_hold_ticker,
    backtest_model_ticker,
    backtest_no_trade_ticker,
    git_state,
    jsonable,
    parse_linear_arch,
    raw_data_stats,
    save_backtest_artifacts,
    split_stats,
    summarize_backtest_results,
)
from utils.util import DEFAULT_TICKERS, filter_tickers_by_start_date, load_cached_ohlc_data, prepare_supervised_splits


DEFAULT_EXPERIMENT_START_DATE = "2023-04-01"


def parse_tickers(value):
    return tuple(ticker.strip().upper() for ticker in value.split(",") if ticker.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a real-data price-prediction experiment.")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--computed-data-dir", default="computed-data")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--start-date", default=DEFAULT_EXPERIMENT_START_DATE)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--backtest-days", type=int, default=14)
    parser.add_argument("--backtest-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--lookahead", type=int, default=256)
    parser.add_argument("--stop-loss", type=float, default=0.4)
    parser.add_argument("--take-profit", type=float, default=0.4)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--min-probability-edge", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--linear-arch", default="128,64")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sampler-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path")
    return parser.parse_args(argv)


def build_contract(args):
    return TradingContract(
        lookahead_steps=args.lookahead,
        stop_loss_percent=args.stop_loss,
        take_profit_percent=args.take_profit,
        commission=args.commission,
        min_probability_edge=args.min_probability_edge,
    )


def build_model_kwargs(args):
    return {"linear_arch": parse_linear_arch(args.linear_arch)}


def build_run_config(args, contract, model_kwargs):
    tickers = parse_tickers(args.tickers)
    return {
        "experiment": ExperimentConfig(
            tickers=tickers,
            contract=contract,
            seed=args.seed,
        ),
        "validation_days": args.validation_days,
        "test_days": args.test_days,
        "start_date": args.start_date,
        "backtest_days": args.backtest_days,
        "backtest_split": args.backtest_split,
        "window_size": args.window_size,
        "stride": args.stride,
        "model_kwargs": model_kwargs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "sampler_fraction": args.sampler_fraction,
        "cash": args.cash,
    }


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_model_artifact(path):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "model_kwargs", "feature_size", "window_size", "contract", "run_id"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model artifact {path} is missing keys: {sorted(missing)}")
    contract = TradingContract(**artifact["contract"])
    model = PricePredictorModel(
        feature_size=artifact["feature_size"],
        window_size=artifact["window_size"],
        **artifact["model_kwargs"],
    )
    model.load_state_dict(artifact["state_dict"])
    return model, contract, artifact


def _save_model_artifact(path, model, model_kwargs, feature_size, window_size, contract, seed, run_id, training_summary):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "PricePredictorModel",
            "model_kwargs": model_kwargs,
            "feature_size": feature_size,
            "window_size": window_size,
            "contract": jsonable(contract),
            "seed": seed,
            "run_id": run_id,
            "training_summary": jsonable(training_summary),
        },
        path,
    )


def _select_split(split_name, validation_split, test_split):
    return validation_split if split_name == "validation" else test_split


def _run_strategy_suite(split, model, contract, window_size, threshold, backtest_days, cash):
    model_results = [
        backtest_model_ticker(ticker, model, contract, window_size, threshold, backtest_days, cash=cash)
        for ticker in split
    ]
    buy_and_hold_results = [
        backtest_buy_and_hold_ticker(ticker, window_size, backtest_days, contract, cash=cash)
        for ticker in split
    ]
    no_trade_results = [
        backtest_no_trade_ticker(ticker, window_size, backtest_days, cash=cash)
        for ticker in split
    ]
    return {
        "model": model_results,
        "buy_and_hold": buy_and_hold_results,
        "no_trade": no_trade_results,
    }


def _save_strategy_artifacts(strategy_results, output_dir):
    backtest_dir = f"{output_dir}/backtest-results"
    for strategy_name, results in strategy_results.items():
        save_backtest_artifacts(results, backtest_dir, strategy_name)


def main(argv=None):
    args = parse_args(argv)
    _set_seed(args.seed)
    tickers = parse_tickers(args.tickers)

    loaded_artifact = None
    if args.model_path:
        model, contract, loaded_artifact = _load_model_artifact(args.model_path)
        model_kwargs = loaded_artifact["model_kwargs"]
        window_size = loaded_artifact["window_size"]
        run_config = build_run_config(args, contract, model_kwargs)
        run_config["loaded_model_path"] = os.path.abspath(args.model_path)
        run_config["window_size"] = window_size
    else:
        contract = build_contract(args)
        model_kwargs = build_model_kwargs(args)
        window_size = args.window_size
        run_config = build_run_config(args, contract, model_kwargs)

    run_id = stable_config_hash(run_config)
    run_dir = f"{args.computed_data_dir}/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    raw_tickers = filter_tickers_by_start_date(
        load_cached_ohlc_data(args.data_dir, tickers=tickers),
        args.start_date,
    )
    train_split, validation_split, test_split = prepare_supervised_splits(
        raw_tickers,
        contract=contract,
        validation_days=args.validation_days,
        test_days=args.test_days,
    )
    selected_split = _select_split(args.backtest_split, validation_split, test_split)
    feature_size = train_split[0][0].shape[1]

    threshold = args.confidence_threshold
    if threshold is None:
        threshold = contract.entry_probability_threshold()

    training_summary = None
    model_path = args.model_path
    if loaded_artifact is not None:
        if loaded_artifact["feature_size"] != feature_size:
            raise ValueError(
                f"Model feature_size={loaded_artifact['feature_size']} does not match data feature_size={feature_size}"
            )
    else:
        model_name = f"mlp_{run_id}_ws{window_size}"
        model_path = f"{run_dir}/model.pt"
        model, training_summary = train_supervised_model(
            PricePredictorModel,
            model_kwargs,
            train_split,
            validation_split,
            window_size,
            args.computed_data_dir,
            model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            log_interval=1,
            save_model=False,
            continue_training=False,
            test_prediction_threshold=threshold,
            window_stride=args.stride,
            random_sampler_samples_percent=args.sampler_fraction,
            checkpoint_path=model_path,
            return_training_history=True,
        )
        _save_model_artifact(
            model_path,
            model,
            model_kwargs,
            feature_size,
            window_size,
            contract,
            args.seed,
            run_id,
            training_summary,
        )

    strategy_results = _run_strategy_suite(
        selected_split,
        model,
        contract,
        window_size,
        threshold,
        args.backtest_days,
        cash=args.cash,
    )
    _save_strategy_artifacts(strategy_results, run_dir)

    summaries = {
        strategy_name: summarize_backtest_results(results)
        for strategy_name, results in strategy_results.items()
    }
    report_type = "tail_validation_report" if args.backtest_split == "validation" else "tail_evaluation_report"
    summary = {
        "run_id": run_id,
        "report_type": report_type,
        "model_path": os.path.abspath(model_path),
        "loaded_model_run_id": loaded_artifact["run_id"] if loaded_artifact else None,
        "config": run_config,
        "threshold": threshold,
        "threshold_source": "explicit" if args.confidence_threshold is not None else "expected_value_uncalibrated",
        "data": raw_data_stats(raw_tickers),
        "splits": {
            "train": split_stats(train_split),
            "validation": split_stats(validation_split),
            "test": split_stats(test_split),
        },
        "selected_backtest_split": args.backtest_split,
        "training": training_summary,
        "strategies": summaries,
        "git": git_state(),
    }

    summary_path = f"{run_dir}/summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(jsonable(summary), file, indent=2, sort_keys=True)

    print(f"{report_type} run_id={run_id}")
    print(f"summary={summary_path}")
    for strategy_name, strategy_summary in summaries.items():
        metrics = strategy_summary["portfolio"]
        print(
            f"{strategy_name}: "
            f"cumulative_return={metrics['cumulative_return']:.4f}, "
            f"trades={metrics['trades']}, "
            f"sharpe_ratio={metrics['sharpe_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
