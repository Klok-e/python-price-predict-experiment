import argparse
import random

import numpy as np
import torch

from models import PricePredictorModel
from train_supervised import train_supervised_model
from utils.experiment import ExperimentConfig, TradingContract, stable_config_hash
from utils.util import (
    create_random_walk_ohlc_data,
    create_synthetic_ohlc_data,
    prepare_supervised_splits,
)


def summarize_split(name, split):
    rows = sum(len(ticker[0]) for ticker in split)
    positives = sum(float(ticker[2]["Label"].sum()) for ticker in split)
    total_labels = sum(len(ticker[2]) for ticker in split)
    positive_rate = positives / total_labels if total_labels else 0
    print(f"{name}: rows={rows}, labels={total_labels}, positive_rate={positive_rate:.4f}")


def parse_tickers(value):
    return tuple(ticker.strip().upper() for ticker in value.split(",") if ticker.strip())


def main():
    parser = argparse.ArgumentParser(description="Run the price-prediction experiment without Jupyter.")
    parser.add_argument("--source", choices=["synthetic", "random-walk"], default="synthetic")
    parser.add_argument("--tickers", default="SYNTH1USDT")
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--validation-days", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=3)
    parser.add_argument("--lookahead", type=int, default=256)
    parser.add_argument("--stop-loss", type=float, default=0.4)
    parser.add_argument("--take-profit", type=float, default=0.4)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sampler-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--computed-data-dir", default="computed-data/local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tickers = parse_tickers(args.tickers)
    contract = TradingContract(
        lookahead_steps=args.lookahead,
        stop_loss_percent=args.stop_loss,
        take_profit_percent=args.take_profit,
        commission=args.commission,
    )
    config = ExperimentConfig(tickers=tickers, contract=contract, seed=args.seed)
    run_id = stable_config_hash(config)

    if args.source == "synthetic":
        raw_tickers = create_synthetic_ohlc_data(days=args.days, tickers=tickers)
    else:
        raw_tickers = create_random_walk_ohlc_data(days=args.days, tickers=tickers)

    train_split, validation_split, test_split = prepare_supervised_splits(
        raw_tickers,
        contract=contract,
        validation_days=args.validation_days,
        test_days=args.test_days,
    )

    print(f"run_id={run_id}")
    print(f"entry_probability_threshold={contract.entry_probability_threshold():.4f}")
    summarize_split("train", train_split)
    summarize_split("validation", validation_split)
    summarize_split("test", test_split)

    if args.no_train:
        return

    model_kwargs = {"linear_arch": [128, 64]}
    model_name = f"local_mlp_{run_id}_ws{args.window_size}"
    train_supervised_model(
        PricePredictorModel,
        model_kwargs,
        train_split,
        validation_split,
        args.window_size,
        args.computed_data_dir,
        model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_interval=1,
        save_model=args.save_model,
        continue_training=False,
        test_prediction_threshold=contract.entry_probability_threshold(),
        window_stride=args.stride,
        random_sampler_samples_percent=args.sampler_fraction,
    )


if __name__ == "__main__":
    main()
