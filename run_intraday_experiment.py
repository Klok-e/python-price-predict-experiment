from __future__ import annotations

import argparse
import os
import random

import numpy as np

from utils.experiment import file_fingerprint, stable_config_hash
from utils.intraday_data import (
    DEFAULT_INTRADAY_START_DATE,
    DEFAULT_INTRADAY_TICKERS,
    build_intraday_datasets,
)
from utils.intraday_policy import (
    CrossValidationConfig,
    IntradayConfig,
    float_grid,
    int_grid,
    write_cross_validation_artifacts,
    run_walk_forward_intraday,
    write_intraday_artifacts,
)
from utils.util import filter_tickers_by_start_date, load_cached_ohlc_data, parse_tickers


INTRADAY_CODE_FINGERPRINT_PATHS = (
    "run_intraday_experiment.py",
    "utils/intraday_data.py",
    "utils/intraday_policy.py",
)


def parse_model_families(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run learned long-only 1m intraday policy backtest.")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--computed-data-dir", default="computed-data")
    parser.add_argument("--tickers", default=",".join(DEFAULT_INTRADAY_TICKERS))
    parser.add_argument("--start-date", default=DEFAULT_INTRADAY_START_DATE)
    parser.add_argument("--end-date")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--evaluation-days", type=int, default=14)
    parser.add_argument("--stride-days", type=int, default=14)
    parser.add_argument("--horizon-grid", default="1,5,15,30,60")
    parser.add_argument("--max-hold-grid", default="1,5,15,30,60")
    parser.add_argument("--threshold-quantiles", default="0.5,0.75,0.9,0.95")
    parser.add_argument("--model-families", default="ridge,hist_gradient_boosting")
    parser.add_argument("--no-futures-metrics", action="store_true")
    parser.add_argument("--no-premium-index", action="store_true")
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--risk-unit", type=float, default=0.25)
    parser.add_argument("--risk-unit-grid")
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--max-drawdown", type=float, default=0.05)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--selection-trade-floor", type=int)
    parser.add_argument("--selection-activity-weight", type=float, default=0.0)
    parser.add_argument("--validation-slices", type=int, default=1)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--cv-windows-per-fold", type=int, default=3)
    parser.add_argument("--min-cv-fold-trades", type=int, default=10)
    parser.add_argument("--config-only-run-id", action="store_true")
    return parser.parse_args(argv)


def build_config(args):
    return IntradayConfig(
        tickers=parse_tickers(args.tickers),
        start_date=args.start_date,
        end_date=args.end_date,
        train_days=args.train_days,
        validation_days=args.validation_days,
        evaluation_days=args.evaluation_days,
        stride_days=args.stride_days,
        horizon_grid=int_grid(args.horizon_grid),
        max_hold_grid=int_grid(args.max_hold_grid),
        threshold_quantiles=float_grid(args.threshold_quantiles),
        model_families=parse_model_families(args.model_families),
        include_futures_metrics=not args.no_futures_metrics,
        include_premium_index=not args.no_premium_index,
        commission=args.commission,
        slippage=args.slippage,
        risk_unit=args.risk_unit,
        risk_unit_grid=float_grid(args.risk_unit_grid) if args.risk_unit_grid else (args.risk_unit,),
        max_exposure=args.max_exposure,
        max_drawdown=args.max_drawdown,
        min_validation_trades=args.min_validation_trades,
        selection_trade_floor=args.selection_trade_floor or args.min_validation_trades,
        selection_activity_weight=args.selection_activity_weight,
        validation_slices=args.validation_slices,
        cash=args.cash,
        seed=args.seed,
    )


def build_cv_config(args):
    return CrossValidationConfig(
        folds=args.cv_folds,
        windows_per_fold=args.cv_windows_per_fold,
        min_fold_trades=args.min_cv_fold_trades,
    )


def build_run_id(config, cv_config, config_only: bool = False):
    payload = {"intraday": config, "cross_validation": cv_config} if cv_config.folds > 0 else config
    if config_only:
        return stable_config_hash(payload)
    return stable_config_hash({
        "config": payload,
        "code_fingerprint": file_fingerprint(INTRADAY_CODE_FINGERPRINT_PATHS),
    })


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = build_config(args)
    cv_config = build_cv_config(args)
    run_id = build_run_id(config, cv_config, config_only=args.config_only_run_id)
    run_dir = os.path.join(args.computed_data_dir, "runs", run_id)
    raw_tickers = filter_tickers_by_start_date(
        load_cached_ohlc_data(args.data_dir, tickers=config.tickers),
        config.start_date,
    )
    datasets = build_intraday_datasets(
        raw_tickers,
        data_dir=args.data_dir,
        start_date=config.start_date,
        end_date=config.end_date,
        include_futures_metrics=config.include_futures_metrics,
        include_premium_index=config.include_premium_index,
    )
    if cv_config.folds > 0:
        report = write_cross_validation_artifacts(datasets, config, cv_config, run_id, run_dir)
        summary = report["summary"]
        print(f"intraday_cv_report run_id={run_id}")
        print(f"summary={report['artifact_paths']['report']}")
        print(
            "intraday_cv: "
            f"folds={summary['fold_count']}, "
            f"median_return={summary['median_cumulative_return']:.4f}, "
            f"median_buy_hold={summary['median_buy_hold_return']:.4f}, "
            f"median_excess={summary['median_excess_return']:.4f}, "
            f"median_sharpe={summary['median_sharpe']:.4f}, "
            f"max_fold_drawdown={summary['max_fold_drawdown']:.4f}, "
            f"pooled_excess={summary['pooled']['excess_return']:.4f}, "
            f"passed_cv={summary['passed_cv']}"
        )
        return

    result = run_walk_forward_intraday(datasets, config)
    report = write_intraday_artifacts(config, run_id, result, run_dir)
    aggregate = report["aggregate"]
    print(f"intraday_report run_id={run_id}")
    print(f"summary={report['artifact_paths']['report']}")
    print(
        "intraday: "
        f"return={aggregate['cumulative_return']:.4f}, "
        f"buy_hold={aggregate['buy_hold']['cumulative_return']:.4f}, "
        f"excess={aggregate['excess_return']:.4f}, "
        f"sharpe={aggregate['sharpe']:.4f}, "
        f"max_drawdown={aggregate['max_drawdown']:.4f}, "
        f"trades_per_day={aggregate['trades_per_day']:.2f}, "
        f"beats_buy_hold={aggregate['beats_buy_hold']}, "
        f"passed={aggregate['passed']}"
    )


if __name__ == "__main__":
    main()
