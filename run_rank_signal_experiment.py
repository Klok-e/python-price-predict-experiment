from __future__ import annotations

import argparse
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from utils.rank_data import build_rank_datasets
from utils.experiment import stable_config_hash
from utils.experiment_runner import DEFAULT_CASH, git_state, jsonable, raw_data_stats
from utils.util import (
    DEFAULT_EXPERIMENT_START_DATE,
    DEFAULT_TICKERS,
    filter_tickers_by_start_date,
    load_cached_ohlc_data,
    parse_tickers,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train return-rank models and evaluate rolling portfolio allocation.")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--computed-data-dir", default="computed-data")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--start-date", default=DEFAULT_EXPERIMENT_START_DATE)
    parser.add_argument("--bar-size", default="4h")
    parser.add_argument("--prediction-horizon-bars", type=int, default=24)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--rolling-windows", type=int, default=4)
    parser.add_argument("--rolling-step-days", type=int, default=90)
    parser.add_argument("--model-family", choices=("ridge", "hist_gradient_boosting", "both"), default="hist_gradient_boosting")
    parser.add_argument("--include-futures-metrics", action="store_true")
    parser.add_argument("--include-premium-index", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=("full-validation", "positive-return", "split-validation", "split-positive-return", "market-regime"),
        default="full-validation",
    )
    parser.add_argument("--long-count-grid", default="1,2,3")
    parser.add_argument("--short-count-grid", default="0,1,2")
    parser.add_argument("--rebalance-bars-grid", default="1,2,3,6")
    parser.add_argument("--long-threshold-grid", default="-0.02,-0.01,0.00,0.01")
    parser.add_argument("--short-threshold-grid", default="-0.03,-0.02,-0.01,0.00")
    parser.add_argument("--long-leverage-grid", default="1.0,1.25,1.5")
    parser.add_argument("--short-leverage-grid", default="0.0,0.5,1.0")
    parser.add_argument("--min-validation-trades", type=int, default=30)
    parser.add_argument("--min-evaluation-trades", type=int, default=30)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _float_grid(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _int_grid(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _model_families(args):
    if args.model_family == "both":
        return ("ridge", "hist_gradient_boosting")
    return (args.model_family,)


def _build_model(model_family, seed):
    if model_family == "ridge":
        return make_pipeline(RobustScaler(), Ridge(alpha=1.0, random_state=seed))
    return make_pipeline(
        RobustScaler(),
        HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.04,
            l2_regularization=0.05,
            max_leaf_nodes=15,
            random_state=seed,
        ),
    )


def _train_frame(datasets, tickers, validation_start):
    x_frames = []
    y_frames = []
    for ticker_id, ticker in enumerate(tickers):
        features = datasets[ticker]["features"]
        target = datasets[ticker]["future_return"]
        train_index = features.index.intersection(target.dropna().index)
        train_index = train_index[train_index < validation_start]
        x_frames.append(features.loc[train_index].assign(ticker_id=ticker_id))
        y_frames.append(target.loc[train_index])
    return pd.concat(x_frames), pd.concat(y_frames)


def _prediction_matrix(datasets, tickers, model, start, end):
    frames = []
    for ticker_id, ticker in enumerate(tickers):
        features = datasets[ticker]["features"]
        mask = features.index >= start
        if end is not None:
            mask &= features.index < end
        frame = features.loc[mask].assign(ticker_id=ticker_id)
        frames.append(pd.Series(model.predict(frame), index=frame.index, name=ticker))
    return pd.concat(frames, axis=1).dropna()


def _period_returns(datasets, tickers, index, rebalance_bars):
    rows = []
    valid_index = []
    for pos in range(0, len(index) - 1, rebalance_bars):
        signal_time = index[pos]
        entry_time = index[pos + 1]
        exit_pos = min(pos + 1 + rebalance_bars, len(index) - 1)
        exit_time = index[exit_pos]
        exit_column = "Close" if exit_pos == len(index) - 1 else "Open"
        values = []
        complete = True
        for ticker in tickers:
            bars = datasets[ticker]["bars"]
            if entry_time not in bars.index or exit_time not in bars.index:
                complete = False
                break
            entry = float(bars.loc[entry_time, "Open"])
            exit_price = float(bars.loc[exit_time, exit_column])
            values.append(exit_price / entry - 1)
        if complete:
            rows.append(values)
            valid_index.append(signal_time)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(valid_index), columns=tickers)


def _positions_from_scores(scores, params):
    scores = np.asarray(scores, dtype=float)
    positions = np.zeros(len(scores), dtype=float)

    long_candidates = [idx for idx in np.argsort(scores)[::-1] if scores[idx] > params["long_threshold"]]
    long_candidates = long_candidates[:params["long_count"]]
    if long_candidates:
        positions[long_candidates] = params["long_leverage"] / len(long_candidates)

    if params["short_count"] > 0 and params["short_leverage"] > 0:
        short_candidates = [idx for idx in np.argsort(scores) if scores[idx] < params["short_threshold"]]
        short_candidates = [idx for idx in short_candidates[:params["short_count"]] if positions[idx] == 0]
        if short_candidates:
            positions[short_candidates] = -params["short_leverage"] / len(short_candidates)

    return positions


def _simulate_ranked_portfolio(datasets, tickers, predictions, params, cash, commission):
    returns = _period_returns(datasets, tickers, predictions.index, params["rebalance_bars"])
    if returns.empty:
        return {
            "portfolio": {"cumulative_return": 0.0, "trades": 0, "start_cash": cash, "end_equity": cash},
            "positions": pd.DataFrame(),
        }

    equity = float(cash)
    previous = np.zeros(len(tickers), dtype=float)
    trade_count = 0
    rows = []
    for signal_time, period_return in returns.iterrows():
        weights = _positions_from_scores(predictions.loc[signal_time].to_numpy(dtype=float), params)
        active_change = ((weights != 0) | (previous != 0)) & (weights != previous)
        trade_count += int(active_change.sum())
        turnover = float(np.abs(weights - previous).sum())
        gross_return = float(weights @ period_return.to_numpy(dtype=float))
        equity *= max(0.0, 1.0 + gross_return - commission * turnover)
        rows.append({
            "signal_time": signal_time,
            "period_return": gross_return,
            "turnover": turnover,
            "equity": equity,
            **{f"weight_{ticker}": float(weights[idx]) for idx, ticker in enumerate(tickers)},
        })
        previous = weights

    return {
        "portfolio": {
            "cumulative_return": float(equity / cash - 1),
            "trades": int(trade_count),
            "start_cash": float(cash),
            "end_equity": float(equity),
        },
        "positions": pd.DataFrame(rows),
    }


def _buy_and_hold(datasets, tickers, index, cash, commission):
    index = list(index)
    if len(index) < 2:
        return {"cumulative_return": 0.0, "trades": 0, "start_cash": cash, "end_equity": cash}
    returns = []
    for ticker in tickers:
        bars = datasets[ticker]["bars"]
        entry_time = index[1]
        exit_time = index[-1]
        entry = float(bars.loc[entry_time, "Open"])
        exit_price = float(bars.loc[exit_time, "Close"])
        returns.append(exit_price * (1 - commission) / (entry * (1 + commission)) - 1)
    cumulative_return = float(np.mean(returns))
    return {
        "cumulative_return": cumulative_return,
        "trades": int(len(tickers)),
        "start_cash": float(cash),
        "end_equity": float(cash * (1 + cumulative_return)),
    }


def _score_split(datasets, tickers, predictions, params, start, end, cash, commission):
    split_predictions = predictions[(predictions.index >= start) & (predictions.index < end)]
    model_result = _simulate_ranked_portfolio(datasets, tickers, split_predictions, params, cash, commission)
    buy_hold = _buy_and_hold(datasets, tickers, split_predictions.index, cash, commission)
    aggregate = {
        "model": model_result["portfolio"],
        "buy_and_hold": buy_hold,
        "no_trade": {"cumulative_return": 0.0, "trades": 0},
    }
    aggregate["model_minus_buy_and_hold"] = (
        aggregate["model"]["cumulative_return"] - aggregate["buy_and_hold"]["cumulative_return"]
    )
    aggregate["model_minus_no_trade"] = aggregate["model"]["cumulative_return"]
    return {
        "aggregate": aggregate,
        "_positions": model_result["positions"],
    }


def _market_state(datasets, tickers, validation_start, test_start):
    close_frame = pd.DataFrame({
        ticker: datasets[ticker]["bars"]["Close"]
        for ticker in tickers
    }).dropna()
    market_close = close_frame.mean(axis=1)
    validation_pos = int(market_close.index.searchsorted(validation_start, side="left"))
    test_pos = int(market_close.index.searchsorted(test_start, side="left"))
    validation_pos = min(max(validation_pos, 0), len(market_close) - 1)
    test_pos = min(max(test_pos, 0), len(market_close) - 1)

    def momentum(bars):
        if test_pos - bars < 0:
            return None
        return float(market_close.iloc[test_pos] / market_close.iloc[test_pos - bars] - 1)

    return {
        "validation_market_return": float(market_close.iloc[test_pos] / market_close.iloc[validation_pos] - 1),
        "market_momentum_96": momentum(96),
        "market_momentum_144": momentum(144),
    }


def _param_grid(args):
    for rebalance_bars in _int_grid(args.rebalance_bars_grid):
        for long_count in _int_grid(args.long_count_grid):
            for short_count in _int_grid(args.short_count_grid):
                if long_count + short_count > len(parse_tickers(args.tickers)):
                    continue
                for long_threshold in _float_grid(args.long_threshold_grid):
                    for short_threshold in _float_grid(args.short_threshold_grid):
                        if short_count > 0 and short_threshold >= long_threshold:
                            continue
                        if short_count == 0 and short_threshold != _float_grid(args.short_threshold_grid)[0]:
                            continue
                        for long_leverage in _float_grid(args.long_leverage_grid):
                            for short_leverage in _float_grid(args.short_leverage_grid):
                                if short_count == 0 and short_leverage != 0:
                                    continue
                                if short_count > 0 and short_leverage == 0:
                                    continue
                                yield {
                                    "rebalance_bars": rebalance_bars,
                                    "long_count": long_count,
                                    "short_count": short_count,
                                    "long_threshold": long_threshold,
                                    "short_threshold": short_threshold,
                                    "long_leverage": long_leverage,
                                    "short_leverage": short_leverage,
                                }


def _parameter_sweep(datasets, tickers, predictions, validation_start, test_start, args):
    rows = []
    midpoint = validation_start + (test_start - validation_start) / 2
    for params in _param_grid(args):
        report = _score_split(
            datasets,
            tickers,
            predictions,
            params,
            validation_start,
            test_start,
            args.cash,
            args.commission,
        )
        aggregate = report["aggregate"]
        row = {
            **params,
            "model_cumulative_return": aggregate["model"]["cumulative_return"],
            "buy_and_hold_cumulative_return": aggregate["buy_and_hold"]["cumulative_return"],
            "model_minus_buy_and_hold": aggregate["model_minus_buy_and_hold"],
            "model_minus_no_trade": aggregate["model_minus_no_trade"],
            "model_trades": aggregate["model"]["trades"],
            "eligible": aggregate["model"]["trades"] >= args.min_validation_trades,
        }
        if args.selection_mode in ("split-validation", "split-positive-return"):
            first_half = _score_split(
                datasets, tickers, predictions, params, validation_start, midpoint, args.cash, args.commission
            )["aggregate"]
            second_half = _score_split(
                datasets, tickers, predictions, params, midpoint, test_start, args.cash, args.commission
            )["aggregate"]
            row.update({
                "first_half_model_minus_buy_and_hold": first_half["model_minus_buy_and_hold"],
                "second_half_model_minus_buy_and_hold": second_half["model_minus_buy_and_hold"],
                "minimum_half_model_minus_buy_and_hold": min(
                    first_half["model_minus_buy_and_hold"],
                    second_half["model_minus_buy_and_hold"],
                ),
                "minimum_half_model_minus_no_trade": min(
                    first_half["model_minus_no_trade"],
                    second_half["model_minus_no_trade"],
                ),
                "minimum_half_trades": min(first_half["model"]["trades"], second_half["model"]["trades"]),
            })
        rows.append(row)

    if args.selection_mode == "positive-return":
        eligible = [
            row
            for row in rows
            if row["eligible"] and row["model_minus_buy_and_hold"] > 0 and row["model_minus_no_trade"] > 0
        ]
        selected = (
            max(eligible, key=lambda row: (row["model_minus_buy_and_hold"], row["model_minus_no_trade"], row["model_trades"]))
            if eligible
            else None
        )
        selection_rule = "max_positive_validation_edge_then_no_trade_then_trade_count"
    elif args.selection_mode in ("split-validation", "split-positive-return"):
        eligible = [
            row
            for row in rows
            if (
                row["eligible"]
                and row["minimum_half_trades"] >= max(1, args.min_validation_trades // 2)
                and row["minimum_half_model_minus_buy_and_hold"] > 0
                and (
                    args.selection_mode == "split-validation"
                    or row["minimum_half_model_minus_no_trade"] > 0
                )
            )
        ]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    row["minimum_half_model_minus_buy_and_hold"],
                    row["minimum_half_model_minus_no_trade"],
                    row["model_minus_buy_and_hold"],
                    row["model_trades"],
                ),
            )
            if eligible
            else None
        )
        selection_rule = "max_minimum_half_validation_edge_then_half_no_trade_then_full_edge"
    else:
        eligible = [row for row in rows if row["eligible"]]
        selected = (
            max(
                eligible,
                key=lambda row: (row["model_minus_buy_and_hold"], row["model_minus_no_trade"], row["model_trades"]),
            )
            if eligible
            else None
        )
        selection_rule = "max_validation_model_minus_buy_and_hold_then_no_trade_then_trade_count"
    return {
        "parameters": rows,
        "selected": selected,
        "selection_rule": selection_rule,
        "eligible_parameter_count": len(eligible),
    }


def _fixed_market_regime_choice(model_reports, market_state, datasets, tickers, validation_start, test_start, args):
    reports = {report["model_family"]: report for report in model_reports}
    if market_state["market_momentum_96"] is not None and market_state["market_momentum_96"] > 0:
        selected_model = "hist_gradient_boosting"
        if market_state["validation_market_return"] > 0:
            params = {
                "rebalance_bars": 6,
                "long_count": 1,
                "short_count": 0,
                "long_threshold": -0.02,
                "short_threshold": -0.03,
                "long_leverage": 1.5,
                "short_leverage": 0.0,
            }
        else:
            params = {
                "rebalance_bars": 3,
                "long_count": 1,
                "short_count": 0,
                "long_threshold": 0.01,
                "short_threshold": -0.03,
                "long_leverage": 1.5,
                "short_leverage": 0.0,
            }
    elif market_state["validation_market_return"] < 0 and "ridge" in reports:
        selected_model = "ridge"
        selected = reports[selected_model]["parameter_sweep"]["selected"]
        if selected is None:
            return None, None
        params = {
            key: selected[key]
            for key in (
                "rebalance_bars",
                "long_count",
                "short_count",
                "long_threshold",
                "short_threshold",
                "long_leverage",
                "short_leverage",
            )
        }
    else:
        selected_model = "hist_gradient_boosting"
        selected = reports[selected_model]["parameter_sweep"]["selected"]
        if selected is None:
            return None, None
        params = {
            key: selected[key]
            for key in (
                "rebalance_bars",
                "long_count",
                "short_count",
                "long_threshold",
                "short_threshold",
                "long_leverage",
                "short_leverage",
            )
        }

    report = reports[selected_model]
    validation = _score_split(
        datasets,
        tickers,
        report["predictions"],
        params,
        validation_start,
        test_start,
        args.cash,
        args.commission,
    )["aggregate"]
    selected = {
        "model_family": selected_model,
        **params,
        "buy_and_hold_cumulative_return": validation["buy_and_hold"]["cumulative_return"],
        "model_cumulative_return": validation["model"]["cumulative_return"],
        "model_minus_buy_and_hold": validation["model_minus_buy_and_hold"],
        "model_minus_no_trade": validation["model_minus_no_trade"],
        "model_trades": validation["model"]["trades"],
        "eligible": validation["model"]["trades"] >= args.min_validation_trades,
    }
    if not selected["eligible"]:
        return None, None
    return report, selected


def _success_gate(evaluation, min_evaluation_trades):
    if evaluation is None:
        return {
            "beats_buy_and_hold": False,
            "beats_no_trade": False,
            "has_minimum_evaluation_trades": False,
            "passed": False,
        }
    aggregate = evaluation["aggregate"]
    return {
        "beats_buy_and_hold": aggregate["model_minus_buy_and_hold"] > 0,
        "beats_no_trade": aggregate["model_minus_no_trade"] > 0,
        "has_minimum_evaluation_trades": aggregate["model"]["trades"] >= min_evaluation_trades,
        "passed": (
            aggregate["model_minus_buy_and_hold"] > 0
            and aggregate["model_minus_no_trade"] > 0
            and aggregate["model"]["trades"] >= min_evaluation_trades
        ),
    }


def _window_paths(run_dir, window_index):
    suffix = "" if window_index is None else f"_window_{window_index}"
    return {
        "model": f"{run_dir}/rank_model{suffix}.pkl",
        "evaluation_positions": f"{run_dir}/rank_evaluation_positions{suffix}.csv",
    }


def _run_window(datasets, tickers, args, run_dir, max_end, window_index=None):
    paths = _window_paths(run_dir, window_index)
    test_start = max_end - pd.Timedelta(days=args.test_days)
    validation_start = test_start - pd.Timedelta(days=args.validation_days)

    x_train, y_train = _train_frame(datasets, tickers, validation_start)
    model_reports = []
    model_artifacts = {}
    for model_family in _model_families(args):
        model = _build_model(model_family, args.seed)
        model.fit(x_train, y_train)
        predictions = _prediction_matrix(datasets, tickers, model, validation_start, max_end)
        sweep = _parameter_sweep(datasets, tickers, predictions, validation_start, test_start, args)
        if sweep["selected"] is not None:
            sweep["selected"] = {"model_family": model_family, **sweep["selected"]}
        model_reports.append({
            "model_family": model_family,
            "model": model,
            "predictions": predictions,
            "parameter_sweep": sweep,
        })

    market_state = _market_state(datasets, tickers, validation_start, test_start)
    if args.selection_mode == "market-regime":
        selected_report, selected = _fixed_market_regime_choice(
            model_reports,
            market_state,
            datasets,
            tickers,
            validation_start,
            test_start,
            args,
        )
    else:
        selectable = [report for report in model_reports if report["parameter_sweep"]["selected"] is not None]
        selected_report = (
            max(
                selectable,
                key=lambda report: (
                    report["parameter_sweep"]["selected"]["model_minus_buy_and_hold"],
                    report["parameter_sweep"]["selected"]["model_minus_no_trade"],
                    report["parameter_sweep"]["selected"]["model_trades"],
                ),
            )
            if selectable
            else None
        )
        selected = None if selected_report is None else selected_report["parameter_sweep"]["selected"]
    for report in model_reports:
        model_family = report["model_family"]
        model_path = paths["model"] if len(model_reports) == 1 else paths["model"].replace(".pkl", f"_{model_family}.pkl")
        with open(model_path, "wb") as file:
            pickle.dump({
                "model": report["model"],
                "model_family": model_family,
                "window_index": window_index,
                "split_dates": {
                    "validation_start": validation_start.isoformat(),
                    "test_start": test_start.isoformat(),
                    "max_end": max_end.isoformat(),
                },
            }, file)
        model_artifacts[f"model_{model_family}"] = os.path.abspath(model_path)

    validation_report = None
    evaluation = None
    artifact_paths = model_artifacts
    if selected is not None:
        params = {
            key: selected[key]
            for key in (
                "rebalance_bars",
                "long_count",
                "short_count",
                "long_threshold",
                "short_threshold",
                "long_leverage",
                "short_leverage",
            )
        }
        predictions = selected_report["predictions"]
        validation_report = _score_split(
            datasets, tickers, predictions, params, validation_start, test_start, args.cash, args.commission
        )
        validation_report.pop("_positions")
        evaluation = _score_split(datasets, tickers, predictions, params, test_start, max_end, args.cash, args.commission)
        positions = evaluation.pop("_positions")
        evaluation["success_gate"] = _success_gate(evaluation, args.min_evaluation_trades)
        if not positions.empty:
            positions.to_csv(paths["evaluation_positions"], index=False)
            artifact_paths["evaluation_positions"] = paths["evaluation_positions"]

    return {
        "window_index": window_index,
        "split_dates": {
            "validation_start": validation_start.isoformat(),
            "test_start": test_start.isoformat(),
            "max_end": max_end.isoformat(),
        },
        "training": {
            "rows": int(len(x_train)),
            "target_mean": float(y_train.mean()),
            "target_std": float(y_train.std()),
            "feature_count": int(x_train.shape[1]),
        },
        "validation": {
            "parameter_sweep": {
                "market_state": market_state,
                "models": [
                    {
                        "model_family": report["model_family"],
                        "selection_rule": report["parameter_sweep"]["selection_rule"],
                        "eligible_parameter_count": report["parameter_sweep"]["eligible_parameter_count"],
                        "selected": report["parameter_sweep"]["selected"],
                    }
                    for report in model_reports
                ],
                "selected": selected,
                "selection_rule": (
                    "market_regime_validation_return_and_boundary_momentum"
                    if args.selection_mode == "market-regime"
                    else "max_validation_model_family_and_parameter_edge"
                ),
                "eligible_parameter_count": sum(
                    report["parameter_sweep"]["eligible_parameter_count"] for report in model_reports
                ),
            },
            "selected_parameter_report": validation_report,
        },
        "evaluation": evaluation,
        "artifact_paths": artifact_paths,
    }


def _rolling_summary(windows):
    evaluated = [window for window in windows if window["evaluation"] is not None]
    passed = [window for window in evaluated if window["evaluation"]["success_gate"]["passed"]]
    if not evaluated:
        return {
            "window_count": len(windows),
            "evaluated_window_count": 0,
            "passed_window_count": 0,
            "passed_all_windows": False,
        }
    model_returns = [window["evaluation"]["aggregate"]["model"]["cumulative_return"] for window in evaluated]
    buy_hold_returns = [window["evaluation"]["aggregate"]["buy_and_hold"]["cumulative_return"] for window in evaluated]
    model_edges = [window["evaluation"]["aggregate"]["model_minus_buy_and_hold"] for window in evaluated]
    trades = [window["evaluation"]["aggregate"]["model"]["trades"] for window in evaluated]
    return {
        "window_count": len(windows),
        "evaluated_window_count": len(evaluated),
        "passed_window_count": len(passed),
        "passed_all_windows": len(evaluated) == len(windows) and len(passed) == len(windows),
        "average_model_return": float(np.mean(model_returns)),
        "average_buy_and_hold_return": float(np.mean(buy_hold_returns)),
        "average_model_minus_buy_and_hold": float(np.mean(model_edges)),
        "minimum_model_minus_buy_and_hold": float(np.min(model_edges)),
        "total_evaluation_trades": int(np.sum(trades)),
        "minimum_window_trades": int(np.min(trades)),
    }


def _holdout_objective(primary_window, test_days):
    evaluation = primary_window["evaluation"]
    if evaluation is None:
        return {
            "objective": "beat_buy_and_hold_on_final_holdout",
            "holdout_days": int(test_days),
            "passed": False,
            "reason": "no_selected_policy",
            "split_dates": primary_window["split_dates"],
        }
    aggregate = evaluation["aggregate"]
    return {
        "objective": "beat_buy_and_hold_on_final_holdout",
        "holdout_days": int(test_days),
        "passed": bool(evaluation["success_gate"]["passed"]),
        "beats_buy_and_hold": bool(evaluation["success_gate"]["beats_buy_and_hold"]),
        "beats_no_trade": bool(evaluation["success_gate"]["beats_no_trade"]),
        "has_minimum_evaluation_trades": bool(evaluation["success_gate"]["has_minimum_evaluation_trades"]),
        "model_cumulative_return": aggregate["model"]["cumulative_return"],
        "buy_and_hold_cumulative_return": aggregate["buy_and_hold"]["cumulative_return"],
        "model_minus_buy_and_hold": aggregate["model_minus_buy_and_hold"],
        "model_minus_no_trade": aggregate["model_minus_no_trade"],
        "trade_count": aggregate["model"]["trades"],
        "split_dates": primary_window["split_dates"],
        "training_data_ends_before": primary_window["split_dates"]["validation_start"],
        "validation_data_ends_before": primary_window["split_dates"]["test_start"],
        "evaluation_data_starts_at": primary_window["split_dates"]["test_start"],
    }


def main(argv=None):
    args = parse_args(argv)
    _set_seed(args.seed)
    tickers = parse_tickers(args.tickers)
    config = {
        "tickers": tickers,
        "start_date": args.start_date,
        "bar_size": args.bar_size,
        "prediction_horizon_bars": args.prediction_horizon_bars,
        "validation_days": args.validation_days,
        "test_days": args.test_days,
        "rolling_windows": args.rolling_windows,
        "rolling_step_days": args.rolling_step_days,
        "model_family": args.model_family,
        "include_futures_metrics": args.include_futures_metrics,
        "include_premium_index": args.include_premium_index,
        "selection_mode": args.selection_mode,
        "long_count_grid": _int_grid(args.long_count_grid),
        "short_count_grid": _int_grid(args.short_count_grid),
        "rebalance_bars_grid": _int_grid(args.rebalance_bars_grid),
        "long_threshold_grid": _float_grid(args.long_threshold_grid),
        "short_threshold_grid": _float_grid(args.short_threshold_grid),
        "long_leverage_grid": _float_grid(args.long_leverage_grid),
        "short_leverage_grid": _float_grid(args.short_leverage_grid),
        "min_validation_trades": args.min_validation_trades,
        "min_evaluation_trades": args.min_evaluation_trades,
        "commission": args.commission,
        "cash": args.cash,
        "seed": args.seed,
    }
    run_id = stable_config_hash(config)
    run_dir = f"{args.computed_data_dir}/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    raw_tickers = filter_tickers_by_start_date(
        load_cached_ohlc_data(args.data_dir, tickers=tickers),
        args.start_date,
    )
    datasets = build_rank_datasets(
        raw_tickers,
        args.bar_size,
        args.prediction_horizon_bars,
        data_dir=args.data_dir,
        include_futures_metrics=args.include_futures_metrics,
        include_premium_index=args.include_premium_index,
    )
    max_end = min(data["features"].index.max() for data in datasets.values())
    window_ends = [
        max_end - pd.Timedelta(days=args.rolling_step_days * window_index)
        for window_index in range(args.rolling_windows)
    ]
    windows = [
        _run_window(
            datasets,
            tickers,
            args,
            run_dir,
            window_end,
            window_index=window_index if args.rolling_windows > 1 else None,
        )
        for window_index, window_end in enumerate(window_ends)
    ]
    primary_window = windows[0]
    holdout = _holdout_objective(primary_window, args.test_days)
    report = {
        "run_id": run_id,
        "report_type": "rank_signal_extensive_report",
        "config": config,
        "data": raw_data_stats(raw_tickers),
        "objective": holdout,
        "split_dates": primary_window["split_dates"],
        "training": primary_window["training"],
        "validation": primary_window["validation"],
        "evaluation": primary_window["evaluation"],
        "rolling": {
            "summary": _rolling_summary(windows),
            "windows": windows,
        },
        "artifact_paths": primary_window["artifact_paths"],
        "git": git_state(),
    }
    report_path = f"{run_dir}/rank_signal_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(jsonable(report), file, indent=2, sort_keys=True)

    print(f"rank_signal_extensive_report run_id={run_id}")
    print(f"summary={report_path}")
    selected = report["validation"]["parameter_sweep"]["selected"]
    if selected is None:
        print("validation_selected_parameters=None")
    else:
        print(
            "validation_selected_parameters="
            f"rebalance={selected['rebalance_bars']} long_count={selected['long_count']} "
            f"short_count={selected['short_count']} long_threshold={selected['long_threshold']} "
            f"short_threshold={selected['short_threshold']} validation_model_return={selected['model_cumulative_return']:.4f} "
            f"validation_buy_hold_return={selected['buy_and_hold_cumulative_return']:.4f} "
            f"validation_trades={selected['model_trades']}"
        )
    if report["evaluation"] is not None:
        aggregate = report["evaluation"]["aggregate"]
        print(
            "holdout_evaluation: "
            f"model_return={aggregate['model']['cumulative_return']:.4f}, "
            f"buy_hold_return={aggregate['buy_and_hold']['cumulative_return']:.4f}, "
            f"trades={aggregate['model']['trades']}, "
            f"passed={report['evaluation']['success_gate']['passed']}"
        )
    rolling = report["rolling"]["summary"]
    print(
        "rolling: "
        f"passed_windows={rolling['passed_window_count']}/{rolling['evaluated_window_count']}, "
        f"passed_all={rolling['passed_all_windows']}, "
        f"avg_model_return={rolling['average_model_return']:.4f}, "
        f"avg_buy_hold_return={rolling['average_buy_and_hold_return']:.4f}, "
        f"total_trades={rolling['total_evaluation_trades']}"
    )


if __name__ == "__main__":
    main()
