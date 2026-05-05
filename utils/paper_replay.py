from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd

from run_rank_signal_experiment import (
    _build_model,
    _fixed_market_regime_choice,
    _market_state,
    _model_families,
    _parameter_sweep,
    _positions_from_scores,
    _prediction_matrix,
    _score_split,
    _success_gate,
)
from utils.experiment_runner import git_state, jsonable
from utils.paper_ledger import idempotency_key, paper_policy_id
from utils.util import validate_ohlc_bars


@dataclass(frozen=True)
class PaperReplayConfig:
    tickers: tuple[str, ...]
    start_date: str
    end_date: str | None
    bar_size: str
    prediction_horizon_bars: int
    validation_days: int
    min_train_days: int
    selection_cadence_days: int
    rebalance_cadence_bars: int
    selector_policy: str
    model_family: str
    include_futures_metrics: bool
    include_premium_index: bool
    long_count_grid: str
    short_count_grid: str
    rebalance_bars_grid: str
    long_threshold_grid: str
    short_threshold_grid: str
    long_leverage_grid: str
    short_leverage_grid: str
    min_validation_trades: int
    min_replay_trades: int
    commission: float
    cash: float
    seed: int


GUARDED_MARKET_REGIME_POLICY = "guarded-market-regime"


def _rank_args(config: PaperReplayConfig):
    selection_mode = config.selector_policy
    if selection_mode == GUARDED_MARKET_REGIME_POLICY:
        selection_mode = "market-regime"
    return SimpleNamespace(
        tickers=",".join(config.tickers),
        validation_days=config.validation_days,
        min_validation_trades=config.min_validation_trades,
        min_evaluation_trades=config.min_replay_trades,
        model_family=config.model_family,
        selection_mode=selection_mode,
        long_count_grid=config.long_count_grid,
        short_count_grid=config.short_count_grid,
        rebalance_bars_grid=config.rebalance_bars_grid,
        long_threshold_grid=config.long_threshold_grid,
        short_threshold_grid=config.short_threshold_grid,
        long_leverage_grid=config.long_leverage_grid,
        short_leverage_grid=config.short_leverage_grid,
        commission=config.commission,
        cash=config.cash,
        seed=config.seed,
    )


def validate_replay_bar_series(raw_tickers, datasets, tickers, bar_size):
    coverage = {}
    for raw_df, ticker in raw_tickers:
        raw = validate_ohlc_bars(raw_df, ticker_name=ticker, frequency="1min")
        bars = validate_ohlc_bars(datasets[ticker]["bars"], ticker_name=ticker, frequency=bar_size)
        coverage[ticker] = {
            "raw_rows": int(len(raw)),
            "raw_start": raw.index.min().isoformat() if len(raw) else None,
            "raw_end": raw.index.max().isoformat() if len(raw) else None,
            "bar_rows": int(len(bars)),
            "bar_start": bars.index.min().isoformat() if len(bars) else None,
            "bar_end": bars.index.max().isoformat() if len(bars) else None,
        }
    missing = sorted(set(tickers).difference(coverage))
    if missing:
        raise ValueError(f"Replay data is missing selected tickers: {missing}")
    return coverage


def decision_timestamps(index, config: PaperReplayConfig):
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date) if config.end_date else index.max()
    first_available = max(start, pd.Timestamp(index.min()))
    first_decision = first_available + pd.Timedelta(days=config.min_train_days + config.validation_days)
    first_pos = int(index.searchsorted(first_decision, side="left"))
    if first_pos >= len(index):
        raise ValueError("Replay range has no decision timestamp after train/validation warmup")

    timestamps = []
    current = index[first_pos]
    while current < end:
        timestamps.append(pd.Timestamp(current))
        current = current + pd.Timedelta(days=config.selection_cadence_days)
        pos = int(index.searchsorted(current, side="left"))
        if pos >= len(index):
            break
        current = index[pos]
    if not timestamps:
        raise ValueError("Replay range produced no strategy decisions")
    return timestamps


def _train_frame_before_decision(datasets, tickers, validation_start):
    x_frames = []
    y_frames = []
    for ticker_id, ticker in enumerate(tickers):
        features = datasets[ticker]["features"]
        target = datasets[ticker]["future_return"]
        train_index = features.index.intersection(target.dropna().index)
        train_index = train_index[train_index < validation_start]
        x_frames.append(features.loc[train_index].assign(ticker_id=ticker_id))
        y_frames.append(target.loc[train_index])
    x_train = pd.concat(x_frames)
    y_train = pd.concat(y_frames)
    if x_train.empty:
        raise ValueError("Replay selection has no training rows before validation period")
    return x_train, y_train


def _copy_params(selected):
    return {
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


def selected_rebalance_bars(params, config: PaperReplayConfig) -> int:
    if params is None:
        return int(config.rebalance_cadence_bars)
    return int(params.get("rebalance_bars", config.rebalance_cadence_bars))


def _no_trade_params() -> dict[str, float | int]:
    return {
        "rebalance_bars": 6,
        "long_count": 0,
        "short_count": 0,
        "long_threshold": 1e9,
        "short_threshold": -1e9,
        "long_leverage": 0.0,
        "short_leverage": 0.0,
    }


def _broad_bull_params(ticker_count: int) -> dict[str, float | int]:
    return {
        "rebalance_bars": 6,
        "long_count": int(ticker_count),
        "short_count": 0,
        "long_threshold": -1e9,
        "short_threshold": -1e9,
        "long_leverage": 1.5,
        "short_leverage": 0.0,
    }


def guarded_market_regime_params(market_state, selected, ticker_count):
    if selected is None:
        return None, None
    validation_return = market_state["validation_market_return"]
    momentum_96 = market_state["market_momentum_96"]
    momentum_96 = -999.0 if momentum_96 is None else momentum_96
    validation_edge = selected["model_minus_buy_and_hold"]

    if validation_return > 0.2 and momentum_96 > 0.1:
        return _broad_bull_params(ticker_count), "strong_bull_broad_long"
    if validation_edge < -0.2:
        return _no_trade_params(), "negative_validation_edge_no_trade"
    if -0.1 < validation_return < 0.0 and -0.05 < momentum_96 < 0.05:
        return _no_trade_params(), "uncertain_flat_negative_regime_no_trade"
    return None, None


def select_replay_strategy(
    datasets,
    tickers,
    config: PaperReplayConfig,
    decision_time,
    run_dir,
    decision_index,
    persist_model_artifacts=True,
):
    args = _rank_args(config)
    decision_time = pd.Timestamp(decision_time)
    validation_start = decision_time - pd.Timedelta(days=config.validation_days)
    train_start = validation_start - pd.Timedelta(days=config.min_train_days)

    x_train, y_train = _train_frame_before_decision(datasets, tickers, validation_start)
    if x_train.index.min() > train_start:
        raise ValueError("Replay selection lacks configured minimum Training Period coverage")

    model_reports = []
    model_artifacts = {}
    for model_family in _model_families(args):
        model = _build_model(model_family, config.seed)
        model.fit(x_train, y_train)
        predictions = _prediction_matrix(datasets, tickers, model, validation_start, decision_time)
        sweep = _parameter_sweep(datasets, tickers, predictions, validation_start, decision_time, args)
        if sweep["selected"] is not None:
            sweep["selected"] = {"model_family": model_family, **sweep["selected"]}
        model_reports.append({
            "model_family": model_family,
            "model": model,
            "predictions": predictions,
            "parameter_sweep": sweep,
        })

        if persist_model_artifacts:
            model_path = os.path.join(run_dir, f"paper_replay_model_decision_{decision_index}_{model_family}.pkl")
            with open(model_path, "wb") as file:
                pickle.dump(
                    {
                        "model": model,
                        "model_family": model_family,
                        "decision_index": decision_index,
                        "decision_timestamp": decision_time.isoformat(),
                        "split_dates": {
                            "training_start": train_start.isoformat(),
                            "validation_start": validation_start.isoformat(),
                            "decision_timestamp": decision_time.isoformat(),
                        },
                    },
                    file,
                )
            model_artifacts[f"model_{model_family}"] = os.path.abspath(model_path)

    market_state = _market_state(datasets, tickers, validation_start, decision_time)
    if config.selector_policy in ("market-regime", GUARDED_MARKET_REGIME_POLICY):
        selected_report, selected = _fixed_market_regime_choice(
            model_reports,
            market_state,
            datasets,
            tickers,
            validation_start,
            decision_time,
            args,
        )
        selection_rule = (
            "guarded_market_regime_validation_return_momentum_and_validation_edge"
            if config.selector_policy == GUARDED_MARKET_REGIME_POLICY
            else "market_regime_validation_return_and_boundary_momentum"
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
        selection_rule = "max_validation_model_family_and_parameter_edge"

    validation_report = None
    if selected is not None:
        validation_report = _score_split(
            datasets,
            tickers,
            selected_report["predictions"],
            _copy_params(selected),
            validation_start,
            decision_time,
            config.cash,
            config.commission,
        )
        validation_report.pop("_positions")

    guard = None
    if config.selector_policy == GUARDED_MARKET_REGIME_POLICY:
        guarded_params, guard_reason = guarded_market_regime_params(market_state, selected, len(tickers))
        if guarded_params is not None:
            guard = {
                "reason": guard_reason,
                "original_allocation_parameters": _copy_params(selected),
                "replacement_allocation_parameters": guarded_params,
            }
            selected = {**selected, **guarded_params, "guard_reason": guard_reason}

    return {
        "decision_index": decision_index,
        "decision_timestamp": decision_time,
        "training_period": {"start": train_start, "end": validation_start},
        "validation_period": {"start": validation_start, "end": decision_time},
        "selected_model_family": None if selected is None else selected["model_family"],
        "allocation_parameters": None if selected is None else _copy_params(selected),
        "selector_policy": config.selector_policy,
        "selection_rule": selection_rule,
        "feature_flags": {
            "include_futures_metrics": config.include_futures_metrics,
            "include_premium_index": config.include_premium_index,
        },
        "selected_tickers": tuple(tickers),
        "market_state": market_state,
        "validation_evidence": {
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
            "selected_parameter_report": validation_report,
            "guard": guard,
        },
        "model": None if selected_report is None else selected_report["model"],
        "artifact_paths": model_artifacts,
        "gate_outcome": {
            "passed": selected is not None,
            "reason": "selected" if selected is not None else "validation_ineligible",
        },
    }


def strategy_selection_record(decision, config: PaperReplayConfig, policy_id, evidence_mode, key):
    return {
        "record_type": "strategy_selection",
        "policy_id": policy_id,
        "evidence_mode": evidence_mode,
        "idempotency_key": key,
        "selection_time": pd.Timestamp(decision["decision_timestamp"]).isoformat(),
        "signal_time": pd.Timestamp(decision["decision_timestamp"]).isoformat(),
        "decision_index": int(decision["decision_index"]),
        "selector_policy": decision.get("selector_policy", config.selector_policy),
        "selection_rule": decision.get("selection_rule"),
        "selection_cadence_days": int(config.selection_cadence_days),
        "rebalance_cadence_bars": int(config.rebalance_cadence_bars),
        "selected_model_family": decision.get("selected_model_family"),
        "allocation_parameters": decision["allocation_parameters"],
        "feature_flags": decision.get("feature_flags", {
            "include_futures_metrics": config.include_futures_metrics,
            "include_premium_index": config.include_premium_index,
        }),
        "selected_tickers": list(decision.get("selected_tickers", config.tickers)),
        "training_period": decision.get("training_period"),
        "validation_period": decision.get("validation_period"),
        "gate_outcome": decision.get("gate_outcome", {"passed": True, "reason": "selected"}),
        "artifact_paths": decision.get("artifact_paths", {}),
    }


def load_model_from_selection(selection_record):
    family = selection_record.get("selected_model_family")
    path = selection_record.get("artifact_paths", {}).get(f"model_{family}")
    if not path:
        return None
    with open(path, "rb") as file:
        artifact = pickle.load(file)
    return artifact["model"]


def prediction_records(datasets, tickers, model, params, config: PaperReplayConfig, signal_time, previous_weights, policy_id, evidence_mode, key):
    signal_time = pd.Timestamp(signal_time)
    next_open_time = signal_time + pd.Timedelta(config.bar_size)
    prediction_end = next_open_time + selected_rebalance_bars(params, config) * pd.Timedelta(config.bar_size)
    predictions = _prediction_matrix(datasets, tickers, model, signal_time, prediction_end)
    predictions = predictions.reindex(pd.DatetimeIndex([signal_time])).dropna()
    if predictions.empty:
        return [], None

    scores = predictions.loc[signal_time].to_numpy(dtype=float)
    weights = _positions_from_scores(scores, params)
    weights_by_ticker = {ticker: float(weights[idx]) for idx, ticker in enumerate(tickers)}
    previous_by_ticker = {ticker: float(previous_weights[idx]) for idx, ticker in enumerate(tickers)}
    records = [
        {
            "record_type": "prediction",
            "policy_id": policy_id,
            "evidence_mode": evidence_mode,
            "idempotency_key": key,
            "signal_time": signal_time.isoformat(),
            "scores": {ticker: float(scores[idx]) for idx, ticker in enumerate(tickers)},
        },
        {
            "record_type": "target_weights",
            "policy_id": policy_id,
            "evidence_mode": evidence_mode,
            "idempotency_key": key,
            "signal_time": signal_time.isoformat(),
            "weights": weights_by_ticker,
            "previous_weights": previous_by_ticker,
        },
    ]
    for idx, ticker in enumerate(tickers):
        delta = float(weights[idx] - previous_weights[idx])
        if delta != 0.0:
            records.append(
                {
                    "record_type": "simulated_order",
                    "policy_id": policy_id,
                    "evidence_mode": evidence_mode,
                    "idempotency_key": key,
                    "signal_time": signal_time.isoformat(),
                    "ticker": ticker,
                    "weight_delta": delta,
                    "target_weight": float(weights[idx]),
                    "previous_weight": float(previous_weights[idx]),
                }
            )
    return records, weights


def fill_records(
    datasets,
    tickers,
    weights,
    previous_weights,
    equity,
    config: PaperReplayConfig,
    signal_time,
    policy_id,
    evidence_mode,
    key,
    rebalance_bars=None,
):
    signal_time = pd.Timestamp(signal_time)
    entry_time = signal_time + pd.Timedelta(config.bar_size)
    cadence = int(rebalance_bars or config.rebalance_cadence_bars)
    exit_time = entry_time + cadence * pd.Timedelta(config.bar_size)
    period_returns = []
    fill_prices = {}
    exit_prices = {}
    for ticker in tickers:
        bars = datasets[ticker]["bars"]
        if entry_time not in bars.index:
            return [], equity, "missing_next_open_fill_data"
        if exit_time not in bars.index:
            return [], equity, "missing_next_open_fill_data"
        entry = float(bars.loc[entry_time, "Open"])
        exit_price = float(bars.loc[exit_time, "Open"])
        fill_prices[ticker] = entry
        exit_prices[ticker] = exit_price
        period_returns.append(exit_price / entry - 1)

    turnover = float(np.abs(weights - previous_weights).sum())
    fee = float(config.commission * turnover * equity)
    gross_return = float(weights @ np.asarray(period_returns, dtype=float))
    new_equity = float(equity * max(0.0, 1.0 + gross_return - config.commission * turnover))
    weights_by_ticker = {ticker: float(weights[idx]) for idx, ticker in enumerate(tickers)}
    common = {
        "policy_id": policy_id,
        "evidence_mode": evidence_mode,
        "idempotency_key": key,
        "signal_time": signal_time.isoformat(),
    }
    return [
        {
            "record_type": "simulated_fill",
            **common,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "fill_prices": fill_prices,
            "exit_prices": exit_prices,
            "period_return": gross_return,
            "turnover": turnover,
            "fee": fee,
            "cash": new_equity,
            "equity": new_equity,
            "weights": weights_by_ticker,
        },
        {
            "record_type": "equity_snapshot",
            **common,
            "cash": new_equity,
            "equity": new_equity,
            "weights": weights_by_ticker,
            "long_exposure": float(sum(weight for weight in weights if weight > 0)),
            "short_exposure": float(sum(abs(weight) for weight in weights if weight < 0)),
            "unrealized_pnl": 0.0,
            "realized_pnl": float(new_equity - config.cash),
            "accumulated_fees": fee,
        },
    ], new_equity, None


def no_action_record(policy_id, evidence_mode, key, signal_time, reason, details=None):
    return {
        "record_type": "no_action",
        "policy_id": policy_id,
        "evidence_mode": evidence_mode,
        "idempotency_key": key,
        "signal_time": pd.Timestamp(signal_time).isoformat(),
        "reason": reason,
        "details": details or {},
    }


def simulate_replay_positions(datasets, tickers, decisions, config: PaperReplayConfig, replay_index):
    equity = float(config.cash)
    previous = np.zeros(len(tickers), dtype=float)
    trade_count = 0
    rows = []
    ledger_records = []
    policy_id = paper_policy_id(config, evidence_mode="historical_replay")

    for decision_pos, decision in enumerate(decisions):
        start = pd.Timestamp(decision["decision_timestamp"])
        key = idempotency_key(policy_id, "historical_replay", start)
        ledger_records.append(strategy_selection_record(decision, config, policy_id, "historical_replay", key))
        end = (
            pd.Timestamp(decisions[decision_pos + 1]["decision_timestamp"])
            if decision_pos + 1 < len(decisions)
            else (
                pd.Timestamp(config.end_date) + pd.Timedelta(config.bar_size)
                if config.end_date
                else replay_index.max() + pd.Timedelta(config.bar_size)
            )
        )
        segment_index = replay_index[(replay_index >= start) & (replay_index < end)]
        if len(segment_index) < 2:
            continue

        params = decision["allocation_parameters"]
        model = decision["model"]
        if params is None or model is None:
            ledger_records.append(no_action_record(policy_id, "historical_replay", key, start, decision["gate_outcome"]["reason"]))
            continue

        end_exclusive = segment_index[-1] + pd.Timedelta(config.bar_size)
        predictions = _prediction_matrix(datasets, tickers, model, segment_index[0], end_exclusive)
        predictions = predictions.reindex(segment_index).dropna()
        if predictions.empty:
            continue

        cadence = selected_rebalance_bars(params, config)
        for pos in range(0, len(predictions.index) - 1, cadence):
            signal_time = predictions.index[pos]
            signal_key = idempotency_key(policy_id, "historical_replay", signal_time)
            entry_time = predictions.index[pos + 1]
            exit_pos = min(pos + 1 + cadence, len(predictions.index) - 1)
            exit_time = predictions.index[exit_pos]
            exit_column = "Close" if exit_pos == len(predictions.index) - 1 else "Open"

            period_returns = []
            for ticker in tickers:
                bars = datasets[ticker]["bars"]
                if entry_time not in bars.index or exit_time not in bars.index:
                    period_returns = []
                    break
                entry = float(bars.loc[entry_time, "Open"])
                exit_price = float(bars.loc[exit_time, exit_column])
                period_returns.append(exit_price / entry - 1)
            if not period_returns:
                ledger_records.append(no_action_record(policy_id, "historical_replay", signal_key, signal_time, "missing_next_open_fill_data"))
                continue

            weights = _positions_from_scores(predictions.loc[signal_time].to_numpy(dtype=float), params)
            records, _ = prediction_records(
                datasets,
                tickers,
                model,
                params,
                config,
                signal_time,
                previous,
                policy_id,
                "historical_replay",
                signal_key,
            )
            ledger_records.extend(records)
            changed = ((weights != 0) | (previous != 0)) & (weights != previous)
            trade_count += int(changed.sum())
            turnover = float(np.abs(weights - previous).sum())
            turnover_cost = float(config.commission * turnover)
            gross_return = float(weights @ np.asarray(period_returns, dtype=float))
            equity *= max(0.0, 1.0 + gross_return - turnover_cost)
            weights_by_ticker = {ticker: float(weights[idx]) for idx, ticker in enumerate(tickers)}
            ledger_records.extend([
                {
                    "record_type": "simulated_fill",
                    "policy_id": policy_id,
                    "evidence_mode": "historical_replay",
                    "idempotency_key": signal_key,
                    "signal_time": signal_time.isoformat(),
                    "entry_time": entry_time.isoformat(),
                    "exit_time": exit_time.isoformat(),
                    "period_return": gross_return,
                    "turnover": turnover,
                    "fee": float(turnover_cost * equity),
                    "cash": equity,
                    "equity": equity,
                    "weights": weights_by_ticker,
                },
                {
                    "record_type": "equity_snapshot",
                    "policy_id": policy_id,
                    "evidence_mode": "historical_replay",
                    "idempotency_key": signal_key,
                    "signal_time": signal_time.isoformat(),
                    "cash": equity,
                    "equity": equity,
                    "weights": weights_by_ticker,
                    "long_exposure": float(sum(weight for weight in weights if weight > 0)),
                    "short_exposure": float(sum(abs(weight) for weight in weights if weight < 0)),
                    "unrealized_pnl": 0.0,
                    "realized_pnl": float(equity - config.cash),
                    "accumulated_fees": float(turnover_cost * equity),
                },
            ])
            rows.append({
                "decision_index": int(decision["decision_index"]),
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "period_return": gross_return,
                "turnover": turnover,
                "turnover_cost": turnover_cost,
                "equity": equity,
                **{f"weight_{ticker}": float(weights[idx]) for idx, ticker in enumerate(tickers)},
            })
            previous = weights

    positions = pd.DataFrame(rows)
    portfolio = {
        "cumulative_return": float(equity / config.cash - 1),
        "trades": int(trade_count),
        "start_cash": float(config.cash),
        "end_equity": float(equity),
    }
    return {"portfolio": portfolio, "positions": positions, "ledger_records": ledger_records}


def buy_and_hold_baseline(datasets, tickers, replay_index, cash, commission):
    if len(replay_index) < 2:
        return {"cumulative_return": 0.0, "trades": 0, "start_cash": cash, "end_equity": cash}
    returns = []
    for ticker in tickers:
        bars = datasets[ticker]["bars"]
        entry_time = replay_index[1]
        exit_time = replay_index[-1]
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


def summarize_positions_by_window(positions):
    if positions.empty:
        return []
    summaries = []
    for decision_index, group in positions.groupby("decision_index", sort=True):
        summaries.append({
            "decision_index": int(decision_index),
            "start": pd.Timestamp(group["signal_time"].min()).isoformat(),
            "end": pd.Timestamp(group["signal_time"].max()).isoformat(),
            "rows": int(len(group)),
            "ending_equity": float(group["equity"].iloc[-1]),
            "turnover": float(group["turnover"].sum()),
        })
    return summaries


def summarize_positions_by_month(positions):
    if positions.empty:
        return []
    data = positions.copy()
    data["month"] = pd.to_datetime(data["signal_time"]).dt.to_period("M").astype(str)
    summaries = []
    for month, group in data.groupby("month", sort=True):
        start_equity = float(group["equity"].iloc[0])
        end_equity = float(group["equity"].iloc[-1])
        summaries.append({
            "month": str(month),
            "rows": int(len(group)),
            "cumulative_return": float(end_equity / start_equity - 1) if start_equity else 0.0,
            "ending_equity": end_equity,
            "turnover": float(group["turnover"].sum()),
        })
    return summaries


def build_replay_report(config, run_id, data_coverage, decisions, replay_result):
    positions = replay_result["positions"]
    replay_index = pd.DatetimeIndex(positions["signal_time"]) if not positions.empty else pd.DatetimeIndex([])
    model = replay_result["portfolio"]
    buy_hold = replay_result["buy_and_hold"]
    aggregate = {
        "model": model,
        "buy_and_hold": buy_hold,
        "no_trade": {"cumulative_return": 0.0, "trades": 0, "start_cash": config.cash, "end_equity": config.cash},
    }
    aggregate["model_minus_buy_and_hold"] = model["cumulative_return"] - buy_hold["cumulative_return"]
    aggregate["model_minus_no_trade"] = model["cumulative_return"]
    evaluation = {"aggregate": aggregate, "success_gate": _success_gate({"aggregate": aggregate}, config.min_replay_trades)}
    return {
        "run_id": run_id,
        "report_type": "historical_paper_replay_report",
        "evidence_mode": "historical_replay",
        "config": asdict(config),
        "data": data_coverage,
        "selection_decisions": [
            {
                key: value
                for key, value in decision.items()
                if key != "model"
            }
            for decision in decisions
        ],
        "replay": {
            "aggregate": aggregate,
            "success_gate": evaluation["success_gate"],
            "summary_by_selection_window": summarize_positions_by_window(positions),
            "summary_by_calendar_month": summarize_positions_by_month(positions),
            "total_trades": int(model["trades"]),
            "minimum_trades": int(min((model["trades"], buy_hold["trades"]))),
            "first_signal_time": replay_index.min().isoformat() if len(replay_index) else None,
            "last_signal_time": replay_index.max().isoformat() if len(replay_index) else None,
        },
        "artifact_paths": replay_result["artifact_paths"],
        "git": git_state(),
    }


def run_historical_paper_replay(raw_tickers, datasets, tickers, config, run_dir, run_id):
    os.makedirs(run_dir, exist_ok=True)
    data_coverage = validate_replay_bar_series(raw_tickers, datasets, tickers, config.bar_size)
    replay_index = None
    for ticker in tickers:
        index = datasets[ticker]["features"].index
        replay_index = index if replay_index is None else replay_index.intersection(index)
    replay_index = pd.DatetimeIndex(replay_index).sort_values()
    end = pd.Timestamp(config.end_date) if config.end_date else replay_index.max()
    replay_index = replay_index[(replay_index >= pd.Timestamp(config.start_date)) & (replay_index <= end)]
    if len(replay_index) < 2:
        raise ValueError("Replay range has fewer than two common feature bars")

    decisions = [
        select_replay_strategy(datasets, tickers, config, timestamp, run_dir, idx)
        for idx, timestamp in enumerate(decision_timestamps(replay_index, config))
    ]
    model_result = simulate_replay_positions(datasets, tickers, decisions, config, replay_index)
    model_result["buy_and_hold"] = buy_and_hold_baseline(
        datasets,
        tickers,
        replay_index,
        config.cash,
        config.commission,
    )

    positions_path = os.path.join(run_dir, "paper_replay_positions.csv")
    decisions_path = os.path.join(run_dir, "selection_decisions.jsonl")
    ledger_path = os.path.join(run_dir, "paper_ledger.jsonl")
    report_path = os.path.join(run_dir, "paper_replay_report.json")
    if not model_result["positions"].empty:
        model_result["positions"].to_csv(positions_path, index=False)
    with open(decisions_path, "w", encoding="utf-8") as file:
        for decision in decisions:
            file.write(json.dumps(jsonable({key: value for key, value in decision.items() if key != "model"})))
            file.write("\n")
    with open(ledger_path, "w", encoding="utf-8") as file:
        for record in model_result["ledger_records"]:
            file.write(json.dumps(jsonable(record), sort_keys=True))
            file.write("\n")

    artifact_paths = {
        "positions": positions_path,
        "selection_decisions": decisions_path,
        "ledger": ledger_path,
        "report": report_path,
    }
    for decision in decisions:
        artifact_paths.update(decision["artifact_paths"])
    model_result["artifact_paths"] = artifact_paths
    report = build_replay_report(config, run_id, data_coverage, decisions, model_result)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(jsonable(report), file, indent=2, sort_keys=True)
    return report
