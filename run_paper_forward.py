from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import pandas as pd

from download_data import main as download_data_main
from download_futures_metrics_direct import main as download_futures_metrics_main
from download_premium_index_klines_direct import main as download_premium_index_main
from run_paper_replay import build_config
from run_rank_signal_experiment import _positions_from_scores, _prediction_matrix
from utils.experiment_runner import DEFAULT_CASH, git_state, jsonable
from utils.rank_data import build_rank_datasets
from utils.paper_ledger import (
    append_ledger_records,
    has_idempotency_key,
    idempotency_key,
    latest_selection,
    ledger_path,
    paper_policy_id,
    pending_paper_orders,
    read_ledger,
    reduce_paper_state,
)
from utils.paper_replay import (
    load_model_from_selection,
    no_action_record,
    select_replay_strategy,
    selected_rebalance_bars,
    strategy_selection_record,
)
from utils.util import (
    DEFAULT_EXPERIMENT_START_DATE,
    DEFAULT_TICKERS,
    filter_tickers_by_start_date,
    load_cached_ohlc_data,
    validate_ohlc_bars,
)


ONLINE_EVIDENCE_MODE = "paper_forward_online"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run online simulated paper trading for the rank-signal policy.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Compute one online cycle without writing ledger records.")
    mode.add_argument("--append", action="store_true", help="Append one idempotent online cycle to the paper ledger.")
    mode.add_argument("--daemon", action="store_true", help="Continuously run idempotent online cycles.")
    mode.add_argument("--state", action="store_true", help="Print the latest reduced paper state.")
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
        default="guarded-market-regime",
    )
    parser.add_argument("--model-family", choices=("ridge", "hist_gradient_boosting", "both"), default="both")
    parser.add_argument("--include-futures-metrics", dest="include_futures_metrics", action="store_true", default=True)
    parser.add_argument("--no-include-futures-metrics", dest="include_futures_metrics", action="store_false")
    parser.add_argument("--include-premium-index", dest="include_premium_index", action="store_true", default=True)
    parser.add_argument("--no-include-premium-index", dest="include_premium_index", action="store_false")
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
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-data", dest="refresh_data", action="store_true", default=False)
    parser.add_argument("--no-refresh-data", dest="refresh_data", action="store_false")
    parser.add_argument("--refresh-lookback-days", type=int, default=14)
    parser.add_argument("--force-reselect", action="store_true")
    parser.add_argument("--force-reselect-reason")
    parser.add_argument("--decision-time")
    parser.add_argument("--max-staleness-hours", type=float, default=6.0)
    parser.add_argument("--max-feature-staleness-hours", type=float, default=48.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--max-cycles", type=int)
    return parser.parse_args(argv)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _anchor_time(args) -> pd.Timestamp:
    if args.decision_time:
        return pd.Timestamp(args.decision_time).tz_localize(None)
    return pd.Timestamp.utcnow().tz_localize(None)


def _date_window(anchor: pd.Timestamp, lookback_days: int) -> tuple[str, str]:
    end = anchor.date()
    start = end - datetime.timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


def _refresh_data(args, config, anchor: pd.Timestamp) -> None:
    if not args.refresh_data:
        return
    tickers = ",".join(config.tickers)
    start_date, end_date = _date_window(anchor, args.refresh_lookback_days)
    download_data_main(["--data-dir", args.data_dir, "--tickers", tickers, "--start-date", start_date])
    if config.include_futures_metrics:
        download_futures_metrics_main([
            "--data-dir",
            args.data_dir,
            "--tickers",
            tickers,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
        ])
    if config.include_premium_index:
        download_premium_index_main([
            "--data-dir",
            args.data_dir,
            "--tickers",
            tickers,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
        ])


def _load_data(args, config):
    anchor = _anchor_time(args)
    _refresh_data(args, config, anchor)
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
    return raw_tickers, datasets


def _latest_common_signal_time(datasets, tickers, anchor_time, bar_size):
    closed_cutoff = pd.Timestamp(anchor_time) - pd.Timedelta(bar_size)
    common = None
    for ticker in tickers:
        index = pd.DatetimeIndex(datasets[ticker]["features"].index).sort_values()
        index = index[index <= closed_cutoff]
        common = index if common is None else common.intersection(index)
    if common is None or len(common) == 0:
        return None
    return pd.Timestamp(common.max())


def _latest_file_date(paths, prefix, suffix):
    latest = None
    for path in paths:
        name = os.path.basename(path)
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        raw_date = name[len(prefix):len(name) - len(suffix)]
        try:
            day = datetime.date.fromisoformat(raw_date)
        except ValueError:
            continue
        latest = day if latest is None else max(latest, day)
    return latest


def _required_feature_staleness(args, config, anchor):
    checks = []
    for ticker in config.tickers:
        if config.include_futures_metrics:
            directory = os.path.join(args.data_dir, "futures", "um", "daily", "metrics", ticker)
            paths = [os.path.join(directory, name) for name in os.listdir(directory)] if os.path.isdir(directory) else []
            latest = _latest_file_date(paths, f"{ticker}-metrics-", ".csv")
            checks.append(("futures_metrics", ticker, latest))
        if config.include_premium_index:
            directory = os.path.join(args.data_dir, "futures", "um", "daily", "premiumIndexKlines", ticker, "1m")
            paths = [os.path.join(directory, name) for name in os.listdir(directory)] if os.path.isdir(directory) else []
            latest = _latest_file_date(paths, f"{ticker}-1m-", ".csv")
            checks.append(("premium_index", ticker, latest))
    stale = []
    for kind, ticker, latest in checks:
        if latest is None:
            stale.append({"feature": kind, "ticker": ticker, "reason": "missing"})
            continue
        age_hours = (pd.Timestamp(anchor.date()) - pd.Timestamp(latest)).total_seconds() / 3600.0
        if age_hours > args.max_feature_staleness_hours:
            stale.append({
                "feature": kind,
                "ticker": ticker,
                "latest_date": latest.isoformat(),
                "age_hours": age_hours,
            })
    if stale:
        return "stale_feature_data", {"stale": stale}
    return None, {}


def _data_no_action(raw_tickers, datasets, tickers, bar_size, signal_time, anchor_time, max_staleness_hours):
    try:
        for raw_df, ticker in raw_tickers:
            validate_ohlc_bars(raw_df, ticker_name=ticker, frequency="1min")
            validate_ohlc_bars(datasets[ticker]["bars"], ticker_name=ticker, frequency=bar_size)
    except ValueError as error:
        return "data_quality_failure", {"error": str(error)}
    if signal_time is None:
        return "stale_data", {"error": "no closed common feature timestamp"}
    age_hours = (pd.Timestamp(anchor_time) - pd.Timestamp(signal_time)).total_seconds() / 3600.0
    if age_hours > max_staleness_hours:
        return "stale_data", {"latest_signal_time": pd.Timestamp(signal_time).isoformat(), "age_hours": age_hours}
    if age_hours < 0:
        return "stale_data", {"latest_signal_time": pd.Timestamp(signal_time).isoformat(), "age_hours": age_hours}
    return None, {}


def _selection_due(selection, signal_time, config, force_reselect):
    if force_reselect or selection is None:
        return True
    last = pd.Timestamp(selection["selection_time"])
    return pd.Timestamp(signal_time) >= last + pd.Timedelta(days=config.selection_cadence_days)


def _last_online_activity_time(records):
    times = []
    for record in records:
        if record.get("record_type") in ("online_paper_fill", "pending_paper_order", "no_action") and record.get("signal_time"):
            times.append(pd.Timestamp(record["signal_time"]))
    return max(times) if times else None


def _rebalance_due(records, signal_time, config, selection):
    last = _last_online_activity_time(records)
    if last is None:
        return True
    cadence = selected_rebalance_bars(selection.get("allocation_parameters") if selection else None, config)
    return pd.Timestamp(signal_time) >= last + cadence * pd.Timedelta(config.bar_size)


def _price_map(datasets, tickers, timestamp, column):
    prices = {}
    for ticker in tickers:
        bars = datasets[ticker]["bars"]
        if timestamp not in bars.index:
            return None
        prices[ticker] = float(bars.loc[timestamp, column])
    return prices


def _weights_from_cash_units(cash, units, prices, tickers):
    values = {ticker: float(units.get(ticker, 0.0) * prices.get(ticker, 0.0)) for ticker in tickers}
    equity = float(cash + sum(values.values()))
    if equity == 0.0:
        return equity, {ticker: 0.0 for ticker in tickers}
    return equity, {ticker: float(values[ticker] / equity) for ticker in tickers}


def _fill_pending_orders(datasets, config, records, policy_id, evidence_mode, anchor_time, slippage):
    batch = []
    for order in pending_paper_orders(records):
        fill_time = pd.Timestamp(order["fill_time"])
        if fill_time >= anchor_time:
            continue
        key = order.get("pending_order_key") or order["idempotency_key"]
        fill_key = idempotency_key(policy_id, evidence_mode, fill_time, suffix=f"fill:{key}")
        if has_idempotency_key(records + batch, fill_key):
            continue
        fill_prices = _price_map(datasets, config.tickers, fill_time, "Open")
        if fill_prices is None:
            continue

        state = reduce_paper_state(records + batch, config.tickers, config.cash)
        cash = float(state["cash"])
        units = {ticker: float(state["units"].get(ticker, 0.0)) for ticker in config.tickers}
        equity, current_weights = _weights_from_cash_units(cash, units, fill_prices, config.tickers)
        target_weights = {ticker: float(order["target_weights"].get(ticker, 0.0)) for ticker in config.tickers}
        unit_deltas = {}
        traded_notional = 0.0
        signed_notional = 0.0
        for ticker in config.tickers:
            price = fill_prices[ticker]
            direction = 1.0 if target_weights[ticker] >= current_weights.get(ticker, 0.0) else -1.0
            execution_price = price * (1.0 + direction * slippage)
            target_value = equity * target_weights[ticker]
            current_value = units[ticker] * execution_price
            delta_units = (target_value - current_value) / execution_price if execution_price else 0.0
            unit_deltas[ticker] = float(delta_units)
            signed_notional += float(delta_units * execution_price)
            traded_notional += abs(float(delta_units * execution_price))
            units[ticker] += float(delta_units)
        fee = float(config.commission * traded_notional)
        cash = float(cash - signed_notional - fee)
        new_equity, weights = _weights_from_cash_units(cash, units, fill_prices, config.tickers)
        batch.append({
            "record_type": "online_paper_fill",
            "policy_id": policy_id,
            "evidence_mode": evidence_mode,
            "idempotency_key": fill_key,
            "pending_order_key": key,
            "signal_time": order["signal_time"],
            "fill_time": fill_time.isoformat(),
            "fill_prices": fill_prices,
            "cash": cash,
            "equity": new_equity,
            "units": units,
            "unit_deltas": unit_deltas,
            "weights": weights,
            "target_weights": target_weights,
            "turnover": float(traded_notional / equity) if equity else 0.0,
            "fee": fee,
            "slippage": float(slippage),
        })
    return batch


def _mark_to_market_record(datasets, config, records, policy_id, evidence_mode, signal_time):
    state = reduce_paper_state(records, config.tickers, config.cash)
    if not any(abs(unit) > 0 for unit in state["units"].values()):
        return None
    key = idempotency_key(policy_id, evidence_mode, signal_time, suffix="mark")
    if has_idempotency_key(records, key):
        return None
    prices = _price_map(datasets, config.tickers, signal_time, "Close")
    if prices is None:
        return None
    equity, weights = _weights_from_cash_units(state["cash"], state["units"], prices, config.tickers)
    return {
        "record_type": "mark_to_market",
        "policy_id": policy_id,
        "evidence_mode": evidence_mode,
        "idempotency_key": key,
        "signal_time": pd.Timestamp(signal_time).isoformat(),
        "mark_time": pd.Timestamp(signal_time).isoformat(),
        "mark_prices": prices,
        "cash": float(state["cash"]),
        "equity": equity,
        "units": state["units"],
        "weights": weights,
        "unrealized_pnl": float(equity - config.cash),
        "realized_pnl": float(state.get("realized_pnl", 0.0)),
    }


def _prediction_and_target_records(datasets, config, model, params, signal_time, previous_weights, policy_id, evidence_mode, key):
    predictions = _prediction_matrix(datasets, config.tickers, model, signal_time, signal_time + pd.Timedelta(config.bar_size))
    predictions = predictions.reindex(pd.DatetimeIndex([signal_time])).dropna()
    if predictions.empty:
        return [], None
    scores = predictions.loc[signal_time].to_numpy(dtype=float)
    weights = _positions_from_scores(scores, params)
    target_weights = {ticker: float(weights[idx]) for idx, ticker in enumerate(config.tickers)}
    previous_by_ticker = {ticker: float(previous_weights.get(ticker, 0.0)) for ticker in config.tickers}
    return [
        {
            "record_type": "prediction",
            "policy_id": policy_id,
            "evidence_mode": evidence_mode,
            "idempotency_key": key,
            "signal_time": pd.Timestamp(signal_time).isoformat(),
            "scores": {ticker: float(scores[idx]) for idx, ticker in enumerate(config.tickers)},
        },
        {
            "record_type": "target_weights",
            "policy_id": policy_id,
            "evidence_mode": evidence_mode,
            "idempotency_key": key,
            "signal_time": pd.Timestamp(signal_time).isoformat(),
            "weights": target_weights,
            "previous_weights": previous_by_ticker,
        },
    ], target_weights


def _target_unchanged(current, target):
    return all(abs(float(current.get(ticker, 0.0)) - float(weight)) < 1e-12 for ticker, weight in target.items())


def _pending_order_record(policy_id, evidence_mode, key, signal_time, fill_time, target_weights, current_weights, selection):
    return {
        "record_type": "pending_paper_order",
        "policy_id": policy_id,
        "evidence_mode": evidence_mode,
        "idempotency_key": key,
        "pending_order_key": key,
        "signal_time": pd.Timestamp(signal_time).isoformat(),
        "fill_time": pd.Timestamp(fill_time).isoformat(),
        "target_weights": target_weights,
        "previous_weights": current_weights,
        "selection_time": selection.get("selection_time"),
        "selected_model_family": selection.get("selected_model_family"),
        "allocation_parameters": selection.get("allocation_parameters"),
    }


def build_forward_decision(args, config, raw_tickers, datasets, records, policy_id, evidence_mode, persist_artifacts):
    anchor_time = _anchor_time(args)
    artifact_id = paper_policy_id(config, evidence_mode, args.force_reselect, args.force_reselect_reason)
    signal_time = _latest_common_signal_time(datasets, config.tickers, anchor_time, config.bar_size)
    batch = _fill_pending_orders(
        datasets,
        config,
        records,
        policy_id,
        evidence_mode,
        anchor_time,
        float(getattr(args, "slippage", 0.0)),
    )
    state_after_fills = reduce_paper_state(records + batch, config.tickers, config.cash)

    if signal_time is None:
        return {
            "policy_id": policy_id,
            "artifact_id": artifact_id,
            "evidence_mode": evidence_mode,
            "provisional": True,
            "signal_time": None,
            "records": batch,
            "state": state_after_fills,
            "no_action": None if batch else {"reason": "no_closed_signal_bar"},
            "git": git_state(),
        }

    mark = _mark_to_market_record(datasets, config, records + batch, policy_id, evidence_mode, signal_time)
    if mark is not None:
        batch.append(mark)

    key = idempotency_key(policy_id, evidence_mode, signal_time, args.force_reselect, args.force_reselect_reason)
    state = reduce_paper_state(records + batch, config.tickers, config.cash)
    if has_idempotency_key(records + batch, key):
        return {
            "policy_id": policy_id,
            "artifact_id": artifact_id,
            "evidence_mode": evidence_mode,
            "provisional": True,
            "appended": False,
            "signal_time": pd.Timestamp(signal_time).isoformat(),
            "idempotency_key": key,
            "records": batch,
            "state": state,
            "no_action": {"reason": "already_processed"},
            "git": git_state(),
        }

    reason, details = _data_no_action(
        raw_tickers,
        datasets,
        config.tickers,
        config.bar_size,
        signal_time,
        anchor_time,
        args.max_staleness_hours,
    )
    if reason is None:
        reason, details = _required_feature_staleness(args, config, anchor_time)
    if reason:
        batch.append(no_action_record(policy_id, evidence_mode, key, signal_time, reason, details))
        return {
            "policy_id": policy_id,
            "artifact_id": artifact_id,
            "evidence_mode": evidence_mode,
            "provisional": True,
            "signal_time": pd.Timestamp(signal_time).isoformat(),
            "idempotency_key": key,
            "records": batch,
            "state": reduce_paper_state(records + batch, config.tickers, config.cash),
            "no_action": {"reason": reason, "details": details},
            "git": git_state(),
        }

    selection = latest_selection(records + batch)
    selected_now = False
    strategy = None
    run_dir = os.path.join(args.computed_data_dir, "runs", policy_id)
    if _selection_due(selection, signal_time, config, args.force_reselect):
        if persist_artifacts:
            os.makedirs(run_dir, exist_ok=True)
        strategy = select_replay_strategy(
            datasets,
            config.tickers,
            config,
            signal_time,
            run_dir,
            len([record for record in records if record.get("record_type") == "strategy_selection"]),
            persist_model_artifacts=persist_artifacts,
        )
        selection = strategy_selection_record(strategy, config, policy_id, evidence_mode, key)
        selection["force_reselect"] = bool(args.force_reselect)
        selection["force_reselect_reason"] = args.force_reselect_reason
        batch.append(selection)
        selected_now = True
    elif not _rebalance_due(records + batch, signal_time, config, selection):
        return {
            "policy_id": policy_id,
            "artifact_id": artifact_id,
            "evidence_mode": evidence_mode,
            "provisional": True,
            "signal_time": signal_time.isoformat(),
            "idempotency_key": key,
            "records": batch,
            "state": reduce_paper_state(records + batch, config.tickers, config.cash),
            "no_action": {"reason": "not_yet_rebalance"},
            "git": git_state(),
        }

    if selection is None or selection.get("selected_model_family") is None:
        batch.append(no_action_record(policy_id, evidence_mode, key, signal_time, "validation_ineligible"))
        return {
            "policy_id": policy_id,
            "artifact_id": artifact_id,
            "evidence_mode": evidence_mode,
            "provisional": True,
            "signal_time": signal_time.isoformat(),
            "idempotency_key": key,
            "records": batch,
            "state": reduce_paper_state(records + batch, config.tickers, config.cash),
            "no_action": {"reason": "validation_ineligible"},
            "git": git_state(),
        }

    try:
        model = strategy["model"] if selected_now and strategy is not None else load_model_from_selection(selection)
    except FileNotFoundError:
        model = None
    if model is None:
        batch.append(no_action_record(policy_id, evidence_mode, key, signal_time, "no_selected_model"))
        return {
            "policy_id": policy_id,
            "artifact_id": artifact_id,
            "evidence_mode": evidence_mode,
            "provisional": True,
            "signal_time": signal_time.isoformat(),
            "idempotency_key": key,
            "records": batch,
            "state": reduce_paper_state(records + batch, config.tickers, config.cash),
            "no_action": {"reason": "no_selected_model"},
            "git": git_state(),
        }

    state = reduce_paper_state(records + batch, config.tickers, config.cash)
    records_for_signal, target_weights = _prediction_and_target_records(
        datasets,
        config,
        model,
        selection["allocation_parameters"],
        signal_time,
        state["current_weights"],
        policy_id,
        evidence_mode,
        key,
    )
    if target_weights is None:
        batch.append(no_action_record(policy_id, evidence_mode, key, signal_time, "no_selected_model"))
        return {
            "policy_id": policy_id,
            "artifact_id": artifact_id,
            "evidence_mode": evidence_mode,
            "provisional": True,
            "signal_time": signal_time.isoformat(),
            "idempotency_key": key,
            "records": batch,
            "state": reduce_paper_state(records + batch, config.tickers, config.cash),
            "no_action": {"reason": "no_selected_model"},
            "git": git_state(),
        }
    batch.extend(records_for_signal)
    if _target_unchanged(state["current_weights"], target_weights):
        batch.append(no_action_record(policy_id, evidence_mode, key, signal_time, "target_unchanged"))
    else:
        fill_time = pd.Timestamp(signal_time) + pd.Timedelta(config.bar_size)
        batch.append(_pending_order_record(
            policy_id,
            evidence_mode,
            key,
            signal_time,
            fill_time,
            target_weights,
            state["current_weights"],
            selection,
        ))

    return {
        "policy_id": policy_id,
        "artifact_id": artifact_id,
        "evidence_mode": evidence_mode,
        "provisional": True,
        "signal_time": signal_time.isoformat(),
        "idempotency_key": key,
        "records": batch,
        "state": reduce_paper_state(records + batch, config.tickers, config.cash),
        "git": git_state(),
    }


def _run_one_cycle(args, config, policy_id, path, evidence_mode, persist_artifacts):
    records = read_ledger(path)
    raw_tickers, datasets = _load_data(args, config)
    result = build_forward_decision(args, config, raw_tickers, datasets, records, policy_id, evidence_mode, persist_artifacts)
    if persist_artifacts:
        appended = append_ledger_records(path, result["records"])
        result["appended"] = appended > 0
        result["appended_records"] = appended
        result["ledger_path"] = path
        if appended:
            result["state"] = reduce_paper_state(read_ledger(path), config.tickers, config.cash)
    return result


def main(argv=None):
    args = parse_args(argv)
    _set_seed(args.seed)
    config = build_config(args)
    policy_id = paper_policy_id(config)
    path = ledger_path(args.computed_data_dir, policy_id)

    if args.state:
        records = read_ledger(path)
        print(json.dumps(jsonable({
            "policy_id": policy_id,
            "ledger_path": path,
            "evidence_mode": ONLINE_EVIDENCE_MODE,
            "state": reduce_paper_state(records, config.tickers, config.cash),
        }), indent=2, sort_keys=True))
        return

    if args.daemon:
        cycles = 0
        while True:
            result = _run_one_cycle(args, config, policy_id, path, ONLINE_EVIDENCE_MODE, True)
            print(json.dumps(jsonable({
                "policy_id": policy_id,
                "signal_time": result.get("signal_time"),
                "appended_records": result.get("appended_records", 0),
                "no_action": result.get("no_action"),
                "state": result.get("state"),
            }), sort_keys=True), flush=True)
            cycles += 1
            if args.max_cycles is not None and cycles >= args.max_cycles:
                return
            time.sleep(args.poll_seconds)

    evidence_mode = "paper_forward_dry_run" if args.dry_run else ONLINE_EVIDENCE_MODE
    result = _run_one_cycle(args, config, policy_id, path, evidence_mode, args.append)
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
