from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from typing import Any

import pandas as pd

from utils.experiment_runner import jsonable


LEDGER_FILENAME = "paper_ledger.jsonl"
LEDGER_RECORD_TYPES = {
    "strategy_selection",
    "prediction",
    "target_weights",
    "simulated_order",
    "simulated_fill",
    "equity_snapshot",
    "pending_paper_order",
    "online_paper_fill",
    "mark_to_market",
    "no_action",
}


def _config_payload(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    return config


def paper_policy_id(
    config: Any,
    evidence_mode: str | None = None,
    force_reselect: bool = False,
    force_reselect_reason: str | None = None,
) -> str:
    payload = {
        "paper_policy_schema": "v1",
        "config": _config_payload(config),
        "evidence_mode": evidence_mode,
        "force_reselect": bool(force_reselect),
        "force_reselect_reason": force_reselect_reason,
    }
    raw = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def ledger_path(computed_data_dir: str, policy_id: str) -> str:
    return os.path.join(computed_data_dir, "runs", policy_id, LEDGER_FILENAME)


def idempotency_key(
    policy_id: str,
    evidence_mode: str,
    signal_time: Any,
    force_reselect: bool = False,
    force_reselect_reason: str | None = None,
    suffix: str | None = None,
) -> str:
    payload = {
        "policy_id": policy_id,
        "evidence_mode": evidence_mode,
        "signal_time": pd.Timestamp(signal_time).isoformat(),
        "force_reselect": bool(force_reselect),
        "force_reselect_reason": force_reselect_reason,
        "suffix": suffix,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_ledger(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record_type = record.get("record_type")
            if record_type not in LEDGER_RECORD_TYPES:
                raise ValueError(f"Unknown paper ledger record_type on line {line_number}: {record_type}")
            records.append(record)
    return records


def has_idempotency_key(records: list[dict[str, Any]], key: str) -> bool:
    return any(record.get("idempotency_key") == key for record in records)


def append_ledger_records(path: str, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    existing = read_ledger(path)
    key = records[0].get("idempotency_key")
    if key and has_idempotency_key(existing, key):
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        for record in records:
            record_type = record.get("record_type")
            if record_type not in LEDGER_RECORD_TYPES:
                raise ValueError(f"Unknown paper ledger record_type: {record_type}")
            file.write(json.dumps(jsonable(record), sort_keys=True))
            file.write("\n")
    return len(records)


def latest_selection(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    selections = [record for record in records if record.get("record_type") == "strategy_selection"]
    if not selections:
        return None
    return max(selections, key=lambda record: pd.Timestamp(record["selection_time"]))


def pending_paper_orders(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filled = {
        record.get("pending_order_key")
        for record in records
        if record.get("record_type") == "online_paper_fill" and record.get("pending_order_key")
    }
    pending = []
    for record in records:
        if record.get("record_type") != "pending_paper_order":
            continue
        key = record.get("pending_order_key") or record.get("idempotency_key")
        if key not in filled:
            pending.append(record)
    return pending


def _zero_tickers(tickers: list[str] | tuple[str, ...]) -> dict[str, float]:
    return {ticker: 0.0 for ticker in tickers}


def _weights_from_units(cash: float, units: dict[str, float], marks: dict[str, float], tickers) -> tuple[float, dict[str, float]]:
    position_values = {ticker: units.get(ticker, 0.0) * marks.get(ticker, 0.0) for ticker in tickers}
    equity = float(cash + sum(position_values.values()))
    if equity == 0.0:
        return equity, _zero_tickers(tickers)
    weights = {ticker: float(position_values[ticker] / equity) for ticker in tickers}
    return equity, weights


def reduce_paper_state(records: list[dict[str, Any]], tickers: list[str] | tuple[str, ...], starting_cash: float) -> dict[str, Any]:
    state = {
        "cash": float(starting_cash),
        "equity": float(starting_cash),
        "units": _zero_tickers(tickers),
        "marks": _zero_tickers(tickers),
        "current_weights": _zero_tickers(tickers),
        "long_exposure": 0.0,
        "short_exposure": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "total_pnl": 0.0,
        "turnover": 0.0,
        "accumulated_fees": 0.0,
        "pending_order_count": 0,
        "last_signal_time": None,
        "last_selection_time": None,
        "last_fill_time": None,
        "last_mark_time": None,
    }
    for record in records:
        record_type = record.get("record_type")
        if "signal_time" in record and record["signal_time"] is not None:
            state["last_signal_time"] = record["signal_time"]
        if record_type == "strategy_selection":
            state["last_selection_time"] = record.get("selection_time")
        if record_type == "simulated_fill":
            state["cash"] = float(record.get("cash", state["cash"]))
            state["equity"] = float(record.get("equity", state["equity"]))
            state["turnover"] += float(record.get("turnover", 0.0))
            state["accumulated_fees"] += float(record.get("fee", 0.0))
            weights = record.get("weights")
            if weights is not None:
                state["current_weights"] = {ticker: float(weights.get(ticker, 0.0)) for ticker in tickers}
        if record_type == "online_paper_fill":
            state["cash"] = float(record.get("cash", state["cash"]))
            units = record.get("units")
            if units is not None:
                state["units"] = {ticker: float(units.get(ticker, 0.0)) for ticker in tickers}
            prices = record.get("fill_prices") or record.get("mark_prices") or {}
            state["marks"].update({ticker: float(prices.get(ticker, state["marks"].get(ticker, 0.0))) for ticker in tickers})
            state["turnover"] += float(record.get("turnover", 0.0))
            state["accumulated_fees"] += float(record.get("fee", 0.0))
            state["last_fill_time"] = record.get("fill_time")
            state["equity"] = float(record.get("equity", state["equity"]))
            if "weights" in record:
                state["current_weights"] = {ticker: float(record["weights"].get(ticker, 0.0)) for ticker in tickers}
            else:
                state["equity"], state["current_weights"] = _weights_from_units(
                    state["cash"], state["units"], state["marks"], tickers
                )
        if record_type in ("equity_snapshot", "mark_to_market"):
            state["cash"] = float(record.get("cash", state["cash"]))
            state["equity"] = float(record.get("equity", state["equity"]))
            if "units" in record:
                state["units"] = {ticker: float(record["units"].get(ticker, 0.0)) for ticker in tickers}
            if "mark_prices" in record:
                state["marks"] = {ticker: float(record["mark_prices"].get(ticker, state["marks"].get(ticker, 0.0))) for ticker in tickers}
            state["current_weights"] = {
                ticker: float(record.get("weights", {}).get(ticker, state["current_weights"].get(ticker, 0.0)))
                for ticker in tickers
            }
            state["realized_pnl"] = float(record.get("realized_pnl", state["realized_pnl"]))
            state["unrealized_pnl"] = float(record.get("unrealized_pnl", state["unrealized_pnl"]))
            if record_type == "mark_to_market":
                state["last_mark_time"] = record.get("mark_time")
    state["pending_order_count"] = len(pending_paper_orders(records))
    weights = state["current_weights"]
    state["long_exposure"] = float(sum(weight for weight in weights.values() if weight > 0))
    state["short_exposure"] = float(sum(abs(weight) for weight in weights.values() if weight < 0))
    state["total_pnl"] = float(state["equity"] - starting_cash)
    return state
