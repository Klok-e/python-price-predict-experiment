from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import torch

from utils.trading_metrics import calculate_metrics
from utils.util import save_pickle, stop_loss_price, take_profit_price


DEFAULT_CASH = 1_000_000
DEFAULT_MODEL_KWARGS = {"linear_arch": [128, 64]}


def jsonable(value: Any):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def parse_linear_arch(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_stats(split):
    rows = sum(len(ticker[0]) for ticker in split)
    labels = sum(len(ticker[2]) for ticker in split)
    positives = sum(float(ticker[2]["Label"].sum()) for ticker in split)
    tickers = {}
    for scaled, original, labels_df, _, ticker in split:
        tickers[ticker] = {
            "rows": len(scaled),
            "labels": len(labels_df),
            "positives": float(labels_df["Label"].sum()) if len(labels_df) else 0.0,
            "positive_rate": float(labels_df["Label"].mean()) if len(labels_df) else 0.0,
            "start": original.index.min().isoformat() if len(original) else None,
            "end": original.index.max().isoformat() if len(original) else None,
        }

    return {
        "rows": rows,
        "labels": labels,
        "positives": positives,
        "positive_rate": positives / labels if labels else 0.0,
        "tickers": tickers,
    }


def raw_data_stats(raw_tickers):
    stats = {}
    for df, ticker in raw_tickers:
        index = pd.to_datetime(df["Open time"]) if "Open time" in df.columns else pd.to_datetime(df.index)
        stats[ticker] = {
            "rows": len(df),
            "start": index.min().isoformat() if len(index) else None,
            "end": index.max().isoformat() if len(index) else None,
        }
    return stats


def git_state():
    def run_git(args):
        try:
            result = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = run_git(["status", "--short"]) or ""
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "changed_files": [line[3:] for line in status.splitlines() if len(line) >= 4],
    }


def metric_dict(equity, trades_count, start_cash):
    cr, mer, mpb, appt, sr = calculate_metrics(equity, trades_count, start_cash)
    return {
        "cumulative_return": float(cr),
        "max_earning_rate": float(mer),
        "maximum_pullback": float(mpb),
        "average_profitability_per_trade": float(appt),
        "sharpe_ratio": float(sr),
        "trades": int(trades_count),
        "start_cash": float(start_cash),
        "end_equity": float(equity.iloc[-1]) if len(equity) else float(start_cash),
    }


def _ticker_bounds(original, window_size, backtest_days):
    if original.empty:
        raise ValueError("Cannot backtest an empty ticker split")

    evaluation_start = original.index.max() - pd.Timedelta(days=backtest_days)
    first_signal_pos = max(window_size - 1, int(original.index.searchsorted(evaluation_start)))
    first_entry_pos = first_signal_pos + 1
    if first_entry_pos >= len(original):
        raise ValueError("Backtest window has no tradable bars after warmup")
    return first_signal_pos, first_entry_pos, len(original) - 1


def _model_probability(model, window):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(window).to(device)
        logits = model(tensor)
        return torch.sigmoid(logits).detach().cpu().reshape(-1)[0].item()


def backtest_model_ticker(
        ticker_tuple,
        model,
        contract,
        window_size,
        threshold,
        backtest_days,
        cash=DEFAULT_CASH,
):
    scaled, original, _, _, ticker = ticker_tuple
    first_signal_pos, first_entry_pos, last_pos = _ticker_bounds(original, window_size, backtest_days)

    cash_balance = float(cash)
    quantity = 0.0
    entry_price = None
    entry_time = None
    entry_bar = None
    pending_entry = None
    trades = []
    equity_values = []
    equity_index = []

    for pos in range(first_signal_pos, last_pos + 1):
        row = original.iloc[pos]
        timestamp = original.index[pos]

        if pending_entry is not None:
            entry_price = float(row["Open"])
            entry_time = timestamp
            entry_bar = pos
            quantity = cash_balance / (entry_price * (1 + contract.commission))
            cash_balance = 0.0
            pending_entry["entry_time"] = entry_time
            pending_entry["entry_price"] = entry_price
            pending_entry = None

        if quantity > 0:
            sl_price = stop_loss_price(entry_price, contract.stop_loss_percent)
            tp_price = take_profit_price(entry_price, contract.take_profit_percent)
            exit_price = None
            exit_reason = None
            if float(row["Low"]) <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
            elif float(row["High"]) >= tp_price:
                exit_price = tp_price
                exit_reason = "take_profit"

            if exit_price is not None:
                cash_balance = quantity * exit_price * (1 - contract.commission)
                trades.append({
                    "Ticker": ticker,
                    "EntryTime": entry_time,
                    "ExitTime": timestamp,
                    "EntryBar": entry_bar,
                    "ExitBar": pos,
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "ExitReason": exit_reason,
                    "Size": quantity,
                    "ReturnPct": (
                        (exit_price * (1 - contract.commission))
                        - (entry_price * (1 + contract.commission))
                    ) / (entry_price * (1 + contract.commission)) * 100,
                })
                quantity = 0.0
                entry_price = None
                entry_time = None
                entry_bar = None

        if pos >= first_entry_pos:
            if quantity > 0:
                equity = quantity * float(row["Close"]) * (1 - contract.commission)
            else:
                equity = cash_balance
            equity_values.append(equity)
            equity_index.append(timestamp)

        if quantity == 0 and pending_entry is None and pos + 1 <= last_pos:
            if pos >= first_signal_pos:
                window = scaled.iloc[pos - window_size + 1:pos + 1].to_numpy(dtype=np.float32).reshape(
                    1, window_size, -1
                )
                probability = _model_probability(model, window)
                if probability > threshold:
                    pending_entry = {
                        "signal_time": timestamp,
                        "probability": probability,
                    }

    equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index), name="Equity")
    trades_df = pd.DataFrame(trades)
    return {
        "ticker": ticker,
        "trades": trades_df,
        "equity": equity,
        "start_cash": float(cash),
        "first_tradable_time": equity.index.min() if len(equity) else None,
    }


def backtest_buy_and_hold_ticker(ticker_tuple, window_size, backtest_days, contract, cash=DEFAULT_CASH):
    _, original, _, _, ticker = ticker_tuple
    _, first_entry_pos, last_pos = _ticker_bounds(original, window_size, backtest_days)
    entry_price = float(original.iloc[first_entry_pos]["Open"])
    quantity = cash / (entry_price * (1 + contract.commission))
    data = original.iloc[first_entry_pos:last_pos + 1]
    equity = data["Close"].astype(float) * quantity * (1 - contract.commission)
    equity.name = "Equity"
    trades = pd.DataFrame([{
        "Ticker": ticker,
        "EntryTime": data.index[0],
        "ExitTime": data.index[-1],
        "EntryBar": first_entry_pos,
        "ExitBar": last_pos,
        "EntryPrice": entry_price,
        "ExitPrice": float(data.iloc[-1]["Close"]),
        "ExitReason": "hold_end",
        "Size": quantity,
        "ReturnPct": (float(equity.iloc[-1]) - cash) / cash * 100,
    }])
    return {
        "ticker": ticker,
        "trades": trades,
        "equity": equity,
        "start_cash": float(cash),
        "first_tradable_time": equity.index.min(),
    }


def backtest_no_trade_ticker(ticker_tuple, window_size, backtest_days, cash=DEFAULT_CASH):
    _, original, _, _, ticker = ticker_tuple
    _, first_entry_pos, last_pos = _ticker_bounds(original, window_size, backtest_days)
    index = original.iloc[first_entry_pos:last_pos + 1].index
    equity = pd.Series(float(cash), index=index, name="Equity")
    return {
        "ticker": ticker,
        "trades": pd.DataFrame(),
        "equity": equity,
        "start_cash": float(cash),
        "first_tradable_time": equity.index.min(),
    }


def portfolio_equity(results):
    series = []
    non_empty_results = []
    for result in results:
        equity = result["equity"]
        if equity.empty:
            continue
        series.append(equity)
        non_empty_results.append(result)
    if not series:
        return pd.Series(dtype=float, name="Equity")

    combined = pd.concat(series, axis=1).sort_index()
    for idx, result in enumerate(non_empty_results):
        combined.iloc[:, idx] = combined.iloc[:, idx].ffill().fillna(result["start_cash"])
    return combined.sum(axis=1).rename("Equity")


def summarize_backtest_results(results):
    ticker_metrics = {}
    total_trades = 0
    total_start_cash = 0
    for result in results:
        trades_count = len(result["trades"])
        total_trades += trades_count
        total_start_cash += result["start_cash"]
        ticker_metrics[result["ticker"]] = metric_dict(
            result["equity"],
            trades_count,
            result["start_cash"],
        )

    portfolio = portfolio_equity(results)
    return {
        "portfolio": metric_dict(portfolio, total_trades, total_start_cash),
        "tickers": ticker_metrics,
    }


def save_backtest_artifacts(results, output_dir, strategy_name):
    for result in results:
        equity_curve = pd.DataFrame({"Equity": result["equity"]})
        save_pickle(
            (result["trades"], equity_curve),
            f"{output_dir}/{strategy_name}_{result['ticker']}.pkl",
        )
