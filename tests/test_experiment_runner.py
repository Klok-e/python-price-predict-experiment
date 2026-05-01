import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from run_experiment import build_contract, parse_args
from utils.experiment import TradingContract
from utils.experiment_runner import (
    backtest_model_ticker,
    backtest_no_trade_ticker,
    summarize_backtest_results,
)
from utils.util import load_cached_ohlc_data


class ConstantLogitModel(nn.Module):
    def __init__(self, logit=10):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))
        self.logit = logit

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.logit, device=x.device)


def make_ticker_tuple(rows):
    index = pd.date_range("2024-01-01", periods=len(rows), freq="1min")
    original = pd.DataFrame(rows, index=index)
    scaled = pd.DataFrame({"feature": np.arange(len(rows), dtype=np.float32)}, index=index)
    labels = pd.DataFrame({"Label": np.zeros(len(rows), dtype=np.float32)}, index=index)
    return scaled, original, labels, None, "TESTUSDT"


def test_run_experiment_defaults_are_validation_tail_report_defaults():
    args = parse_args([])
    contract = build_contract(args)

    assert args.backtest_split == "validation"
    assert args.validation_days == 14
    assert args.test_days == 14
    assert args.backtest_days == 14
    assert args.start_date == "2023-04-01"
    assert args.window_size == 64
    assert args.epochs == 5
    assert contract.entry_probability_threshold() == pytest.approx(0.7502502502502481)


def test_run_experiment_does_not_download_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run download_data.py first"):
        load_cached_ohlc_data(tmp_path, tickers=("MISSINGUSDT",))


def test_no_trade_baseline_is_flat_and_zero_trade():
    ticker = make_ticker_tuple([
        {"Open": 100, "High": 100, "Low": 100, "Close": 100},
        {"Open": 101, "High": 101, "Low": 101, "Close": 101},
        {"Open": 102, "High": 102, "Low": 102, "Close": 102},
    ])

    result = backtest_no_trade_ticker(ticker, window_size=1, backtest_days=1, cash=1000)
    summary = summarize_backtest_results([result])

    assert result["trades"].empty
    assert (result["equity"] == 1000).all()
    assert summary["portfolio"]["trades"] == 0
    assert summary["portfolio"]["cumulative_return"] == 0


def test_model_backtest_uses_stop_loss_first_when_both_barriers_hit_same_candle():
    ticker = make_ticker_tuple([
        {"Open": 100, "High": 100, "Low": 100, "Close": 100},
        {"Open": 100, "High": 101.5, "Low": 98.5, "Close": 100},
    ])
    contract = TradingContract(lookahead_steps=1, stop_loss_percent=1, take_profit_percent=1, commission=0)
    model = ConstantLogitModel()

    result = backtest_model_ticker(
        ticker,
        model,
        contract,
        window_size=1,
        threshold=0.5,
        backtest_days=1,
        cash=1000,
    )

    assert len(result["trades"]) == 1
    trade = result["trades"].iloc[0]
    assert trade["ExitReason"] == "stop_loss"
    assert trade["ExitPrice"] == pytest.approx(99)
    assert result["equity"].iloc[-1] == pytest.approx(990)
