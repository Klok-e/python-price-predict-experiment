import numpy as np
import pandas as pd
import pytest

from utils.experiment import TradingContract
from utils.util import generate_labels_for_supervised, validate_ohlc_bars


def make_bars(opens, highs, lows, closes):
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
        },
        index=pd.date_range("2024-01-01", periods=len(opens), freq="1min"),
    )


def test_barrier_label_positive_when_take_profit_hits_first():
    bars = make_bars(
        opens=[100, 100, 100, 100],
        highs=[100, 100.5, 101.2, 100],
        lows=[100, 99.5, 99.8, 100],
        closes=[100, 100, 100, 100],
    )
    contract = TradingContract(lookahead_steps=2, stop_loss_percent=1, take_profit_percent=1)

    labels = generate_labels_for_supervised(bars, contract=contract)

    assert labels.iloc[0, 0] == 1


def test_barrier_label_negative_when_stop_loss_hits_first():
    bars = make_bars(
        opens=[100, 100, 100, 100],
        highs=[100, 100.5, 101.2, 100],
        lows=[100, 98.9, 99.8, 100],
        closes=[100, 100, 100, 100],
    )
    contract = TradingContract(lookahead_steps=2, stop_loss_percent=1, take_profit_percent=1)

    labels = generate_labels_for_supervised(bars, contract=contract)

    assert labels.iloc[0, 0] == 0


def test_barrier_label_negative_when_both_barriers_hit_same_candle():
    bars = make_bars(
        opens=[100, 100, 100],
        highs=[100, 101.2, 100],
        lows=[100, 98.9, 100],
        closes=[100, 100, 100],
    )
    contract = TradingContract(lookahead_steps=1, stop_loss_percent=1, take_profit_percent=1)

    labels = generate_labels_for_supervised(bars, contract=contract)

    assert labels.iloc[0, 0] == 0


def test_barrier_label_uses_next_open_as_entry_anchor():
    bars = make_bars(
        opens=[100, 200, 200],
        highs=[100, 201, 200],
        lows=[100, 199, 200],
        closes=[100, 200, 200],
    )
    contract = TradingContract(lookahead_steps=1, stop_loss_percent=1, take_profit_percent=1)

    labels = generate_labels_for_supervised(bars, contract=contract)

    assert labels.iloc[0, 0] == 0


def test_barrier_labels_drop_incomplete_lookahead_rows():
    bars = make_bars(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 100, 100],
        lows=[100, 100, 100, 100],
        closes=[100, 100, 100, 100],
    )
    contract = TradingContract(lookahead_steps=2, stop_loss_percent=1, take_profit_percent=1)

    labels = generate_labels_for_supervised(bars, contract=contract)

    assert list(labels.index) == list(bars.index[:2])


def test_validate_ohlc_bars_rejects_missing_minutes():
    bars = make_bars(
        opens=[100, 100, 100],
        highs=[100, 100, 100],
        lows=[100, 100, 100],
        closes=[100, 100, 100],
    )
    bars = bars.drop(bars.index[1])

    with pytest.raises(ValueError, match="missing or irregular bars"):
        validate_ohlc_bars(bars, ticker_name="TEST")


def test_validate_ohlc_bars_rejects_invalid_ohlc():
    bars = make_bars(
        opens=[100],
        highs=[99],
        lows=[100],
        closes=[100],
    )

    with pytest.raises(ValueError, match="invalid OHLC"):
        validate_ohlc_bars(bars, ticker_name="TEST")
