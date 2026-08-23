from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from netgrowth.config import load_config
from netgrowth.market_data import CanonicalDataset, InMemoryMarketData, InstrumentData
from netgrowth.torch_backend import TorchEvaluationBackend
from netgrowth.workflow import NetGrowthWorkflow

TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


def positive_development_pattern() -> CanonicalDataset:
    index = pd.date_range("2026-01-01", periods=24 * 60 * 24, freq="min", tz="UTC")
    trend = 100.0 * np.exp(np.arange(len(index)) * np.log1p(0.002) / 15.0)
    instruments = {}
    for ticker in TICKERS:
        bars = pd.DataFrame(
            {
                "open": trend,
                "high": trend * 1.0001,
                "low": trend * 0.9999,
                "close": trend,
                "volume": 100.0,
                "taker_buy_volume": 75.0,
                "trades": 100,
            },
            index=index,
        )
        instruments[ticker] = InstrumentData(
            perpetual=bars,
            spot=bars * 0.999,
            funding=pd.Series(0.0, index=index[::480], name="funding_rate"),
            open_interest=pd.Series(1_000.0, index=index[::5], name="open_interest"),
            premium=pd.Series(0.001, index=index, name="premium"),
        )
    return CanonicalDataset(instruments=instruments, tickers=TICKERS)


def test_direct_policy_trains_and_validates_positive_development_evidence(tmp_path) -> None:
    canonical = positive_development_pattern()
    config = replace(
        load_config("policy.toml"),
        validation_folds=2,
        fold_days=2,
        development_evidence_end=date(2026, 1, 25),
        training_episode_days=7,
        training_epochs=4,
    )
    workflow = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        backend=TorchEvaluationBackend(tmp_path),
        output_directory=tmp_path,
        device="cpu",
    )

    result = workflow.validate()

    report = pd.read_json(result.artifact_directory / "report.json", typ="series")
    assert result.summary.startswith("Validated Policy Protocol")
    assert report["compounded_net_return"] > 0.0
    assert report["maximum_drawdown"] <= config.drawdown_limit
    assert report["transaction_cost"] > 0.0
    assert {path.name for path in result.artifact_directory.iterdir()} == {
        "manifest.json",
        "report.json",
        "equity.csv",
        "trades.csv",
        "model.pt",
    }
