from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from netgrowth.torch_backend import TorchEvaluationBackend, _average_targets  # noqa: E402
from netgrowth.training import (  # noqa: E402
    LinearDirectPolicy,
    TemporalConvPolicy,
    net_log_growth_loss,
    walk_forward_folds,
)


def test_walk_forward_has_twelve_disjoint_ninety_day_folds_with_purged_training() -> None:
    folds = walk_forward_folds(
        common_start=pd.Timestamp("2020-09-01", tz="UTC"),
        holdout_start=pd.Timestamp("2026-05-01", tz="UTC"),
        fold_count=12,
        fold_days=90,
        purge_days=7,
    )

    assert len(folds) == 12
    assert all((fold.validation_end - fold.validation_start).days == 90 for fold in folds)
    assert all(folds[i].validation_end == folds[i + 1].validation_start for i in range(11))
    assert all(fold.training_end <= fold.validation_start - pd.Timedelta(days=7) for fold in folds)
    assert folds[-1].validation_end == pd.Timestamp("2026-05-01", tz="UTC")


def test_net_log_growth_objective_rewards_profit_after_turnover_and_funding() -> None:
    profitable_weights = torch.tensor([[0.0], [0.5], [0.5]], dtype=torch.float64)
    cash_weights = torch.zeros_like(profitable_weights)
    returns = torch.tensor([[0.0], [0.02], [0.02]], dtype=torch.float64)
    funding = torch.tensor([[0.0], [0.001], [0.001]], dtype=torch.float64)

    profitable = net_log_growth_loss(
        profitable_weights,
        returns,
        funding,
        transaction_cost_rate=0.0007,
        drawdown_limit=0.20,
        constraint_multiplier=torch.tensor(0.0),
    )
    cash = net_log_growth_loss(
        cash_weights,
        returns,
        funding,
        transaction_cost_rate=0.0007,
        drawdown_limit=0.20,
        constraint_multiplier=torch.tensor(0.0),
    )

    assert profitable < cash


def test_declared_policies_emit_one_continuous_weight_per_instrument() -> None:
    market = torch.zeros((2, 16, 8))
    current = torch.zeros((2, 4))

    linear = LinearDirectPolicy(feature_count=8, ticker_count=4)
    temporal = TemporalConvPolicy(feature_count=8, ticker_count=4, width=32, receptive_bars=8)

    assert linear(market, current).shape == (2, 4)
    assert temporal(market, current).shape == (2, 4)
    assert torch.all(linear(market, current).abs() <= 0.5)
    assert torch.all(temporal(market, current).abs() <= 0.5)


def test_temporal_ensemble_averages_member_weights_before_replay() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    members = [
        {timestamp: {"BTCUSDT": 0.5, "ETHUSDT": 0.1}},
        {timestamp: {"BTCUSDT": -0.5, "ETHUSDT": 0.2}},
        {timestamp: {"BTCUSDT": 0.5, "ETHUSDT": -0.3}},
    ]

    averaged = _average_targets(members, ("BTCUSDT", "ETHUSDT"))

    assert averaged[timestamp] == pytest.approx({"BTCUSDT": 1 / 6, "ETHUSDT": 0.0})


def test_paper_backend_refuses_to_relabel_archive_replay_as_forward_proof(tmp_path) -> None:
    backend = TorchEvaluationBackend(tmp_path)

    with pytest.raises(RuntimeError, match="contemporaneous midpoint"):
        backend.paper(None, None, "cpu")  # type: ignore[arg-type]
