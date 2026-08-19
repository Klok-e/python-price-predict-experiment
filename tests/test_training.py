from __future__ import annotations

import io
from dataclasses import replace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from netgrowth.config import load_config  # noqa: E402
from netgrowth.market_data import build_market_state  # noqa: E402
from netgrowth.torch_backend import (  # noqa: E402
    TorchEvaluationBackend,
    _ensemble_target,
    _market_minutes,
    _training_funding,
    _training_rollout,
)
from netgrowth.training import (  # noqa: E402
    LinearDirectPolicy,
    TemporalConvPolicy,
    net_log_growth_loss,
    sunday_retraining_boundaries,
    training_episode_slices,
    update_drawdown_multiplier,
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


def test_prequential_refits_are_scheduled_on_sunday_utc() -> None:
    boundaries = sunday_retraining_boundaries(
        pd.Timestamp("2026-05-01 00:00", tz="UTC"),
        pd.Timestamp("2026-05-20 00:00", tz="UTC"),
    )

    assert boundaries == (
        pd.Timestamp("2026-05-01 00:00", tz="UTC"),
        pd.Timestamp("2026-05-03 00:00", tz="UTC"),
        pd.Timestamp("2026-05-10 00:00", tz="UTC"),
        pd.Timestamp("2026-05-17 00:00", tz="UTC"),
        pd.Timestamp("2026-05-20 00:00", tz="UTC"),
    )


def test_training_episodes_cover_expanding_revealed_history() -> None:
    episodes = training_episode_slices(revealed_bars=40_000, episode_bars=8_640, epochs=4)

    assert episodes[0] == slice(0, 8_640)
    assert episodes[-1] == slice(31_360, 40_000)
    assert all(item.stop - item.start == 8_640 for item in episodes)
    assert all(left.stop < right.stop for left, right in zip(episodes, episodes[1:], strict=False))


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


def test_drawdown_multiplier_updates_only_from_constraint_violation() -> None:
    multiplier = torch.tensor(0.4)

    unchanged = update_drawdown_multiplier(multiplier, torch.tensor(0.0))
    increased = update_drawdown_multiplier(multiplier, torch.tensor(0.1))

    assert unchanged.item() == pytest.approx(0.4)
    assert increased.item() == pytest.approx(0.401)


def test_declared_policies_emit_one_continuous_weight_per_instrument() -> None:
    market = torch.zeros((2, 16, 8))
    current = torch.zeros((2, 4))

    linear = LinearDirectPolicy(feature_count=8, ticker_count=4)
    temporal = TemporalConvPolicy(feature_count=8, ticker_count=4, width=32, receptive_bars=8)

    assert linear(market, current).shape == (2, 4)
    assert temporal(market, current).shape == (2, 4)
    assert torch.all(linear(market, current).abs() <= 0.5)
    assert torch.all(temporal(market, current).abs() <= 0.5)


def test_temporal_ensemble_members_receive_one_shared_marked_portfolio() -> None:
    observed: list[torch.Tensor] = []

    class CurrentAwarePolicy(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def target_from_encoded(self, encoded: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
            observed.append(current.clone())
            return current + self.offset

    target = _ensemble_target(
        (CurrentAwarePolicy(0.1), CurrentAwarePolicy(-0.1), CurrentAwarePolicy(0.0)),
        (torch.zeros(2), torch.ones(2), torch.full((2,), 2.0)),
        {"BTCUSDT": 0.25, "ETHUSDT": -0.2},
        ("BTCUSDT", "ETHUSDT"),
    )

    expected_current = torch.tensor([[0.25, -0.2]])
    assert len(observed) == 3
    assert all(torch.equal(current, expected_current) for current in observed)
    assert target == pytest.approx({"BTCUSDT": 0.25, "ETHUSDT": -0.2})


def test_training_policy_receives_actual_marked_current_portfolio() -> None:
    path = _training_rollout(
        torch.full((2, 1), 10.0, requires_grad=True),
        torch.zeros((1, 1)),
        torch.tensor([[0.1], [0.0]]),
        torch.tensor([[0.1], [0.0]]),
        torch.zeros((2, 1)),
        transaction_cost_rate=0.0,
        minimum_turnover=0.01,
    )

    assert path.current_portfolios[0, 0].item() == 0.0
    first_weight = path.weights[0, 0].item()
    assert path.current_portfolios[1, 0].item() == pytest.approx(first_weight * 1.1 / (1.0 + first_weight * 0.1))


def test_training_latency_return_belongs_to_the_pre_fill_portfolio() -> None:
    path = _training_rollout(
        torch.full((2, 1), 10.0, requires_grad=True),
        torch.zeros((1, 1)),
        torch.tensor([[1.0], [0.0]]),
        torch.zeros((2, 1)),
        torch.zeros((2, 1)),
        transaction_cost_rate=0.0,
        minimum_turnover=0.01,
    )

    assert path.simple_growth[0].item() == pytest.approx(0.0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="requires a ROCm GPU")
def test_full_training_episode_backward_is_stable_on_rocm() -> None:
    torch.manual_seed(17)
    device = "cuda"
    steps = 90 * 96
    logits = torch.randn(steps, 4, device=device, requires_grad=True)
    current_matrix = torch.randn(4, 4, device=device, requires_grad=True)
    latency = torch.randn(steps, 4, device=device) * 0.001
    returns = torch.randn(steps, 4, device=device) * 0.01
    funding = torch.zeros_like(returns)

    path = _training_rollout(
        logits,
        current_matrix,
        latency,
        returns,
        funding,
        transaction_cost_rate=0.0007,
        minimum_turnover=0.01,
    )
    (-path.simple_growth.sum()).backward()
    torch.cuda.synchronize()

    assert logits.grad is not None
    assert current_matrix.grad is not None


def test_training_funding_uses_event_mark_notional() -> None:
    from tests.test_market_data import dataset

    canonical = dataset()
    ticker = "BTCUSDT"
    event = canonical.instruments[ticker].funding.index[1]
    canonical.instruments[ticker].funding.loc[event] = 0.001
    canonical.instruments[ticker].perpetual.loc[event, "close"] = 150.0
    decision_time = event - pd.Timedelta(minutes=15)
    closes = pd.DataFrame(100.0, index=pd.DatetimeIndex([decision_time]), columns=canonical.tickers)

    effective = _training_funding(canonical, closes)

    assert effective.loc[decision_time, ticker] == pytest.approx(0.0015)


def test_historical_signal_minute_uses_only_the_just_closed_candle() -> None:
    from tests.test_market_data import dataset

    canonical = dataset()
    config = load_config("policy.toml")
    signal_time = canonical.instruments["BTCUSDT"].perpetual.index[-3]
    future_close = canonical.instruments["BTCUSDT"].perpetual.loc[signal_time, "close"]
    previous_close = canonical.instruments["BTCUSDT"].perpetual.loc[signal_time - pd.Timedelta(minutes=1), "close"]

    rows = _market_minutes(
        canonical,
        signal_time,
        signal_time + pd.Timedelta(minutes=2),
        config,
    )

    assert pd.Timestamp(rows[0].timestamp) == signal_time
    assert rows[0].mark_prices["BTCUSDT"] == previous_close
    assert rows[0].mark_prices["BTCUSDT"] != future_close
    assert (rows[0].historical_open or {})["BTCUSDT"] == canonical.instruments["BTCUSDT"].perpetual.loc[
        signal_time, "open"
    ]


def test_paper_backend_fits_selected_architecture_and_emits_current_portfolio_target(tmp_path) -> None:
    from tests.test_market_data import dataset

    backend = TorchEvaluationBackend(tmp_path)
    canonical = dataset()
    config = replace(load_config("policy.toml"), training_episode_days=1, training_epochs=1)
    buffer = io.BytesIO()
    torch.save({"kind": "linear", "metadata": {}, "state_dicts": []}, buffer)
    observed_at = build_market_state(canonical).index[-1].to_pydatetime()

    decision = backend.paper(
        canonical,
        config,
        "cpu",
        validated_model=buffer.getvalue(),
        fitted_model=None,
        observed_at=observed_at,
        current_weights=dict.fromkeys(config.tickers, 0.0),
    )

    assert set(decision.target_weights) == set(config.tickers)
    assert sum(abs(weight) for weight in decision.target_weights.values()) <= 1.0
    assert all(abs(weight) <= 0.5 for weight in decision.target_weights.values())
    assert decision.refitted is True
    assert decision.model_bytes
