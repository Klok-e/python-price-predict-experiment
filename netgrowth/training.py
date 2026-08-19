"""Small direct-policy models, Net Log Growth objective, and chronological folds."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


def constrain_weights(raw: Tensor) -> Tensor:
    concentrated = 0.5 * torch.tanh(raw)
    gross = concentrated.abs().sum(dim=-1, keepdim=True)
    scale = torch.clamp(1.0 / gross.clamp_min(1e-12), max=1.0)
    return concentrated * scale


class LinearDirectPolicy(nn.Module):
    def __init__(self, feature_count: int, ticker_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count + ticker_count, ticker_count)

    def forward(self, market_state: Tensor, current_weights: Tensor) -> Tensor:
        return self.target_from_encoded(market_state[:, -1, :], current_weights)

    def encode_trajectory(self, market_state: Tensor) -> Tensor:
        return market_state.squeeze(0)

    def target_from_encoded(self, encoded: Tensor, current_weights: Tensor) -> Tensor:
        return constrain_weights(self.linear(torch.cat((encoded, current_weights), dim=-1)))

    def rollout_components(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        feature_count = encoded.shape[-1]
        market_logits = functional.linear(encoded, self.linear.weight[:, :feature_count], self.linear.bias)
        return market_logits, self.linear.weight[:, feature_count:]

    def trajectory(self, market_state: Tensor, current_weights: Tensor) -> Tensor:
        return self.target_from_encoded(self.encode_trajectory(market_state), current_weights)


class TemporalConvPolicy(nn.Module):
    def __init__(self, feature_count: int, ticker_count: int, *, width: int, receptive_bars: int) -> None:
        super().__init__()
        if width not in (32, 64):
            raise ValueError("temporal width must be 32 or 64")
        if receptive_bars < 1:
            raise ValueError("receptive field must contain at least one bar")
        self.receptive_bars = receptive_bars
        layer_count = max(1, ceil(log2(max(1, receptive_bars - 1) / 2 + 1)))
        layers: list[nn.Module] = []
        incoming = feature_count
        self.dilations = tuple(2**index for index in range(layer_count))
        for dilation in self.dilations:
            layers.append(nn.Conv1d(incoming, width, kernel_size=3, dilation=dilation))
            incoming = width
        self.convolutions = nn.ModuleList(layers)
        self.output = nn.Linear(width + ticker_count, ticker_count)

    def _encode(self, market_state: Tensor) -> Tensor:
        sequence = market_state.transpose(1, 2)
        for convolution, dilation in zip(self.convolutions, self.dilations, strict=True):
            sequence = functional.gelu(convolution(functional.pad(sequence, (2 * dilation, 0))))
        return sequence.transpose(1, 2)

    def forward(self, market_state: Tensor, current_weights: Tensor) -> Tensor:
        encoded = self._encode(market_state)[:, -1, :]
        return self.target_from_encoded(encoded, current_weights)

    def encode_trajectory(self, market_state: Tensor) -> Tensor:
        return self._encode(market_state).squeeze(0)

    def target_from_encoded(self, encoded: Tensor, current_weights: Tensor) -> Tensor:
        return constrain_weights(self.output(torch.cat((encoded, current_weights), dim=-1)))

    def rollout_components(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        feature_count = encoded.shape[-1]
        market_logits = functional.linear(encoded, self.output.weight[:, :feature_count], self.output.bias)
        return market_logits, self.output.weight[:, feature_count:]

    def trajectory(self, market_state: Tensor, current_weights: Tensor) -> Tensor:
        return self.target_from_encoded(self.encode_trajectory(market_state), current_weights)


def _growth_and_drawdown_violation(
    target_weights: Tensor,
    asset_returns: Tensor,
    funding_rates: Tensor,
    *,
    transaction_cost_rate: float,
    drawdown_limit: float,
    turnover: Tensor | None = None,
    simple_growth: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if target_weights.shape != asset_returns.shape or target_weights.shape != funding_rates.shape:
        raise ValueError("weights, returns, and funding must have identical time-by-ticker shape")
    if turnover is None:
        previous = torch.cat((torch.zeros_like(target_weights[:1]), target_weights[:-1]), dim=0)
        turnover = (target_weights - previous).abs().sum(dim=-1)
    elif turnover.shape != target_weights.shape[:1]:
        raise ValueError("turnover must contain one value per portfolio transition")
    if simple_growth is None:
        held_return = (target_weights * asset_returns).sum(dim=-1)
        funding = (target_weights * funding_rates).sum(dim=-1)
        simple_growth = held_return - funding - transaction_cost_rate * turnover
    elif simple_growth.shape != target_weights.shape[:1]:
        raise ValueError("simple growth must contain one value per portfolio transition")
    net_logs = torch.log1p(simple_growth.clamp_min(-0.999999))
    equity = torch.exp(torch.cumsum(net_logs, dim=0))
    peaks = torch.cummax(equity, dim=0).values.clamp_min(1e-12)
    maximum_drawdown = torch.max(1.0 - equity / peaks)
    violation = torch.relu(maximum_drawdown - drawdown_limit)
    return net_logs, violation


def update_drawdown_multiplier(multiplier: Tensor, violation: Tensor, *, step_size: float = 0.01) -> Tensor:
    """Perform one non-negative augmented-Lagrangian dual update."""
    return (multiplier + step_size * violation.detach()).clamp_min(0.0)


def drawdown_constraint_violation(
    target_weights: Tensor,
    asset_returns: Tensor,
    funding_rates: Tensor,
    *,
    transaction_cost_rate: float,
    drawdown_limit: float,
    turnover: Tensor | None = None,
    simple_growth: Tensor | None = None,
) -> Tensor:
    """Return the realized training-episode drawdown excess after costs and funding."""
    _, violation = _growth_and_drawdown_violation(
        target_weights,
        asset_returns,
        funding_rates,
        transaction_cost_rate=transaction_cost_rate,
        drawdown_limit=drawdown_limit,
        turnover=turnover,
        simple_growth=simple_growth,
    )
    return violation


def net_log_growth_loss(
    target_weights: Tensor,
    asset_returns: Tensor,
    funding_rates: Tensor,
    *,
    transaction_cost_rate: float,
    drawdown_limit: float,
    constraint_multiplier: Tensor,
    penalty_coefficient: float = 10.0,
    turnover: Tensor | None = None,
    simple_growth: Tensor | None = None,
) -> Tensor:
    """Negative contiguous Net Log Growth plus an augmented drawdown constraint."""
    net_logs, violation = _growth_and_drawdown_violation(
        target_weights,
        asset_returns,
        funding_rates,
        transaction_cost_rate=transaction_cost_rate,
        drawdown_limit=drawdown_limit,
        turnover=turnover,
        simple_growth=simple_growth,
    )
    return -net_logs.sum() + constraint_multiplier * violation + 0.5 * penalty_coefficient * violation.square()


@dataclass(frozen=True)
class WalkForwardFold:
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def sunday_retraining_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    """Return the initial fit, each revealed Sunday 00:00 UTC, and the period end."""
    if start >= end:
        raise ValueError("evaluation period must have positive duration")
    days_until_sunday = (6 - start.dayofweek) % 7
    first_sunday = start.normalize() + pd.Timedelta(days=days_until_sunday)
    if first_sunday <= start:
        first_sunday += pd.Timedelta(days=7)
    boundaries = [start]
    boundary = first_sunday
    while boundary < end:
        boundaries.append(boundary)
        boundary += pd.Timedelta(days=7)
    boundaries.append(end)
    return tuple(boundaries)


def training_episode_slices(*, revealed_bars: int, episode_bars: int, epochs: int) -> tuple[slice, ...]:
    """Spread fixed-length causal episodes across all revealed expanding history."""
    if revealed_bars < 1 or episode_bars < 1 or epochs < 1:
        raise ValueError("revealed bars, episode bars, and epochs must be positive")
    if revealed_bars <= episode_bars:
        return (slice(0, revealed_bars),) * epochs
    if epochs == 1:
        return (slice(revealed_bars - episode_bars, revealed_bars),)
    span = revealed_bars - episode_bars
    ends = (episode_bars + round(index * span / (epochs - 1)) for index in range(epochs))
    return tuple(slice(end - episode_bars, end) for end in ends)


def walk_forward_folds(
    *,
    common_start: pd.Timestamp,
    holdout_start: pd.Timestamp,
    fold_count: int = 12,
    fold_days: int = 90,
    purge_days: int = 7,
) -> tuple[WalkForwardFold, ...]:
    validation_span = pd.Timedelta(days=fold_days)
    first_validation = holdout_start - fold_count * validation_span
    if first_validation - pd.Timedelta(days=purge_days) <= common_start:
        raise ValueError("insufficient expanding history for the requested purged folds")
    return tuple(
        WalkForwardFold(
            training_start=common_start,
            training_end=first_validation + index * validation_span - pd.Timedelta(days=purge_days),
            validation_start=first_validation + index * validation_span,
            validation_end=first_validation + (index + 1) * validation_span,
        )
        for index in range(fold_count)
    )
