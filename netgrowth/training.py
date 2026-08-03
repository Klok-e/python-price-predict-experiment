"""Small direct-policy models, Net Log Growth objective, and chronological folds."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


def _constrain_weights(raw: Tensor) -> Tensor:
    concentrated = 0.5 * torch.tanh(raw)
    gross = concentrated.abs().sum(dim=-1, keepdim=True)
    scale = torch.clamp(1.0 / gross.clamp_min(1e-12), max=1.0)
    return concentrated * scale


class LinearDirectPolicy(nn.Module):
    def __init__(self, feature_count: int, ticker_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count + ticker_count, ticker_count)

    def forward(self, market_state: Tensor, current_weights: Tensor) -> Tensor:
        latest_state = market_state[:, -1, :]
        return _constrain_weights(self.linear(torch.cat((latest_state, current_weights), dim=-1)))


class TemporalConvPolicy(nn.Module):
    def __init__(self, feature_count: int, ticker_count: int, *, width: int, receptive_bars: int) -> None:
        super().__init__()
        if width not in (32, 64):
            raise ValueError("temporal width must be 32 or 64")
        if receptive_bars < 1:
            raise ValueError("receptive field must contain at least one bar")
        self.receptive_bars = receptive_bars
        self.convolution = nn.Conv1d(feature_count, width, kernel_size=receptive_bars)
        self.output = nn.Linear(width + ticker_count, ticker_count)

    def forward(self, market_state: Tensor, current_weights: Tensor) -> Tensor:
        sequence = market_state.transpose(1, 2)
        causal = functional.pad(sequence, (self.receptive_bars - 1, 0))
        encoded = torch.nn.functional.gelu(self.convolution(causal))[:, :, -1]
        return _constrain_weights(self.output(torch.cat((encoded, current_weights), dim=-1)))


def net_log_growth_loss(
    target_weights: Tensor,
    asset_returns: Tensor,
    funding_rates: Tensor,
    *,
    transaction_cost_rate: float,
    drawdown_limit: float,
    constraint_multiplier: Tensor,
    penalty_coefficient: float = 10.0,
) -> Tensor:
    """Negative contiguous Net Log Growth plus an augmented drawdown constraint."""
    if target_weights.shape != asset_returns.shape or target_weights.shape != funding_rates.shape:
        raise ValueError("weights, returns, and funding must have identical time-by-ticker shape")
    previous = torch.cat((torch.zeros_like(target_weights[:1]), target_weights[:-1]), dim=0)
    turnover = (target_weights - previous).abs().sum(dim=-1)
    held_return = (target_weights * asset_returns).sum(dim=-1)
    funding = (target_weights * funding_rates).sum(dim=-1)
    simple_growth = held_return - funding - transaction_cost_rate * turnover
    net_logs = torch.log1p(simple_growth.clamp_min(-0.999999))
    equity = torch.exp(torch.cumsum(net_logs, dim=0))
    peaks = torch.cummax(equity, dim=0).values.clamp_min(1e-12)
    maximum_drawdown = torch.max(1.0 - equity / peaks)
    violation = torch.relu(maximum_drawdown - drawdown_limit)
    return -net_logs.sum() + constraint_multiplier * violation + 0.5 * penalty_coefficient * violation.square()


@dataclass(frozen=True)
class WalkForwardFold:
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


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
