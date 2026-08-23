"""Deterministic Integrated Gradients summaries for Fitted Policy decisions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

AttributionGroupKind = Literal["ticker", "feature_family", "temporal_region", "current_portfolio"]


@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    ticker: str | None
    family: str


@dataclass(frozen=True)
class AttributionGroup:
    kind: AttributionGroupKind
    name: str
    value: float


@dataclass(frozen=True)
class SignedInfluence:
    label: str
    value: float


@dataclass(frozen=True)
class TargetAttribution:
    target: str
    output_at_input: float
    output_at_baseline: float
    output_delta: float
    attribution_sum: float
    completeness_error: float
    groups: tuple[AttributionGroup, ...]
    top_influences: tuple[SignedInfluence, ...]


@dataclass(frozen=True)
class ModelAttribution:
    method: str
    baseline: str
    approximation_label: str
    steps: int
    top_k: int
    input_hash: str
    model_hash: str
    targets: tuple[TargetAttribution, ...]


def feature_metadata_from_names(
    feature_names: tuple[str, ...],
    tickers: tuple[str, ...],
) -> tuple[FeatureMetadata, ...]:
    """Map the canonical Market State names to readable attribution families."""
    result: list[FeatureMetadata] = []
    for name in feature_names:
        ticker = next((candidate for candidate in tickers if name.startswith(f"{candidate}_")), None)
        if name.endswith("_available"):
            family = "availability"
        elif any(token in name for token in ("realized_volatility", "candle_range", "candle_location")):
            family = "volatility/range"
        elif any(token in name for token in ("volume", "taker_imbalance", "trades")):
            family = "flow/activity"
        elif any(token in name for token in ("basis", "funding", "open_interest", "premium")):
            family = "derivatives positioning/carry"
        elif "return" in name:
            family = "momentum"
        elif name.startswith("clock_"):
            family = "time/context"
        else:
            family = "other market state"
        result.append(FeatureMetadata(name=name, ticker=ticker, family=family))
    return tuple(result)


def _update_tensor_hash(digest: hashlib._Hash, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(repr(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def _input_hash(
    market_input: Tensor,
    current_portfolio: Tensor,
    feature_metadata: tuple[FeatureMetadata, ...],
    portfolio_names: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    _update_tensor_hash(digest, "market", market_input)
    _update_tensor_hash(digest, "current_portfolio", current_portfolio)
    for feature in feature_metadata:
        digest.update(repr((feature.name, feature.ticker, feature.family)).encode())
    for name in portfolio_names:
        digest.update(name.encode())
    return digest.hexdigest()


def _model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    model_type = type(model)
    digest.update(f"{model_type.__module__}.{model_type.__qualname__}".encode())
    for name, value in sorted(model.state_dict().items()):
        _update_tensor_hash(digest, name, value)
    return digest.hexdigest()


def _evaluate(model: nn.Module, market_input: Tensor, current_portfolio: Tensor) -> Tensor:
    output = cast(Tensor, model(market_input, current_portfolio))
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("Fitted Policy attribution requires one batch of target weights")
    return output


def _temporal_masks(length: int, device: torch.device) -> tuple[tuple[str, Tensor], ...]:
    region_width = (length + 2) // 3
    positions = torch.arange(length, device=device)
    recent_start = max(0, length - region_width)
    middle_start = max(0, recent_start - region_width)
    return (
        ("distant", positions < middle_start),
        ("middle", (positions >= middle_start) & (positions < recent_start)),
        ("recent", positions >= recent_start),
    )


def _group_attributions(
    market_attribution: Tensor,
    portfolio_attribution: Tensor,
    feature_metadata: tuple[FeatureMetadata, ...],
    portfolio_names: tuple[str, ...],
) -> tuple[AttributionGroup, ...]:
    market = market_attribution.squeeze(0)
    portfolio = portfolio_attribution.squeeze(0)
    groups: list[AttributionGroup] = []

    ticker_names = tuple(dict.fromkeys(feature.ticker or "market" for feature in feature_metadata))
    for ticker in ticker_names:
        indices = [index for index, feature in enumerate(feature_metadata) if (feature.ticker or "market") == ticker]
        groups.append(AttributionGroup("ticker", ticker, float(market[:, indices].sum().item())))

    family_names = tuple(dict.fromkeys(feature.family for feature in feature_metadata))
    for family in family_names:
        indices = [index for index, feature in enumerate(feature_metadata) if feature.family == family]
        groups.append(AttributionGroup("feature_family", family, float(market[:, indices].sum().item())))

    for region, mask in _temporal_masks(market.shape[0], market.device):
        groups.append(AttributionGroup("temporal_region", region, float(market[mask].sum().item())))

    groups.extend(
        AttributionGroup("current_portfolio", name, float(portfolio[index].item()))
        for index, name in enumerate(portfolio_names)
    )
    return tuple(groups)


def integrated_gradients(
    model: nn.Module,
    market_input: Tensor,
    current_portfolio: Tensor,
    *,
    feature_metadata: tuple[FeatureMetadata, ...],
    target_names: tuple[str, ...],
    portfolio_names: tuple[str, ...],
    steps: int = 64,
    top_k: int = 10,
) -> ModelAttribution:
    """Attribute all Target Weights from a neutral normalized-market/portfolio baseline."""
    if market_input.ndim != 3 or market_input.shape[0] != 1:
        raise ValueError("market input must have shape (1, time, features)")
    if current_portfolio.ndim != 2 or current_portfolio.shape[0] != 1:
        raise ValueError("Current Portfolio must have shape (1, tickers)")
    if not market_input.is_floating_point() or not current_portfolio.is_floating_point():
        raise ValueError("Integrated Gradients inputs must be floating point")
    if market_input.device != current_portfolio.device:
        raise ValueError("market input and Current Portfolio must share one device")
    if len(feature_metadata) != market_input.shape[2]:
        raise ValueError("feature metadata must identify every market feature")
    if len(portfolio_names) != current_portfolio.shape[1]:
        raise ValueError("portfolio names must identify every Current Portfolio weight")
    if steps < 1:
        raise ValueError("Integrated Gradients steps must be positive")
    if top_k < 1:
        raise ValueError("top influence count must be positive")

    baseline_market = torch.zeros_like(market_input)
    baseline_portfolio = torch.zeros_like(current_portfolio)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output_at_input = _evaluate(model, market_input, current_portfolio)
            output_at_baseline = _evaluate(model, baseline_market, baseline_portfolio)
        if output_at_input.shape[1] != len(target_names):
            raise ValueError("target names must identify every Fitted Policy output")

        targets: list[TargetAttribution] = []
        for target_index, target_name in enumerate(target_names):
            market_gradients = torch.zeros_like(market_input)
            portfolio_gradients = torch.zeros_like(current_portfolio)
            for step in range(steps):
                alpha = (step + 0.5) / steps
                interpolated_market = (baseline_market + alpha * (market_input - baseline_market)).detach()
                interpolated_portfolio = (
                    baseline_portfolio + alpha * (current_portfolio - baseline_portfolio)
                ).detach()
                interpolated_market.requires_grad_(True)
                interpolated_portfolio.requires_grad_(True)
                output = _evaluate(model, interpolated_market, interpolated_portfolio)[0, target_index]
                if output.requires_grad:
                    gradients = torch.autograd.grad(
                        output,
                        (interpolated_market, interpolated_portfolio),
                        allow_unused=True,
                    )
                    market_gradients += gradients[0] if gradients[0] is not None else 0.0
                    portfolio_gradients += gradients[1] if gradients[1] is not None else 0.0

            market_attribution = (market_input - baseline_market) * market_gradients / steps
            portfolio_attribution = (current_portfolio - baseline_portfolio) * portfolio_gradients / steps
            groups = _group_attributions(
                market_attribution,
                portfolio_attribution,
                feature_metadata,
                portfolio_names,
            )
            top = tuple(
                SignedInfluence(f"{group.kind}:{group.name}", group.value)
                for group in sorted(groups, key=lambda group: (-abs(group.value), group.kind, group.name))[:top_k]
            )
            input_value = float(output_at_input[0, target_index].item())
            baseline_value = float(output_at_baseline[0, target_index].item())
            delta = input_value - baseline_value
            attribution_sum = float((market_attribution.sum() + portfolio_attribution.sum()).item())
            targets.append(
                TargetAttribution(
                    target=target_name,
                    output_at_input=input_value,
                    output_at_baseline=baseline_value,
                    output_delta=delta,
                    attribution_sum=attribution_sum,
                    completeness_error=attribution_sum - delta,
                    groups=groups,
                    top_influences=top,
                )
            )
    finally:
        model.train(was_training)

    return ModelAttribution(
        method="Integrated Gradients",
        baseline="zero normalized market plus zero Current Portfolio",
        approximation_label="approximate post-hoc influence evidence",
        steps=steps,
        top_k=top_k,
        input_hash=_input_hash(market_input, current_portfolio, feature_metadata, portfolio_names),
        model_hash=_model_hash(model),
        targets=tuple(targets),
    )
