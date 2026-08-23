from __future__ import annotations

import torch
from torch import nn

from netgrowth.attribution import FeatureMetadata, feature_metadata_from_names, integrated_gradients


class SmallLinearPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.market_coefficients = nn.Parameter(
            torch.tensor(
                [
                    [[1.0, -2.0], [0.5, 1.0], [-1.0, 3.0]],
                    [[-0.5, 1.0], [2.0, -1.0], [1.0, 0.25]],
                ]
            )
        )
        self.portfolio_coefficients = nn.Parameter(torch.tensor([[4.0, -1.0], [0.5, 2.0]]))
        self.bias = nn.Parameter(torch.tensor([0.25, -0.75]))

    def forward(self, market: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        market_output = torch.einsum("btf,otf->bo", market, self.market_coefficients)
        return market_output + current @ self.portfolio_coefficients.T + self.bias


def test_integrated_gradients_is_complete_grouped_and_signed_against_zero_baselines() -> None:
    model = SmallLinearPolicy()
    market = torch.tensor([[[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]]])
    current = torch.tensor([[0.25, -0.5]])
    metadata = (
        FeatureMetadata("BTCUSDT_return_15m", ticker="BTCUSDT", family="momentum"),
        FeatureMetadata("ETHUSDT_volume", ticker="ETHUSDT", family="flow/activity"),
    )

    attribution = integrated_gradients(
        model,
        market,
        current,
        feature_metadata=metadata,
        target_names=("BTCUSDT", "ETHUSDT"),
        portfolio_names=("BTCUSDT", "ETHUSDT"),
        steps=8,
        top_k=5,
    )

    assert attribution.method == "Integrated Gradients"
    assert attribution.baseline == "zero normalized market plus zero Current Portfolio"
    assert attribution.approximation_label == "approximate post-hoc influence evidence"
    assert attribution.steps == 8
    assert attribution.top_k == 5
    assert len(attribution.input_hash) == 64
    assert len(attribution.model_hash) == 64

    btc = attribution.targets[0]
    assert abs(btc.completeness_error) < 1e-6
    assert abs(btc.attribution_sum - btc.output_delta) < 1e-6
    groups = {(group.kind, group.name): group.value for group in btc.groups}
    assert groups[("ticker", "BTCUSDT")] == -3.0
    assert groups[("ticker", "ETHUSDT")] == -6.0
    assert groups[("feature_family", "momentum")] == -3.0
    assert groups[("feature_family", "flow/activity")] == -6.0
    assert groups[("temporal_region", "recent")] == -6.0
    assert groups[("current_portfolio", "BTCUSDT")] == 1.0
    assert groups[("current_portfolio", "ETHUSDT")] == 0.5
    assert btc.top_influences[0].value == -6.0


def test_integrated_gradients_is_deterministic_and_complete_for_a_quadratic_model() -> None:
    class QuadraticPolicy(nn.Module):
        def forward(self, market: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
            return (market.square().sum(dim=(1, 2)) + 2.0 * current.square().sum(dim=1)).unsqueeze(1)

    kwargs = {
        "feature_metadata": (FeatureMetadata("market_return_15m", ticker=None, family="momentum"),),
        "target_names": ("BTCUSDT",),
        "portfolio_names": ("BTCUSDT", "ETHUSDT"),
        "steps": 16,
    }
    market = torch.tensor([[[1.0], [-2.0], [3.0]]])
    current = torch.tensor([[0.5, -0.25]])

    first = integrated_gradients(QuadraticPolicy(), market, current, **kwargs)
    second = integrated_gradients(QuadraticPolicy(), market, current, **kwargs)
    renamed = integrated_gradients(
        QuadraticPolicy(),
        market,
        current,
        **{
            **kwargs,
            "feature_metadata": (FeatureMetadata("renamed_return", ticker=None, family="momentum"),),
        },
    )

    assert first == second
    assert first.input_hash != renamed.input_hash
    assert abs(first.targets[0].completeness_error) < 1e-6
    assert first.targets[0].output_delta == 14.625


def test_feature_metadata_inference_uses_dashboard_feature_families() -> None:
    metadata = feature_metadata_from_names(
        (
            "BTCUSDT_return_15m",
            "BTCUSDT_realized_volatility",
            "ETHUSDT_taker_imbalance",
            "ETHUSDT_open_interest",
            "SOLUSDT_funding_available",
            "clock_week_sin",
        ),
        ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    )

    assert [(feature.ticker, feature.family) for feature in metadata] == [
        ("BTCUSDT", "momentum"),
        ("BTCUSDT", "volatility/range"),
        ("ETHUSDT", "flow/activity"),
        ("ETHUSDT", "derivatives positioning/carry"),
        ("SOLUSDT", "availability"),
        (None, "time/context"),
    ]
