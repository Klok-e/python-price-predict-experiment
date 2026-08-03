"""Trading Policy implementations and portfolio constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TargetPortfolio:
    weights: dict[str, float]
    cash_weight: float

    @property
    def gross_exposure(self) -> float:
        return sum(abs(weight) for weight in self.weights.values())


def project_target_weights(
    raw_weights: NDArray[np.floating],
    tickers: tuple[str, ...],
    *,
    max_instrument_weight: float = 0.50,
    max_gross_exposure: float = 1.0,
) -> TargetPortfolio:
    """Euclidean projection onto the capped signed L1 portfolio constraint."""
    raw = np.nan_to_num(np.asarray(raw_weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if raw.shape != (len(tickers),):
        raise ValueError(f"expected {len(tickers)} weights, received shape {raw.shape}")

    clipped = np.clip(raw, -max_instrument_weight, max_instrument_weight)
    if float(np.abs(clipped).sum()) > max_gross_exposure:
        lower, upper = 0.0, float(np.abs(raw).max())
        for _ in range(80):
            threshold = (lower + upper) / 2.0
            candidate = np.minimum(np.maximum(np.abs(raw) - threshold, 0.0), max_instrument_weight)
            if float(candidate.sum()) > max_gross_exposure:
                lower = threshold
            else:
                upper = threshold
        clipped = np.sign(raw) * np.minimum(np.maximum(np.abs(raw) - upper, 0.0), max_instrument_weight)

    weights = {ticker: float(weight) for ticker, weight in zip(tickers, clipped, strict=True)}
    gross = sum(abs(weight) for weight in weights.values())
    return TargetPortfolio(weights=weights, cash_weight=max(0.0, 1.0 - gross))


class RawPolicy(Protocol):
    def raw_weights(
        self, market_state: NDArray[np.floating], current_weights: NDArray[np.floating]
    ) -> NDArray[np.floating]: ...


@dataclass(frozen=True)
class FixedPolicy:
    output: NDArray[np.floating]

    def raw_weights(
        self, market_state: NDArray[np.floating], current_weights: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        del market_state, current_weights
        return np.asarray(self.output, dtype=float)


@dataclass(frozen=True)
class EnsemblePolicy:
    policies: tuple[RawPolicy, ...]
    tickers: tuple[str, ...]

    def target_weights(
        self, market_state: NDArray[np.floating], current_weights: NDArray[np.floating]
    ) -> TargetPortfolio:
        if not self.policies:
            raise ValueError("an ensemble requires at least one Fitted Policy")
        outputs = [policy.raw_weights(market_state, current_weights) for policy in self.policies]
        return project_target_weights(np.mean(outputs, axis=0), self.tickers)
