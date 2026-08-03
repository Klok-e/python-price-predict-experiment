from __future__ import annotations

import numpy as np
import pytest

from netgrowth.policy import EnsemblePolicy, FixedPolicy, project_target_weights


def test_target_weights_are_projected_into_the_no_leverage_portfolio() -> None:
    portfolio = project_target_weights(
        np.array([1.0, -0.75, 0.25, -0.10]),
        tickers=("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"),
    )

    assert portfolio.weights == pytest.approx({"BTCUSDT": 0.5, "ETHUSDT": -0.5, "BNBUSDT": 0.0, "SOLUSDT": 0.0})
    assert portfolio.gross_exposure == pytest.approx(1.0)
    assert portfolio.cash_weight == pytest.approx(0.0)


def test_ensemble_averages_fitted_policies_before_projection() -> None:
    tickers = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
    ensemble = EnsemblePolicy(
        policies=(
            FixedPolicy(np.array([1.0, 0.0, 0.0, 0.0])),
            FixedPolicy(np.array([0.0, -1.0, 0.0, 0.0])),
        ),
        tickers=tickers,
    )

    target = ensemble.target_weights(market_state=np.zeros((1, 1)), current_weights=np.zeros(4))

    assert target.weights == pytest.approx({"BTCUSDT": 0.5, "ETHUSDT": -0.5, "BNBUSDT": 0.0, "SOLUSDT": 0.0})
    assert target.cash_weight == pytest.approx(0.0)
