from __future__ import annotations

import pytest

from netgrowth.evidence import (
    CandidateResult,
    EvidenceState,
    FoldResult,
    TemporalCandidate,
    choose_candidate,
)


def candidate(name: str, returns: list[float], drawdowns: list[float]) -> CandidateResult:
    return CandidateResult(
        name=name,
        folds=tuple(
            FoldResult(net_return=value, max_drawdown=drawdown)
            for value, drawdown in zip(returns, drawdowns, strict=True)
        ),
    )


def test_architecture_uses_median_seed_but_actual_ensemble_competes_with_linear() -> None:
    linear = candidate("linear", [0.01, 0.01], [0.05, 0.05])
    lucky_seed = candidate("seed-lucky", [0.50, 0.50], [0.10, 0.10])
    weak_seed = candidate("seed-weak", [-0.02, -0.02], [0.10, 0.10])
    stable_seeds = tuple(candidate(f"stable-{i}", [0.03, 0.03], [0.10, 0.10]) for i in range(3))
    temporal = (
        TemporalCandidate(
            "unstable",
            (lucky_seed, weak_seed, weak_seed),
            candidate("unstable-ensemble", [0.20, 0.20], [0.10, 0.10]),
        ),
        TemporalCandidate("stable", stable_seeds, candidate("stable-ensemble", [0.04, 0.04], [0.10, 0.10])),
    )

    selected = choose_candidate(linear, temporal, drawdown_limit=0.20)

    assert selected.name == "stable-ensemble"


def test_non_positive_return_or_one_fold_drawdown_makes_candidate_ineligible() -> None:
    linear = candidate("linear", [-0.01, 0.0], [0.01, 0.01])
    breached = candidate("ensemble", [0.50, 0.50], [0.01, 0.21])
    seeds = tuple(candidate(f"seed-{i}", [0.01, 0.01], [0.01, 0.01]) for i in range(3))

    with pytest.raises(ValueError, match="no eligible"):
        choose_candidate(linear, (TemporalCandidate("tcn", seeds, breached),), drawdown_limit=0.20)


def test_holdout_is_single_use_and_failed_result_is_consumed() -> None:
    state = EvidenceState()
    with pytest.raises(ValueError, match="validation"):
        state.consume_holdout(protocol_hash="p1", net_return=0.10, max_drawdown=0.10)

    state.record_validation(protocol_hash="p1", passed=True)
    first = state.consume_holdout(protocol_hash="p1", net_return=-0.01, max_drawdown=0.10)
    repeated = state.consume_holdout(protocol_hash="p1", net_return=0.50, max_drawdown=0.01)

    assert first is repeated
    assert repeated.status == "consumed"
    assert repeated.net_return == -0.01
