"""Candidate selection and irreversible evidence-state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from statistics import median


@dataclass(frozen=True)
class FoldResult:
    net_return: float
    max_drawdown: float


@dataclass(frozen=True)
class CandidateResult:
    name: str
    folds: tuple[FoldResult, ...]

    @property
    def compounded_net_return(self) -> float:
        return prod(1.0 + fold.net_return for fold in self.folds) - 1.0

    def eligible(self, drawdown_limit: float) -> bool:
        return self.compounded_net_return > 0.0 and all(fold.max_drawdown <= drawdown_limit for fold in self.folds)


@dataclass(frozen=True)
class TemporalCandidate:
    architecture: str
    seeds: tuple[CandidateResult, CandidateResult, CandidateResult]
    ensemble: CandidateResult

    @property
    def median_seed_return(self) -> float:
        return median(seed.compounded_net_return for seed in self.seeds)


def choose_candidate(
    linear: CandidateResult,
    temporal: tuple[TemporalCandidate, ...],
    *,
    drawdown_limit: float,
) -> CandidateResult:
    """Select temporal stability first, then replayed ensemble versus linear."""
    contenders = [linear]
    if temporal:
        chosen_architecture = max(temporal, key=lambda item: item.median_seed_return)
        contenders.append(chosen_architecture.ensemble)
    eligible = [candidate for candidate in contenders if candidate.eligible(drawdown_limit)]
    if not eligible:
        raise ValueError("no eligible candidate: return must be positive and every fold within Drawdown Limit")
    return max(eligible, key=lambda candidate: candidate.compounded_net_return)


@dataclass(frozen=True)
class HoldoutEvidence:
    protocol_hash: str
    net_return: float
    max_drawdown: float
    status: str
    artifact: str | None = None


@dataclass
class EvidenceState:
    validated_protocol: str | None = None
    validation_passed: bool = False
    validated_artifact: str | None = None
    validated_model_hash: str | None = None
    holdout: HoldoutEvidence | None = None
    drawdown_limit: float = 0.20

    def record_validation(
        self,
        protocol_hash: str,
        *,
        passed: bool,
        artifact: str | None = None,
        model_hash: str | None = None,
    ) -> None:
        self.validated_protocol = protocol_hash
        self.validation_passed = passed
        self.validated_artifact = artifact if passed else None
        self.validated_model_hash = model_hash if passed else None

    def consume_holdout(
        self,
        *,
        protocol_hash: str,
        net_return: float,
        max_drawdown: float,
        artifact: str | None = None,
    ) -> HoldoutEvidence:
        if self.holdout is not None:
            return self.holdout
        if not self.validation_passed or self.validated_protocol != protocol_hash:
            raise ValueError("successful validation is required before Historical Holdout")
        passed = net_return > 0.0 and max_drawdown <= self.drawdown_limit
        self.holdout = HoldoutEvidence(
            protocol_hash=protocol_hash,
            net_return=net_return,
            max_drawdown=max_drawdown,
            status="passed" if passed else "consumed",
            artifact=artifact,
        )
        return self.holdout
