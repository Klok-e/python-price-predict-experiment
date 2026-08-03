"""Candidate selection and irreversible evidence-state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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


@dataclass
class PaperEvidence:
    protocol_hash: str
    started_at: datetime
    observed_at: datetime
    changes: int = 0
    net_return: float = 0.0
    max_drawdown: float = 0.0
    fitted_policy_hash: str | None = None
    passed: bool = False
    failed: bool = False


@dataclass
class EvidenceState:
    validated_protocol: str | None = None
    validation_passed: bool = False
    holdout: HoldoutEvidence | None = None
    paper: PaperEvidence | None = None
    proof_days: int = 60
    proof_changes: int = 100
    drawdown_limit: float = 0.20

    def record_validation(self, protocol_hash: str, *, passed: bool) -> None:
        self.validated_protocol = protocol_hash
        self.validation_passed = passed

    def consume_holdout(self, *, protocol_hash: str, net_return: float, max_drawdown: float) -> HoldoutEvidence:
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
        )
        return self.holdout

    def start_paper(self, protocol_hash: str, started_at: datetime) -> PaperEvidence:
        if self.paper is None or self.paper.protocol_hash != protocol_hash:
            self.paper = PaperEvidence(
                protocol_hash=protocol_hash,
                started_at=started_at,
                observed_at=started_at,
            )
        return self.paper

    def record_paper_progress(
        self,
        protocol_hash: str,
        observed_at: datetime,
        *,
        changes: int,
        net_return: float,
        max_drawdown: float,
    ) -> PaperEvidence:
        if self.paper is None or self.paper.protocol_hash != protocol_hash:
            raise ValueError("Forward Paper Proof has not started for this Policy Protocol")
        self.paper.observed_at = observed_at
        self.paper.changes = changes
        self.paper.net_return = net_return
        self.paper.max_drawdown = max_drawdown
        elapsed_days = (observed_at - self.paper.started_at).total_seconds() / 86_400.0
        self.paper.failed = max_drawdown > self.drawdown_limit
        complete = elapsed_days >= self.proof_days and changes >= self.proof_changes
        if complete and net_return <= 0.0:
            self.paper.failed = True
        self.paper.passed = complete and net_return > 0.0 and not self.paper.failed
        return self.paper

    def record_fitted_policy_handoff(self, protocol_hash: str, *, fitted_policy_hash: str) -> None:
        if self.paper is None or self.paper.protocol_hash != protocol_hash:
            raise ValueError("Policy Handoff requires an active matching Proof Clock")
        self.paper.fitted_policy_hash = fitted_policy_hash
