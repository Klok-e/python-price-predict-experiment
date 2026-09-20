"""Validation gate shared by fresh startup and in-account Policy Revision staging."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from math import isfinite, prod
from typing import Any


def validate_revision_evidence(
    evidence: Mapping[str, Any],
    *,
    protocol_id: str,
    model_id: str,
    drawdown_limit: float,
    fold_count: int = 12,
) -> None:
    folds = evidence.get("folds")
    if (
        evidence.get("protocol_id") != protocol_id
        or evidence.get("trained_protocol_id") != protocol_id
        or evidence.get("model_id") != model_id
        or not isinstance(folds, list)
        or len(folds) != fold_count
        or any(not isinstance(fold, dict) for fold in folds)
    ):
        raise ValueError("candidate requires matching training and validation evidence")
    try:
        returns = [float(fold["net_return"]) for fold in folds]
        drawdowns = [float(fold["max_drawdown"]) for fold in folds]
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("candidate has malformed validation folds") from error
    growth = prod(1.0 + value for value in returns)
    if (
        not all(isfinite(value) for value in returns + drawdowns)
        or not isfinite(growth)
        or any(value <= -1 for value in returns)
        or growth <= 1
        or any(value < 0 or value > drawdown_limit for value in drawdowns)
    ):
        raise ValueError("candidate failed validation gate")
    fitted_at = evidence.get("fitted_at")
    if fitted_at is not None:
        fitted = datetime.fromisoformat(str(fitted_at))
        if fitted.tzinfo is None or fitted.utcoffset() is None:
            raise ValueError("candidate fitted_at must be timezone-aware")
