"""The checked-in, non-overridable Policy Protocol configuration."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class PolicyConfig:
    tickers: tuple[str, ...]
    history_start: str
    decision_minutes: int
    decision_latency_seconds: int
    mark_minutes: int
    transaction_cost_rate: float
    minimum_turnover: float
    initial_equity: float
    max_gross_exposure: float
    max_instrument_weight: float
    drawdown_limit: float
    contexts: tuple[str, ...]
    normalization_window_bars: int
    temporal_widths: tuple[int, ...]
    receptive_field_days: tuple[int, ...]
    seeds: tuple[int, ...]
    training_episode_days: int
    training_epochs: int
    validation_folds: int
    fold_days: int
    development_evidence_end: date
    holdout_start: date
    holdout_end: date
    proof_days: int
    proof_changes: int

    @property
    def identity_hash(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: value.isoformat(),
        )
        return sha256(payload.encode()).hexdigest()


def load_config(path: str | Path = "policy.toml") -> PolicyConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    config = PolicyConfig(
        tickers=tuple(raw["universe"]["tickers"]),
        history_start=raw["universe"]["history_start"],
        decision_minutes=raw["timing"]["decision_minutes"],
        decision_latency_seconds=raw["timing"]["decision_latency_seconds"],
        mark_minutes=raw["timing"]["mark_minutes"],
        transaction_cost_rate=raw["execution"]["transaction_cost_rate"],
        minimum_turnover=raw["execution"]["minimum_turnover"],
        initial_equity=raw["execution"]["initial_equity"],
        max_gross_exposure=raw["risk"]["max_gross_exposure"],
        max_instrument_weight=raw["risk"]["max_instrument_weight"],
        drawdown_limit=raw["risk"]["drawdown_limit"],
        contexts=tuple(raw["features"]["contexts"]),
        normalization_window_bars=raw["features"]["normalization_window_bars"],
        temporal_widths=tuple(raw["models"]["temporal_widths"]),
        receptive_field_days=tuple(raw["models"]["receptive_field_days"]),
        seeds=tuple(raw["models"]["seeds"]),
        training_episode_days=raw["models"]["training_episode_days"],
        training_epochs=raw["models"]["training_epochs"],
        validation_folds=raw["validation"]["folds"],
        fold_days=raw["validation"]["fold_days"],
        development_evidence_end=date.fromisoformat(raw["validation"]["development_evidence_end"]),
        holdout_start=date.fromisoformat(raw["validation"]["holdout_start"]),
        holdout_end=date.fromisoformat(raw["validation"]["holdout_end"]),
        proof_days=raw["proof"]["minimum_days"],
        proof_changes=raw["proof"]["minimum_qualifying_changes"],
    )
    if config.tickers != ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"):
        raise ValueError("the first Policy Protocol has a fixed Trading Universe")
    if config.temporal_widths != (32, 64) or config.receptive_field_days != (1, 7):
        raise ValueError("temporal-convolution search surface is fixed")
    if len(config.seeds) != 3:
        raise ValueError("each temporal-convolution configuration requires exactly three seeds")
    return config
