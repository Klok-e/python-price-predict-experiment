"""The checked-in, non-overridable Policy Protocol configuration."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

_POLICY_RUNTIME_FILES = (
    "config.py",
    "market_data.py",
    "policy.py",
    "simulation.py",
    "torch_backend.py",
    "training.py",
)


def _policy_code_hash() -> str:
    """Hash Policy Protocol behavior without coupling it to UI or dependency metadata."""
    digest = sha256()
    package = Path(__file__).parent
    for name in _POLICY_RUNTIME_FILES:
        path = package / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicyConfig:
    tickers: tuple[str, ...]
    market: str
    account_currency: str
    history_start: str
    decision_minutes: int
    decision_latency_seconds: int
    mark_minutes: int
    retrain_weekday_utc: str
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

    @property
    def protocol_manifest(self) -> dict[str, object]:
        """The explicit Policy Protocol revision, independent of dashboard/runtime code."""
        return {
            "trading_universe": list(self.tickers),
            "market": self.market,
            "account_currency": self.account_currency,
            "history_start": self.history_start,
            "timing": {
                "decision_minutes": self.decision_minutes,
                "decision_latency_seconds": self.decision_latency_seconds,
                "mark_minutes": self.mark_minutes,
                "retrain_weekday_utc": self.retrain_weekday_utc,
            },
            "execution": {
                "transaction_cost_rate": self.transaction_cost_rate,
                "minimum_turnover": self.minimum_turnover,
                "initial_equity": self.initial_equity,
            },
            "risk": {
                "max_gross_exposure": self.max_gross_exposure,
                "max_instrument_weight": self.max_instrument_weight,
                "drawdown_limit": self.drawdown_limit,
            },
            "features": {
                "contexts": list(self.contexts),
                "normalization_window_bars": self.normalization_window_bars,
            },
            "models": {
                "temporal_widths": list(self.temporal_widths),
                "receptive_field_days": list(self.receptive_field_days),
                "seeds": list(self.seeds),
                "training_episode_days": self.training_episode_days,
                "training_epochs": self.training_epochs,
            },
            "validation": {
                "folds": self.validation_folds,
                "fold_days": self.fold_days,
                "development_evidence_end": self.development_evidence_end.isoformat(),
                "holdout_start": self.holdout_start.isoformat(),
                "holdout_end": self.holdout_end.isoformat(),
            },
        }

    @property
    def compatibility_manifest(self) -> dict[str, object]:
        """The Paper Account semantics that a Policy Revision must preserve."""
        return {
            "trading_universe": list(self.tickers),
            "account_currency": self.account_currency,
            "position_semantics": "signed-perpetual-target-weights-v1",
            "execution_semantics": "delayed-midpoint-adverse-cost-v1",
            "risk_semantics": "marked-equity-no-leverage-drawdown-stop-v1",
        }

    @property
    def identity_hash(self) -> str:
        payload = json.dumps(self.protocol_manifest, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    @property
    def protocol_id(self) -> str:
        """Policy Revision identity shared by evidence and Paper Account operation."""
        return sha256(f"{self.identity_hash}:{self.code_hash}".encode()).hexdigest()

    @property
    def code_hash(self) -> str:
        """Identity of the source files that implement Policy Protocol behavior."""
        return _policy_code_hash()

    @property
    def compatibility_hash(self) -> str:
        payload = json.dumps(self.compatibility_manifest, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


def load_config(path: str | Path = "policy.toml") -> PolicyConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    config = PolicyConfig(
        tickers=tuple(raw["universe"]["tickers"]),
        market=raw["universe"]["market"],
        account_currency=raw["account"]["currency"],
        history_start=raw["universe"]["history_start"],
        decision_minutes=raw["timing"]["decision_minutes"],
        decision_latency_seconds=raw["timing"]["decision_latency_seconds"],
        mark_minutes=raw["timing"]["mark_minutes"],
        retrain_weekday_utc=raw["timing"]["retrain_weekday_utc"],
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
    )
    if config.tickers != ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"):
        raise ValueError("the first Policy Protocol has a fixed Trading Universe")
    if config.temporal_widths != (32, 64) or config.receptive_field_days != (1, 7):
        raise ValueError("temporal-convolution search surface is fixed")
    if len(config.seeds) != 3:
        raise ValueError("each temporal-convolution configuration requires exactly three seeds")
    return config
