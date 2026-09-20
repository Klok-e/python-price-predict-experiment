from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch

from netgrowth.config import load_config
from netgrowth.market_data import CanonicalDataset, InstrumentData
from netgrowth.paper_dashboard.application import create_application
from netgrowth.paper_dashboard.domain import MarketObservation, PolicyDecision
from netgrowth.paper_dashboard.production import ProductionPolicyBackend
from netgrowth.training import LinearDirectPolicy

TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
SIGNAL_TIME = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)


def _canonical(*, periods: int, correction_before_signal: bool = False) -> CanonicalDataset:
    index = pd.date_range("2026-08-23T14:00:00Z", periods=periods, freq="min")
    instruments: dict[str, InstrumentData] = {}
    for offset, ticker in enumerate(TICKERS):
        close = pd.Series([100.0 + offset + step / 10 for step in range(periods)], index=index)
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 10.0,
                "taker_buy_volume": 6.0,
                "trades": 20,
            },
            index=index,
        )
        if correction_before_signal:
            bars.loc[index[-1], "close"] += 1.0
        instruments[ticker] = InstrumentData(
            perpetual=bars,
            spot=bars.copy(),
            funding=pd.Series([0.0], index=index[:1], name="funding_rate"),
            open_interest=pd.Series([1_000.0], index=index[:1], name="open_interest"),
            premium=pd.Series([0.001], index=index[:1], name="premium"),
        )
    return CanonicalDataset(instruments=instruments, tickers=TICKERS)


def _observation() -> MarketObservation:
    prices = dict.fromkeys(TICKERS, 100.0)
    return MarketObservation(
        timestamp=SIGNAL_TIME,
        mark_prices=prices,
        bid=prices,
        ask=prices,
        quote_exchange_times=dict.fromkeys(TICKERS, SIGNAL_TIME),
        quote_observed_at=SIGNAL_TIME,
        input_id="feed-input",
        decision_bar_closed=True,
    )


def _record(input_id: str, model_id: str, current_weights: dict[str, float]) -> dict[str, Any]:
    return {
        "input_id": input_id,
        "model_id": model_id,
        "signal_time": SIGNAL_TIME.isoformat(),
        "current_portfolio": current_weights,
    }


@dataclass
class _Adapter:
    canonical: CanonicalDataset
    loads: int = 0

    def load(self) -> CanonicalDataset:
        self.loads += 1
        return self.canonical


class _Backend:
    @staticmethod
    def _selected_metadata(model: bytes) -> dict[str, object]:
        if model == b"provisional-model":
            return {"kind": "linear", "metadata": {}}
        return torch.load(BytesIO(model), map_location="cpu", weights_only=True)

    def paper(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            refitted=False,
            model_bytes=b"unused",
            target_weights=dict.fromkeys(TICKERS, 0.0),
        )


def _backend(tmp_path: Path, canonical: CanonicalDataset, fitted_model: bytes) -> ProductionPolicyBackend:
    adapter = _Adapter(canonical)
    return ProductionPolicyBackend(
        adapter=adapter,  # type: ignore[arg-type]
        backend=_Backend(),  # type: ignore[arg-type]
        fitting_adapter=adapter,  # type: ignore[arg-type]
        attribution_adapter=adapter,  # type: ignore[arg-type]
        fitting_backend=_Backend(),  # type: ignore[arg-type]
        checkpoint_directory=tmp_path,
        config=load_config("policy.toml"),
        device="cpu",
        fitted_model=fitted_model,
    )


def _linear_model_bytes(tmp_path: Path, feature_count: int) -> bytes:
    model = LinearDirectPolicy(feature_count, len(TICKERS))
    checkpoint = tmp_path / "fitted-linear.pt"
    torch.save(
        {"kind": "linear", "metadata": {}, "state_dicts": [model.state_dict()]},
        checkpoint,
    )
    return checkpoint.read_bytes()


def _prepared_backend(tmp_path: Path, canonical: CanonicalDataset) -> ProductionPolicyBackend:
    provisional = _backend(tmp_path, canonical, b"provisional-model")
    provisional_decision = provisional.decide(_observation(), dict.fromkeys(TICKERS, 0.0))
    metadata, market_values = provisional._deserialize_attribution_input(
        provisional_decision.input_id,
        _record(provisional_decision.input_id, provisional_decision.model_id, dict.fromkeys(TICKERS, 0.0)),
        provisional.export_attribution_input(provisional_decision.input_id),
    )
    assert metadata["input_id"] == provisional_decision.input_id
    return _backend(tmp_path, canonical, _linear_model_bytes(tmp_path, market_values.shape[1]))


def test_durable_attribution_replays_exact_prepared_input_after_new_rows_or_corrections(tmp_path) -> None:
    original = _canonical(periods=240)
    current = {ticker: index / 10 for index, ticker in enumerate(TICKERS)}
    decision_backend = _prepared_backend(tmp_path, original)
    decision = decision_backend.decide(_observation(), current)
    durable_payload = decision_backend.export_attribution_input(decision.input_id)
    record = _record(decision.input_id, decision.model_id, current)

    initial = decision_backend.attribute(
        decision.input_id,
        {**record, "_durable_attribution_input": durable_payload},
    )

    resumed = _backend(
        tmp_path,
        _canonical(periods=246, correction_before_signal=True),
        decision_backend.fitted_model,
    )
    restored = resumed.attribute(
        decision.input_id,
        {**record, "_durable_attribution_input": durable_payload},
    )

    assert initial == restored
    assert resumed.adapter.loads == 0


def test_durable_attribution_identity_changes_when_the_causal_market_slice_changes(tmp_path) -> None:
    current = dict.fromkeys(TICKERS, 0.0)
    original = _prepared_backend(tmp_path, _canonical(periods=240))
    corrected = _prepared_backend(tmp_path, _canonical(periods=240, correction_before_signal=True))

    original_decision = original.decide(_observation(), current)
    corrected_decision = corrected.decide(_observation(), current)
    original_payload = original.export_attribution_input(original_decision.input_id)
    corrected_payload = corrected.export_attribution_input(corrected_decision.input_id)

    assert original_decision.input_id != corrected_decision.input_id
    original_metadata, original_market = original._deserialize_attribution_input(
        original_decision.input_id,
        _record(original_decision.input_id, original_decision.model_id, current),
        original_payload,
    )
    corrected_metadata, corrected_market = corrected._deserialize_attribution_input(
        corrected_decision.input_id,
        _record(corrected_decision.input_id, corrected_decision.model_id, current),
        corrected_payload,
    )
    assert original_metadata["observed_at"] == corrected_metadata["observed_at"] == SIGNAL_TIME.isoformat()
    assert not (original_market == corrected_market).all()


def test_durable_attribution_rejects_payload_bound_to_a_different_signal_portfolio_or_time(tmp_path) -> None:
    current = dict.fromkeys(TICKERS, 0.0)
    backend = _prepared_backend(tmp_path, _canonical(periods=240))
    decision = backend.decide(_observation(), current)
    payload = backend.export_attribution_input(decision.input_id)
    record = _record(decision.input_id, decision.model_id, current)

    with pytest.raises(RuntimeError, match="Current Portfolio differs"):
        backend._deserialize_attribution_input(
            decision.input_id,
            {**record, "current_portfolio": {**current, "BTCUSDT": 0.5}},
            payload,
        )
    with pytest.raises(RuntimeError, match="Signal Time differs"):
        backend._deserialize_attribution_input(
            decision.input_id,
            {**record, "signal_time": (SIGNAL_TIME.replace(minute=1)).isoformat()},
            payload,
        )


@pytest.mark.anyio
async def test_restart_attributes_from_the_retained_exact_input_then_releases_it_on_success(tmp_path) -> None:
    payload = b"exact prepared attribution input"

    @dataclass
    class Policy:
        available: dict[str, bytes]

        def decide(
            self,
            observed: MarketObservation,
            current_weights: dict[str, float],
        ) -> PolicyDecision:
            del current_weights
            target = dict.fromkeys(TICKERS, 0.0)
            return PolicyDecision(target, target, "model-1", observed.input_id, "protocol-1")

        def export_attribution_input(self, input_id: str) -> bytes:
            return self.available.pop(input_id)

    @dataclass
    class Attribution:
        inputs: list[bytes]

        def attribute(self, input_id: str, decision: dict[str, Any]) -> dict[str, Any]:
            assert input_id == decision["input_id"]
            self.inputs.append(decision["_durable_attribution_input"])
            return {"input_hash": "explained-from-retained-input"}

    database = tmp_path / "paper.sqlite3"
    first = create_application(
        database_path=database,
        tickers=TICKERS,
        policy_backend=Policy({_observation().input_id: payload}),
    ).state.paper_dashboard
    first.advance_once_sync(_observation())
    decision = next(event for event in first.store.events() if event.event_type == "DecisionRecord")
    assert first.store.attribution_input(_observation().input_id)["payload"] == payload
    first.close()

    attribution = Attribution([])
    resumed = create_application(
        database_path=database,
        tickers=TICKERS,
        policy_backend=Policy({}),
        attribution_backend=attribution,
    ).state.paper_dashboard
    resumed.resume_pending_attributions()
    await asyncio.gather(*resumed._attribution_tasks)

    assert attribution.inputs == [payload]
    assert resumed.event_snapshot(decision.event_id)["attribution"]["status"] == "complete"
    with pytest.raises(RuntimeError, match="released after successful completion"):
        resumed.store.attribution_input(_observation().input_id)
    resumed.close()


def test_historical_decision_without_exact_durable_input_is_explicitly_unavailable() -> None:
    backend = object.__new__(ProductionPolicyBackend)

    with pytest.raises(RuntimeError, match="exact durable attribution input is unavailable"):
        backend.attribute("legacy-input", {"input_id": "legacy-input"})
