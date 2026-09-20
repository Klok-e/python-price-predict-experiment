from __future__ import annotations

from netgrowth.config import load_config


def test_policy_protocol_manifest_separates_revision_from_account_compatibility() -> None:
    config = load_config("policy.toml")

    assert config.market == "Binance USD-M perpetuals"
    assert config.account_currency == "USD"
    assert config.retrain_weekday_utc == "Sunday"
    assert not hasattr(config, "proof_days")
    assert not hasattr(config, "proof_changes")
    assert config.protocol_manifest["trading_universe"] == list(config.tickers)
    assert config.compatibility_manifest == {
        "trading_universe": list(config.tickers),
        "account_currency": "USD",
        "position_semantics": "signed-perpetual-target-weights-v1",
        "execution_semantics": "signal-time-eligibility-delayed-midpoint-v2",
        "risk_semantics": "marked-equity-no-leverage-drawdown-stop-v1",
    }
    assert config.compatibility_hash != config.identity_hash
    assert len(config.protocol_id) == 64
    assert config.protocol_id != config.identity_hash
