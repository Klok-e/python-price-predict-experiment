"""Canonical Binance-native market data and causal Market State construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol, cast

import numpy as np
import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "volume", "taker_buy_volume", "trades")


@dataclass
class InstrumentData:
    perpetual: pd.DataFrame
    spot: pd.DataFrame
    funding: pd.Series
    open_interest: pd.Series
    premium: pd.Series


@dataclass
class CanonicalDataset:
    instruments: dict[str, InstrumentData]
    tickers: tuple[str, ...]

    def validate(self) -> None:
        if set(self.instruments) != set(self.tickers):
            raise ValueError("canonical data must cover the fixed Trading Universe")
        for ticker, instrument in self.instruments.items():
            for source_name, bars in (("perpetual", instrument.perpetual), ("spot", instrument.spot)):
                if not isinstance(bars.index, pd.DatetimeIndex) or str(bars.index.tz) != "UTC":
                    raise ValueError(f"{ticker} {source_name} must use timezone-aware UTC")
                if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
                    raise ValueError(f"{ticker} {source_name} timestamps must be ordered and unique")
                missing_columns = set(BAR_COLUMNS) - set(bars.columns)
                if missing_columns:
                    raise ValueError(f"{ticker} {source_name} missing {sorted(missing_columns)}")
                if bars.loc[:, list(BAR_COLUMNS)].isna().any().any():
                    raise ValueError(f"{ticker} missing required execution price or bar value")
            if instrument.funding.isna().any():
                raise ValueError(f"{ticker} missing required funding cashflow")
            if instrument.funding.empty:
                raise ValueError(f"{ticker} missing required funding cashflow archive")
            for optional in (instrument.open_interest, instrument.premium, instrument.funding):
                if not isinstance(optional.index, pd.DatetimeIndex) or str(optional.index.tz) != "UTC":
                    raise ValueError(f"{ticker} timestamped inputs must use timezone-aware UTC")

    @property
    def common_trading_start(self) -> pd.Timestamp:
        self.validate()
        return pd.Timestamp(max(instrument.perpetual.index.min() for instrument in self.instruments.values()))

    def execution_complete(self) -> dict[str, InstrumentData]:
        start = self.common_trading_start
        return {
            ticker: InstrumentData(
                perpetual=data.perpetual.loc[start:],
                spot=data.spot.loc[start:],
                funding=data.funding.loc[start:],
                open_interest=data.open_interest.loc[start:],
                premium=data.premium.loc[start:],
            )
            for ticker, data in self.instruments.items()
        }

    @property
    def identity_hash(self) -> str:
        self.validate()
        digest = sha256()
        for ticker in self.tickers:
            digest.update(ticker.encode())
            data = self.instruments[ticker]
            for value in (
                data.perpetual,
                data.spot,
                data.funding,
                data.open_interest,
                data.premium,
            ):
                hashed = pd.util.hash_pandas_object(value, index=True).to_numpy()
                digest.update(hashed.tobytes())
        return digest.hexdigest()


class MarketDataAdapter(Protocol):
    @property
    def mode(self) -> Literal["historical", "live"]: ...

    def load(self) -> CanonicalDataset: ...


@dataclass(frozen=True)
class InMemoryMarketData:
    dataset: CanonicalDataset
    mode: Literal["historical", "live"]

    def load(self) -> CanonicalDataset:
        self.dataset.validate()
        return self.dataset


def _closed_decision_bars(bars: pd.DataFrame) -> pd.DataFrame:
    counts = bars["close"].resample("15min", closed="left", label="right").count()
    aggregated = bars.resample("15min", closed="left", label="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "taker_buy_volume": "sum",
            "trades": "sum",
        }
    )
    return aggregated.loc[counts == 15]


def _asof_optional(values: pd.Series, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    if values.empty:
        aligned = pd.Series(np.nan, index=index, dtype=float)
    else:
        aligned = values.sort_index().reindex(index, method="ffill")
    return aligned.fillna(0.0), aligned.notna().astype(float)


def _instrument_features(data: InstrumentData, ticker: str) -> pd.DataFrame:
    perpetual = _closed_decision_bars(data.perpetual)
    spot = _closed_decision_bars(data.spot).reindex(perpetual.index)
    result = pd.DataFrame(index=perpetual.index)
    prefix = f"{ticker}_"
    result[prefix + "return_15m"] = np.log(perpetual["close"]).diff()
    result[prefix + "return_1h"] = np.log(perpetual["close"]).diff(4)
    result[prefix + "return_4h"] = np.log(perpetual["close"]).diff(16)
    result[prefix + "return_1d"] = np.log(perpetual["close"]).diff(96)
    result[prefix + "realized_volatility"] = result[prefix + "return_15m"].rolling(96).std()
    result[prefix + "candle_range"] = (perpetual["high"] - perpetual["low"]) / perpetual["open"]
    denominator = (perpetual["high"] - perpetual["low"]).replace(0.0, np.nan)
    result[prefix + "candle_location"] = ((perpetual["close"] - perpetual["low"]) / denominator).fillna(0.5)
    result[prefix + "volume"] = perpetual["volume"]
    result[prefix + "taker_imbalance"] = (
        2.0 * perpetual["taker_buy_volume"] / perpetual["volume"].replace(0.0, np.nan) - 1.0
    ).fillna(0.0)
    result[prefix + "trades"] = perpetual["trades"]
    result[prefix + "basis"] = perpetual["close"] / spot["close"] - 1.0

    result_index = cast(pd.DatetimeIndex, result.index)
    funding, funding_mask = _asof_optional(data.funding, result_index)
    interest, interest_mask = _asof_optional(data.open_interest, result_index)
    premium, premium_mask = _asof_optional(data.premium, result_index)
    result[prefix + "funding"] = funding
    result[prefix + "funding_available"] = funding_mask
    result[prefix + "open_interest"] = interest
    result[prefix + "open_interest_change"] = interest.pct_change().replace([np.inf, -np.inf], 0.0)
    result[prefix + "open_interest_available"] = interest_mask
    result[prefix + "premium"] = premium
    result[prefix + "premium_available"] = premium_mask
    return result


def _causal_robust_normalize(values: pd.DataFrame) -> pd.DataFrame:
    masks = values.filter(regex="_available$")
    continuous = values.drop(columns=masks.columns)
    median = continuous.rolling(96, min_periods=4).median()
    deviation = (continuous - median).abs().rolling(96, min_periods=4).median()
    normalized = (continuous - median) / (1.4826 * deviation.replace(0.0, np.nan))
    normalized = normalized.clip(-10.0, 10.0).fillna(0.0)
    return pd.concat((normalized, masks), axis=1).sort_index(axis=1)


def build_market_state(dataset: CanonicalDataset) -> pd.DataFrame:
    """Build fixed causal 15-minute Market State from execution-complete history."""
    dataset.validate()
    complete = dataset.execution_complete()
    state = pd.concat(
        [_instrument_features(complete[ticker], ticker) for ticker in dataset.tickers], axis=1, join="inner"
    )
    return_columns = [f"{ticker}_return_15m" for ticker in dataset.tickers]
    market_return = state[return_columns].mean(axis=1)
    state["market_return_15m"] = market_return
    for ticker in dataset.tickers:
        state[f"{ticker}_relative_return_15m"] = state[f"{ticker}_return_15m"] - market_return
    state_index = cast(pd.DatetimeIndex, state.index)
    minute = state_index.hour * 60 + state_index.minute
    state["clock_day_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    state["clock_day_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    state["clock_week_sin"] = np.sin(2.0 * np.pi * (state_index.dayofweek * 1440 + minute) / (7 * 1440.0))
    state["clock_week_cos"] = np.cos(2.0 * np.pi * (state_index.dayofweek * 1440 + minute) / (7 * 1440.0))
    return _causal_robust_normalize(state)
