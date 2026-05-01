from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class TradingContract:
    timeframe: str = "1m"
    lookahead_steps: int = 256
    stop_loss_percent: float = 0.4
    take_profit_percent: float = 0.4
    commission: float = 0.001
    same_candle_policy: str = "negative"
    entry_timing: str = "next_open"
    min_probability_edge: float = 0.0

    def net_take_profit_return(self) -> float:
        buy_cost = 1 + self.commission
        sell_proceeds = (1 + self.take_profit_percent / 100) * (1 - self.commission)
        return (sell_proceeds - buy_cost) / buy_cost

    def net_stop_loss_return(self) -> float:
        buy_cost = 1 + self.commission
        sell_proceeds = (1 - self.stop_loss_percent / 100) * (1 - self.commission)
        return (sell_proceeds - buy_cost) / buy_cost

    def break_even_probability(self) -> float:
        win = self.net_take_profit_return()
        loss = -self.net_stop_loss_return()
        if win <= 0:
            return 1.0
        if loss <= 0:
            return 0.0
        return loss / (win + loss)

    def entry_probability_threshold(self) -> float:
        return min(1.0, self.break_even_probability() + self.min_probability_edge)


DEFAULT_TRADING_CONTRACT = TradingContract()


@dataclass(frozen=True)
class ExperimentConfig:
    tickers: tuple[str, ...]
    contract: TradingContract = field(default_factory=TradingContract)
    feature_set: str = "single_ticker_ohlc"
    split_policy: str = "strict_temporal_train_val_test"
    sampling_policy: str = "ticker_class_balanced"
    model_family: str = "mlp"
    seed: int = 42


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_config_hash(config: Any, length: int = 12) -> str:
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
