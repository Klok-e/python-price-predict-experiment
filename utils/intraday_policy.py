from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.experiment import stable_config_hash
from utils.experiment_runner import git_state, jsonable


MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass(frozen=True)
class IntradayConfig:
    tickers: tuple[str, ...]
    start_date: str
    end_date: str | None
    train_days: int
    validation_days: int
    evaluation_days: int
    stride_days: int
    horizon_grid: tuple[int, ...]
    max_hold_grid: tuple[int, ...]
    threshold_quantiles: tuple[float, ...]
    model_families: tuple[str, ...]
    include_futures_metrics: bool
    include_premium_index: bool
    commission: float
    slippage: float
    risk_unit: float
    risk_unit_grid: tuple[float, ...]
    max_exposure: float
    max_drawdown: float
    min_validation_trades: int
    selection_trade_floor: int
    selection_activity_weight: float
    validation_slices: int
    cash: float
    seed: int


@dataclass(frozen=True)
class CrossValidationConfig:
    folds: int
    windows_per_fold: int
    min_fold_trades: int


class PositiveReturnClassifier:
    def __init__(self, seed: int):
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=seed),
        )
        self.constant_probability = None

    def fit(self, x, y):
        labels = np.asarray(y > 0.0, dtype=int)
        if len(np.unique(labels)) < 2:
            self.constant_probability = float(labels[0]) if len(labels) else 0.0
            return self
        self.constant_probability = None
        self.model.fit(x, labels)
        return self

    def predict(self, x):
        if self.constant_probability is not None:
            return np.full(len(x), self.constant_probability)
        return self.model.predict_proba(x)[:, 1]


class MarketFeatureRegressor:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.feature_columns = None

    def fit(self, x, y):
        market_x = self._market_features(x)
        market_y = pd.Series(y).groupby(level=0).mean().reindex(market_x.index)
        valid = market_y.dropna().index
        self.feature_columns = tuple(market_x.columns)
        self.model.fit(market_x.loc[valid], market_y.loc[valid])
        return self

    def predict_matrix(self, datasets, tickers, start, end):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        common_index = None
        for ticker in tickers:
            features = datasets[ticker]["features"]
            index = features.index[(features.index >= start) & (features.index < end)]
            common_index = index if common_index is None else common_index.intersection(index)
        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()
        frames = [datasets[ticker]["features"].loc[common_index, self.feature_columns] for ticker in tickers]
        market_x = sum(frames) / len(frames)
        scores = self.model.predict(market_x)
        return pd.DataFrame({ticker: scores for ticker in tickers}, index=common_index)

    @staticmethod
    def _market_features(x):
        return x.drop(columns=["ticker_id"], errors="ignore").groupby(level=0).mean()


class DropColumnsRegressor:
    def __init__(self, columns):
        self.columns = tuple(columns)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.feature_columns = None

    def fit(self, x, y):
        frame = x.drop(columns=list(self.columns), errors="ignore")
        self.feature_columns = tuple(frame.columns)
        self.model.fit(frame, y)
        return self

    def predict(self, x):
        frame = x.drop(columns=list(self.columns), errors="ignore").reindex(columns=self.feature_columns)
        return self.model.predict(frame)


def int_grid(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def float_grid(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def after_cost_forward_return(bars: pd.DataFrame, horizon_minutes: int, commission: float, slippage: float):
    entry = bars["Open"].shift(-1)
    exit_price = bars["Open"].shift(-(1 + horizon_minutes))
    return exit_price / entry - 1.0 - 2.0 * (commission + slippage)


def after_cost_path_mean_forward_return(bars: pd.DataFrame, horizon_minutes: int, commission: float, slippage: float):
    returns = [
        after_cost_forward_return(bars, horizon, commission, slippage)
        for horizon in range(1, horizon_minutes + 1)
    ]
    return pd.concat(returns, axis=1).mean(axis=1, skipna=False)


def model_target_kind(model_family: str):
    if model_family.endswith("_path_mean_vol_scaled"):
        return "path_mean_vol_scaled"
    if model_family.endswith("_path_mean"):
        return "path_mean"
    if model_family.endswith("_market_excess_vol_scaled"):
        return "market_excess_vol_scaled"
    if base_model_family(model_family) == "market_ridge" and model_family.endswith("_vol_scaled"):
        return "market_return_vol_scaled"
    if base_model_family(model_family) == "market_ridge":
        return "market_return"
    if model_family.endswith("_vol_scaled"):
        return "vol_scaled"
    if model_family.endswith("_market_excess"):
        return "market_excess"
    if model_family.endswith("_market_return"):
        return "market_return"
    return "absolute"


def base_model_family(model_family: str):
    return (
        model_family
        .removesuffix("_vol_scaled")
        .removesuffix("_path_mean")
        .removesuffix("_market_excess")
        .removesuffix("_market_return")
    )


def decay_half_life_days(model_family: str):
    base = base_model_family(model_family)
    if not base.startswith("ridge_decay_"):
        return None
    return float(base.removeprefix("ridge_decay_").replace("_", "."))


def recency_sample_weights(index: pd.Index, half_life_days: float):
    if half_life_days <= 0.0:
        raise ValueError("half_life_days must be positive")
    timestamps = pd.DatetimeIndex(index)
    age_days = (timestamps.max() - timestamps).total_seconds() / 86400.0
    return np.power(0.5, age_days / half_life_days)


def forward_return_frame(datasets, tickers, horizon_minutes, commission, slippage):
    return pd.DataFrame(
        {
            ticker: after_cost_forward_return(datasets[ticker]["bars"], horizon_minutes, commission, slippage)
            for ticker in tickers
        }
    )


def path_mean_forward_return_frame(datasets, tickers, horizon_minutes, commission, slippage):
    return pd.DataFrame(
        {
            ticker: after_cost_path_mean_forward_return(
                datasets[ticker]["bars"],
                horizon_minutes,
                commission,
                slippage,
            )
            for ticker in tickers
        }
    )


def train_frame(datasets, tickers, horizon_minutes, start, end, commission, slippage, target_kind="absolute"):
    x_frames = []
    y_frames = []
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if target_kind in {"path_mean", "path_mean_vol_scaled"}:
        forward_returns = path_mean_forward_return_frame(datasets, tickers, horizon_minutes, commission, slippage)
    else:
        forward_returns = forward_return_frame(datasets, tickers, horizon_minutes, commission, slippage)
    market_return = forward_returns.mean(axis=1)
    for ticker_id, ticker in enumerate(tickers):
        features = datasets[ticker]["features"]
        target = forward_returns[ticker]
        if target_kind == "market_excess":
            target = target - market_return
        elif target_kind == "market_return":
            target = market_return
        elif target_kind == "market_return_vol_scaled":
            target = market_return / (features["market_volatility_60m"].abs() + 1e-6)
        elif target_kind == "market_excess_vol_scaled":
            target = (target - market_return) / (features["volatility_60m"].abs() + 1e-6)
        elif target_kind == "vol_scaled":
            target = target / (features["volatility_60m"].abs() + 1e-6)
        elif target_kind == "path_mean_vol_scaled":
            target = target / (features["volatility_60m"].abs() + 1e-6)
        elif target_kind == "path_mean":
            target = target
        elif target_kind != "absolute":
            raise ValueError(f"Unknown intraday target kind: {target_kind}")
        index = features.index.intersection(target.dropna().index)
        index = index[(index >= start) & (index < end)]
        x_frames.append(features.loc[index].assign(ticker_id=ticker_id))
        y_frames.append(target.loc[index])
    if not x_frames:
        return pd.DataFrame(), pd.Series(dtype=float)
    return pd.concat(x_frames), pd.concat(y_frames)


def prediction_matrix(datasets, tickers, model, start, end):
    if hasattr(model, "predict_matrix"):
        return model.predict_matrix(datasets, tickers, start, end).dropna()
    frames = []
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    for ticker_id, ticker in enumerate(tickers):
        features = datasets[ticker]["features"]
        index = features.index[(features.index >= start) & (features.index < end)]
        if len(index) == 0:
            continue
        frame = features.loc[index].assign(ticker_id=ticker_id)
        frames.append(pd.Series(model.predict(frame), index=index, name=ticker))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna()


def build_model(model_family: str, seed: int):
    model_family = base_model_family(model_family)
    if model_family == "market_ridge":
        return MarketFeatureRegressor()
    if model_family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_family == "ridge_shared":
        return DropColumnsRegressor(("ticker_id",))
    if model_family.startswith("ridge_decay_"):
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_family.startswith("ridge_alpha_"):
        alpha = float(model_family.removeprefix("ridge_alpha_").replace("_", "."))
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if model_family == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=200))
    if model_family == "logistic_positive":
        return PositiveReturnClassifier(seed)
    if model_family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=seed)
    raise ValueError(f"Unknown intraday model family: {model_family}")


def fit_model(model, model_family: str, x: pd.DataFrame, y: pd.Series):
    half_life_days = decay_half_life_days(model_family)
    if half_life_days is None:
        return model.fit(x, y)
    weights = recency_sample_weights(x.index, half_life_days)
    return model.fit(x, y, standardscaler__sample_weight=weights, ridge__sample_weight=weights)


def equity_metrics(equity: pd.Series, trades: int, start_cash: float, days: float):
    if equity.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades": int(trades),
            "trades_per_day": 0.0,
            "start_cash": float(start_cash),
            "end_equity": float(start_cash),
        }
    returns = equity.pct_change().dropna()
    sharpe = 0.0
    if len(returns) and float(returns.std()) > 0.0:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(MINUTES_PER_YEAR))
    drawdown = equity / equity.cummax() - 1.0
    return {
        "cumulative_return": float(equity.iloc[-1] / start_cash - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(abs(drawdown.min())) if len(drawdown) else 0.0,
        "trades": int(trades),
        "trades_per_day": float(trades / days) if days > 0 else 0.0,
        "start_cash": float(start_cash),
        "end_equity": float(equity.iloc[-1]),
    }


def common_open_frame(datasets, tickers, start, end):
    frame = pd.DataFrame({ticker: datasets[ticker]["bars"]["Open"] for ticker in tickers}).dropna()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return frame.loc[(frame.index >= start) & (frame.index < end)]


def buy_hold_replay(datasets, tickers, config: IntradayConfig, start, end, start_cash=None):
    opens = common_open_frame(datasets, tickers, start, end)
    cash = float(config.cash if start_cash is None else start_cash)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400.0
    if len(opens) < 2:
        return {
            "metrics": equity_metrics(pd.Series(dtype=float), 0, cash, days),
            "positions": pd.DataFrame(),
        }

    weight = float(config.max_exposure / len(tickers))
    entry_cost = config.max_exposure * (config.commission + config.slippage)
    exit_cost = config.max_exposure * (config.commission + config.slippage)
    returns = opens.pct_change().dropna().mul(weight).sum(axis=1)
    equity = cash * max(0.0, 1.0 - entry_cost) * (1.0 + returns).cumprod()
    equity.iloc[-1] = equity.iloc[-1] * max(0.0, 1.0 - exit_cost)
    positions = pd.DataFrame(
        {
            "signal_time": opens.index[:-1],
            "entry_time": opens.index[:-1],
            "exit_time": opens.index[1:],
            "equity": equity.to_numpy(),
            "period_return": returns.to_numpy(),
            "turnover": 0.0,
            "cost": 0.0,
            "long_exposure": float(config.max_exposure),
            **{f"weight_{ticker}": weight for ticker in tickers},
        }
    )
    positions.loc[positions.index[0], "turnover"] = float(config.max_exposure)
    positions.loc[positions.index[0], "cost"] = float(entry_cost)
    positions.loc[positions.index[-1], "turnover"] += float(config.max_exposure)
    positions.loc[positions.index[-1], "cost"] += float(exit_cost)
    trades = len(tickers) * 2
    return {"metrics": equity_metrics(equity, trades, cash, days), "positions": positions}


def flat_replay(common_index, tickers, start, end, equity):
    index = common_index[(common_index >= pd.Timestamp(start)) & (common_index < pd.Timestamp(end))]
    if len(index) == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "signal_time": index,
            "entry_time": index,
            "exit_time": index,
            "equity": float(equity),
            "period_return": 0.0,
            "turnover": 0.0,
            "cost": 0.0,
            "long_exposure": 0.0,
            **{f"weight_{ticker}": 0.0 for ticker in tickers},
        }
    )


def _target_weights(
    scores,
    previous_weights,
    held_minutes,
    entry_threshold,
    exit_threshold,
    horizon_minutes,
    max_hold_minutes,
    config: IntradayConfig,
):
    if max_hold_minutes < horizon_minutes:
        raise ValueError("max_hold_minutes must be greater than or equal to horizon_minutes")
    weights = {}
    active_count = 0
    max_active = int(config.max_exposure / config.risk_unit)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for ticker, score in ranked:
        was_active = previous_weights.get(ticker, 0.0) > 0.0
        held = held_minutes.get(ticker, 0)
        must_hold = was_active and held < horizon_minutes
        can_extend = was_active and held < max_hold_minutes and score >= exit_threshold
        can_open = (not was_active) and score >= entry_threshold and active_count < max_active
        weights[ticker] = float(config.risk_unit if (must_hold or can_extend or can_open) else 0.0)
        if weights[ticker] > 0:
            active_count += 1
    return weights


def replay_long_only(
    datasets,
    tickers,
    predictions,
    threshold,
    max_hold_minutes,
    config: IntradayConfig,
    start,
    end,
    record_ledger: bool = True,
    horizon_minutes: int = 1,
    exit_threshold: float | None = None,
):
    index = pd.DatetimeIndex(predictions.index)
    index = index[(index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))]
    if len(index) < 2:
        return {"metrics": equity_metrics(pd.Series(dtype=float), 0, config.cash, 0), "positions": pd.DataFrame(), "ledger_records": []}

    equity = float(config.cash)
    previous_weights = {ticker: 0.0 for ticker in tickers}
    held_minutes = {ticker: 0 for ticker in tickers}
    rows = []
    equity_values = []
    ledger_records = []
    trades = 0
    entry_threshold = float(threshold)
    exit_threshold = entry_threshold if exit_threshold is None else float(exit_threshold)

    for signal_time in index[:-2]:
        entry_time = signal_time + pd.Timedelta(minutes=1)
        exit_time = signal_time + pd.Timedelta(minutes=2)
        scores = {ticker: float(predictions.loc[signal_time, ticker]) for ticker in tickers}
        weights = _target_weights(
            scores,
            previous_weights,
            held_minutes,
            entry_threshold,
            exit_threshold,
            horizon_minutes,
            max_hold_minutes,
            config,
        )
        turnover = float(sum(abs(weights[ticker] - previous_weights[ticker]) for ticker in tickers))
        trades += int(sum(weights[ticker] != previous_weights[ticker] for ticker in tickers))
        period_return = 0.0
        complete = True
        for ticker in tickers:
            bars = datasets[ticker]["bars"]
            if entry_time not in bars.index or exit_time not in bars.index:
                complete = False
                break
            one_minute_return = float(bars.loc[exit_time, "Open"] / bars.loc[entry_time, "Open"] - 1.0)
            period_return += weights[ticker] * one_minute_return
        if not complete:
            if record_ledger:
                ledger_records.append({
                    "record_type": "no_action",
                    "signal_time": signal_time.isoformat(),
                    "reason": "missing_next_open_fill_data",
                })
            continue
        cost = turnover * (config.commission + config.slippage)
        equity *= max(0.0, 1.0 + period_return - cost)
        equity_values.append(equity)
        decisions = []
        for ticker in tickers:
            delta = weights[ticker] - previous_weights[ticker]
            if record_ledger:
                action = "buy" if delta > 0 else "sell" if delta < 0 else "hold"
                decisions.append({
                    "ticker": ticker,
                    "action": action,
                    "score": scores[ticker],
                    "previous_weight": previous_weights[ticker],
                    "target_weight": weights[ticker],
                    "weight_delta": delta,
                })
            held_minutes[ticker] = held_minutes[ticker] + 1 if weights[ticker] > 0 else 0
        if record_ledger:
            rows.append({
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "equity": equity,
                "period_return": period_return,
                "turnover": turnover,
                "cost": cost,
                "long_exposure": float(sum(weights.values())),
                **{f"weight_{ticker}": weights[ticker] for ticker in tickers},
            })
            ledger_records.append({
                "record_type": "intraday_decision",
                "signal_time": signal_time.isoformat(),
                "scores": scores,
                "decisions": decisions,
                "threshold": float(threshold),
                "exit_threshold": float(exit_threshold),
                "horizon_minutes": int(horizon_minutes),
                "max_hold_minutes": int(max_hold_minutes),
            })
        previous_weights = weights

    positions = pd.DataFrame(rows)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400.0
    equity_series = positions["equity"] if not positions.empty else pd.Series(equity_values, dtype=float)
    metrics = equity_metrics(equity_series, trades, config.cash, days)
    return {"metrics": metrics, "positions": positions, "ledger_records": ledger_records}


def threshold_candidates(predictions: pd.DataFrame, quantiles: tuple[float, ...]):
    values = predictions.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (0.0,)
    candidates = {0.0}
    for quantile in quantiles:
        candidates.add(float(np.quantile(values, quantile)))
    return tuple(sorted(candidates))


def exit_threshold_candidates(thresholds: tuple[float, ...], threshold: float, horizon: int, max_hold: int):
    if max_hold == horizon:
        return (threshold,)
    return tuple(candidate for candidate in thresholds if candidate <= threshold)


def selected_generalization(selection, evaluation_metrics, evaluation_buy_hold_metrics):
    validation_metrics = selection["validation_metrics"]
    validation_excess = float(selection["validation_excess_return"])
    evaluation_excess = float(evaluation_metrics["cumulative_return"] - evaluation_buy_hold_metrics["cumulative_return"])
    return {
        "validation_excess_return": validation_excess,
        "evaluation_excess_return": evaluation_excess,
        "excess_return_delta": float(evaluation_excess - validation_excess),
        "validation_cumulative_return": float(validation_metrics["cumulative_return"]),
        "evaluation_cumulative_return": float(evaluation_metrics["cumulative_return"]),
        "cumulative_return_delta": float(
            evaluation_metrics["cumulative_return"] - validation_metrics["cumulative_return"]
        ),
        "validation_sharpe": float(validation_metrics["sharpe"]),
        "evaluation_sharpe": float(evaluation_metrics["sharpe"]),
        "sharpe_delta": float(evaluation_metrics["sharpe"] - validation_metrics["sharpe"]),
        "validation_trades": int(validation_metrics["trades"]),
        "evaluation_trades": int(evaluation_metrics["trades"]),
        "trades_delta": int(evaluation_metrics["trades"] - validation_metrics["trades"]),
        "validation_max_drawdown": float(validation_metrics["max_drawdown"]),
        "evaluation_max_drawdown": float(evaluation_metrics["max_drawdown"]),
        "max_drawdown_delta": float(evaluation_metrics["max_drawdown"] - validation_metrics["max_drawdown"]),
    }


def validation_slice_periods(start, end, slice_count: int):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if slice_count <= 1:
        return ((start, end),)
    edges = pd.date_range(start=start, end=end, periods=slice_count + 1)
    return tuple((edges[i], edges[i + 1]) for i in range(slice_count))


def validation_slice_summary(
    datasets,
    tickers,
    predictions,
    threshold,
    max_hold,
    config: IntradayConfig,
    validation_start,
    evaluation_start,
    horizon,
    exit_threshold,
):
    summaries = []
    for slice_start, slice_end in validation_slice_periods(
        validation_start,
        evaluation_start,
        config.validation_slices,
    ):
        replay = replay_long_only(
            datasets,
            tickers,
            predictions,
            threshold,
            max_hold,
            config,
            slice_start,
            slice_end,
            record_ledger=False,
            horizon_minutes=horizon,
            exit_threshold=exit_threshold,
        )
        buy_hold = buy_hold_replay(datasets, tickers, config, slice_start, slice_end)["metrics"]
        metrics = replay["metrics"]
        summaries.append({
            "start": slice_start.isoformat(),
            "end": slice_end.isoformat(),
            "cumulative_return": float(metrics["cumulative_return"]),
            "buy_hold_return": float(buy_hold["cumulative_return"]),
            "excess_return": float(metrics["cumulative_return"] - buy_hold["cumulative_return"]),
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "trades": int(metrics["trades"]),
        })
    return summaries


def select_window_policy(datasets, tickers, config: IntradayConfig, train_start, validation_start, evaluation_start):
    best = None
    best_candidate = None
    candidate_summaries = []
    candidate_count = 0
    validation_buy_hold = buy_hold_replay(datasets, tickers, config, validation_start, evaluation_start)["metrics"]
    for horizon in config.horizon_grid:
        for model_family in config.model_families:
            x_train, y_train = train_frame(
                datasets,
                tickers,
                horizon,
                train_start,
                validation_start,
                config.commission,
                config.slippage,
                target_kind=model_target_kind(model_family),
            )
            if x_train.empty:
                continue
            model = build_model(model_family, config.seed)
            fit_model(model, model_family, x_train, y_train)
            validation_predictions = prediction_matrix(datasets, tickers, model, validation_start, evaluation_start)
            thresholds = threshold_candidates(validation_predictions, config.threshold_quantiles)
            for threshold in thresholds:
                for max_hold in config.max_hold_grid:
                    if max_hold < horizon:
                        continue
                    exit_thresholds = exit_threshold_candidates(thresholds, threshold, horizon, max_hold)
                    for risk_unit in config.risk_unit_grid:
                        candidate_config = replace(config, risk_unit=risk_unit)
                        for exit_threshold in exit_thresholds:
                            replay = replay_long_only(
                                datasets,
                                tickers,
                                validation_predictions,
                                threshold,
                                max_hold,
                                candidate_config,
                                validation_start,
                                evaluation_start,
                                record_ledger=False,
                                horizon_minutes=horizon,
                                exit_threshold=exit_threshold,
                            )
                            metrics = replay["metrics"]
                            candidate_count += 1
                            validation_excess_return = metrics["cumulative_return"] - validation_buy_hold["cumulative_return"]
                            passed_activity_gate = (
                                metrics["trades"] >= config.min_validation_trades
                                and metrics["max_drawdown"] <= config.max_drawdown
                                and metrics["sharpe"] > 0.0
                                and metrics["cumulative_return"] > 0.0
                            )
                            slice_summary = ()
                            slice_excess_median = None
                            slice_excess_min = None
                            if config.validation_slices > 1 and passed_activity_gate:
                                slice_summary = validation_slice_summary(
                                    datasets,
                                    tickers,
                                    validation_predictions,
                                    threshold,
                                    max_hold,
                                    candidate_config,
                                    validation_start,
                                    evaluation_start,
                                    horizon,
                                    exit_threshold,
                                )
                                slice_excesses = [item["excess_return"] for item in slice_summary]
                                slice_excess_median = float(np.median(slice_excesses))
                                slice_excess_min = float(np.min(slice_excesses))
                            passed_gate = (
                                passed_activity_gate
                                and validation_excess_return > 0.0
                            )
                            passed_selection_trade_floor = metrics["trades"] >= config.selection_trade_floor
                            selection_score = (
                                (slice_excess_median if slice_excess_median is not None else validation_excess_return)
                                + config.selection_activity_weight * math.log1p(metrics["trades"])
                            )
                            candidate_summary = {
                                "model_family": model_family,
                                "horizon_minutes": int(horizon),
                                "threshold": float(threshold),
                                "exit_threshold": float(exit_threshold),
                                "max_hold_minutes": int(max_hold),
                                "risk_unit": float(risk_unit),
                                "validation_metrics": metrics,
                                "validation_buy_hold_metrics": validation_buy_hold,
                                "validation_excess_return": float(validation_excess_return),
                                "validation_slice_excess_median": slice_excess_median,
                                "validation_slice_excess_min": slice_excess_min,
                                "validation_slices": slice_summary,
                                "selection_score": float(selection_score),
                                "passed_activity_gate": passed_activity_gate,
                                "passed_selection_trade_floor": passed_selection_trade_floor,
                                "passed_gate": passed_gate,
                            }
                            candidate_summaries.append(candidate_summary)
                            if (
                                best_candidate is None
                                or (
                                    passed_activity_gate
                                    and not best_candidate["passed_activity_gate"]
                                )
                                or (
                                    passed_activity_gate == best_candidate["passed_activity_gate"]
                                    and passed_selection_trade_floor
                                    and not best_candidate["passed_selection_trade_floor"]
                                )
                                or (
                                    passed_activity_gate == best_candidate["passed_activity_gate"]
                                    and passed_selection_trade_floor == best_candidate["passed_selection_trade_floor"]
                                    and selection_score > best_candidate["selection_score"]
                                )
                            ):
                                best_candidate = candidate_summary
                            candidate = {
                                "model": model,
                                "model_family": model_family,
                                "horizon_minutes": horizon,
                                "threshold": threshold,
                                "exit_threshold": exit_threshold,
                                "max_hold_minutes": max_hold,
                                "risk_unit": risk_unit,
                                "validation_metrics": metrics,
                                "validation_buy_hold_metrics": validation_buy_hold,
                                "validation_excess_return": float(validation_excess_return),
                                "validation_slice_excess_median": slice_excess_median,
                                "validation_slice_excess_min": slice_excess_min,
                                "validation_slices": slice_summary,
                                "selection_score": float(selection_score),
                                "passed_activity_gate": passed_activity_gate,
                                "passed_selection_trade_floor": passed_selection_trade_floor,
                                "passed_gate": passed_gate,
                            }
                            if passed_activity_gate:
                                candidate_key = (
                                    passed_selection_trade_floor,
                                    selection_score,
                                    metrics["sharpe"],
                                )
                                best_key = (
                                    best["passed_selection_trade_floor"],
                                    best["selection_score"],
                                    best["validation_metrics"]["sharpe"],
                                ) if best else None
                                if best is None or candidate_key > best_key:
                                    best = candidate
    return {
        "selected": best,
        "candidate_count": candidate_count,
        "validation_buy_hold_metrics": validation_buy_hold,
        "best_candidate": best_candidate,
        "top_candidates": sorted(
            candidate_summaries,
            key=lambda item: (
                item["passed_activity_gate"],
                item["passed_selection_trade_floor"],
                item["selection_score"],
                item["validation_metrics"]["sharpe"],
            ),
            reverse=True,
        )[:10],
    }


def walk_forward_windows(index: pd.DatetimeIndex, config: IntradayConfig):
    start = max(pd.Timestamp(config.start_date), index.min())
    end = pd.Timestamp(config.end_date) if config.end_date else index.max()
    cursor = start + pd.Timedelta(days=config.train_days + config.validation_days)
    while cursor + pd.Timedelta(days=config.evaluation_days) <= end:
        train_start = cursor - pd.Timedelta(days=config.validation_days + config.train_days)
        validation_start = cursor - pd.Timedelta(days=config.validation_days)
        evaluation_start = cursor
        evaluation_end = cursor + pd.Timedelta(days=config.evaluation_days)
        yield train_start, validation_start, evaluation_start, evaluation_end
        cursor += pd.Timedelta(days=config.stride_days)


def max_common_feature_index(datasets, tickers):
    common = None
    for ticker in tickers:
        index = datasets[ticker]["features"].index
        common = index if common is None else common.intersection(index)
    return pd.DatetimeIndex(common).sort_values()


def cross_validation_fold_span(config: IntradayConfig, cv_config: CrossValidationConfig):
    return pd.Timedelta(
        days=(
            config.evaluation_days
            + config.stride_days * (cv_config.windows_per_fold - 1)
        )
    )


def rolling_cross_validation_folds(
    common_index: pd.DatetimeIndex,
    config: IntradayConfig,
    cv_config: CrossValidationConfig,
):
    if cv_config.folds <= 0:
        return []
    if cv_config.windows_per_fold <= 0:
        raise ValueError("cv_windows_per_fold must be greater than 0")
    if config.stride_days != config.evaluation_days:
        raise ValueError("Cross-validation requires stride_days to equal evaluation_days")
    if common_index.empty:
        raise ValueError("Cannot build cross-validation folds without common feature rows")

    data_start = max(pd.Timestamp(config.start_date), common_index.min())
    requested_end = pd.Timestamp(config.end_date) if config.end_date else common_index.max()
    data_end = min(requested_end, common_index.max())
    evaluation_span = cross_validation_fold_span(config, cv_config)
    first_evaluation_start = data_end - evaluation_span * cv_config.folds
    first_fold_start = first_evaluation_start - pd.Timedelta(days=config.train_days + config.validation_days)
    if first_fold_start < data_start:
        raise ValueError(
            "Not enough data for requested cross-validation folds: "
            f"need start <= {first_fold_start.isoformat()}, available start is {data_start.isoformat()}"
        )

    folds = []
    for fold_index in range(cv_config.folds):
        evaluation_start = first_evaluation_start + evaluation_span * fold_index
        evaluation_end = evaluation_start + evaluation_span
        fold_start = evaluation_start - pd.Timedelta(days=config.train_days + config.validation_days)
        fold_end = evaluation_end
        fold_config = replace(
            config,
            start_date=fold_start.isoformat(),
            end_date=fold_end.isoformat(),
        )
        folds.append({
            "fold_index": fold_index,
            "fold_start": fold_start,
            "fold_end": fold_end,
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            "config": fold_config,
        })
    return folds


def run_walk_forward_intraday(datasets, config: IntradayConfig):
    common_index = max_common_feature_index(datasets, config.tickers)
    windows = []
    all_positions = []
    all_benchmark_positions = []
    ledger_records = []
    global_equity = float(config.cash)
    benchmark_equity = float(config.cash)
    for window_index, (train_start, validation_start, evaluation_start, evaluation_end) in enumerate(
        walk_forward_windows(common_index, config)
    ):
        selection_result = select_window_policy(
            datasets, config.tickers, config, train_start, validation_start, evaluation_start
        )
        selection = selection_result["selected"]
        window_buy_hold = buy_hold_replay(datasets, config.tickers, config, evaluation_start, evaluation_end)
        global_buy_hold = buy_hold_replay(
            datasets, config.tickers, config, evaluation_start, evaluation_end, start_cash=benchmark_equity
        )
        benchmark_positions = global_buy_hold["positions"].copy()
        if not benchmark_positions.empty:
            benchmark_equity = float(benchmark_positions["equity"].iloc[-1])
            benchmark_positions["window_index"] = window_index
            all_benchmark_positions.append(benchmark_positions)
        if selection is None:
            flat_positions = flat_replay(common_index, config.tickers, evaluation_start, evaluation_end, global_equity)
            if not flat_positions.empty:
                flat_positions["window_index"] = window_index
                all_positions.append(flat_positions)
            windows.append({
                "window_index": window_index,
                "train_start": train_start.isoformat(),
                "validation_start": validation_start.isoformat(),
                "evaluation_start": evaluation_start.isoformat(),
                "evaluation_end": evaluation_end.isoformat(),
                "candidate_count": selection_result["candidate_count"],
                "best_candidate": selection_result["best_candidate"],
                "top_candidates": selection_result["top_candidates"],
                "selected": None,
                "buy_hold_metrics": window_buy_hold["metrics"],
                "evaluation_metrics": None,
                "excess_return": None,
                "beats_buy_hold": False,
                "passed_gate": False,
            })
            continue
        predictions = prediction_matrix(datasets, config.tickers, selection["model"], evaluation_start, evaluation_end)
        replay_config = replace(config, risk_unit=selection["risk_unit"])
        replay = replay_long_only(
            datasets,
            config.tickers,
            predictions,
            selection["threshold"],
            selection["max_hold_minutes"],
            replay_config,
            evaluation_start,
            evaluation_end,
            horizon_minutes=selection["horizon_minutes"],
            exit_threshold=selection["exit_threshold"],
        )
        positions = replay["positions"].copy()
        if not positions.empty:
            scale = global_equity / config.cash
            positions["equity"] = positions["equity"] * scale
            replay["metrics"]["start_cash"] = float(global_equity)
            replay["metrics"]["end_equity"] = float(positions["equity"].iloc[-1])
            replay["metrics"]["cumulative_return"] = float(positions["equity"].iloc[-1] / global_equity - 1.0)
            global_equity = float(positions["equity"].iloc[-1])
            positions["window_index"] = window_index
            all_positions.append(positions)
        ledger_records.extend(replay["ledger_records"])
        excess_return = replay["metrics"]["cumulative_return"] - window_buy_hold["metrics"]["cumulative_return"]
        windows.append({
            "window_index": window_index,
            "train_start": train_start.isoformat(),
            "validation_start": validation_start.isoformat(),
            "evaluation_start": evaluation_start.isoformat(),
            "evaluation_end": evaluation_end.isoformat(),
            "candidate_count": selection_result["candidate_count"],
            "best_candidate": selection_result["best_candidate"],
            "top_candidates": selection_result["top_candidates"],
            "selected": {
                "model_family": selection["model_family"],
                "horizon_minutes": int(selection["horizon_minutes"]),
                "threshold": float(selection["threshold"]),
                "exit_threshold": float(selection["exit_threshold"]),
                "max_hold_minutes": int(selection["max_hold_minutes"]),
                "risk_unit": float(selection["risk_unit"]),
                "validation_metrics": selection["validation_metrics"],
                "validation_buy_hold_metrics": selection["validation_buy_hold_metrics"],
                "validation_excess_return": float(selection["validation_excess_return"]),
                "validation_slice_excess_median": selection["validation_slice_excess_median"],
                "validation_slice_excess_min": selection["validation_slice_excess_min"],
                "validation_slices": selection["validation_slices"],
                "selection_score": float(selection["selection_score"]),
                "passed_activity_gate": bool(selection["passed_activity_gate"]),
                "passed_selection_trade_floor": bool(selection["passed_selection_trade_floor"]),
                "passed_gate": bool(selection["passed_gate"]),
            },
            "buy_hold_metrics": window_buy_hold["metrics"],
            "evaluation_metrics": replay["metrics"],
            "selected_generalization": selected_generalization(selection, replay["metrics"], window_buy_hold["metrics"]),
            "excess_return": float(excess_return),
            "beats_buy_hold": bool(excess_return > 0.0),
            "passed_gate": (
                replay["metrics"]["sharpe"] > 0.0
                and replay["metrics"]["max_drawdown"] <= config.max_drawdown
                and excess_return > 0.0
            ),
        })
    positions = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()
    benchmark_positions = (
        pd.concat(all_benchmark_positions, ignore_index=True) if all_benchmark_positions else pd.DataFrame()
    )
    aggregate = aggregate_windows(windows, positions, benchmark_positions, config)
    return {
        "windows": windows,
        "positions": positions,
        "benchmark_positions": benchmark_positions,
        "ledger_records": ledger_records,
        "aggregate": aggregate,
    }


def aggregate_windows(windows, positions, benchmark_positions, config: IntradayConfig):
    evaluated = [window for window in windows if window["evaluation_metrics"] is not None]
    benchmark_trades = int(sum(window["buy_hold_metrics"]["trades"] for window in windows))
    total_days = len(windows) * config.evaluation_days
    benchmark_metrics = equity_metrics(
        benchmark_positions["equity"] if not benchmark_positions.empty else pd.Series(dtype=float),
        benchmark_trades,
        config.cash,
        total_days,
    )
    if positions.empty:
        return {
            "window_count": len(windows),
            "evaluated_window_count": len(evaluated),
            "passed_window_count": 0,
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "trades_per_day": 0.0,
            "buy_hold": benchmark_metrics,
            "excess_return": -benchmark_metrics["cumulative_return"],
            "beats_buy_hold": False,
            "passed": False,
        }
    metrics = equity_metrics(
        positions["equity"],
        int(sum(window["evaluation_metrics"]["trades"] for window in evaluated)),
        config.cash,
        total_days,
    )
    excess_return = metrics["cumulative_return"] - benchmark_metrics["cumulative_return"]
    beats_buy_hold = excess_return > 0.0
    return {
        "window_count": len(windows),
        "evaluated_window_count": len(evaluated),
        "passed_window_count": int(sum(1 for window in windows if window.get("passed_gate"))),
        **metrics,
        "buy_hold": benchmark_metrics,
        "excess_return": float(excess_return),
        "beats_buy_hold": bool(beats_buy_hold),
        "passed": (
            metrics["sharpe"] > 0.0
            and metrics["max_drawdown"] <= config.max_drawdown
            and beats_buy_hold
        ),
    }


def write_intraday_timeline_html(positions: pd.DataFrame, benchmark_positions: pd.DataFrame, ledger_records, output_path):
    figure = go.Figure()
    if not positions.empty:
        figure.add_trace(go.Scatter(x=positions["signal_time"], y=positions["equity"], mode="lines", name="Equity"))
    if not benchmark_positions.empty:
        figure.add_trace(
            go.Scatter(
                x=benchmark_positions["signal_time"],
                y=benchmark_positions["equity"],
                mode="lines",
                name="Buy and hold",
                line={"dash": "dot", "color": "#9467bd"},
            )
        )
    grouped = {
        "buy": {"x": [], "y": [], "text": [], "name": "Buy", "color": "#2ca02c", "symbol": "triangle-up"},
        "sell": {"x": [], "y": [], "text": [], "name": "Sell to cash", "color": "#d62728", "symbol": "triangle-down"},
        "hold": {"x": [], "y": [], "text": [], "name": "Hold", "color": "#7f7f7f", "symbol": "circle-open"},
    }
    equity_by_time = {}
    if not positions.empty:
        equity_by_time = {pd.Timestamp(row["signal_time"]).isoformat(): float(row["equity"]) for _, row in positions.iterrows()}
    for record in ledger_records:
        if record.get("record_type") != "intraday_decision":
            continue
        decisions = record["decisions"]
        action = "buy" if any(item["action"] == "buy" for item in decisions) else "sell" if any(item["action"] == "sell" for item in decisions) else "hold"
        signal_time = pd.Timestamp(record["signal_time"])
        grouped[action]["x"].append(signal_time)
        grouped[action]["y"].append(equity_by_time.get(signal_time.isoformat(), None))
        summary = ", ".join(
            f"{item['ticker']} {item['action']} {item['weight_delta']:.4f}" for item in decisions
        )
        grouped[action]["text"].append(
            f"<b>{action}</b><br>signal_time: {signal_time.isoformat()}<br>actions: {summary}<br>"
            f"threshold: {record['threshold']:.8f}<br>exit_threshold: {record['exit_threshold']:.8f}<br>"
            f"horizon: {record['horizon_minutes']}m<br>"
            f"max_hold: {record['max_hold_minutes']}m"
        )
    for group in grouped.values():
        figure.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers",
                name=group["name"],
                marker={"color": group["color"], "symbol": group["symbol"], "size": 8},
                text=group["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Learned 1m Intraday Decision Timeline",
        xaxis_title="Time",
        yaxis_title="Equity",
        template="plotly_white",
        hovermode="closest",
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def build_intraday_report(config: IntradayConfig, run_id: str, result, artifact_paths):
    return {
        "run_id": run_id,
        "report_type": "intraday_report",
        "config": asdict(config),
        "aggregate": result["aggregate"],
        "windows": result["windows"],
        "artifact_paths": artifact_paths,
        "git": git_state(),
    }


def write_intraday_artifacts(config: IntradayConfig, run_id: str, result, run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    positions_path = os.path.join(run_dir, "intraday_positions.csv")
    benchmark_positions_path = os.path.join(run_dir, "intraday_buy_hold_positions.csv")
    ledger_path = os.path.join(run_dir, "intraday_ledger.jsonl")
    timeline_path = os.path.join(run_dir, "intraday_decision_timeline.html")
    report_path = os.path.join(run_dir, "intraday_report.json")
    if not result["positions"].empty:
        result["positions"].to_csv(positions_path, index=False)
    if not result["benchmark_positions"].empty:
        result["benchmark_positions"].to_csv(benchmark_positions_path, index=False)
    with open(ledger_path, "w", encoding="utf-8") as file:
        for record in result["ledger_records"]:
            file.write(json.dumps(jsonable(record), sort_keys=True))
            file.write("\n")
    write_intraday_timeline_html(
        result["positions"], result["benchmark_positions"], result["ledger_records"], timeline_path
    )
    artifact_paths = {
        "positions": positions_path,
        "buy_hold_positions": benchmark_positions_path,
        "ledger": ledger_path,
        "decision_timeline": timeline_path,
        "report": report_path,
    }
    report = build_intraday_report(config, run_id, result, artifact_paths)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(jsonable(report), file, indent=2, sort_keys=True)
    return report


def _median(values):
    return float(np.median(values)) if values else 0.0


def summarize_cross_validation(fold_reports, pooled_metrics, config: IntradayConfig, cv_config: CrossValidationConfig):
    aggregates = [fold["aggregate"] for fold in fold_reports]
    median_excess_return = _median([item["excess_return"] for item in aggregates])
    median_sharpe = _median([item["sharpe"] for item in aggregates])
    max_fold_drawdown = max((item["max_drawdown"] for item in aggregates), default=0.0)
    min_fold_excess_return = min((item["excess_return"] for item in aggregates), default=0.0)
    min_fold_trades = min((item["trades"] for item in aggregates), default=0)
    return {
        "fold_count": len(fold_reports),
        "passed_fold_count": int(sum(1 for item in aggregates if item["passed"])),
        "beats_buy_hold_fold_count": int(sum(1 for item in aggregates if item["beats_buy_hold"])),
        "active_fold_count": int(sum(1 for item in aggregates if item["trades"] >= cv_config.min_fold_trades)),
        "median_cumulative_return": _median([item["cumulative_return"] for item in aggregates]),
        "median_buy_hold_return": _median([item["buy_hold"]["cumulative_return"] for item in aggregates]),
        "median_excess_return": median_excess_return,
        "median_sharpe": median_sharpe,
        "max_fold_drawdown": float(max_fold_drawdown),
        "min_fold_excess_return": float(min_fold_excess_return),
        "min_fold_trades": int(min_fold_trades),
        "min_cv_fold_trades": int(cv_config.min_fold_trades),
        "pooled": pooled_metrics,
        "passed_cv": (
            median_excess_return > 0.0
            and median_sharpe > 0.0
            and max_fold_drawdown <= config.max_drawdown
            and min_fold_excess_return >= 0.0
            and min_fold_trades >= cv_config.min_fold_trades
            and pooled_metrics["excess_return"] > 0.0
        ),
    }


def fold_report_summary(fold, report):
    aggregate = report["aggregate"]
    return {
        "fold_index": int(fold["fold_index"]),
        "fold_start": fold["fold_start"].isoformat(),
        "fold_end": fold["fold_end"].isoformat(),
        "evaluation_start": fold["evaluation_start"].isoformat(),
        "evaluation_end": fold["evaluation_end"].isoformat(),
        "run_id": report["run_id"],
        "artifact_paths": report["artifact_paths"],
        "cumulative_return": aggregate["cumulative_return"],
        "buy_hold": aggregate["buy_hold"],
        "excess_return": aggregate["excess_return"],
        "sharpe": aggregate["sharpe"],
        "max_drawdown": aggregate["max_drawdown"],
        "trades": aggregate["trades"],
        "trades_per_day": aggregate["trades_per_day"],
        "beats_buy_hold": aggregate["beats_buy_hold"],
        "passed": aggregate["passed"],
    }


def build_cross_validation_report(config, cv_config, run_id, fold_summaries, summary, artifact_paths):
    return {
        "run_id": run_id,
        "report_type": "intraday_cv_report",
        "config": asdict(config),
        "cross_validation": asdict(cv_config),
        "summary": summary,
        "folds": fold_summaries,
        "artifact_paths": artifact_paths,
        "git": git_state(),
    }


def pooled_cross_validation_equity(fold_summaries, start_cash):
    learned_rows = []
    buy_hold_rows = []
    learned_equity = None
    buy_hold_equity = None
    for fold in fold_summaries:
        positions_path = fold["artifact_paths"]["positions"]
        buy_hold_path = fold["artifact_paths"]["buy_hold_positions"]
        if os.path.exists(positions_path):
            positions = pd.read_csv(positions_path)
            if not positions.empty:
                scale = 1.0 if learned_equity is None else learned_equity / start_cash
                positions["equity"] = positions["equity"] * scale
                learned_equity = float(positions["equity"].iloc[-1])
                positions["fold_index"] = fold["fold_index"]
                learned_rows.append(positions)
        if os.path.exists(buy_hold_path):
            buy_hold = pd.read_csv(buy_hold_path)
            if not buy_hold.empty:
                scale = 1.0 if buy_hold_equity is None else buy_hold_equity / start_cash
                buy_hold["equity"] = buy_hold["equity"] * scale
                buy_hold_equity = float(buy_hold["equity"].iloc[-1])
                buy_hold["fold_index"] = fold["fold_index"]
                buy_hold_rows.append(buy_hold)

    learned = pd.concat(learned_rows, ignore_index=True) if learned_rows else pd.DataFrame()
    buy_hold = pd.concat(buy_hold_rows, ignore_index=True) if buy_hold_rows else pd.DataFrame()
    return learned, buy_hold


def pooled_cross_validation_metrics(learned, buy_hold, learned_trades, buy_hold_trades, start_cash, days):
    learned_metrics = equity_metrics(
        learned["equity"] if not learned.empty else pd.Series(dtype=float),
        learned_trades,
        start_cash,
        days,
    )
    buy_hold_metrics = equity_metrics(
        buy_hold["equity"] if not buy_hold.empty else pd.Series(dtype=float),
        buy_hold_trades,
        start_cash,
        days,
    )
    excess_return = learned_metrics["cumulative_return"] - buy_hold_metrics["cumulative_return"]
    return {
        "learned": learned_metrics,
        "buy_hold": buy_hold_metrics,
        "excess_return": float(excess_return),
        "beats_buy_hold": bool(excess_return > 0.0),
    }


def write_cross_validation_artifacts(
    datasets,
    config: IntradayConfig,
    cv_config: CrossValidationConfig,
    run_id: str,
    run_dir: str,
):
    os.makedirs(run_dir, exist_ok=True)
    common_index = max_common_feature_index(datasets, config.tickers)
    folds = rolling_cross_validation_folds(common_index, config, cv_config)
    fold_summaries = []
    fold_reports = []

    for fold in folds:
        fold_config = fold["config"]
        fold_run_id = stable_config_hash(fold_config)
        fold_dir = os.path.join(run_dir, "cv_folds", f"fold_{fold['fold_index']:02d}")
        result = run_walk_forward_intraday(datasets, fold_config)
        report = write_intraday_artifacts(fold_config, fold_run_id, result, fold_dir)
        fold_reports.append(report)
        fold_summaries.append(fold_report_summary(fold, report))

    pooled_positions, pooled_buy_hold_positions = pooled_cross_validation_equity(fold_summaries, config.cash)
    pooled_positions_path = os.path.join(run_dir, "intraday_cv_pooled_positions.csv")
    if not pooled_positions.empty:
        pooled_positions.to_csv(pooled_positions_path, index=False)
    pooled_buy_hold_positions_path = os.path.join(run_dir, "intraday_cv_pooled_buy_hold_positions.csv")
    if not pooled_buy_hold_positions.empty:
        pooled_buy_hold_positions.to_csv(pooled_buy_hold_positions_path, index=False)
    total_days = cv_config.folds * cross_validation_fold_span(config, cv_config).total_seconds() / 86400.0
    pooled_metrics = pooled_cross_validation_metrics(
        pooled_positions,
        pooled_buy_hold_positions,
        int(sum(fold["trades"] for fold in fold_summaries)),
        int(sum(fold["buy_hold"]["trades"] for fold in fold_summaries)),
        config.cash,
        total_days,
    )
    summary = summarize_cross_validation(fold_reports, pooled_metrics, config, cv_config)
    report_path = os.path.join(run_dir, "intraday_cv_report.json")
    artifact_paths = {
        "report": report_path,
        "folds_dir": os.path.join(run_dir, "cv_folds"),
        "pooled_positions": pooled_positions_path,
        "pooled_buy_hold_positions": pooled_buy_hold_positions_path,
    }
    report = build_cross_validation_report(
        config, cv_config, run_id, fold_summaries, summary, artifact_paths
    )
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(jsonable(report), file, indent=2, sort_keys=True)
    return report
