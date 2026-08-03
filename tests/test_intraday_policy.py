import numpy as np
import pandas as pd
import pytest

from utils.intraday_policy import (
    CrossValidationConfig,
    IntradayConfig,
    after_cost_forward_return,
    after_cost_path_mean_forward_return,
    base_model_family,
    build_model,
    buy_hold_replay,
    decay_half_life_days,
    exit_threshold_candidates,
    fit_model,
    model_target_kind,
    prediction_matrix,
    recency_sample_weights,
    replay_long_only,
    rolling_cross_validation_folds,
    run_walk_forward_intraday,
    summarize_cross_validation,
    train_frame,
    validation_slice_periods,
)


def config(**overrides):
    values = {
        "tickers": ("AAA", "BBB"),
        "start_date": "2025-01-01",
        "end_date": "2025-01-01 01:00",
        "train_days": 0.01,
        "validation_days": 0.005,
        "evaluation_days": 0.005,
        "stride_days": 0.005,
        "horizon_grid": (1,),
        "max_hold_grid": (2,),
        "threshold_quantiles": (0.5,),
        "model_families": ("ridge",),
        "include_futures_metrics": False,
        "include_premium_index": False,
        "commission": 0.001,
        "slippage": 0.0002,
        "risk_unit": 0.25,
        "risk_unit_grid": (0.25,),
        "max_exposure": 1.0,
        "max_drawdown": 0.05,
        "min_validation_trades": 1,
        "selection_trade_floor": 1,
        "selection_activity_weight": 0.0,
        "validation_slices": 1,
        "cash": 1.0,
        "seed": 42,
    }
    values.update(overrides)
    return IntradayConfig(**values)


def make_dataset(index, prices, feature_scale=1.0):
    bars = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": np.ones(len(index)),
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "feature": np.arange(len(index), dtype=float) * feature_scale,
            "momentum": pd.Series(prices, index=index).pct_change().fillna(0.0).to_numpy(),
            "volatility_60m": np.ones(len(index), dtype=float) * 0.01,
            "market_volatility_60m": np.ones(len(index), dtype=float) * 0.02,
        },
        index=index,
    )
    return {"bars": bars, "features": features}


def test_after_cost_forward_return_uses_next_open_entry_and_horizon_exit():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    bars = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 104.0, 108.0],
            "High": [100.0, 101.0, 104.0, 108.0],
            "Low": [100.0, 101.0, 104.0, 108.0],
            "Close": [100.0, 101.0, 104.0, 108.0],
        },
        index=index,
    )

    label = after_cost_forward_return(bars, horizon_minutes=2, commission=0.001, slippage=0.0005)

    assert label.loc[index[0]] == pytest.approx(108.0 / 101.0 - 1.0 - 0.003)


def test_path_mean_forward_return_averages_all_horizons_from_next_open():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    bars = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 104.0, 108.0],
            "High": [100.0, 101.0, 104.0, 108.0],
            "Low": [100.0, 101.0, 104.0, 108.0],
            "Close": [100.0, 101.0, 104.0, 108.0],
        },
        index=index,
    )

    label = after_cost_path_mean_forward_return(bars, horizon_minutes=2, commission=0.0, slippage=0.0)

    expected = ((104.0 / 101.0 - 1.0) + (108.0 / 101.0 - 1.0)) / 2.0
    assert label.loc[index[0]] == pytest.approx(expected)


def test_train_frame_can_use_path_mean_target():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 101.0, 104.0, 108.0]),
    }

    _, target = train_frame(
        datasets,
        ("AAA",),
        horizon_minutes=2,
        start=index[0],
        end=index[-1],
        commission=0.0,
        slippage=0.0,
        target_kind="path_mean",
    )

    expected = ((104.0 / 101.0 - 1.0) + (108.0 / 101.0 - 1.0)) / 2.0
    assert target.iloc[0] == pytest.approx(expected)


def test_train_frame_can_use_market_excess_target():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 101.0, 104.0, 108.0]),
        "BBB": make_dataset(index, [100.0, 100.0, 100.0, 100.0]),
    }

    _, target = train_frame(
        datasets,
        ("AAA", "BBB"),
        horizon_minutes=2,
        start=index[0],
        end=index[-1],
        commission=0.0,
        slippage=0.0,
        target_kind="market_excess",
    )

    assert target.iloc[0] == pytest.approx((108.0 / 101.0 - 1.0) / 2.0)
    assert target.iloc[1] == pytest.approx(-target.iloc[0])


def test_train_frame_can_use_market_return_target():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 101.0, 104.0, 108.0]),
        "BBB": make_dataset(index, [100.0, 100.0, 100.0, 100.0]),
    }

    _, target = train_frame(
        datasets,
        ("AAA", "BBB"),
        horizon_minutes=2,
        start=index[0],
        end=index[-1],
        commission=0.0,
        slippage=0.0,
        target_kind="market_return",
    )

    assert target.iloc[0] == pytest.approx((108.0 / 101.0 - 1.0) / 2.0)
    assert target.iloc[1] == pytest.approx(target.iloc[0])


def test_train_frame_can_use_market_return_volatility_scaled_target():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 101.0, 104.0, 108.0]),
        "BBB": make_dataset(index, [100.0, 100.0, 100.0, 100.0]),
    }

    _, target = train_frame(
        datasets,
        ("AAA", "BBB"),
        horizon_minutes=2,
        start=index[0],
        end=index[-1],
        commission=0.0,
        slippage=0.0,
        target_kind="market_return_vol_scaled",
    )

    expected_market_return = (108.0 / 101.0 - 1.0) / 2.0
    assert target.iloc[0] == pytest.approx(expected_market_return / 0.020001)
    assert target.iloc[1] == pytest.approx(target.iloc[0])


def test_train_frame_can_use_volatility_scaled_target():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 101.0, 104.0, 108.0]),
    }

    _, target = train_frame(
        datasets,
        ("AAA",),
        horizon_minutes=2,
        start=index[0],
        end=index[-1],
        commission=0.0,
        slippage=0.0,
        target_kind="vol_scaled",
    )

    assert target.iloc[0] == pytest.approx((108.0 / 101.0 - 1.0) / 0.010001)


def test_train_frame_can_use_market_excess_volatility_scaled_target():
    index = pd.date_range("2025-01-01", periods=4, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 101.0, 104.0, 108.0]),
        "BBB": make_dataset(index, [100.0, 100.0, 100.0, 100.0]),
    }

    _, target = train_frame(
        datasets,
        ("AAA", "BBB"),
        horizon_minutes=2,
        start=index[0],
        end=index[-1],
        commission=0.0,
        slippage=0.0,
        target_kind="market_excess_vol_scaled",
    )

    expected_excess = (108.0 / 101.0 - 1.0) / 2.0
    assert target.iloc[0] == pytest.approx(expected_excess / 0.010001)
    assert target.iloc[1] == pytest.approx(-expected_excess / 0.010001)


def test_market_excess_volatility_scaled_model_family_uses_ridge_base():
    model_family = "ridge_market_excess_vol_scaled"

    assert model_target_kind(model_family) == "market_excess_vol_scaled"
    assert base_model_family(model_family) == "ridge"


def test_path_mean_volatility_scaled_model_family_uses_ridge_base():
    model_family = "ridge_path_mean_vol_scaled"

    assert model_target_kind(model_family) == "path_mean_vol_scaled"
    assert base_model_family(model_family) == "ridge"


def test_market_ridge_uses_shared_market_features_and_scores():
    index = pd.date_range("2025-01-01", periods=6, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0], feature_scale=1.0),
        "BBB": make_dataset(index, [100.0, 99.0, 98.0, 97.0, 96.0, 95.0], feature_scale=3.0),
    }
    model_family = "market_ridge"

    x, y = train_frame(
        datasets,
        ("AAA", "BBB"),
        horizon_minutes=2,
        start=index[0],
        end=index[-1],
        commission=0.0,
        slippage=0.0,
        target_kind=model_target_kind(model_family),
    )
    model = build_model(model_family, seed=42).fit(x, y)
    predictions = prediction_matrix(datasets, ("AAA", "BBB"), model, index[0], index[-1])

    assert model_target_kind(model_family) == "market_return"
    assert not predictions.empty
    assert predictions["AAA"].to_list() == pytest.approx(predictions["BBB"].to_list())


def test_market_ridge_vol_scaled_uses_market_return_vol_scaled_target():
    model_family = "market_ridge_vol_scaled"

    assert model_target_kind(model_family) == "market_return_vol_scaled"
    assert base_model_family(model_family) == "market_ridge"


def test_replay_is_long_only_and_caps_total_exposure():
    index = pd.date_range("2025-01-01", periods=8, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100, 101, 102, 103, 104, 105, 106, 107]),
        "BBB": make_dataset(index, [100, 100, 99, 98, 97, 96, 95, 94]),
    }
    predictions = pd.DataFrame(
        {
            "AAA": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "BBB": [1.0] * 8,
        },
        index=index,
    )

    replay = replay_long_only(datasets, ("AAA", "BBB"), predictions, threshold=0.5, max_hold_minutes=2, config=config(), start=index[0], end=index[-1])
    positions = replay["positions"]

    weight_columns = [column for column in positions.columns if column.startswith("weight_")]
    assert (positions[weight_columns] >= 0.0).all().all()
    assert positions["long_exposure"].max() <= 1.0
    assert positions["weight_AAA"].max() == pytest.approx(0.25)
    assert any(
        decision["ticker"] == "AAA" and decision["action"] == "sell"
        for record in replay["ledger_records"]
        if record["record_type"] == "intraday_decision"
        for decision in record["decisions"]
    )


def test_replay_can_skip_ledgers_for_validation_search():
    index = pd.date_range("2025-01-01", periods=8, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100, 101, 102, 103, 104, 105, 106, 107]),
        "BBB": make_dataset(index, [100, 100, 99, 98, 97, 96, 95, 94]),
    }
    predictions = pd.DataFrame({"AAA": [1.0] * 8, "BBB": [0.0] * 8}, index=index)

    replay = replay_long_only(
        datasets,
        ("AAA", "BBB"),
        predictions,
        threshold=0.5,
        max_hold_minutes=2,
        config=config(),
        start=index[0],
        end=index[-1],
        record_ledger=False,
    )

    assert replay["metrics"]["trades"] > 0
    assert replay["ledger_records"] == []


def test_replay_holds_through_selected_forecast_horizon():
    index = pd.date_range("2025-01-01", periods=8, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100, 101, 102, 103, 104, 105, 106, 107]),
        "BBB": make_dataset(index, [100, 100, 100, 100, 100, 100, 100, 100]),
    }
    predictions = pd.DataFrame(
        {
            "AAA": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "BBB": [0.0] * 8,
        },
        index=index,
    )

    replay = replay_long_only(
        datasets,
        ("AAA", "BBB"),
        predictions,
        threshold=0.5,
        max_hold_minutes=3,
        config=config(),
        start=index[0],
        end=index[-1],
        horizon_minutes=3,
    )

    assert replay["positions"].loc[:2, "weight_AAA"].tolist() == [pytest.approx(0.25)] * 3
    assert replay["positions"].loc[3, "weight_AAA"] == pytest.approx(0.0)
    assert any(record.get("horizon_minutes") == 3 for record in replay["ledger_records"])


def test_replay_uses_separate_exit_threshold_to_reduce_churn():
    index = pd.date_range("2025-01-01", periods=8, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100, 101, 102, 103, 104, 105, 106, 107]),
        "BBB": make_dataset(index, [100, 100, 100, 100, 100, 100, 100, 100]),
    }
    predictions = pd.DataFrame(
        {
            "AAA": [1.0, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            "BBB": [0.0] * 8,
        },
        index=index,
    )

    replay = replay_long_only(
        datasets,
        ("AAA", "BBB"),
        predictions,
        threshold=0.5,
        max_hold_minutes=3,
        config=config(),
        start=index[0],
        end=index[-1],
        horizon_minutes=1,
        exit_threshold=0.25,
    )

    assert replay["positions"].loc[:2, "weight_AAA"].tolist() == [pytest.approx(0.25)] * 3
    assert replay["positions"].loc[3, "weight_AAA"] == pytest.approx(0.0)
    assert any(record.get("exit_threshold") == 0.25 for record in replay["ledger_records"])


def test_buy_hold_replay_uses_same_cash_exposure_and_entry_exit_costs():
    index = pd.date_range("2025-01-01", periods=3, freq="min")
    datasets = {
        "AAA": make_dataset(index, [100.0, 110.0, 121.0]),
        "BBB": make_dataset(index, [200.0, 200.0, 200.0]),
    }

    replay = buy_hold_replay(datasets, ("AAA", "BBB"), config(cash=100.0), index[0], index[-1])

    expected = 100.0 * (1.0 - 0.0012) * (1.0 + 0.05) * (1.0 - 0.0012)
    assert replay["metrics"]["end_equity"] == pytest.approx(expected)
    assert replay["metrics"]["trades"] == 4


def test_exit_threshold_candidates_skip_duplicates_when_max_hold_equals_horizon():
    thresholds = (-1.0, 0.0, 1.0)

    assert exit_threshold_candidates(thresholds, threshold=1.0, horizon=2, max_hold=2) == (1.0,)
    assert exit_threshold_candidates(thresholds, threshold=1.0, horizon=2, max_hold=3) == thresholds


def test_positive_return_classifier_predicts_probabilities():
    model = build_model("logistic_positive", seed=42)
    x = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
    y = pd.Series([-0.01, -0.005, 0.005, 0.01])

    model.fit(x, y)
    predictions = model.predict(x)

    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()
    assert predictions[-1] > predictions[0]


def test_huber_model_family_predicts_regression_scores():
    model = build_model("huber", seed=42)
    x = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
    y = pd.Series([-0.01, -0.005, 0.005, 0.01])

    model.fit(x, y)
    predictions = model.predict(x)

    assert predictions[-1] > predictions[0]


def test_ridge_alpha_model_family_predicts_regression_scores():
    model = build_model("ridge_alpha_10", seed=42)
    x = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
    y = pd.Series([-0.01, -0.005, 0.005, 0.01])

    model.fit(x, y)
    predictions = model.predict(x)

    assert predictions[-1] > predictions[0]


def test_ridge_shared_model_family_drops_ordinal_ticker_id():
    model = build_model("ridge_shared", seed=42)
    x = pd.DataFrame(
        {
            "feature": [-2.0, -1.0, 1.0, 2.0],
            "ticker_id": [0.0, 0.0, 1.0, 1.0],
        }
    )
    y = pd.Series([-0.01, -0.005, 0.005, 0.01])
    same_feature = pd.DataFrame(
        {
            "feature": [1.0, 1.0],
            "ticker_id": [0.0, 999.0],
        }
    )

    model.fit(x, y)
    predictions = model.predict(same_feature)

    assert predictions[0] == pytest.approx(predictions[1])


def test_ridge_decay_model_family_weights_recent_training_rows():
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    x = pd.DataFrame({"feature": [1.0, 1.0, 1.0]}, index=index)
    y = pd.Series([0.0, 0.0, 1.0], index=index)
    weights = recency_sample_weights(x.index, half_life_days=1.0)
    model = build_model("ridge_decay_1", seed=42)

    fit_model(model, "ridge_decay_1", x, y)

    assert decay_half_life_days("ridge_decay_1_vol_scaled") == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([0.25, 0.5, 1.0])
    assert model.predict(x)[0] > y.mean()


def test_walk_forward_intraday_writes_selected_risk_adjusted_windows():
    index = pd.date_range("2025-01-01", periods=120, freq="min")
    prices = np.linspace(100.0, 130.0, len(index))
    datasets = {
        "AAA": make_dataset(index, prices),
        "BBB": make_dataset(index, np.linspace(100.0, 80.0, len(index)), feature_scale=-0.5),
    }

    result = run_walk_forward_intraday(datasets, config(end_date="2025-01-01 02:00"))

    assert result["aggregate"]["window_count"] > 0
    assert "sharpe" in result["aggregate"]
    assert "max_drawdown" in result["aggregate"]
    assert "top_candidates" in result["windows"][0]
    assert len(result["windows"][0]["top_candidates"]) <= 10
    assert "selected_generalization" in result["windows"][0]
    assert "excess_return_delta" in result["windows"][0]["selected_generalization"]
    assert "buy_hold" in result["aggregate"]
    assert "beats_buy_hold" in result["aggregate"]
    assert result["aggregate"]["passed"] == (
        result["aggregate"]["sharpe"] > 0
        and result["aggregate"]["max_drawdown"] <= config().max_drawdown
        and result["aggregate"]["beats_buy_hold"]
    )


def test_walk_forward_intraday_can_select_positive_candidate_without_validation_excess():
    index = pd.date_range("2025-01-01", periods=120, freq="min")
    prices = np.linspace(100.0, 130.0, len(index))
    datasets = {
        "AAA": make_dataset(index, prices),
        "BBB": make_dataset(index, prices * 0.95, feature_scale=0.5),
    }

    result = run_walk_forward_intraday(datasets, config(end_date="2025-01-01 02:00"))
    window = result["windows"][0]

    assert window["selected"] is not None
    assert window["best_candidate"]["passed_activity_gate"] is True
    assert window["best_candidate"]["passed_gate"] is False
    assert window["selected"]["passed_gate"] is False
    assert result["aggregate"]["trades"] > 0


def test_validation_slices_add_stability_fields_to_selection():
    index = pd.date_range("2025-01-01", periods=160, freq="min")
    prices = np.linspace(100.0, 130.0, len(index))
    datasets = {
        "AAA": make_dataset(index, prices),
        "BBB": make_dataset(index, np.linspace(100.0, 80.0, len(index)), feature_scale=-0.5),
    }

    result = run_walk_forward_intraday(datasets, config(end_date="2025-01-01 02:20", validation_slices=2))
    selected = result["windows"][0]["selected"]

    assert validation_slice_periods(index[0], index[2], 2)[0][1] == index[1]
    assert selected["validation_slice_excess_median"] is not None
    assert selected["validation_slice_excess_min"] is not None
    assert len(selected["validation_slices"]) == 2


def test_aggregate_trade_rate_counts_flat_walk_forward_windows():
    index = pd.date_range("2025-01-01", periods=120, freq="min")
    prices = np.linspace(100.0, 130.0, len(index))
    datasets = {
        "AAA": make_dataset(index, prices),
        "BBB": make_dataset(index, prices * 0.95, feature_scale=0.5),
    }

    result = run_walk_forward_intraday(datasets, config(end_date="2025-01-01 02:00"))

    assert result["aggregate"]["trades_per_day"] == pytest.approx(
        result["aggregate"]["trades"] / (result["aggregate"]["window_count"] * config().evaluation_days)
    )


def test_rolling_cross_validation_folds_are_recent_and_non_overlapping():
    common_index = pd.date_range("2025-01-01", "2025-01-20", freq="D")
    cv_config = CrossValidationConfig(folds=2, windows_per_fold=3, min_fold_trades=10)

    folds = rolling_cross_validation_folds(
        common_index,
        config(
            start_date="2025-01-01",
            end_date="2025-01-20",
            train_days=1,
            validation_days=1,
            evaluation_days=1,
            stride_days=1,
        ),
        cv_config,
    )

    assert len(folds) == 2
    assert folds[0]["evaluation_end"] == folds[1]["evaluation_start"]
    assert folds[1]["evaluation_end"] == pd.Timestamp("2025-01-20")
    assert folds[0]["fold_end"] == folds[0]["evaluation_end"]
    assert folds[0]["fold_start"] < folds[0]["evaluation_start"]
    assert folds[1]["fold_start"] < folds[0]["fold_end"]
    assert folds[0]["config"].start_date == folds[0]["fold_start"].isoformat()
    assert folds[0]["config"].end_date == folds[0]["fold_end"].isoformat()


def test_rolling_cross_validation_rejects_gapped_or_overlapping_coverage():
    common_index = pd.date_range("2025-01-01", "2025-01-20", freq="D")

    with pytest.raises(ValueError, match="stride_days to equal evaluation_days"):
        rolling_cross_validation_folds(
            common_index,
            config(
                start_date="2025-01-01",
                end_date="2025-01-20",
                train_days=1,
                validation_days=1,
                evaluation_days=1,
                stride_days=2,
            ),
            CrossValidationConfig(folds=2, windows_per_fold=1, min_fold_trades=10),
        )


def test_cross_validation_summary_requires_robust_excess_sharpe_and_drawdown():
    def report(excess, sharpe, drawdown, trades=10, passed=False):
        return {
            "aggregate": {
                "cumulative_return": excess,
                "buy_hold": {"cumulative_return": 0.0},
                "excess_return": excess,
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "trades": trades,
                "beats_buy_hold": excess > 0.0,
                "passed": passed,
            }
        }

    pooled = {"excess_return": 0.03}
    summary = summarize_cross_validation(
        [report(0.02, 1.0, 0.01), report(0.04, 0.5, 0.02), report(-0.01, -0.2, 0.03)],
        pooled,
        config(max_drawdown=0.05),
        CrossValidationConfig(folds=3, windows_per_fold=1, min_fold_trades=10),
    )
    failed = summarize_cross_validation(
        [report(0.02, 1.0, 0.01), report(0.04, -0.5, 0.02), report(-0.01, -0.2, 0.03)],
        pooled,
        config(max_drawdown=0.05),
        CrossValidationConfig(folds=3, windows_per_fold=1, min_fold_trades=10),
    )

    assert summary["median_excess_return"] == pytest.approx(0.02)
    assert summary["median_sharpe"] == pytest.approx(0.5)
    assert summary["max_fold_drawdown"] == pytest.approx(0.03)
    assert summary["min_fold_excess_return"] == pytest.approx(-0.01)
    assert summary["passed_cv"] is False
    assert failed["passed_cv"] is False


def test_cross_validation_summary_requires_fold_activity_and_pooled_edge():
    def report(excess, trades):
        return {
            "aggregate": {
                "cumulative_return": excess,
                "buy_hold": {"cumulative_return": 0.0},
                "excess_return": excess,
                "sharpe": 1.0,
                "max_drawdown": 0.01,
                "trades": trades,
                "beats_buy_hold": excess > 0.0,
                "passed": True,
            }
        }

    cv_config = CrossValidationConfig(folds=2, windows_per_fold=1, min_fold_trades=10)
    active = summarize_cross_validation(
        [report(0.02, 10), report(0.03, 11)],
        {"excess_return": 0.04},
        config(max_drawdown=0.05),
        cv_config,
    )
    inactive = summarize_cross_validation(
        [report(0.02, 10), report(0.03, 0)],
        {"excess_return": 0.04},
        config(max_drawdown=0.05),
        cv_config,
    )
    poor_pooled = summarize_cross_validation(
        [report(0.02, 10), report(0.03, 11)],
        {"excess_return": -0.01},
        config(max_drawdown=0.05),
        cv_config,
    )

    assert active["passed_cv"] is True
    assert inactive["passed_cv"] is False
    assert poor_pooled["passed_cv"] is False
