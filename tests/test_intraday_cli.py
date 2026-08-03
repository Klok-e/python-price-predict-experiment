from run_intraday_experiment import build_config, build_cv_config, build_run_id, parse_args


def test_intraday_cli_accepts_cross_validation_flags():
    args = parse_args(["--cv-folds", "2", "--cv-windows-per-fold", "1", "--min-cv-fold-trades", "3"])

    assert args.cv_folds == 2
    assert args.cv_windows_per_fold == 1
    assert args.min_cv_fold_trades == 3


def test_intraday_cli_accepts_risk_unit_grid():
    args = parse_args(["--risk-unit-grid", "0.25,0.5,1.0"])
    config = build_config(args)

    assert config.risk_unit_grid == (0.25, 0.5, 1.0)


def test_intraday_cli_accepts_selection_trade_floor():
    args = parse_args(["--min-validation-trades", "1", "--selection-trade-floor", "10"])
    config = build_config(args)

    assert config.min_validation_trades == 1
    assert config.selection_trade_floor == 10


def test_intraday_cli_accepts_selection_activity_weight():
    args = parse_args(["--selection-activity-weight", "0.001"])
    config = build_config(args)

    assert config.selection_activity_weight == 0.001


def test_intraday_cli_accepts_config_only_run_id():
    args = parse_args(["--config-only-run-id"])

    assert args.config_only_run_id is True


def test_intraday_run_id_includes_code_fingerprint_by_default():
    args = parse_args(["--cv-folds", "1"])
    config = build_config(args)
    cv_config = build_cv_config(args)

    default_run_id = build_run_id(config, cv_config)
    config_only_run_id = build_run_id(config, cv_config, config_only=True)

    assert len(default_run_id) == 12
    assert len(config_only_run_id) == 12
    assert default_run_id != config_only_run_id
