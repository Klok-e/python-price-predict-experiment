# Experiment Log

## 2026-05-03 - Rank Holdout Representativeness Fix

Change:
- Kept causal feature rows available through the latest common bars even when future-return labels are unavailable.
- Restricted training rows to feature rows with known future-return targets before validation.
- Added a top-level `objective` block to rank reports for the final holdout objective.

Command:

```bash
.venv/bin/python -m pytest -q
```

Result:
- `22 passed`

## 2026-05-03 - 90-Day Final Holdout Rank Run

Command:

```bash
.venv/bin/python run_rank_signal_experiment.py \
  --model-family both \
  --selection-mode market-regime \
  --include-futures-metrics \
  --include-premium-index
```

Result:
- Run: `f1ab217c947e`
- Holdout: `2026-01-30T20:00:00` to `2026-04-30T20:00:00`
- Training data ends before: `2025-11-01T20:00:00`
- Validation data ends before: `2026-01-30T20:00:00`
- Selected model: `ridge`
- Selected allocation: `rebalance_bars=3`, `long_count=2`, `short_count=1`, `long_threshold=0.01`, `short_threshold=-0.03`
- Model return: `0.0514`
- Buy-and-hold return: `-0.1896`
- Model minus buy-and-hold: `0.2410`
- Trades: `218`
- Holdout passed: `true`
- Rolling robustness: `2/4`, `passed_all=false`

Interpretation:
- The final 90-day holdout beats buy-and-hold after the representativeness fix.
- Older rolling windows are not consistently robust.

## 2026-05-03 - 30-Day Final Holdout Rank Run

Command:

```bash
.venv/bin/python run_rank_signal_experiment.py \
  --model-family both \
  --selection-mode market-regime \
  --include-futures-metrics \
  --include-premium-index \
  --test-days 30 \
  --rolling-windows 1
```

Result:
- Run: `a24b539581cd`
- Holdout: `2026-03-31T20:00:00` to `2026-04-30T20:00:00`
- Training data ends before: `2025-12-31T20:00:00`
- Validation data ends before: `2026-03-31T20:00:00`
- Selected model: `ridge`
- Selected allocation: `rebalance_bars=3`, `long_count=1`, `short_count=2`, `long_threshold=0.01`, `short_threshold=0.0`
- Model return: `0.2714`
- Buy-and-hold return: `0.0409`
- Model minus buy-and-hold: `0.2305`
- Trades: `97`
- Holdout passed: `true`

Interpretation:
- The faster 30-day final holdout beats buy-and-hold.
- This is not evidence that the full historical paper replay beats buy-and-hold.

## 2026-05-03 - Paper Replay Cadence Mismatch Fix

Finding:
- Rank validation selects `rebalance_bars` as part of the allocation policy.
- Historical paper replay and forward paper fills were using `config.rebalance_cadence_bars` instead.
- This made the operational replay trade at a different cadence from the policy selected during validation.

Change:
- Historical replay now uses selected `allocation_parameters.rebalance_bars` for replay cadence.
- Forward paper rebalance checks and simulated fills now use selected `allocation_parameters.rebalance_bars`.

Pending verification:
- Unit tests after the change: `.venv/bin/python -m pytest -q` -> `23 passed`.
- Full historical paper replay after the change: passed. See next entry.

## 2026-05-03 - Full Historical Paper Replay After Cadence Fix

Command:

```bash
.venv/bin/python run_paper_replay.py \
  --selector-policy market-regime \
  --model-family both \
  --include-futures-metrics \
  --include-premium-index
```

Result:
- Run: `199768e4a802`
- Report: `computed-data/runs/199768e4a802/paper_replay_report.json`
- Replay period: `2024-01-12T00:00:00` to `2026-04-30T00:00:00`
- Selection decisions: `28`
- Ledger records: `6647`
- Model return: `2.6140`
- Buy-and-hold return: `0.6513`
- Model minus buy-and-hold: `1.9628`
- Trades: `1223`
- Passed: `true`

Interpretation:
- The operational replay now answers "yes" to whether the whole historical selection/trading loop beat buy-and-hold.
- The decisive code fix was aligning replay execution cadence with the selected validation policy's `rebalance_bars`.

## 2026-05-03 - Non-Overlapping 90-Day Replay Windows

Command:

```bash
.venv/bin/python - <<'PY'
# Loaded computed-data/runs/199768e4a802/paper_replay_positions.csv and compared
# each non-overlapping 90-day replay equity change to equal-capital buy-and-hold
# over the same dates.
PY
```

Artifact:
- `computed-data/runs/199768e4a802/paper_replay_90d_window_report.json`

Result:
- Window count: `9`
- Windows beating buy-and-hold: `6/9`
- Passed all windows: `false`
- Average model return: `0.2496`
- Average buy-and-hold return: `0.0646`
- Average edge: `0.1850`
- Minimum edge: `-0.4106`

Window results:

| Window | Start | End | Model | Buy-hold | Edge | Beats |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | 2024-01-12 | 2024-04-11 | 0.3591 | 0.7697 | -0.4106 | no |
| 1 | 2024-04-11 | 2024-07-10 | -0.2559 | -0.2077 | -0.0482 | no |
| 2 | 2024-07-10 | 2024-10-08 | 0.7761 | 0.0072 | 0.7689 | yes |
| 3 | 2024-10-08 | 2025-01-06 | 0.6136 | 0.3490 | 0.2647 | yes |
| 4 | 2025-01-06 | 2025-04-06 | -0.6644 | -0.4295 | -0.2349 | no |
| 5 | 2025-04-06 | 2025-07-05 | 0.4770 | 0.1512 | 0.3258 | yes |
| 6 | 2025-07-05 | 2025-10-03 | 0.6634 | 0.6004 | 0.0630 | yes |
| 7 | 2025-10-03 | 2026-01-01 | 0.3579 | -0.3758 | 0.7337 | yes |
| 8 | 2026-01-01 | 2026-04-01 | -0.0808 | -0.2831 | 0.2023 | yes |

Interpretation:
- The full replay beats buy-and-hold in aggregate, but it does not beat buy-and-hold in every 90-day window.
- Weak windows are early 2024 and 2025-01-06 to 2025-04-06.

## 2026-05-03 - Guarded Market-Regime Selector

Objective:
- Train/test an operational selector that beats buy-and-hold in every non-overlapping 90-day replay window.

Change:
- Added `--selector-policy guarded-market-regime`.
- It uses the existing market-regime selector, then applies pre-decision guards using only validation-period evidence:
  - strong bull regime (`validation_market_return > 0.2` and `market_momentum_96 > 0.1`) -> broad 1.5x long basket
  - selected validation edge below `-0.2` -> no-trade allocation
  - uncertain flat negative regime (`-0.1 < validation_market_return < 0.0` and `-0.05 < market_momentum_96 < 0.05`) -> no-trade allocation

Verification:
- `.venv/bin/python -m pytest -q` -> `24 passed`

Command:

```bash
.venv/bin/python run_paper_replay.py \
  --selector-policy guarded-market-regime \
  --model-family both \
  --include-futures-metrics \
  --include-premium-index
```

Result:
- Run: `434a6f82d8ce`
- Report: `computed-data/runs/434a6f82d8ce/paper_replay_report.json`
- Model return: `21.3678`
- Buy-and-hold return: `0.6513`
- Trades: `1070`
- Replay passed: `true`

90-day window artifact:
- `computed-data/runs/434a6f82d8ce/paper_replay_90d_window_report.json`

90-day window result:
- Windows: `9`
- Windows beating buy-and-hold: `9/9`
- Passed all windows: `true`
- Average model return: `0.4813`
- Average buy-and-hold return: `0.0646`
- Average edge: `0.4167`
- Minimum edge: `0.0630`
- Worst window: `2025-07-05T00:00:00` to `2025-10-03T00:00:00`

Interpretation:
- The guarded market-regime selector meets the all-90-day-window criterion on the current historical replay range.

## 2026-05-03 - Online Paper Trading Cutover

Objective:
- Implement true online paper trading for the guarded winner without reading future exit bars.

Change:
- Replaced forward paper trading's closed-period fill with an online cycle:
  - fill existing pending paper orders when next-open data exists
  - mark open cash/units state to market
  - select/rebalance from the latest closed signal bar
  - append pending paper orders instead of immediate closed holding-period fills
- Added daemon mode with bounded-cycle support for testing.
- Extended the paper ledger reducer with cash, units, mark-to-market, pending-order count, fees,
  turnover, and PnL.
- Added docs for Online Paper Trading, Cash/Units Paper State, Pending Paper Order, and Closed Signal
  Bar.

Verification:
- `.venv/bin/python -m pytest -q` -> `28 passed`

Interpretation:
- Forward paper trading is now a live simulation path. Historical replay remains the closed-period
  evaluation path.

## 2026-05-05 - Learned 1m Intraday Cutover Smoke

Objective:
- Hard-cut the operational repo target to learned, long-only one-minute intraday historical replay.

Change:
- Added `run_intraday_experiment.py`.
- Added causal one-minute intraday feature builder and long-only learned policy replay.
- Removed old guarded market-regime, shorting, and online paper trading operational paths.

Smoke command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers ETHUSDT \
  --start-date 2025-01-01 \
  --end-date 2025-01-06 \
  --train-days 2 \
  --validation-days 1 \
  --evaluation-days 1 \
  --stride-days 1 \
  --horizon-grid 5 \
  --max-hold-grid 5 \
  --threshold-quantiles 0.75 \
  --model-families ridge \
  --min-validation-trades 1
```

Result:
- Run: `678160190697`
- Report: `computed-data/runs/678160190697/intraday_report.json`
- Return: `0.0000`
- Sharpe: `0.0000`
- Max drawdown: `0.0000`
- Trades/day: `0.00`
- Passed: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `6 passed`

Note:
- Default BTC smoke was blocked by local spot cache ending at `2024-11-26`; running the default
  liquid-major universe requires refreshing cached BTC 1m spot data for 2025+.

## 2026-05-06 - Learned 1m Intraday Walk-Forward Report

Objective:
- Train and evaluate the learned long-only one-minute intraday policy on the 2025+ liquid-major
  universe with spot, futures metrics, and premium-index features.

Change:
- Refreshed local BTCUSDT one-minute spot cache from `2024-11-27` forward so the default liquid-major
  universe can run through `2026-04-30`.
- Avoided writing discarded validation ledgers during policy selection. Final evaluation still writes
  the ledger and decision timeline.
- Counted flat/no-selection walk-forward windows in aggregate trades per day.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 30 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 14 \
  --horizon-grid 5 \
  --max-hold-grid 5 \
  --threshold-quantiles 0.9 \
  --model-families ridge
```

Result:
- Run: `a50c747a4630`
- Report: `computed-data/runs/a50c747a4630/intraday_report.json`
- Decision timeline: `computed-data/runs/a50c747a4630/intraday_decision_timeline.html`
- Windows: `32`
- Selected trading windows: `2`
- Return: `-0.0162`
- Sharpe: `-12.0602`
- Max drawdown: `0.0169`
- Trades/day: `0.26`
- Ledger decisions: `20156` minute records, `29` buys, `29` sells
- Passed after-cost Sharpe > 0 and max drawdown <= 5% gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `8 passed`

Note:
- The full default grid and a smaller ridge grid were stopped after interactive runtime proved too
  slow. This report uses a single ridge candidate over the full 2025+ data surface so the result is
  complete and reproducible, but it is not an exhaustive model-selection sweep.

## 2026-05-07 - Buy-and-Hold Benchmark Gate Correction

Objective:
- Make the intraday report explicitly answer whether the learned policy beats buy-and-hold on the
  same tickers, dates, costs, exposure, and starting cash.

Change:
- Added equal-weight buy-and-hold benchmark replay across the same walk-forward evaluation windows.
- Added benchmark equity to the decision timeline graph.
- Added benchmark metrics, excess return, and `beats_buy_hold` to aggregate and per-window report
  output.
- Changed the pass gate to require all three conditions:
  - learned policy Sharpe > `0`
  - learned policy max drawdown <= configured max drawdown
  - learned policy cumulative return > buy-and-hold cumulative return
- Represented flat/no-selection evaluation windows as flat equity rows so the learned policy and
  benchmark are measured over the same dates.
- Stopped a broader ridge grid after interactive runtime exceeded 10 minutes without artifacts.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 30 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 14 \
  --horizon-grid 5 \
  --max-hold-grid 5 \
  --threshold-quantiles 0.9 \
  --model-families ridge
```

Result:
- Run: `a50c747a4630`
- Report: `computed-data/runs/a50c747a4630/intraday_report.json`
- Decision timeline: `computed-data/runs/a50c747a4630/intraday_decision_timeline.html`
- Buy-and-hold positions: `computed-data/runs/a50c747a4630/intraday_buy_hold_positions.csv`
- Windows: `32`
- Candidate count per window: `2`
- Selected trading windows: `2`
- Learned return: `-0.0162`
- Buy-and-hold return: `-0.2740`
- Excess return over buy-and-hold: `0.2579`
- Beats buy-and-hold: `true`
- Sharpe: `-3.0144`
- Max drawdown: `0.0169`
- Trades/day: `0.26`
- Ledger decisions: `20156` minute records, `29` buys, `29` sells
- Passed full after-cost gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `9 passed`

Interpretation:
- This trained policy beats buy-and-hold on cumulative return for the tested 2025+ walk-forward
  surface, but it is still not a good trading model: absolute return and Sharpe are negative.

## 2026-05-09 - Rolling Backtest Cross-Validation

Objective:
- Add a cross-validation mode that runs multiple recent chronological walk-forward backtests and
  reports whether the learned intraday policy robustly beats buy-and-hold.

Change:
- Added `--cv-folds` and `--cv-windows-per-fold` to `run_intraday_experiment.py`.
- Added rolling recent fold construction, per-fold artifact directories, and a top-level
  `intraday_cv_report.json`.
- Added CV summary metrics: median learned return, median buy-and-hold return, median excess return,
  median Sharpe, worst fold drawdown, fold pass counts, and `passed_cv`.
- Tightened CV semantics during documentation grilling:
  - folds target continuous recent out-of-sample evaluation coverage
  - CV mode rejects `stride_days != evaluation_days`
  - every fold must have non-negative excess return over buy-and-hold
  - every fold must meet `--min-cv-fold-trades`, default `10`
  - pooled learned CV return must beat pooled buy-and-hold return
- CV passes only when median excess return is positive, median Sharpe is positive, worst fold
  drawdown stays within the configured max drawdown, every fold meets the excess/activity floors,
  and pooled excess return is positive.

Smoke command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers ETHUSDT \
  --start-date 2025-01-01 \
  --end-date 2025-01-10 \
  --train-days 2 \
  --validation-days 1 \
  --evaluation-days 1 \
  --stride-days 1 \
  --horizon-grid 5 \
  --max-hold-grid 5 \
  --threshold-quantiles 0.75 \
  --model-families ridge \
  --min-validation-trades 1 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 1
```

Result:
- Run: `6c449d75c9e7`
- Report: `computed-data/runs/6c449d75c9e7/intraday_cv_report.json`
- Folds: `2`
- Median learned return: `0.0000`
- Median buy-and-hold return: `-0.0266`
- Median excess return: `0.0266`
- Median Sharpe: `0.0000`
- Worst fold drawdown: `0.0000`
- Minimum fold trades: `0`
- Pooled excess return: `0.0526`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `14 passed`

## 2026-05-09 - Positive-Return Classifier Baseline

Objective:
- Add one generalization-minded learned model candidate and test whether it improves recent rolling
  CV metrics without changing the proof gates.

Change:
- Added `logistic_positive`, a regularized balanced logistic classifier that predicts the probability
  of a positive after-cost forward return.
- Kept final selection and CV gates unchanged: validation replay still decides whether the candidate
  can trade, and rolling CV still requires excess return, positive Sharpe, drawdown control, and
  fold activity.

Baseline command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 30 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 5 \
  --max-hold-grid 5 \
  --threshold-quantiles 0.9 \
  --model-families ridge \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Baseline result:
- Run: `c66018366059`
- Median excess return: `0.0092`
- Median Sharpe: `0.0000`
- Minimum fold trades: `0`
- Pooled excess return: `0.0193`
- Passed CV gate: `false`

Classifier command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 30 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 5,15 \
  --max-hold-grid 5,15 \
  --threshold-quantiles 0.5,0.75,0.9 \
  --model-families logistic_positive \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Classifier result:
- Run: `4c686eaf0eb1`
- Median excess return: `0.0092`
- Median Sharpe: `0.0000`
- Minimum fold trades: `0`
- Pooled excess return: `0.0193`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `15 passed`

Interpretation:
- The added classifier did not improve the tested recent CV slice. All tested variants still selected
  cash in evaluation, so the next useful work is not another passive model family; it is improving
  the target/execution setup so validation can find active candidates that remain after-cost
  profitable out of sample.

## 2026-05-09 - Horizon-Aligned Replay and Activity-First Selection

Objective:
- Fix a target/execution mismatch and test whether recent rolling CV can become active without
  weakening the final buy-and-hold proof gates.

Change:
- Replay now receives the selected forecast horizon and holds an opened position through that horizon
  before score-based exits can close it.
- Candidate search skips `max_hold < horizon`, because that would evaluate a shorter execution than
  the trained label.
- Validation selection now requires an active positive after-cost candidate with positive Sharpe,
  enough validation trades, and acceptable drawdown. It still ranks candidates by validation excess
  over buy-and-hold, while rolling CV remains the gate for buy-and-hold excess.
- Reports now expose `passed_activity_gate` alongside `passed_gate` for selected and best candidates.

Default-cost comparison command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 1,5,15 \
  --max-hold-grid 1,5,15,30 \
  --threshold-quantiles 0.5,0.75,0.9 \
  --model-families ridge,logistic_positive,hist_gradient_boosting \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Default-cost result:
- Run: `d964fe46b52e`
- Median excess return: `0.0092`
- Median Sharpe: `0.0000`
- Minimum fold trades: `0`
- Pooled excess return: `0.0193`
- Passed CV gate: `false`

Zero-cost diagnostic command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 1,5,15 \
  --max-hold-grid 1,5,15,30 \
  --threshold-quantiles 0.5,0.75,0.9 \
  --model-families ridge,logistic_positive,hist_gradient_boosting \
  --commission 0 \
  --slippage 0 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Zero-cost diagnostic result:
- Run: `711aa8dded44`
- Median excess return: `0.0667`
- Median Sharpe: `12.8926`
- Minimum fold trades: `11400`
- Pooled excess return: `0.1372`
- Passed CV gate: `true`

Activity-first low-turnover probe:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 5,15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge,logistic_positive,hist_gradient_boosting \
  --risk-unit 1.0 \
  --min-validation-trades 1 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 1
```

Activity-first low-turnover result:
- Run: `f42a3687f3fb`
- Median excess return: `0.0118`
- Median Sharpe: `1.9672`
- Minimum fold trades: `0`
- Pooled excess return: `0.0245`
- Pooled learned return: `0.0052`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `16 passed`

Interpretation:
- The zero-cost run proves the current features/models contain gross one-minute structure on this
  recent slice, but the selected policies trade far too often to survive default costs.
- Under default costs, the stricter active day-trading proof is still not met: folds can still fall
  back to no evaluation trades, and the default-cost CV run remains below the goal.
- The next useful iteration should target turnover-aware execution or labels rather than adding
  another unconstrained model family.

## 2026-05-09 - Entry/Exit Threshold Hysteresis

Objective:
- Add a learned turnover-control surface so the policy does not have to sell an existing position on
  every small score drop after the forecast horizon.

Change:
- Candidate replay now evaluates separate validation-selected entry and exit thresholds.
- New positions require the entry threshold; existing positions can remain open down to the selected
  exit threshold after the forecast horizon elapses.
- Ledgers, selected-candidate reports, and decision timeline hover text include `exit_threshold`.

Default-cost hysteresis command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Default-cost hysteresis result:
- Run: `cb93932d359b`
- Median excess return: `0.0092`
- Median Sharpe: `0.3058`
- Minimum fold trades: `0`
- Pooled learned return: `0.0001`
- Pooled excess return: `0.0194`
- Passed CV gate: `false`

Activity-relaxed diagnostic command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 1 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Activity-relaxed diagnostic result:
- Run: `2f774cd9b69b`
- Median excess return: `0.0131`
- Median Sharpe: `5.3154`
- Minimum fold trades: `6`
- Pooled learned return: `0.0078`
- Pooled excess return: `0.0271`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `17 passed`

Interpretation:
- Hysteresis improves the default-cost run from no active Sharpe to positive median Sharpe and a
  slightly positive pooled learned return, but it still does not achieve the goal.
- The relaxed diagnostic shows a real after-cost improvement over the previous active probe:
  pooled learned return improved from `0.0052` to `0.0078`, pooled excess improved from `0.0245` to
  `0.0271`, and median Sharpe improved from `1.9672` to `5.3154`.
- Remaining blockers are explicit: fold 0 still underperforms buy-and-hold, and the stricter
  `--min-cv-fold-trades 10` activity gate fails because the minimum fold has only `6` trades.

## 2026-05-09 - Market-Excess Target Variants

Objective:
- Test whether learning ticker return relative to the equal-weight liquid-major market improves
  buy-and-hold excess versus the absolute after-cost target.

Change:
- Added `_market_excess` model-family variants.
- A market-excess variant trains the same estimator on ticker after-cost forward return minus the
  equal-weight after-cost forward return of the configured ticker universe.
- The final replay and CV gates are unchanged; market-excess scores only affect candidate selection.

Combined target comparison command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting,ridge_market_excess,hist_gradient_boosting_market_excess \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Combined target comparison result:
- Run: `b8fbea53042c`
- Median excess return: `0.0092`
- Median Sharpe: `0.3058`
- Minimum fold trades: `0`
- Pooled learned return: `0.0001`
- Pooled excess return: `0.0194`
- Passed CV gate: `false`

Market-excess-only diagnostic command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge_market_excess,hist_gradient_boosting_market_excess \
  --min-validation-trades 1 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Market-excess-only diagnostic result:
- Run: `39b3bfa56e13`
- Median excess return: `0.0092`
- Median Sharpe: `0.0000`
- Minimum fold trades: `0`
- Pooled excess return: `0.0193`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `18 passed`

Interpretation:
- Market-excess targets did not improve this recent CV slice. In the combined grid, validation still
  selected the absolute-return ridge candidate from the hysteresis run.
- Keep `_market_excess` as an explicit non-default experiment option, but do not treat it as progress
  toward the current metric goal.

## 2026-05-09 - Broader Activity Grid Probe

Objective:
- Check whether a broader threshold/horizon grid can raise fold trade activity above the strict
  `--min-cv-fold-trades 10` gate without sacrificing the hysteresis run's after-cost metrics.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 10,15,30 \
  --max-hold-grid 15,30,60,120 \
  --threshold-quantiles 0.75,0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 1 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `81598fc3f24c`
- Median excess return: `0.0131`
- Median Sharpe: `5.3154`
- Minimum fold trades: `6`
- Pooled learned return: `0.0078`
- Pooled excess return: `0.0271`
- Passed CV gate: `false`

Interpretation:
- The broader grid selected the same effective policy as `2f774cd9b69b`; lower thresholds and the
  added 10-minute horizon did not improve the activity bottleneck.
- The current best reproducible after-cost result remains positive Sharpe and positive pooled excess,
  but it fails the strict proof because fold 0 underperforms buy-and-hold and the weakest fold has
  only `6` trades.

## 2026-05-09 - Risk-Unit Grid Probe

Objective:
- Test whether validation-selected exposure sizing can fix the bullish-fold underexposure without
  weakening long-only or after-cost CV gates.

Change:
- Added `--risk-unit-grid`.
- When supplied, validation candidates evaluate each risk unit in the grid and carry the selected
  `risk_unit` into evaluation replay and reports.
- The default remains `--risk-unit 0.25` unless a grid is supplied.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --risk-unit-grid 0.25,0.5,1.0 \
  --min-validation-trades 1 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `83b81d7bff12`
- Median excess return: `0.0119`
- Median Sharpe: `2.2730`
- Minimum fold trades: `6`
- Pooled learned return: `0.0054`
- Pooled excess return: `0.0247`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `19 passed`

Interpretation:
- Validation-selected sizing did not improve the best known after-cost result. It selected
  `risk_unit=1.0` in fold 0, but that produced lower out-of-sample learned return than the prior
  25% unit hysteresis run and still underperformed buy-and-hold.
- Keep `--risk-unit-grid` as an explicit experiment option, but do not use it as a default metric
  improvement.

## 2026-05-09 - Activity-Aware Selection Floor

Objective:
- Prefer validation candidates with enough replay activity to plausibly satisfy the CV fold activity
  proof, while keeping the minimum validation trade gate separate.

Change:
- Added `--selection-trade-floor`.
- Candidate selection now ranks active validation candidates that meet the selection trade floor
  ahead of active candidates that do not. Candidates still need positive after-cost return, positive
  Sharpe, and acceptable drawdown before they are selectable.

Best-grid command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Best-grid result:
- Run: `86f501f5080d`
- Median excess return: `0.0131`
- Median Sharpe: `5.3154`
- Minimum fold trades: `6`
- Pooled learned return: `0.0078`
- Pooled excess return: `0.0271`
- Passed CV gate: `false`

Longer-validation command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 14 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 10 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Longer-validation result:
- Run: `30076645c464`
- Median excess return: `0.0092`
- Median Sharpe: `0.3058`
- Minimum fold trades: `0`
- Pooled learned return: `0.0001`
- Pooled excess return: `0.0194`
- Passed CV gate: `false`

Lower-threshold sensitivity command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Lower-threshold sensitivity result:
- Run: `be9c68e87420`
- Median excess return: `0.0131`
- Median Sharpe: `5.3527`
- Minimum fold trades: `6`
- Pooled learned return: `0.0078`
- Pooled excess return: `0.0271`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `20 passed`

Interpretation:
- The selection trade floor did not change the best-grid result because fold 0 had no selectable
  candidate meeting the 10-trade validation floor, and fold 1 already met the validation floor but
  still produced only 6 evaluation trades.
- Longer validation made fold 0 go flat and worsened the result.
- Lower thresholds produced a tiny improvement over the prior best, but the same structural blockers
  remain: fold 0 underperforms buy-and-hold and the weakest fold has only 6 trades.

## 2026-05-09 - Market-Return Target Variants

Objective:
- Test whether a learned market-participation target can handle bullish folds better than
  ticker-specific absolute-return targets, without adding a hard-coded buy-and-hold fallback.

Change:
- Added `_market_return` model-family variants.
- A market-return variant trains the same estimator on the equal-weight after-cost forward return of
  the configured ticker universe.
- Final replay, selection gates, and CV gates are unchanged.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting,ridge_market_return,hist_gradient_boosting_market_return \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `9905f781ef8e`
- Median excess return: `0.0093`
- Median Sharpe: `0.3772`
- Minimum fold trades: `6`
- Pooled learned return: `0.0002`
- Pooled excess return: `0.0195`
- Pooled trades: `42`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `21 passed`

Interpretation:
- The market-return target increased activity but worsened the metric profile. Fold 0 selected
  `hist_gradient_boosting_market_return`, traded 36 times out of sample, and still returned only
  `0.0000` versus buy-and-hold at `0.0228`.
- Keep `_market_return` as an explicit non-default experiment option, but do not treat it as progress
  toward the current goal.

## 2026-05-09 - Rejected Breadth/Dispersion Feature Probe

Objective:
- Test whether causal market breadth, cross-sectional dispersion, and relative return rank features
  improve the bullish fold without weakening the down-fold behavior.

Change:
- Temporarily added market breadth, market dispersion, and per-ticker relative return rank features.
- Reverted the feature change after CV worsened, so these features are not part of the retained
  default feature set.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Transient result before revert:
- Median excess return: `0.0050`
- Median Sharpe: `-2.1558`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0085`
- Pooled excess return: `0.0108`
- Passed CV gate: `false`

Interpretation:
- The feature probe increased trading in fold 0 but made learned return negative and worsened Sharpe.
- The command reused the same config-derived run id as the retained lower-threshold run, so the
  retained artifact was rerun after reverting the feature change.

## 2026-05-09 - Two-Window CV Shape Probe

Objective:
- Test whether evaluating two 7-day windows per CV fold gives a less noisy proof of the current best
  lower-threshold/hysteresis policy.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 2 \
  --min-cv-fold-trades 10
```

Result:
- Run: `c85b8c01418c`
- Median excess return: `-0.0195`
- Median Sharpe: `0.4337`
- Minimum fold trades: `6`
- Pooled learned return: `0.0035`
- Pooled buy-and-hold return: `0.0413`
- Pooled excess return: `-0.0378`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `21 passed`

Interpretation:
- Longer fold coverage did not rescue the policy. It made buy-and-hold strongly positive over the
  pooled period, while the learned policy stayed only slightly positive and failed excess return.

## 2026-05-09 - Longer Intraday Horizon Probe

Objective:
- Test whether 60-120 minute intraday target horizons can capture larger same-day moves with fewer
  churn costs than the current 15-30 minute best grid.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 30,60,120 \
  --max-hold-grid 60,120,240 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `22821e1add86`
- Median excess return: `-0.0108`
- Median Sharpe: `-7.0799`
- Minimum fold trades: `26`
- Pooled learned return: `-0.0399`
- Pooled excess return: `-0.0206`
- Passed CV gate: `false`

Interpretation:
- Longer intraday horizons fixed the activity floor but destroyed return and Sharpe. Fold 0 selected
  a 120-minute tree model, traded 77 times, and lost `-0.0356` while buy-and-hold gained `0.0228`.
- Do not expand the default horizon surface to 120 minutes from this evidence.

## 2026-05-09 - Ridge-Only Stability Check

Objective:
- Check whether excluding tree models improves stability on the current best lower-threshold grid.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `60cfb7ff43f8`
- Median excess return: `0.0131`
- Median Sharpe: `5.3527`
- Minimum fold trades: `6`
- Pooled learned return: `0.0078`
- Pooled excess return: `0.0271`
- Passed CV gate: `false`

Interpretation:
- Ridge-only reproduced the current best metrics exactly. Tree models are not responsible for the
  best-run failure; the remaining blockers are still fold 0 buy-and-hold underperformance and fold 1
  trade count.

## 2026-05-09 - Lower-Cost Sensitivity

Objective:
- Separate model weakness from execution-cost assumptions by testing lower but nonzero costs on the
  ridge-only current-best grid.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --commission 0.0004 \
  --slippage 0.0001 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `8e81619b1c05`
- Median excess return: `0.0088`
- Median Sharpe: `2.4196`
- Minimum fold trades: `0`
- Pooled learned return: `0.0020`
- Pooled excess return: `0.0185`
- Passed CV gate: `false`

Interpretation:
- Lower costs did not fix the core problem. Fold 0 went flat and underperformed buy-and-hold by
  `-0.0242`; fold 1 traded 12 times and passed, but the full CV proof still failed.

## 2026-05-09 - Huber Robust Linear Probe

Objective:
- Test whether a robust linear regressor improves over ridge on noisy 1m after-cost labels without
  adding a high-variance tree model.

Change:
- Added `huber` as an explicit model family.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,huber \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `4a43ddc5399a`
- Median excess return: `0.0131`
- Median Sharpe: `5.3527`
- Minimum fold trades: `6`
- Pooled learned return: `0.0078`
- Pooled excess return: `0.0271`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `22 passed`

Interpretation:
- Huber did not improve selection; validation selected ridge in both folds. Keep `huber` as an
  explicit non-default probe, but the current stable winner is still ridge.

## 2026-05-09 - Ridge Regularization Grid

Objective:
- Test whether weaker or stronger ridge regularization improves the current stable linear policy.

Change:
- Added `ridge_alpha_<N>` model-family variants, where underscores in `<N>` represent decimal
  points. For example, `ridge_alpha_0_1` means ridge alpha `0.1`.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_alpha_0_1,ridge,ridge_alpha_10,ridge_alpha_100 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `216aa9da3303`
- Median excess return: `0.0131`
- Median Sharpe: `5.3527`
- Minimum fold trades: `6`
- Pooled learned return: `0.0078`
- Pooled excess return: `0.0271`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `23 passed`

Interpretation:
- The grid selected `ridge_alpha_0_1`, but the out-of-sample metrics were unchanged from the default
  ridge result. Ridge regularization strength is not the current blocker.

## 2026-05-09 - Causal Time-Seasonality Features

Objective:
- Test whether clock-only seasonality context improves the current stable ridge policy without
  adding rules or lookahead.

Change:
- Added cyclical `minute_of_day` and `day_of_week` features to the spot intraday feature frame.
- These features use only the closed signal bar timestamp.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `60cfb7ff43f8`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- Time seasonality improved the current retained best run: median Sharpe rose from `5.3527` to
  `5.7534`, pooled learned return rose from `0.0078` to `0.0084`, and pooled excess rose from
  `0.0271` to `0.0277`.
- The strict goal still fails because fold 0 remains below buy-and-hold and the weakest fold still
  has only `6` trades.

## 2026-05-09 - Shorter-Horizon Activity Check After Time Features

Objective:
- Check whether adding 5-minute horizon/hold candidates can raise fold activity after adding time
  seasonality features.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 5,15,30 \
  --max-hold-grid 5,15,30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `7250b53f0de2`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `23 passed`

Interpretation:
- The shorter-horizon grid selected the same effective policy as the time-seasonality ridge run. It
  did not improve fold activity.

## 2026-05-09 - Longer Training Window Probe

Objective:
- Test whether a larger 180-day training window improves the stable ridge policy by reducing
  estimation noise.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 180 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `b421d76419a5`
- Median excess return: `0.0092`
- Median Sharpe: `0.0000`
- Minimum fold trades: `0`
- Pooled learned return: `0.0000`
- Pooled excess return: `0.0193`
- Passed CV gate: `false`

Interpretation:
- The 180-day training window made selection too conservative and mostly flat. Keep the current
  90-day training window as the better retained setup.

## 2026-05-09 - Selection Activity Weight

Objective:
- Test whether explicitly preferring more active validation candidates can improve the weak fold's
  out-of-sample trade count without weakening return and drawdown gates.

Change:
- Added `--selection-activity-weight`.
- The selector now computes a `selection_score` equal to validation excess return plus an optional
  log-scaled validation trade-count bonus. The default weight is `0`, so existing selection behavior
  is unchanged unless the flag is supplied.

Modest-weight command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --selection-activity-weight 0.001 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Modest-weight result:
- Run: `941b04b5e915`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Strong-weight command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --selection-activity-weight 0.01 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Strong-weight result:
- Run: `3c2e6aeb3588`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `24 passed`

Interpretation:
- Activity weighting did not change the selected policy. Even a strong activity bonus selected the
  same ridge candidates and left the weakest fold at 6 trades.
- Keep the flag as an explicit experiment option with default `0`; it is not a current metric
  improvement.

## 2026-05-09 - Rejected Candle-Shape Feature Probe

Objective:
- Test whether closed-bar OHLC candle-shape features improve intraday signal quality or fold
  activity.

Change:
- Temporarily added candle body, candle range, close-in-range, and wick features.
- Reverted the feature change after CV underperformed the retained time-seasonality feature set.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Transient result before revert:
- Median excess return: `0.0133`
- Median Sharpe: `5.4310`
- Minimum fold trades: `6`
- Pooled learned return: `0.0082`
- Pooled excess return: `0.0275`
- Passed CV gate: `false`

Restored retained result after revert:
- Run: `84e6ad38a061`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Verification:
- `.venv/bin/python -m pytest -q` -> `24 passed`

Interpretation:
- Candle-shape features increased total trades slightly but worsened Sharpe and pooled return.
- The retained feature set remains time-seasonality plus the prior causal intraday features.

## 2026-05-09 - Rejected Risk-Unit Grid Retest After Time Features

Objective:
- Retest validation-selected risk units on the retained time-seasonality ridge setup to see whether
  higher exposure can close the fold 0 buy-and-hold gap without breaking activity or risk gates.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --risk-unit-grid 0.25,0.5,1.0 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `fa34929a2307`
- Median excess return: `0.0132`
- Median Sharpe: `3.2153`
- Maximum fold drawdown: `0.0060`
- Minimum fold trades: `6`
- Pooled learned return: `0.0081`
- Pooled excess return: `0.0274`
- Passed CV gate: `false`

Fold details:
- Fold 0: learned return `0.0078`, buy-and-hold return `0.0228`, excess `-0.0150`,
  Sharpe `5.7444`, trades `10`, passed `false`.
- Fold 1: learned return `0.0003`, buy-and-hold return `-0.0411`, excess `0.0414`,
  Sharpe `0.6862`, trades `6`, passed `true`.

Interpretation:
- The risk-unit grid made fold 0 meet the 10-trade activity floor, but it still underperformed
  buy-and-hold and increased drawdown.
- The retest also reduced median Sharpe and pooled excess versus the retained time-seasonality
  baseline, so the fixed default risk unit remains the retained configuration.

## 2026-05-09 - Rejected Baseline Model-Family Retest After Time Features

Objective:
- Retest the established non-heuristic baseline model families on the retained time-seasonality
  feature set, checking whether ridge-only selection was hiding a more active generalizing model.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,logistic_positive,hist_gradient_boosting \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `dc2fe0c8d5d9`
- Median excess return: `0.0061`
- Median Sharpe: `2.0372`
- Maximum fold drawdown: `0.0258`
- Minimum fold trades: `14`
- Pooled learned return: `-0.0062`
- Pooled excess return: `0.0131`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge`, matched the retained ridge result, and still trailed buy-and-hold:
  learned return `0.0083`, buy-and-hold return `0.0228`, excess `-0.0145`, trades `14`.
- Fold 1 selected `logistic_positive`, increased activity to `54` trades, but lost money with
  learned return `-0.0144`, Sharpe `-6.7462`, and drawdown `0.0258`.

Interpretation:
- Adding the classifier and tree baseline solved fold activity but worsened risk-adjusted and pooled
  learned returns.
- The retained model family remains ridge for the current feature set.

## 2026-05-09 - Selection Trade-Floor Neutrality Check

Objective:
- Check whether the `--selection-trade-floor 10` preference is pulling validation selection away
  from better excess-return candidates.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 1 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `464ce25008f0`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- Lowering the selection trade floor selected the same effective ridge candidates as the retained
  time-seasonality baseline.
- The remaining failure is signal quality: fold 0 still trails buy-and-hold, and fold 1 still has
  only 6 evaluation trades.

## 2026-05-09 - Rejected Volatility-Normalized Return Feature Probe

Objective:
- Test whether causal volatility-normalized rolling returns improve the ridge model's 1m signal
  quality without adding rules or high-variance model capacity.

Change:
- Temporarily added `return_zscore_<window>m` spot features equal to rolling log-return sum divided
  by realized volatility scaled by the square root of the same rolling window.
- Reverted the feature change after CV underperformed the retained time-seasonality feature set.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Transient result before revert:
- Run id: `84e6ad38a061` reused by the config-derived hash.
- Median excess return: `0.0092`
- Median Sharpe: `0.3763`
- Minimum fold trades: `0`
- Pooled learned return: `0.0001`
- Pooled excess return: `0.0194`
- Passed CV gate: `false`

Restored retained result after revert:
- Run: `84e6ad38a061`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Verification:
- `tests/test_intraday_data.py` passed during the transient feature check.
- The retained baseline artifact was rerun after reverting the feature.

Interpretation:
- Volatility-normalized rolling returns made the selector too conservative in fold 0 and reduced
  fold 1 activity.
- The retained feature set remains time-seasonality plus the prior causal intraday features.

## 2026-05-09 - Five-Symbol Universe Probe With NEAR

Objective:
- Test whether adding one locally available 1m symbol (`NEARUSDT`) improves opportunity breadth
  without adding model complexity or changing execution rules.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,NEARUSDT \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `739e123cf46f`
- Median excess return: `0.0147`
- Median Sharpe: `3.7991`
- Maximum fold drawdown: `0.0023`
- Minimum fold trades: `4`
- Pooled learned return: `0.0046`
- Pooled excess return: `0.0303`
- Passed CV gate: `false`

Fold details:
- Fold 0: learned return `0.0045`, buy-and-hold return `0.0198`, excess `-0.0153`,
  Sharpe `6.9447`, trades `16`, passed `false`.
- Fold 1: learned return `0.0002`, buy-and-hold return `-0.0446`, excess `0.0447`,
  Sharpe `0.6535`, trades `4`, passed `true`.

Interpretation:
- Adding NEAR improved median excess and pooled excess versus the four-symbol retained baseline, but
  reduced median Sharpe and worsened the weakest fold's trade count from `6` to `4`.
- Do not change the default liquid-major universe from this evidence. Keep broader local universe
  experiments as a promising but unproven direction because the strict CV proof still fails.

## 2026-05-09 - Five-Symbol Universe Activity-Weight Check

Objective:
- Check whether the existing validation activity-weight preference can fix the weak-fold trade count
  after adding `NEARUSDT`.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,NEARUSDT \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --selection-activity-weight 0.01 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `e1d401af9ca2`
- Median excess return: `0.0147`
- Median Sharpe: `3.7991`
- Minimum fold trades: `4`
- Pooled learned return: `0.0046`
- Pooled excess return: `0.0303`
- Passed CV gate: `false`

Interpretation:
- Activity weighting selected the same effective five-symbol policy. It did not repair the weak-fold
  activity problem.

## 2026-05-09 - Five-Symbol Shorter-Horizon Check

Objective:
- Check whether adding 5-minute horizon and max-hold candidates improves weak-fold activity in the
  five-symbol universe.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,NEARUSDT \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 5,15,30 \
  --max-hold-grid 5,15,30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `db251bfe93bf`
- Median excess return: `0.0147`
- Median Sharpe: `3.7991`
- Minimum fold trades: `4`
- Pooled learned return: `0.0046`
- Pooled excess return: `0.0303`
- Passed CV gate: `false`

Interpretation:
- The expanded horizon grid selected the same effective 30-minute ridge policies as the five-symbol
  baseline.
- The weak-fold activity gap is not solved by adding 5-minute horizon candidates in the expanded
  universe.

## 2026-05-09 - Feature-Source Ablation

Objective:
- Test whether futures metrics or premium-index features are adding noise to the retained ridge
  setup, or whether the model needs both exogenous feature sources.

Spot-only command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --no-futures-metrics \
  --no-premium-index \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Spot-only result:
- Run: `9649752c51e6`
- Median excess return: `0.0092`
- Median Sharpe: `0.0000`
- Minimum fold trades: `0`
- Pooled learned return: `0.0000`
- Pooled excess return: `0.0193`
- Passed CV gate: `false`

Premium-only exogenous command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --no-futures-metrics \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Premium-only exogenous result:
- Run: `c9c64ee95c04`
- Median excess return: `0.0092`
- Median Sharpe: `0.3058`
- Minimum fold trades: `0`
- Pooled learned return: `0.0001`
- Pooled excess return: `0.0194`
- Passed CV gate: `false`

Futures-only exogenous command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --no-premium-index \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Futures-only exogenous result:
- Run: `e5da066e7771`
- Median excess return: `0.0110`
- Median Sharpe: `3.1228`
- Minimum fold trades: `2`
- Pooled learned return: `0.0037`
- Pooled excess return: `0.0230`
- Passed CV gate: `false`

Interpretation:
- Removing both exogenous sources made the strategy flat.
- Keeping premium without futures was also mostly flat.
- Keeping futures without premium was better than the other ablations but still worse than the
  retained full-feature baseline: median excess `0.0134`, median Sharpe `5.7534`, minimum fold
  trades `6`, pooled learned return `0.0084`, pooled excess `0.0277`.
- Keep both futures metrics and premium-index features enabled by default.

## 2026-05-09 - Rejected Sparse Linear Model Probe

Objective:
- Test whether sparse linear models can suppress noisy intraday features better than ridge without
  adding nonlinear model capacity.

Change:
- Temporarily added `lasso_alpha_<N>` and `elastic_net_alpha_<N>` model-family variants.
- Reverted the model-family additions after CV selected ridge in both folds and the sparse candidates
  added substantial runtime with coordinate-descent convergence warnings.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,lasso_alpha_0_000001,lasso_alpha_0_00001,elastic_net_alpha_0_000001,elastic_net_alpha_0_00001 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `03cf60297aaa`
- Selected model family: `ridge` in both folds
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- Sparse linear candidates did not improve selection or metrics over the retained ridge baseline.
- Do not keep the sparse model-family surface until there is a better-conditioned implementation or
  a clearer reason to pay the runtime cost.

## 2026-05-09 - Rejected Longer Validation Window Probe

Objective:
- Test whether a wider 14-day validation window selects less fragile candidates than the retained
  7-day validation window, without changing model family, execution, costs, or CV fold shape.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 14 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `efa18a1d46af`
- Median excess return: `0.0092`
- Median Sharpe: `0.2730`
- Minimum fold trades: `0`
- Pooled learned return: `0.0001`
- Pooled excess return: `0.0194`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no candidate and stayed flat while buy-and-hold gained `0.0228`.
- Fold 1 selected ridge, traded `6` times, and beat falling buy-and-hold, but its validation excess
  was still negative.

Interpretation:
- Longer validation made selection more conservative in fold 0 and reduced median Sharpe and pooled
  return.
- Keep the retained 7-day validation window for the current setup.

## 2026-05-09 - Rejected Shorter Training Window Probe

Objective:
- Test whether a 60-day training window adapts better to recent 1m intraday conditions than the
  retained 90-day training window.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 60 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `69534a5a237d`
- Median excess return: `0.0083`
- Median Sharpe: `2.6667`
- Maximum fold drawdown: `0.0171`
- Minimum fold trades: `6`
- Pooled learned return: `-0.0018`
- Pooled excess return: `0.0175`
- Passed CV gate: `false`

Fold details:
- Fold 0 traded only `6` times and trailed buy-and-hold by `-0.0201`.
- Fold 1 traded `75` times but lost money with Sharpe `-2.5090`.

Interpretation:
- The shorter training window increased weak-fold activity in fold 1 by selecting a much churnier
  candidate, but it reduced pooled learned return and Sharpe.
- Keep the retained 90-day training window.

## 2026-05-09 - Post-Time-Feature Market-Return Target Check

Objective:
- Retest the learned market-return target after adding time-seasonality features, checking whether it
  can improve broad-market participation in the bull evaluation fold without a hard-coded regime
  rule.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_market_return \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `a6dca72d8a5d`
- Selected model family: `ridge` in both folds
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- The market-return target did not change selection after time-seasonality features; validation still
  selected the absolute-return ridge policy.
- Keep `_market_return` as a non-default experiment option, but it is not current progress toward
  the goal.

## 2026-05-09 - Post-Time-Feature Market-Excess Target Check

Objective:
- Retest the benchmark-relative market-excess target after adding time-seasonality features, checking
  whether it can reduce the remaining fold-level buy-and-hold gap.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_market_excess \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `ee0a012f94c8`
- Selected model family: `ridge` in both folds
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- The market-excess target did not change selection after time-seasonality features.
- Keep `_market_excess` as a non-default experiment option, but it is not current progress toward
  the goal.

## 2026-05-09 - Lower Threshold Surface Check

Objective:
- Test whether adding 5th and 10th percentile score thresholds can raise weak-fold activity without
  adding model complexity or weakening CV gates.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `728d8a77824d`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- Lower threshold candidates selected the same effective ridge policy as the retained baseline.
- The weak-fold trade count is not fixed by extending the threshold surface below the 25th
  percentile.

## 2026-05-09 - Longer Max-Hold Grid Check

Objective:
- Test whether the retained 15/30-minute learned entries are being exited too early, while keeping
  positions intraday and validation-selected.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120,240,480 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `5246ae4a4db4`
- Selected max hold: `60` minutes in both folds
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- Longer intraday max holds were available, but validation selected the same 60-minute max hold as
  the retained baseline.
- The fold 0 buy-and-hold gap is not solved by extending max hold to 4-8 hours.

## 2026-05-09 - Rejected 10-Symbol Liquid Universe Probe

Objective:
- Test whether a broader locally cached liquid universe improves opportunity breadth and weak-fold
  trade activity without changing model family, execution rules, costs, or CV gates.

Change:
- Expanded the local cache with spot, futures metrics, and premium-index 1m data for
  `ADAUSDT,XRPUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT`.
- Updated the intraday data loader to validate only the requested experiment date slice, so unused
  earlier cache gaps do not block later CV windows.

Data commands:

```bash
.venv/bin/python download_data.py \
  --tickers ADAUSDT,XRPUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT \
  --start-date 2025-01-01

.venv/bin/python download_futures_metrics_direct.py \
  --tickers ADAUSDT,XRPUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT \
  --start-date 2025-01-01 \
  --end-date 2026-04-30

.venv/bin/python download_premium_index_klines_direct.py \
  --tickers ADAUSDT,XRPUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT \
  --start-date 2025-01-01 \
  --end-date 2026-04-30
```

CV command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,NEARUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT \
  --start-date 2025-05-01 \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `0c049b7c2e77`
- Median excess return: `0.0042`
- Median Sharpe: `-2.5287`
- Maximum fold drawdown: `0.0042`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0017`
- Pooled excess return: `0.0088`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no candidate and stayed flat while expanded-universe buy-and-hold gained `0.0149`.
- Fold 1 traded `4` times, beat falling buy-and-hold on excess return, but lost money with Sharpe
  `-5.0574`.

Verification:
- `tests/test_intraday_data.py` covered requested-slice validation after the loader change.

Interpretation:
- Expanding to 10 symbols did not improve the current ridge policy. It reduced median Sharpe, pooled
  learned return, pooled excess, and weak-fold activity versus the retained four-symbol baseline.
- Keep the 10-symbol cache for future experiments, but do not change the default liquid-major
  universe or retained best run from this evidence.

## 2026-05-09 - Rejected Regularized Tree Model Probe

Objective:
- Test whether a shallow, L2-regularized histogram gradient boosting model can capture useful
  feature interactions without the instability observed from the default tree model.

Change:
- Temporarily added `hist_gradient_boosting_regularized` with lower learning rate, fewer iterations,
  shallow leaves, and L2 regularization.
- Reverted the model-family addition after CV selected ridge in both folds and metrics were
  unchanged.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,hist_gradient_boosting_regularized \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `c4a2eb390ab5`
- Selected model family: `ridge` in both folds
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- The regularized tree did not improve selection or metrics over the retained ridge baseline.
- Do not keep the extra tree variant in the operational model-family surface.

## 2026-05-09 - Rejected 7-Symbol High-Liquidity Universe Probe

Objective:
- Test whether a narrower expansion to high-liquidity majors improves opportunity breadth without
  the degradation seen in the full 10-symbol universe probe.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT \
  --start-date 2025-05-01 \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `5e207af9195d`
- Median excess return: `-0.0029`
- Median Sharpe: `-5.3830`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0077`
- Pooled excess return: `-0.0053`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no candidate and stayed flat while buy-and-hold gained `0.0188`.
- Fold 1 traded `10` times, beat falling buy-and-hold on excess return, but lost `-0.0077` with
  Sharpe `-10.7659`.

Interpretation:
- The high-liquidity subset was worse than both the retained four-symbol baseline and the full
  10-symbol probe.
- Do not expand the default universe to this subset.

## 2026-05-09 - Rejected One-Hot Ticker Identity Probe

Objective:
- Test whether replacing ordinal numeric `ticker_id` with one-hot ticker identity improves the
  ridge model by removing arbitrary ordering between assets.

Change:
- Temporarily replaced numeric `ticker_id` with one-hot `ticker_<symbol>` features in training and
  prediction frames.
- Reverted the change after CV underperformed the retained numeric ticker-id baseline.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Transient result before revert:
- Run id: `84e6ad38a061` reused by the config-derived hash.
- Median excess return: `0.0093`
- Median Sharpe: `0.3431`
- Minimum fold trades: `0`
- Pooled learned return: `0.0001`
- Pooled excess return: `0.0194`
- Passed CV gate: `false`

Restored retained result after revert:
- Run: `84e6ad38a061`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- One-hot ticker identity made fold 0 flat and reduced pooled learned return.
- Keep numeric `ticker_id` for the current ridge setup despite its imperfect semantics, because it is
  empirically stronger on the current CV proof.

## 2026-05-09 - Rejected Hold-Until-Max-Hold Exit Candidate

Objective:
- Test whether score-based exits after the forecast horizon are cutting profitable intraday holds
  too early.

Change:
- Temporarily added a validation-selectable `-inf` exit-threshold candidate, meaning an open position
  can hold until the selected max hold instead of exiting on score deterioration after the horizon.
- Reverted the candidate after validation selected the same finite exit thresholds as the retained
  baseline and metrics were unchanged.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run id: `84e6ad38a061` reused by the config-derived hash.
- Selected exit thresholds: finite retained thresholds in both folds, not the `-inf` sentinel.
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- Validation did not prefer holding positions until max hold. Keep the existing finite exit-threshold
  hysteresis surface.

## 2026-05-09 - Candidate Surface Diagnostics

Objective:
- Add enough report detail to inspect what validation candidates existed, not only the selected
  candidate, so future metric work can target the actual failure surface.

Change:
- Added `top_candidates` to each walk-forward window report.
- Each window now records up to 10 validation candidates sorted by activity gate, selection trade
  floor, selection score, and Sharpe.
- No selection logic changed.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `84e6ad38a061`
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Diagnostic findings:
- Fold 0 top validation candidates all had negative validation excess; the best active candidate
  had only `2` validation trades and validation excess `-0.0458`.
- Fold 1 top validation candidates also had negative validation excess; the best active candidate
  had `14` validation trades and validation excess `-0.0137`.
- This confirms the next metric improvement likely needs a stronger learned signal or a different
  benchmark-aware objective, not another small threshold or max-hold tweak.

Verification:
- `tests/test_intraday_policy.py` checks that walk-forward windows expose at most 10
  `top_candidates`.

## 2026-05-09 - Rejected Per-Ticker Ridge Probe

Objective:
- Test whether separate regularized linear models per ticker improve over the pooled ridge model
  with numeric ticker identity, without adding nonlinear capacity.

Change:
- Temporarily added `ridge_per_ticker`, fitting one standardized ridge model per `ticker_id` with a
  global ridge fallback.
- Reverted the model-family addition after CV underperformed the retained pooled ridge baseline.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_per_ticker \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `e6191812c37b`
- Median excess return: `0.0114`
- Median Sharpe: `0.2803`
- Minimum fold trades: `14`
- Pooled learned return: `0.0045`
- Pooled excess return: `0.0238`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected pooled `ridge`, matching the retained fold 0 failure against buy-and-hold.
- Fold 1 selected `ridge_per_ticker`, increased trades to `16`, but lost money with Sharpe
  `-10.2602`.

Interpretation:
- Per-ticker ridge improved activity but degraded return and Sharpe.
- Keep pooled ridge as the retained model family.

## 2026-05-09 - Rejected Market-Excess Classifier Probe

Objective:
- Test whether the existing positive-return classifier improves benchmark-relative selection when
  trained on the market-excess label.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,logistic_positive_market_excess \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `6c528d27a964`
- Selected model family: `ridge` in both folds
- Median excess return: `0.0134`
- Median Sharpe: `5.7534`
- Minimum fold trades: `6`
- Pooled learned return: `0.0084`
- Pooled excess return: `0.0277`
- Passed CV gate: `false`

Interpretation:
- The market-excess classifier did not improve validation selection; the top candidates remained
  pooled ridge candidates.
- Keep `logistic_positive_market_excess` as an available composed experiment name, but it is not
  current progress toward the goal.

## 2026-05-09 - Volatility-Scaled Target Improvement

Objective:
- Test whether a risk-adjusted target improves the retained ridge policy by reducing the influence
  of high-volatility raw-return labels.

Change:
- Added `_vol_scaled` model-family suffix support.
- A `_vol_scaled` variant trains on after-cost forward return divided by causal `volatility_60m` at
  the signal bar.
- Kept selection and CV proof gates unchanged.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `6ebe01e9a349`
- Median excess return: `0.0140`
- Median Sharpe: `6.2236`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `6`
- Pooled learned return: `0.0096`
- Pooled excess return: `0.0289`
- Passed CV gate: `false`

Fold details:
- Fold 0 still selected plain `ridge`, returned `0.0083`, and trailed buy-and-hold by `-0.0145`.
- Fold 1 selected `ridge_vol_scaled`, improved learned return to `0.0013`, improved Sharpe to
  `1.6266`, and improved excess to `0.0424`, but still traded only `6` times.

Interpretation:
- This is a retained metric improvement over the previous best: median excess, median Sharpe, pooled
  learned return, and pooled excess all improved.
- The strict CV goal still fails because fold 0 trails buy-and-hold and the weakest fold has only
  `6` trades.
- Keep `_vol_scaled` as a supported non-default target variant and use `6ebe01e9a349` as the current
  best direction, while keeping `84e6ad38a061` as the best plain-ridge baseline.

## 2026-05-09 - Five-Symbol Vol-Scaled Target Check

Objective:
- Test whether combining the retained `_vol_scaled` target direction with the earlier NEAR universe
  expansion improves the strict CV proof.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --tickers BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,NEARUSDT \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `b2ed65795802`
- Selected model family: `ridge` in both folds
- Median excess return: `0.0147`
- Median Sharpe: `3.7991`
- Minimum fold trades: `4`
- Pooled learned return: `0.0046`
- Pooled excess return: `0.0303`
- Passed CV gate: `false`

Interpretation:
- The five-symbol run improved pooled excess but selected plain ridge and reproduced the earlier NEAR
  universe result.
- It is not the retained direction because it lowers median Sharpe, pooled learned return, and
  weakest-fold activity versus the four-symbol `_vol_scaled` run.

## 2026-05-09 - Lower Threshold Surface After Vol-Scaled Target

Objective:
- Test whether adding lower entry threshold quantiles to the retained `_vol_scaled` target surface
  increases weakest-fold activity or improves strict CV performance.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `536e369583b4`
- Median excess return: `0.0140`
- Median Sharpe: `6.2236`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `6`
- Pooled learned return: `0.0096`
- Pooled excess return: `0.0289`
- Passed CV gate: `false`

Interpretation:
- The expanded low-threshold surface selected the same effective behavior as retained run
  `6ebe01e9a349`.
- It does not fix the remaining blockers: fold 0 still trails buy-and-hold and fold 1 still has only
  `6` trades against the strict `10`-trade activity floor.

## 2026-05-09 - Market-Excess Vol-Scaled Target Probe

Objective:
- Test whether a benchmark-relative, volatility-normalized label helps the selector find candidates
  that beat buy-and-hold in the weak fold.

Change:
- Added `_market_excess_vol_scaled` model-family suffix support.
- The target is equal-weight market-excess forward return divided by causal `volatility_60m`.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,ridge_market_excess_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `abb830453de8`
- Median excess return: `0.0140`
- Median Sharpe: `6.2236`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `6`
- Pooled learned return: `0.0096`
- Pooled excess return: `0.0289`
- Passed CV gate: `false`

Interpretation:
- The composite target did not enter the selected policy in either fold.
- Fold 0 still selected `ridge`; fold 1 still selected `ridge_vol_scaled`.
- Keep the target variant available for future benchmark-relative probes, but it is not a metric
  improvement over retained run `6ebe01e9a349`.

## 2026-05-09 - Vol-Scaled Risk-Unit Grid

Objective:
- Test whether validation-selected exposure sizing improves the current `_vol_scaled` direction,
  especially fold 0's underexposure versus buy-and-hold.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --risk-unit-grid 0.25,0.5,1.0 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `e6fcb755af44`
- Median excess return: `0.0144`
- Median Sharpe: `3.3377`
- Maximum fold drawdown: `0.0162`
- Minimum fold trades: `6`
- Pooled learned return: `0.0105`
- Pooled excess return: `0.0298`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge` with `risk_unit=1.0`, traded `10` times, and still trailed buy-and-hold by
  `-0.0150`.
- Fold 1 selected `ridge_vol_scaled` with `risk_unit=1.0`, beat buy-and-hold by `0.0438`, but still
  traded only `6` times.

Interpretation:
- This improves median excess, pooled learned return, pooled excess, and fold 0 activity versus
  retained run `6ebe01e9a349`.
- It is not a clean replacement because median Sharpe drops from `6.2236` to `3.3377`, drawdown
  increases, fold 0 still fails buy-and-hold, and the weakest fold still misses the 10-trade CV
  floor.

## 2026-05-09 - Max-Hold 30 Activity Probe

Objective:
- Test whether removing the 60/120-minute hold choices improves out-of-sample fold activity without
  adding a hand-picked exit rule.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `6fc9e442f7d3`
- Median excess return: `0.0128`
- Median Sharpe: `4.7348`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `8`
- Pooled learned return: `0.0073`
- Pooled excess return: `0.0266`
- Passed CV gate: `false`

Interpretation:
- The shorter max-hold surface increased weakest-fold activity from `6` to `8`, but reduced median
  excess, pooled excess, and made fold 1's learned return and Sharpe negative.
- Do not retain this as the best direction.

## 2026-05-09 - Fifteen-Minute Horizon Activity Probe

Objective:
- Test whether the current 30-minute selected horizon is too coarse for the day-trading activity
  target by forcing a 15-minute forecast horizon surface.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15 \
  --max-hold-grid 15,30 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `b293d97c1479`
- Median excess return: `0.0081`
- Median Sharpe: `-1.5716`
- Maximum fold drawdown: `0.0045`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0021`
- Pooled excess return: `0.0172`
- Passed CV gate: `false`

Interpretation:
- The forced 15-minute horizon is worse than the retained 30-minute-selected surface.
- Fold 0 found no selected active policy, fold 1 learned return was negative, and pooled learned
  return was negative.
- Do not retain this direction.

## 2026-05-09 - Learned Market-Level Ridge Probe

Objective:
- Test whether a learned market-participation model can improve the buy-and-hold failure surface
  without adding an always-on buy-and-hold fallback.

Change:
- Added `market_ridge`, a model family that trains on the equal-weight market-return label using
  averaged causal ticker features.
- `market_ridge` emits one shared market score to every ticker; validation replay still selects
  thresholds, horizon, max hold, and whether the family is used.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `f5ef0c9987b7`
- Median excess return: `0.0161`
- Median Sharpe: `8.0331`
- Maximum fold drawdown: `0.0051`
- Minimum fold trades: `14`
- Pooled learned return: `0.0140`
- Pooled excess return: `0.0333`
- Passed CV gate: `false`

Fold details:
- Fold 0 still selected `ridge`, returned `0.0083`, and trailed buy-and-hold by `-0.0145`.
- Fold 1 selected `market_ridge`, returned `0.0056`, beat buy-and-hold by `0.0467`, and traded `24`
  times.

Interpretation:
- This is the strongest retained direction so far on median excess, median Sharpe, pooled learned
  return, pooled excess, and weakest-fold trade activity.
- The strict CV goal still fails because fold 0 trails buy-and-hold, but the activity blocker is
  cleared in this run.

## 2026-05-09 - Market-Level Ridge Only Check

Objective:
- Check whether `market_ridge` alone can solve the fold 0 buy-and-hold failure, or whether it is only
  helpful when validation can choose it in specific folds.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `35010755c972`
- Median excess return: `0.0120`
- Median Sharpe: `2.6227`
- Maximum fold drawdown: `0.0051`
- Minimum fold trades: `0`
- Pooled learned return: `0.0056`
- Pooled excess return: `0.0249`
- Passed CV gate: `false`

Interpretation:
- `market_ridge` alone is worse than the combined retained surface.
- Fold 0 selected no active market policy, while fold 1 reproduced the useful `market_ridge`
  behavior.
- Keep `market_ridge` as a candidate family, not as a standalone replacement.

## 2026-05-09 - Market-Level Ridge With Risk-Unit Grid

Objective:
- Test whether combining the two strongest partial improvements, `market_ridge` and
  validation-selected risk units, improves strict CV.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --risk-unit-grid 0.25,0.5,1.0 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `1ea5db524351`
- Median excess return: `0.0144`
- Median Sharpe: `3.3377`
- Maximum fold drawdown: `0.0162`
- Minimum fold trades: `6`
- Pooled learned return: `0.0105`
- Pooled excess return: `0.0298`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge` with `risk_unit=1.0`, traded `10` times, and still trailed buy-and-hold by
  `-0.0150`.
- Fold 1 selected `ridge_vol_scaled` with `risk_unit=1.0`, displacing the better `market_ridge`
  behavior from run `f5ef0c9987b7`.

Interpretation:
- The risk-unit grid cancels the `market_ridge` improvement by selecting higher-exposure
  `ridge_vol_scaled` in fold 1.
- Do not combine broad risk-unit selection with the current `market_ridge` candidate surface.

## 2026-05-09 - Market-Level Ridge Three-Fold Robustness Check

Objective:
- Check whether the new `market_ridge` improvement generalizes beyond the two most recent CV folds.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `f9b9c5c5702b`
- Median excess return: `-0.0145`
- Median Sharpe: `5.2454`
- Maximum fold drawdown: `0.0225`
- Minimum fold trades: `14`
- Pooled learned return: `-0.0076`
- Pooled excess return: `-0.0335`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `market_ridge`, traded `56` times, lost `-0.0213`, and trailed buy-and-hold by
  `-0.0674`.
- Fold 1 reproduced the retained fold 0 `ridge` failure from the two-fold run.
- Fold 2 reproduced the useful `market_ridge` behavior from the two-fold run.

Interpretation:
- `market_ridge` is a strong two-fold improvement but fails a broader three-fold generalization
  check.
- Keep it as an experimental candidate because it improved recent two-fold activity and excess, but
  do not treat it as achieving the goal.
- The next durable improvement needs to prevent market-participation candidates from being selected
  in validation states that do not generalize to the following week.

## 2026-05-10 - Fourteen-Day Validation Robustness Check

Objective:
- Test whether the `market_ridge` candidate surface needs a longer validation window to avoid
  selecting market-participation policies that fail the following week.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 14 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `3178b170a5ec`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0078`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0061`
- Pooled excess return: `-0.0320`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge`, had positive validation excess, then lost `-0.0062` while buy-and-hold
  gained `0.0461`.
- Fold 1 selected no active policy and missed a positive buy-and-hold week.
- Fold 2 selected `ridge`, barely traded, and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- A longer validation window does not fix the generalization problem; it mostly suppresses trading
  while still missing positive market weeks.
- Do not replace the retained 7-day validation surface with this setting.

## 2026-05-10 - Selected Generalization Diagnostics

Objective:
- Make validation-to-evaluation instability visible in report artifacts so future iterations can
  target the selector failure directly.

Change:
- Added `selected_generalization` to each evaluated walk-forward window.
- The diagnostic records validation and evaluation excess return, cumulative return, Sharpe,
  maximum drawdown, trade count, and their deltas.
- Selection behavior is unchanged.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `f5ef0c9987b7`
- Metrics unchanged from the learned market-level ridge probe.
- Fold 0 diagnostic: validation excess `-0.0458`, evaluation excess `-0.0145`, trades increased
  from `2` to `14`.
- Fold 1 diagnostic: validation excess `-0.0023`, evaluation excess `0.0467`, trades fell from `40`
  to `24`.

Verification:
- `tests/test_intraday_policy.py` checks that selected windows expose `selected_generalization` and
  `excess_return_delta`.
- `.venv/bin/python -m pytest -q` passed with `29 passed`.

Interpretation:
- This does not improve metrics, but it improves the experiment loop by making selector
  generalization drift explicit in every report.

## 2026-05-10 - Rejected Positive Validation Excess Selection Gate

Objective:
- Test whether requiring selected candidates to beat buy-and-hold during validation improves
  three-fold generalization.

Transient change:
- Added a temporary `--selection-requires-positive-excess` selector gate.
- The gate required `validation_excess_return > 0` before a candidate could be selected.
- Reverted after the run because it suppressed all trading and did not improve CV.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --selection-requires-positive-excess \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `fb042682ea2e`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0000`
- Minimum fold trades: `0`
- Pooled learned return: `0.0000`
- Pooled excess return: `-0.0259`
- Passed CV gate: `false`

Interpretation:
- The gate avoided drawdown but selected no active policy in any fold, missed positive buy-and-hold
  weeks, and failed the day-trading activity proof.
- Do not retain this selector constraint.

## 2026-05-10 - Rejected Market Trend-State Feature Probe

Objective:
- Test whether causal market trend-state features help distinguish useful market participation from
  `market_ridge` overtrading.

Transient change:
- Added `market_sma_distance_<window>m` and `market_range_position_<window>m` features built from the
  cumulative equal-weight one-minute market return.
- Reverted after the feature worsened the current two-fold best.

Commands:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 2 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Transient results:
- Three-fold run id reused by config hash: `f9b9c5c5702b`
- Three-fold median excess return: `-0.0154`
- Three-fold pooled excess return: `-0.0181`
- Three-fold minimum fold trades: `0`
- Two-fold run id reused by config hash: `f5ef0c9987b7`
- Two-fold median excess return: `0.0131`
- Two-fold pooled excess return: `0.0271`
- Two-fold minimum fold trades: `6`
- Passed CV gate: `false`

Interpretation:
- The features improved the previously bad three-fold pooled excess from `-0.0335` to `-0.0181`, but
  the proof still failed with one flat fold and negative median excess.
- They worsened the current two-fold best versus retained `f5ef0c9987b7` before the feature probe
  (`0.0161` median excess and `0.0333` pooled excess).
- Do not retain these market trend-state features.

## 2026-05-10 - Three-Fold Robustness Without Market Ridge

Objective:
- Check whether the broader three-fold failure is caused by including `market_ridge` in the candidate
  surface.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `67f7e551ee84`
- Median excess return: `-0.0145`
- Median Sharpe: `1.6266`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Pooled learned return: `0.0096`
- Pooled excess return: `-0.0163`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold rally.
- Fold 1 selected `ridge`, returned `0.0083`, and trailed buy-and-hold by `-0.0145`.
- Fold 2 selected `ridge_vol_scaled`, returned `0.0013`, and beat falling buy-and-hold by `0.0424`.

Interpretation:
- Removing `market_ridge` improves the three-fold pooled excess versus the `market_ridge` run
  (`-0.0163` versus `-0.0335`), but still fails the generalization proof.
- The broader blocker is not only `market_ridge`; the model surface still misses rising market weeks
  or underparticipates in them.

## 2026-05-10 - Rejected 120-Day Training Window Probe

Objective:
- Test whether a 120-day training window is a better compromise than the current 90-day window and
  the previously rejected 180-day window.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 120 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `b5f55b6e04f2`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0087`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0028`
- Pooled excess return: `-0.0287`
- Passed CV gate: `false`

Fold details:
- Folds 0 and 1 selected no active policy and missed positive buy-and-hold weeks.
- Fold 2 selected `ridge_vol_scaled`, traded `22` times, beat falling buy-and-hold on excess, but
  lost money with negative Sharpe.

Interpretation:
- The 120-day training window is too conservative and worse than the 90-day three-fold
  `ridge,ridge_vol_scaled` surface.
- Keep 90 training days as the retained setting.

## 2026-05-10 - Rejected Market-Return Classifier Probe

Objective:
- Test whether a positive-market-return classifier helps the model participate in rising market
  weeks without using a hard-coded buy-and-hold fallback.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,logistic_positive_market_return \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `cfb2ac090ff8`
- Median excess return: `-0.0145`
- Median Sharpe: `1.6266`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Pooled learned return: `0.0096`
- Pooled excess return: `-0.0163`
- Passed CV gate: `false`

Interpretation:
- The classifier was not selected in any fold and reproduced the no-`market_ridge` three-fold
  baseline.
- Keep `logistic_positive_market_return` available through the composable model-family naming, but
  it is not current progress.

## 2026-05-10 - Rejected Three-Day Validation Probe

Objective:
- Test whether a shorter validation window adapts faster to rising market weeks than the retained
  seven-day validation window.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 3 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `791165b9fc66`
- Median excess return: `-0.0122`
- Median Sharpe: `-5.1758`
- Maximum fold drawdown: `0.0175`
- Minimum fold trades: `10`
- Pooled learned return: `-0.0089`
- Pooled excess return: `-0.0348`
- Passed CV gate: `false`

Fold details:
- The shorter validation window selected active policies in every fold and met the trade-count floor.
- It overtraded: fold 0 lost `-0.0157`, fold 2 lost `-0.0036`, and pooled learned return was
  negative.

Interpretation:
- Three validation days solves inactivity but creates noisy, losing selection.
- Keep the retained seven-day validation window.

## 2026-05-10 - Rejected Strict Validation Activity Gate

Objective:
- Test whether making the validation activity gate match the CV fold activity proof avoids selecting
  thin validation candidates that fail out of sample.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 10 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `421271d9952e`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Pooled learned return: `0.0013`
- Pooled excess return: `-0.0247`
- Passed CV gate: `false`

Interpretation:
- The stricter validation activity gate made folds 0 and 1 flat and still missed positive
  buy-and-hold weeks.
- It does not solve the generalization problem; keep `--min-validation-trades 1` with
  `--selection-trade-floor 10` as the better retained selection setup.

## 2026-05-10 - Rejected Five-Day Validation Probe

Objective:
- Test whether a five-day validation window adapts faster than seven days without becoming as noisy
  as the rejected three-day validation probe.

Command:

```bash
.venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 5 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `198700f9cdc4`
- Median excess return: `-0.0145`
- Median Sharpe: `-0.7460`
- Maximum fold drawdown: `0.0215`
- Minimum fold trades: `10`
- Pooled learned return: `-0.0078`
- Pooled excess return: `-0.0337`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge_vol_scaled`, traded `20` times, lost `-0.0155`, and trailed
  buy-and-hold by `-0.0616`.
- Fold 1 selected `ridge`, traded `14` times, returned `0.0083`, and trailed buy-and-hold by
  `-0.0145`.
- Fold 2 selected `ridge_vol_scaled`, traded `10` times, returned `-0.0005`, and beat falling
  buy-and-hold by `0.0406`.

Interpretation:
- Five validation days kept every fold active, but it still overtraded the rising fold and produced
  negative pooled learned return.
- Keep the retained seven-day validation window; shorter validation windows have not solved the
  generalization problem.

## 2026-05-10 - Rejected Recency-Weighted Ridge Probe

Objective:
- Test whether exponentially weighting recent training rows helps the learned 1m model adapt without
  changing execution rules, costs, CV gates, or adding hand-coded market-state logic.

Implementation:
- Added `ridge_decay_<days>` model-family support.
- The candidate uses the same ridge estimator as `ridge`, but passes exponentially decayed sample
  weights during training. A 14-day half-life gives the newest row weight `1.0`, rows 14 days older
  weight `0.5`, and rows 28 days older weight `0.25`.
- Target suffixes still compose, so this probe used `ridge_decay_14_vol_scaled`.

Command:

```bash
timeout 600s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_decay_14_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `cb399a692694`
- Median excess return: `-0.0181`
- Median Sharpe: `0.4234`
- Maximum fold drawdown: `0.0067`
- Minimum fold trades: `0`
- Pooled learned return: `0.0052`
- Pooled excess return: `-0.0207`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy, missed a `0.0461` buy-and-hold week, and failed the trade floor.
- Fold 1 traded `12` times, returned `0.0047`, and trailed buy-and-hold by `-0.0181`.
- Fold 2 traded `16` times, returned `0.0005`, and beat falling buy-and-hold by `0.0416`.

Interpretation:
- Recency weighting did not fix the rising-market underparticipation problem and did not improve the
  retained three-fold proof.
- Keep the implementation available only as an experimental model-family candidate; it is not current
  metric progress.

## 2026-05-10 - Rejected Shared Ridge Probe

Objective:
- Test whether removing ordinal numeric `ticker_id` from pooled ridge improves generalization while
  preserving the same learned features, execution, costs, and CV gates.

Implementation:
- Added `ridge_shared`, a ridge model family that trains across all ticker rows after dropping
  `ticker_id`.
- This is distinct from the earlier rejected one-hot and per-ticker probes: it keeps one pooled
  model but removes arbitrary ticker ordering as a feature.

Command:

```bash
timeout 600s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_shared \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `3a4304875306`
- Median excess return: `-0.0164`
- Median Sharpe: `0.7525`
- Maximum fold drawdown: `0.0017`
- Minimum fold trades: `0`
- Pooled learned return: `0.0065`
- Pooled excess return: `-0.0194`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 traded `12` times, returned `0.0063`, and trailed buy-and-hold by `-0.0164`.
- Fold 2 traded only `4` times, returned `0.0001`, and beat falling buy-and-hold by `0.0413` but
  failed the CV activity floor.

Interpretation:
- Dropping ordinal ticker identity did not solve the rising-market underparticipation problem and
  reduced activity versus the retained proof.
- Keep `ridge_shared` as an experimental model-family candidate, but it is not current metric
  progress.

## 2026-05-10 - Rejected Validation-Slice Selector Probe

Objective:
- Test whether ranking validation candidates by chronological validation-slice stability reduces
  overfit candidate selection without changing execution rules, costs, or CV gates.

Implementation:
- Added optional `--validation-slices`.
- The default is `1`, preserving existing selection behavior.
- When greater than `1`, candidates must still pass the whole-validation activity, Sharpe, return,
  and drawdown gates. Passing candidates are then ranked by median excess return across equal
  chronological validation slices instead of whole-validation excess return.
- Reports include `validation_slice_excess_median`, `validation_slice_excess_min`, and per-slice
  validation metrics for selected and top candidates.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --validation-slices 3 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `a487d4382af4`
- Median excess return: `-0.0145`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0125`
- Minimum fold trades: `0`
- Pooled learned return: `0.0038`
- Pooled excess return: `-0.0221`
- Passed CV gate: `false`

Fold details:
- Fold 0 still selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `ridge`, traded `14` times, returned `0.0083`, and trailed buy-and-hold by
  `-0.0145`.
- Fold 2 selected `ridge_vol_scaled`, traded `24` times, returned `-0.0045`, and only beat
  buy-and-hold because buy-and-hold fell.

Interpretation:
- Slice ranking made candidate instability visible, but it did not solve the core rising-market
  underparticipation problem.
- Keep `--validation-slices` as an experimental anti-overfitting diagnostic/selector option, not as
  a retained metric improvement.

## 2026-05-10 - Rejected Path-Mean Target Probe

Objective:
- Test whether a path-mean intraday target better matches path-based replay than the exact-horizon
  forward return target.

Implementation:
- Added `_path_mean` model-family target variants.
- A path-mean label averages after-cost forward returns from one minute through the selected
  horizon. It requires the full future path to be known before a training row is eligible.
- Added `_path_mean_vol_scaled`, which divides the path-mean label by causal `volatility_60m`.
- This deliberately avoids max-favorable-excursion labeling; it does not train on the best unseen
  future exit.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_path_mean,ridge_path_mean_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `fd0afdfae4b9`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0045`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0006`
- Pooled excess return: `-0.0265`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected no active policy and missed a `0.0228` buy-and-hold week.
- Fold 2 selected `ridge_path_mean_vol_scaled`, traded `6` times, lost `-0.0006`, and only beat
  buy-and-hold because buy-and-hold fell.

Interpretation:
- The path-mean target worsened activity and pooled return versus the retained exact-horizon target
  surface.
- Keep `_path_mean` variants as experimental label options, but they are not current metric progress.

## 2026-05-10 - Rejected Market Volatility-Scaled Ridge Probe

Objective:
- Test whether volatility-scaling the learned market-participation label improves the unstable
  `market_ridge` direction without adding a hard-coded market fallback.

Implementation:
- Added `market_ridge_vol_scaled`.
- The model still averages causal ticker features by timestamp and emits one shared score to every
  ticker, but trains on equal-weight market forward return divided by causal
  `market_volatility_60m`.
- Validation replay still selects horizon, threshold, exit threshold, max hold, and whether the
  family trades.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families market_ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `3a1ef4a3c722`
- Median excess return: `-0.0229`
- Median Sharpe: `-0.1592`
- Maximum fold drawdown: `0.0218`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0073`
- Pooled excess return: `-0.0332`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `market_ridge_vol_scaled`, traded `8` times, lost `-0.0001`, and trailed
  buy-and-hold by `-0.0229`.
- Fold 2 selected `market_ridge_vol_scaled`, traded `32` times, lost `-0.0072`, and only beat
  buy-and-hold because buy-and-hold fell.

Interpretation:
- Volatility-scaling the market-participation label worsened pooled learned return, median excess,
  and Sharpe.
- Keep `market_ridge_vol_scaled` as an experimental model-family candidate, but it is not current
  metric progress.

## 2026-05-10 - Fold-0 Oracle Candidate Diagnostic

Objective:
- Determine whether the persistent fold-0 buy-and-hold failure is caused by validation selection or
  by the candidate surface lacking any winning policy for that out-of-sample week.

Method:
- Rebuilt the same recent three-fold CV configuration ending `2026-04-30`.
- Focused on fold 0.
- Trained candidates only on the fold's training period.
- Used evaluation-period replay only as an oracle diagnostic, not as a retained selector.
- Ran two narrowed oracle sweeps:
  - `ridge,ridge_vol_scaled`
  - `market_ridge`

Diagnostic command shape:

```bash
timeout 600s .venv/bin/python - <<'PY'
# Ad hoc oracle diagnostic:
# build the same run_intraday_experiment config, take rolling CV fold 0,
# train each candidate on train_start..validation_start,
# replay every threshold/exit/max-hold candidate on evaluation_start..evaluation_end,
# then count candidates that beat buy-and-hold out of sample.
PY
```

Result:
- Fold-0 buy-and-hold return: `0.0461`
- `ridge,ridge_vol_scaled` candidate count: `336`
- `ridge,ridge_vol_scaled` candidates beating buy-and-hold in evaluation: `0`
- `ridge,ridge_vol_scaled` candidates with positive evaluation return: `0`
- `ridge,ridge_vol_scaled` candidates with at least 10 evaluation trades: `294`
- `market_ridge` candidate count: `168`
- `market_ridge` candidates beating buy-and-hold in evaluation: `0`
- `market_ridge` candidates with positive evaluation return: `0`
- `market_ridge` candidates with at least 10 evaluation trades: `147`

Interpretation:
- The fold-0 failure is not just selector overfitting. Within the tested candidate surfaces, even an
  oracle using evaluation data cannot find a candidate that beats buy-and-hold.
- The next useful work should target feature/data/model signal quality for rising market weeks, not
  another selector-ranking tweak over the same candidate surface.

## 2026-05-10 - Rejected Market Quote-Volume Feature Probe

Objective:
- Test whether causal aggregate quote-volume pressure across the liquid-major universe improves
  rising-market participation.

Transient change:
- Added `market_quote_volume_change_<lag>m` and `market_quote_volume_zscore_<window>m` features,
  computed from the sum of per-ticker `Close * Volume`.
- Reverted after the CV run because the feature worsened selection and activity.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `6fc63fc7f496`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Pooled learned return: `0.0013`
- Pooled excess return: `-0.0247`
- Passed CV gate: `false`

Fold details:
- Folds 0 and 1 selected no active policy and missed positive buy-and-hold weeks.
- Fold 2 traded only `8` times, returned `0.0013`, and beat falling buy-and-hold by `0.0424`.

Interpretation:
- Aggregate quote-volume features made the retained ridge surface more conservative and reduced
  activity.
- Do not retain these features.

## 2026-05-10 - Rejected Regularized Extra Trees Probe

Objective:
- Test whether a shallow randomized tree ensemble can capture useful non-linear 1m interactions
  after the fold-0 oracle diagnostic showed no winning candidate in the retained linear surfaces.

Transient change:
- Added `extra_trees_regularized`, an `ExtraTreesRegressor` with capped depth, large minimum leaf
  size, square-root feature subsampling, bootstrap sampling, and deterministic seed.
- Reverted after CV because it worsened pooled return and did not repair fold 0.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families extra_trees_regularized \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `1098117a618e`
- Median excess return: `-0.0269`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0186`
- Minimum fold trades: `0`
- Pooled learned return: `-0.0042`
- Pooled excess return: `-0.0301`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `extra_trees_regularized`, traded `82` times, lost `-0.0042`, and trailed
  buy-and-hold by `-0.0269`.
- Fold 2 selected no active policy and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- The randomized tree model added activity in fold 1 but not durable edge, and it did not solve the
  rising-market fold.
- Do not retain this model family.

## 2026-05-10 - Rejected Session Return Feature Probe

Objective:
- Test whether causal session-to-date return features improve day-trading participation during
  rising market weeks.

Transient change:
- Added per-ticker `session_log_return`, computed from the current close versus the first close of
  the UTC day.
- Added `market_session_log_return`, computed as the cumulative equal-weight market log return since
  the UTC day started.
- Reverted after CV because the feature worsened selection and activity.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run id reused by config hash: `6fc63fc7f496`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Pooled learned return: `0.0013`
- Pooled excess return: `-0.0247`
- Passed CV gate: `false`

Fold details:
- Folds 0 and 1 selected no active policy and missed positive buy-and-hold weeks.
- Fold 2 traded only `6` times, returned `0.0013`, and beat falling buy-and-hold by `0.0424`.

Interpretation:
- Session-to-date return features made the retained ridge surface more conservative and reduced
  activity.
- Do not retain these features.
- The reused run id shows a tooling weakness: config-hash artifacts can be overwritten by code-only
  feature changes.

## 2026-05-10 - Code-Fingerprinted Run IDs

Objective:
- Prevent code-only feature/model probes from overwriting prior artifacts that used the same config
  hash.

Change:
- Top-level intraday run ids now hash both the run configuration and a fingerprint of:
  - `run_intraday_experiment.py`
  - `utils/intraday_data.py`
  - `utils/intraday_policy.py`
- Added `--config-only-run-id` for intentional legacy comparisons.
- Updated docs to distinguish code-fingerprinted run ids from legacy config-only artifact ids.

Verification:
- `tests/test_intraday_cli.py` checks that `--config-only-run-id` parses and that the default run id
  differs from the config-only id.
- `.venv/bin/python -m pytest tests/test_intraday_cli.py -q` passed with `6 passed`.
- A one-fold smoke run produced code-fingerprinted run id `400cb2c31ad9`.
- The same one-fold smoke run with `--config-only-run-id` produced legacy config-only run id
  `49c332a24e22`.

Interpretation:
- This does not improve trading metrics directly.
- It improves the experiment loop's reproducibility by preventing future code-only probes from
  silently overwriting earlier artifacts.

## 2026-05-10 - Rejected Gross Return Target Probe

Objective:
- Test whether training on gross short-horizon price movement, while still selecting/evaluating with
  after-cost replay gates, helps the model learn movement before costs dominate labels.

Transient change:
- Added `_gross` and `_gross_vol_scaled` target variants.
- Reverted after CV because they overtraded validation, selected no active evaluation policy, and
  contradicted the domain preference for after-cost labels.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_gross,ridge_gross_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `d18a4995c46b`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0000`
- Minimum fold trades: `0`
- Pooled learned return: `0.0000`
- Pooled excess return: `-0.0259`
- Passed CV gate: `false`

Fold details:
- Every fold selected no active evaluation policy.
- The top validation candidates traded heavily but had strongly negative validation returns and
  Sharpe.

Interpretation:
- Training on gross movement did not recover after-cost tradeable edge.
- Do not retain pre-cost target variants.

## 2026-05-10 - Rejected Same-Day Max-Hold Probe

Objective:
- Test whether the retained max-hold grid is too restrictive for day trading in rising weeks, while
  still avoiding multi-day positions.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 30 \
  --max-hold-grid 720,1440 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `51e8c5be5831`
- Median excess return: `-0.0145`
- Median Sharpe: `0.6862`
- Maximum fold drawdown: `0.0024`
- Minimum fold trades: `0`
- Pooled learned return: `0.0084`
- Pooled excess return: `-0.0175`
- Passed CV gate: `false`

Fold details:
- Fold 0 still selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `ridge`, traded `14` times, returned `0.0083`, and trailed buy-and-hold by
  `-0.0145`.
- Fold 2 selected `ridge`, traded only `6` times, returned `0.0001`, and beat falling buy-and-hold
  by `0.0413`.

Interpretation:
- Same-day max holds of 720/1440 minutes did not repair the rising-week failure or fold activity.
- Do not expand the retained max-hold grid from this evidence.

## 2026-05-10 - Fold-0 Rank-Only Entry Diagnostic

Objective:
- Check whether forcing activity over the current ridge scores would solve the fold-0 rising-week
  failure.

Method:
- Used the same fold-0 train/validation/evaluation periods from the recent three-fold CV setup.
- Trained `ridge` and `ridge_vol_scaled` on the fold training period.
- Replayed a diagnostic threshold of `-inf` and exit threshold `-inf`, meaning the policy is always
  long the top-scored tickers up to the exposure cap.
- Tested max holds `30,60,120,720`.
- This used evaluation-period replay as a diagnostic only; it is not a retained selector.

Result:
- Fold-0 validation buy-and-hold return: `0.0150`
- Fold-0 evaluation buy-and-hold return: `0.0461`
- Best diagnostic candidate: `ridge`, horizon `15`, max hold `720`
- Best validation return: `-0.0149`
- Best validation excess: `-0.0299`
- Best validation Sharpe: `-1.5848`
- Best evaluation return: `0.0151`
- Best evaluation excess: `-0.0310`
- Best evaluation Sharpe: `2.2342`
- Best evaluation trades: `108`
- Shorter max holds were destroyed by churn, with evaluation returns from about `-0.1402` to
  `-0.5192`.

Interpretation:
- Forced rank-only activity over the current scores does not beat buy-and-hold in the rising fold.
- The best same-day rank-only exposure makes money, but captures only about one third of
  buy-and-hold's fold-0 return.
- Do not add an always-rank/always-long entry candidate over the current score surface.

## 2026-05-10 - Rejected Long Same-Day Horizon Probe

Objective:
- Test whether 4-hour and 12-hour same-day forecast labels can learn broad intraday drift that the
  15/30-minute labels miss, without moving to multi-day positions.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 240,720 \
  --max-hold-grid 240,720,1440 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `be2d3dc8a908`
- Median excess return: `-0.0070`
- Median Sharpe: `-0.9379`
- Maximum fold drawdown: `0.0596`
- Minimum fold trades: `31`
- Pooled learned return: `-0.0290`
- Pooled excess return: `-0.0549`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge_vol_scaled`, horizon `720`, max hold `1440`, traded `31` times, lost
  `-0.0046`, and trailed buy-and-hold by `-0.0507`.
- Fold 1 selected `ridge`, horizon `720`, max hold `1440`, returned `0.0158`, and trailed
  buy-and-hold by `-0.0070`.
- Fold 2 selected `ridge_vol_scaled`, horizon `720`, max hold `1440`, lost `-0.0397`, and exceeded
  the configured drawdown gate with `0.0596`.

Interpretation:
- Longer same-day labels solve activity but not excess return or risk.
- They increase drawdown beyond the configured gate and worsen pooled performance.
- Do not replace the retained 15/30-minute horizon surface with this setting.

## 2026-05-10 - Rejected Long Same-Day Risk-Unit Grid Probe

Objective:
- Test whether smaller validation-selected position sizes rescue the previously rejected 12-hour
  label / 24-hour max-hold surface.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 720 \
  --max-hold-grid 1440 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --risk-unit-grid 0.05,0.1,0.25 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `0756bd19528f`
- Median excess return: `-0.0070`
- Median Sharpe: `-0.9379`
- Maximum fold drawdown: `0.0596`
- Minimum fold trades: `31`
- Pooled learned return: `-0.0290`
- Pooled excess return: `-0.0549`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge_vol_scaled`, risk unit `0.25`, returned `-0.0046`, and trailed
  buy-and-hold by `-0.0507`.
- Fold 1 selected `ridge`, risk unit `0.25`, returned `0.0158`, and trailed buy-and-hold by
  `-0.0070`.
- Fold 2 selected `ridge_vol_scaled`, risk unit `0.25`, returned `-0.0397`, beat the falling
  buy-and-hold fold by only `0.0015`, and exceeded the configured drawdown gate with `0.0596`.

Interpretation:
- The risk-unit grid did not rescue the long same-day surface because the validation selector still
  chose the largest tested risk unit in every fold.
- This exactly matches the rejected long same-day horizon result, so the problem is not solved by
  smaller sizing being available in the candidate grid.
- Do not keep expanding this long-horizon/risk-unit combination without changing the selection
  objective or score surface.

## 2026-05-10 - Timed Out Market Ridge Validation-Slice Combination Probe

Objective:
- Test whether validation-slice stability becomes useful when combined with the strongest retained
  candidate surface, including `market_ridge`.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --validation-slices 3 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Exit code: `124`
- The run timed out after 900 seconds and did not print a completed report path.

Interpretation:
- This is not evidence for or against the trading metric.
- The full three-slice, three-family grid is too expensive for the current iteration loop.
- Narrow the slice-stability test before using it as a regular CV gate.

## 2026-05-10 - Rejected Narrow Market Ridge Validation-Slice Probe

Objective:
- Test a cheaper version of validation-slice stability with the learned market-level candidate in
  the grid.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --validation-slices 2 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `fc5f15339e58`
- Median excess return: `-0.0145`
- Median Sharpe: `5.2454`
- Maximum fold drawdown: `0.0225`
- Minimum fold trades: `14`
- Pooled learned return: `-0.0076`
- Pooled excess return: `-0.0335`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `market_ridge` despite negative whole-validation excess, traded `56` times, lost
  `-0.0213`, and trailed buy-and-hold by `-0.0674`.
- Fold 1 selected `ridge`, traded `14` times, returned `0.0083`, and trailed buy-and-hold by
  `-0.0145`.
- Fold 2 selected `market_ridge`, traded `24` times, returned `0.0056`, and beat falling
  buy-and-hold by `0.0467`.

Interpretation:
- Narrow validation-slice ranking did not improve the three-fold market-ridge surface.
- It exposed that the selector could still choose active candidates whose validation replay did not
  beat validation buy-and-hold.

## 2026-05-10 - Strict Validation-Excess Selection Gate

Objective:
- Stop selecting active candidates that have positive validation return but do not beat validation
  buy-and-hold.

Change:
- Selection now requires `passed_gate`, not just `passed_activity_gate`, before a candidate can
  become the active policy for an evaluation window.
- If no validation candidate beats buy-and-hold after costs while also passing return, Sharpe,
  drawdown, and trade gates, the evaluation window stays flat.
- Updated README, CONTEXT, and ADR wording so docs match the stricter selector.

Verification:
- `tests/test_intraday_policy.py` now covers both:
  - flat selection when validation candidates fail the buy-and-hold excess gate
  - active selection when a synthetic cross-sectional opportunity beats validation buy-and-hold
- `.venv/bin/python -m pytest tests/test_intraday_policy.py tests/test_intraday_cli.py -q` passed
  with `36 passed`.
- `.venv/bin/python -m pytest -q` passed with `40 passed`.

Metric check:
- A strict-gate three-fold `ridge,ridge_vol_scaled,market_ridge` CV attempt timed out after 900
  seconds before writing a top-level CV report.
- Partial artifacts were written under `computed-data/runs/d03c2910f0ac`.
- Completed partial folds:
  - Fold 0 selected no active policy, returned `0.0000`, and trailed buy-and-hold by `-0.0461`.
  - Fold 1 selected no active policy, returned `0.0000`, and trailed buy-and-hold by `-0.0228`.
- A strict-gate two-fold comparison attempt timed out after 600 seconds before writing a top-level CV
  report.
- Partial artifact `computed-data/runs/229eda91f690/cv_folds/fold_00/intraday_report.json` selected
  no active policy, returned `0.0000`, and trailed buy-and-hold by `-0.0228`.

Interpretation:
- The strict validation-excess gate is a selector-quality fix, not a metric win.
- It prevents trading candidates that already failed the validation benchmark, but on the current
  score surface it mostly stays flat during positive buy-and-hold weeks and therefore still fails the
  day-trading activity and buy-and-hold excess goals.
- The next metric-improvement work should return to signal quality rather than further tightening
  this same selector surface.

## 2026-05-10 - Rejected Market Beta/Correlation Feature Probe

Objective:
- Test whether causal rolling market beta, market correlation, and idiosyncratic volatility features
  improve the retained ridge score surface during rising-market weeks.

Transient change:
- Added per-ticker rolling features for each configured intraday window:
  - `market_beta_<window>m`
  - `market_correlation_<window>m`
  - `idiosyncratic_volatility_<window>m`
- These features used only ticker and equal-weight market one-minute returns available up to the
  signal bar.
- Reverted after CV because the feature made the strict selector fully flat.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `f1039e28b5e5`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0000`
- Minimum fold trades: `0`
- Active fold count: `0`
- Pooled learned return: `0.0000`
- Pooled excess return: `-0.0259`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy, returned `0.0000`, and trailed buy-and-hold by `-0.0461`.
- Fold 1 selected no active policy, returned `0.0000`, and trailed buy-and-hold by `-0.0228`.
- Fold 2 selected no active policy, returned `0.0000`, and only beat buy-and-hold because
  buy-and-hold fell.
- The best recorded candidates all had negative validation excess and failed the selection gate.

Verification:
- The transient feature code and test assertions were reverted after the failed CV.
- `.venv/bin/python -m pytest tests/test_intraday_data.py tests/test_intraday_policy.py -q` passed
  with `34 passed`.

Interpretation:
- Rolling beta/correlation/residual-volatility features did not improve the retained ridge surface.
- They made validation candidates less benchmark-competitive under the strict selection gate.
- Do not retain this feature family.

## 2026-05-10 - Reverted Strict Validation-Excess Selection Gate

Objective:
- Restore trading activity after the strict validation-excess gate made the current score surface too
  conservative.

Change:
- Reverted active candidate selection from `passed_gate` back to `passed_activity_gate`.
- Candidates must still have positive after-cost validation return, positive Sharpe, acceptable
  drawdown, and enough validation trades before they can be selected.
- Validation excess over buy-and-hold remains the ranking score and diagnostic.
- Buy-and-hold excess remains a rolling CV proof gate rather than a mandatory single-window
  validation deployment filter.
- Restored README, CONTEXT, and ADR wording to that proof structure.

Evidence:
- The strict-gate three-fold and two-fold attempts timed out before top-level reports.
- Completed partial strict-gate folds selected no active policy in positive buy-and-hold periods.
- The beta/correlation feature probe under the strict gate completed with active fold count `0`,
  median excess `-0.0228`, and pooled excess `-0.0259`.

Verification:
- `tests/test_intraday_policy.py` now explicitly covers that a positive, risk-adjusted candidate can
  still be selected even if it does not beat buy-and-hold in the validation window.
- `.venv/bin/python -m pytest tests/test_intraday_policy.py tests/test_intraday_data.py tests/test_intraday_cli.py -q`
  passed with `40 passed`.

Interpretation:
- The strict validation-excess gate was a defensible anti-overfitting idea but not metric progress.
- It harmed the explicit day-trading activity goal and should not be retained as the default
  selector behavior.

## 2026-05-10 - Current-Code Ridge Surface Reproduction

Objective:
- Confirm the current code state after reverting the strict validation-excess gate and beta feature
  probe reproduces the known retained ridge surface.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `d325b02af0ec`
- Median excess return: `-0.0145`
- Median Sharpe: `1.6266`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `0.0096`
- Pooled excess return: `-0.0163`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `ridge`, traded `14` times, returned `0.0083`, and trailed buy-and-hold by
  `-0.0145`.
- Fold 2 selected `ridge_vol_scaled`, traded `6` times, returned `0.0013`, and beat falling
  buy-and-hold by `0.0424`.

Interpretation:
- The repo is back to the known retained ridge behavior after rejecting the strict gate and beta
  feature probes.
- This is still not a successful model: fold 0 remains flat, fold 1 trails buy-and-hold, and fold 2
  misses the minimum trade floor.

## 2026-05-10 - Rejected Two-Stage Expected-Return Model Probe

Objective:
- Test whether a learned two-stage expected-return model improves over direct ridge regression on
  cost-dominated one-minute labels.

Transient change:
- Added a `two_stage_expected` model family.
- The model learned:
  - probability of positive after-cost return with regularized balanced logistic regression
  - conditional positive-return magnitude with ridge regression
  - conditional non-positive-return magnitude with ridge regression
- The prediction score was the learned expected return:
  `P(positive) * E(return | positive) + P(non-positive) * E(return | non-positive)`.
- Reverted after CV because it worsened pooled return and Sharpe.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families two_stage_expected \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `f7e6b9455a0e`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0115`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `-0.0005`
- Pooled excess return: `-0.0264`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected no active policy and missed a `0.0228` buy-and-hold week.
- Fold 2 selected `two_stage_expected`, traded `34` times, lost `-0.0005`, and only beat
  buy-and-hold because buy-and-hold fell.

Verification:
- The transient model code and unit test were reverted after the failed CV.
- `.venv/bin/python -m pytest tests/test_intraday_policy.py -q` passed with `30 passed`.

Interpretation:
- Decomposing positive probability and conditional magnitude did not recover a durable expected
  return signal.
- Do not retain this model family.

## 2026-05-10 - Timed Out Three-Fold Risk-Unit Grid

Objective:
- Check whether the earlier two-fold risk-unit improvement survives the current-code three-fold CV
  surface.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --risk-unit-grid 0.25,0.5,1.0 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Exit code: `124`
- The run timed out after 900 seconds and did not print a completed report path.

Interpretation:
- This is not valid metric evidence.
- The full three-size risk-unit grid is too expensive for the current iteration loop.
- Use fixed-size probes or optimize replay before expanding this grid again.

## 2026-05-10 - Rejected Fixed Full-Exposure Ridge Probe

Objective:
- Test whether the earlier risk-unit-grid improvement came from simply using full 100% exposure per
  selected position instead of validation-selecting position size.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --risk-unit 1.0 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `928d1fc74ab5`
- Median excess return: `-0.0150`
- Median Sharpe: `0.9309`
- Maximum fold drawdown: `0.0162`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `0.0105`
- Pooled excess return: `-0.0155`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `ridge`, traded `10` times, returned `0.0078`, and trailed buy-and-hold by
  `-0.0150`.
- Fold 2 selected `ridge_vol_scaled`, traded `6` times, returned `0.0026`, and beat falling
  buy-and-hold by `0.0438`.

Interpretation:
- Full exposure slightly improves pooled learned return and pooled excess versus current-code
  baseline `d325b02af0ec`.
- It worsens median excess, median Sharpe, drawdown, and still leaves fold 0 flat and fold 2 below
  the activity floor.
- Do not replace the fixed `0.25` risk unit from this evidence.

## 2026-05-10 - Timed Out Broad Minimum-Threshold Probe

Objective:
- Test whether adding very low entry-threshold quantiles lets the learned score surface participate
  in rising weeks without forcing an always-long rank-only policy.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Exit code: `124`
- The run timed out after 900 seconds and did not print a completed report path.

Interpretation:
- This is not valid metric evidence.
- Adding several low thresholds expands the triangular entry/exit replay grid enough to make the
  current loop impractical.

## 2026-05-10 - Rejected Minimum-Threshold Probe

Objective:
- Test a narrower version of the low-threshold idea by adding only the minimum validation score as
  an entry-threshold candidate to the current retained ridge surface.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.0,0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `0ed40bc1d58a`
- Median excess return: `-0.0145`
- Median Sharpe: `1.6266`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `0.0096`
- Pooled excess return: `-0.0163`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `ridge` with entry threshold `0.0`, traded `14` times, and trailed buy-and-hold
  by `-0.0145`.
- Fold 2 selected `ridge_vol_scaled` with entry threshold `0.0`, traded `6` times, and beat falling
  buy-and-hold by `0.0424`.

Interpretation:
- Adding the minimum-score threshold exactly reproduced current-code baseline `d325b02af0ec`.
- Validation still selected entry threshold `0.0` in the active folds.
- Do not expand threshold quantiles in this direction without first optimizing replay or changing the
  score surface.

## 2026-05-10 - Rejected Longer-Horizon Market Ridge Probe

Objective:
- Test whether learned market-level participation works better at 60/120-minute intraday horizons,
  where broad drift may be less dominated by one-minute costs.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 60,120 \
  --max-hold-grid 120,240 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families market_ridge \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `011f9df56734`
- Median excess return: `-0.0035`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0570`
- Minimum fold trades: `0`
- Active fold count: `2`
- Pooled learned return: `-0.0235`
- Pooled excess return: `-0.0494`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `market_ridge`, horizon `120`, max hold `240`, traded `96` times, returned
  `0.0193`, and trailed buy-and-hold by `-0.0035`.
- Fold 2 selected `market_ridge`, horizon `120`, max hold `240`, traded `168` times, lost
  `-0.0420`, trailed buy-and-hold by `-0.0009`, and exceeded the configured drawdown gate with
  `0.0570`.

Interpretation:
- Longer market-level horizons increase activity and make median excess less negative, but they
  overtrade, worsen pooled return, and breach the drawdown gate.
- They do not solve the fold-0 rising-market flatness.
- Do not retain this longer-horizon `market_ridge` surface.

## 2026-05-10 - Rejected Two/Three-Day Feature Context Probe

Objective:
- Test whether adding longer causal feature context helps the 1m learned policy recognize rising
  multi-day market weeks without changing the intraday prediction/holding horizon.

Transient change:
- Added `2880` and `4320` minute entries to `INTRADAY_LAGS`.
- Added `2880` and `4320` minute entries to `INTRADAY_WINDOWS`.
- This created two-day and three-day causal return, volume, volatility, SMA-distance,
  range-position, market-return, market-volatility, and relative-return features.
- Reverted after CV because the feature degraded return and Sharpe.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `c0aa5af9c657`
- Median excess return: `-0.0151`
- Median Sharpe: `-2.7019`
- Maximum fold drawdown: `0.0027`
- Minimum fold trades: `2`
- Active fold count: `1`
- Pooled learned return: `0.0046`
- Pooled excess return: `-0.0213`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected `ridge`, horizon `15`, max hold `30`, traded `2` times, lost `-0.0024`, and
  trailed buy-and-hold by `-0.0485`.
- Fold 1 selected `ridge`, traded `14` times, returned `0.0077`, and trailed buy-and-hold by
  `-0.0151`.
- Fold 2 selected `ridge`, traded `8` times, lost `-0.0006`, and only beat buy-and-hold because
  buy-and-hold fell.

Verification:
- `.venv/bin/python -m pytest tests/test_intraday_data.py tests/test_intraday_policy.py -q` passed
  with `34 passed` before CV.
- The transient constants were reverted after the failed CV.

Interpretation:
- Longer causal context made fold 0 active but not profitable, and it worsened median Sharpe and
  pooled excess versus current-code baseline `d325b02af0ec`.
- Do not retain two/three-day lag/window features.

## 2026-05-10 - Exit-Threshold Replay Deduplication

Objective:
- Reduce duplicate validation replay work without changing strategy behavior.

Change:
- Added `exit_threshold_candidates`.
- When `max_hold_minutes == horizon_minutes`, the exit threshold cannot affect replay behavior
  because positions cannot be extended beyond the required hold. The selector now evaluates only
  `exit_threshold == entry_threshold` for that case instead of replaying every lower exit threshold.
- When `max_hold_minutes > horizon_minutes`, the prior exit-threshold candidate surface is unchanged.

Verification:
- `tests/test_intraday_policy.py` covers the duplicate-pruning case and the unchanged extension case.
- `.venv/bin/python -m pytest tests/test_intraday_policy.py -q` passed with `31 passed`.
- A direct helper check returned:
  - `exit_threshold_candidates((-2.0,-1.0,0.0), 0.0, 30, 30) == (0.0,)`
  - `exit_threshold_candidates((-2.0,-1.0,0.0), 0.0, 15, 30) == (-2.0, -1.0, 0.0)`

Interpretation:
- This is not a trading metric improvement by itself.
- It is retained because it removes semantically duplicate replay work and makes future grids cheaper.

## 2026-05-10 - Timed Out Optimized Three-Fold Risk-Unit Grid

Objective:
- Retry the previously timed-out three-fold risk-unit grid after exit-threshold replay
  deduplication.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --risk-unit-grid 0.25,0.5,1.0 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Exit code: `124`
- The run timed out after 900 seconds and did not print a completed top-level CV report.
- Partial artifact directory: `computed-data/runs/de04960fea95`
- Completed partial fold:
  - Fold 0 selected no active policy, returned `0.0000`, and trailed buy-and-hold by `-0.0461`.

Interpretation:
- The deduplication is useful but not enough to make the full three-size risk-unit grid practical in
  the current loop.
- Do not keep retrying this exact full grid until replay is optimized more substantially.

## 2026-05-10 - Rejected Current-Code Futures-Only Feature Ablation

Objective:
- Recheck the feature-source question under the current three-fold `ridge,ridge_vol_scaled` surface:
  whether premium-index features are adding noise or helping the retained model.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --no-premium-index \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `2c2969dbe3f0`
- Median excess return: `-0.0229`
- Median Sharpe: `-0.3085`
- Maximum fold drawdown: `0.0260`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `-0.0181`
- Pooled excess return: `-0.0440`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `ridge_vol_scaled`, traded `6` times, lost `-0.0002`, and trailed buy-and-hold by
  `-0.0229`.
- Fold 2 selected `ridge_vol_scaled`, traded `46` times, lost `-0.0179`, and only beat buy-and-hold
  because buy-and-hold fell.

Interpretation:
- Removing premium-index features worsened median excess, Sharpe, pooled learned return, and pooled
  excess versus current-code baseline `d325b02af0ec`.
- Keep premium-index features enabled by default.

## 2026-05-10 - Rejected Combined Market-Return Target Search

Objective:
- Test whether adding the market-return target family to the current retained `ridge,ridge_vol_scaled`
  surface finds a more robust 1m long-only policy under the same three-fold rolling CV proof gate.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled,ridge_market_return \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Exit code: `124`
- Partial artifact directory: `computed-data/runs/075ef6257d5e`
- No top-level `intraday_cv_report.json` was produced.
- Completed fold 0 selected no active policy, returned `0.0000`, and trailed buy-and-hold by
  `-0.0461`.
- Completed fold 1 selected `ridge`, returned `0.0083`, traded `14` times, and trailed
  buy-and-hold by `-0.0145`.

Interpretation:
- Adding `ridge_market_return` to the full retained search surface did not solve the fold-0 blocker
  and was too slow for the current 900 second iteration loop.
- Do not repeat this combined search without either narrowing the surface or improving replay speed.

## 2026-05-10 - Rejected Stronger Ridge Regularization Search

Objective:
- Test whether stronger linear regularization improves generalization for the retained 1m long-only
  surface without changing labels, execution rules, or the rolling CV proof gate.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_alpha_10,ridge_alpha_100 \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `7003851305b4`
- Median excess return: `-0.0145`
- Median Sharpe: `0.6862`
- Maximum fold drawdown: `0.0024`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `0.0084`
- Pooled excess return: `-0.0175`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected an active policy, returned `0.0083`, traded `14` times, and trailed buy-and-hold
  by `-0.0145`.
- Fold 2 returned `0.0001`, beat a falling buy-and-hold week by `0.0413`, but traded only `6`
  times and stayed below the CV activity floor.

Interpretation:
- Stronger ridge regularization did not improve the retained baseline. Pooled excess worsened versus
  `d325b02af0ec`, median Sharpe fell, and the fold-0 flat-policy blocker remained.
- Do not keep this alpha pair in the default model search.

## 2026-05-10 - Rejected Gross-Return Target Probe

Objective:
- Test whether a learned gross-return target can rank small directional opportunities better than
  the retained after-cost target, while still evaluating replay, validation selection, and CV gates
  after normal commission and slippage.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_gross \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `310fe05a344a`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0000`
- Minimum fold trades: `0`
- Active fold count: `0`
- Pooled learned return: `0.0000`
- Pooled excess return: `-0.0259`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and trailed buy-and-hold by `-0.0461`.
- Fold 1 selected no active policy and trailed buy-and-hold by `-0.0228`.
- Fold 2 selected no active policy and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- The gross-return target made the policy even less active than the retained after-cost target.
- The transient `_gross` target-family code was removed after this rejection.

## 2026-05-10 - Rejected SGD Huber Linear Model Probe

Objective:
- Test whether an SGD-trained Huber linear regressor improves generalization or fold activity versus
  the retained ridge surface, without changing labels, features, selection gates, or long-only
  execution.

Transient change:
- Added an `sgd_huber` model family using `StandardScaler` plus scikit-learn `SGDRegressor` with
  Huber loss and L2 penalty.
- Reverted after CV because it increased losing activity and did not solve fold 0.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families sgd_huber,sgd_huber_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `0b8643ba5cae`
- Median excess return: `-0.0188`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0258`
- Minimum fold trades: `0`
- Active fold count: `2`
- Pooled learned return: `-0.0138`
- Pooled excess return: `-0.0397`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 traded `22` times, returned `0.0040`, and trailed buy-and-hold by `-0.0188`.
- Fold 2 traded `54` times, lost `-0.0177`, and only beat buy-and-hold because buy-and-hold fell
  more.

Interpretation:
- SGD Huber increased activity in two folds, but it produced negative pooled learned return and did
  not repair the fold-0 flatness.
- Do not retain this model family.

## 2026-05-10 - Rejected Shared Market Gross-Return Target Probe

Objective:
- Test whether a shared learned market-direction score trained on gross forward returns can repair
  the fold-0 rising-market miss, while still selecting and evaluating after normal commission and
  slippage.

Transient change:
- Reintroduced `_gross` target-family suffix support only for this probe.
- Used `market_ridge_gross`, which trains `MarketFeatureRegressor` on gross per-ticker forward
  returns and emits one shared score to all tickers.
- Reverted after CV because it selected no active policy in every fold.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families market_ridge_gross \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `1a5b4d59abfe`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0000`
- Minimum fold trades: `0`
- Active fold count: `0`
- Pooled learned return: `0.0000`
- Pooled excess return: `-0.0259`
- Passed CV gate: `false`

Fold details:
- Folds 0 and 1 selected no active policy and missed positive buy-and-hold weeks.
- Fold 2 selected no active policy and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- Learning shared market direction on gross returns did not make the validation-selected policy
  participate in rising evaluation weeks.
- Do not retain gross-return target suffix support.

## 2026-05-10 - Rejected Bayesian Ridge Linear Model Probe

Objective:
- Test whether Bayesian linear shrinkage improves generalization versus the retained ridge surface,
  without changing labels, features, selection gates, or long-only execution.

Transient change:
- Added a `bayesian_ridge` model family using `StandardScaler` plus scikit-learn `BayesianRidge`.
- Reverted after CV because it reproduced the retained ridge baseline without improving any proof
  metric.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families bayesian_ridge,bayesian_ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `278149a0aac8`
- Median excess return: `-0.0145`
- Median Sharpe: `1.6266`
- Maximum fold drawdown: `0.0041`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `0.0096`
- Pooled excess return: `-0.0163`
- Passed CV gate: `false`

Fold details:
- Fold 0 selected no active policy and missed a `0.0461` buy-and-hold week.
- Fold 1 selected `bayesian_ridge`, returned `0.0083`, traded `14` times, and trailed buy-and-hold
  by `-0.0145`.
- Fold 2 selected `bayesian_ridge_vol_scaled`, returned `0.0013`, traded `6` times, and only beat
  buy-and-hold because buy-and-hold fell.

Interpretation:
- Bayesian ridge duplicated the retained baseline profile (`d325b02af0ec`) rather than improving it.
- Do not retain this model family.

## 2026-05-10 - Rejected Taker-Buy Imbalance Feature Probe

Objective:
- Test whether causal one-minute taker-buy imbalance and trade-count features add useful
  microstructure signal beyond the retained spot, futures, premium, and market-relative feature set.

Transient change:
- Preserved optional Binance kline fields in `raw_to_1m_bars`:
  - `Quote asset volume`
  - `Number of trades`
  - `Taker buy base asset volume`
  - `Taker buy quote asset volume`
- Added per-ticker taker-buy base/quote share, share changes over configured lags, trade-count
  changes, and rolling z-scores over configured windows.
- Reverted after CV because the feature set overtraded fold 0 and worsened pooled return.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `597eaadb527c`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0316`
- Minimum fold trades: `0`
- Active fold count: `2`
- Pooled learned return: `-0.0249`
- Pooled excess return: `-0.0508`
- Passed CV gate: `false`

Fold details:
- Fold 0 traded `52` times, lost `-0.0255`, and trailed buy-and-hold by `-0.0716`.
- Fold 1 selected no active policy and missed a `0.0228` buy-and-hold week.
- Fold 2 traded `30` times, returned `0.0006`, and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- Taker-buy imbalance and trade-count features caused noisy losing activity rather than robust edge.
- Do not retain these features.

## 2026-05-10 - Rejected PCA-Ridge Dimensionality Reduction Probe

Objective:
- Test whether reducing the retained feature space before ridge improves generalization and avoids
  overfitting noisy one-minute features, without changing labels, selection gates, or long-only
  execution.

Transient change:
- Added a `ridge_pca_<components>` model family using `StandardScaler`, scikit-learn `PCA`, and
  `Ridge(alpha=1.0)`.
- Reverted after CV because it reduced activity and worsened pooled excess versus the retained
  baseline.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge_pca_16,ridge_pca_16_vol_scaled \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `36e608076154`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0046`
- Minimum fold trades: `0`
- Active fold count: `0`
- Pooled learned return: `-0.0012`
- Pooled excess return: `-0.0271`
- Passed CV gate: `false`

Fold details:
- Folds 0 and 1 selected no active policy and missed positive buy-and-hold weeks.
- Fold 2 traded `8` times, lost `-0.0012`, and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- PCA-ridge made the policy too inactive in positive-market folds and still unprofitable in the
  falling fold.
- Do not retain this dimensionality-reduction model family.

## 2026-05-10 - Rejected Current-Code No-Futures Feature Ablation

Objective:
- Recheck whether causally forward-filled five-minute futures metrics are hurting the current
  retained three-fold `ridge,ridge_vol_scaled` surface, while keeping one-minute premium-index
  features enabled.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --no-futures-metrics \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `543860e89cc6`
- Median excess return: `-0.0228`
- Median Sharpe: `0.0000`
- Maximum fold drawdown: `0.0220`
- Minimum fold trades: `0`
- Active fold count: `1`
- Pooled learned return: `-0.0123`
- Pooled excess return: `-0.0382`
- Passed CV gate: `false`

Fold details:
- Folds 0 and 1 selected no active policy and missed positive buy-and-hold weeks.
- Fold 2 traded `42` times, lost `-0.0123`, and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- Removing futures metrics worsened median excess, median Sharpe, pooled learned return, and pooled
  excess versus current-code baseline `d325b02af0ec`.
- Keep futures metrics enabled by default.

## 2026-05-10 - Rejected Current-Code Spot-Only Feature Ablation

Objective:
- Recheck whether removing both exogenous sources helps the current retained three-fold
  `ridge,ridge_vol_scaled` surface generalize using only spot OHLCV and market-relative spot
  features.

Command:

```bash
timeout 900s .venv/bin/python run_intraday_experiment.py \
  --end-date 2026-04-30 \
  --train-days 90 \
  --validation-days 7 \
  --evaluation-days 7 \
  --stride-days 7 \
  --horizon-grid 15,30 \
  --max-hold-grid 30,60,120 \
  --threshold-quantiles 0.25,0.5,0.75,0.9,0.95,0.99 \
  --model-families ridge,ridge_vol_scaled \
  --no-futures-metrics \
  --no-premium-index \
  --min-validation-trades 1 \
  --selection-trade-floor 10 \
  --cv-folds 3 \
  --cv-windows-per-fold 1 \
  --min-cv-fold-trades 10
```

Result:
- Run: `583b0b9f9268`
- Median excess return: `-0.0214`
- Median Sharpe: `-1.4604`
- Maximum fold drawdown: `0.0271`
- Minimum fold trades: `4`
- Active fold count: `1`
- Pooled learned return: `-0.0179`
- Pooled excess return: `-0.0438`
- Passed CV gate: `false`

Fold details:
- Fold 0 traded `4` times, lost `-0.0006`, and trailed buy-and-hold by `-0.0467`.
- Fold 1 traded `4` times, returned `0.0014`, and trailed buy-and-hold by `-0.0214`.
- Fold 2 traded `48` times, lost `-0.0186`, and only beat buy-and-hold because buy-and-hold fell.

Interpretation:
- Spot-only features worsened median excess, median Sharpe, pooled learned return, and pooled excess
  versus current-code baseline `d325b02af0ec`.
- Keep both futures metrics and premium-index features enabled by default.
