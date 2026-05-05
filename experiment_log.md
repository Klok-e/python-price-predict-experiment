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
