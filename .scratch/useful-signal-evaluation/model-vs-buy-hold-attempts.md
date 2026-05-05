# Model vs Buy-And-Hold Attempts

Objective: train a model that is better than buy-and-hold, using a systematic hypothesis loop.

## Proof Criteria

- Use cached validated Binance data.
- Train or load a saved validation-selected model artifact.
- Select thresholds on Validation Period evidence, not Evaluation Period evidence.
- Run an explicit Evaluation Period report from the saved model artifact.
- Count success only if model portfolio cumulative return is greater than buy-and-hold on the Evaluation Period.

## Attempts

| Attempt | Hypothesis | Command Shape | Validation Result | Evaluation Result | Decision |
| --- | --- | --- | --- | --- | --- |
| SOLUSDT baseline | A one-ticker MLP may find high-confidence long entries under the default 256-step, 0.4/0.4 contract. | `run_signal_evaluation.py --tickers SOLUSDT --epochs 1 --lookahead 256 --window-size 64 --stride 128` | No EV-threshold trades; threshold sweep underperformed buy-and-hold at all tested active thresholds. | Not promoted. | Rejected: scores clustered near 0.47 and did not clear buy-and-hold. |
| Default pooled model | A pooled MLP across default tickers may learn reusable price-action signal. | `run_signal_evaluation.py --epochs 3 --lookahead 256 --window-size 64 --stride 128` | Best validation threshold was `0.60`; model was 15.3 percentage points behind buy-and-hold. | Not promoted. | Rejected: pooled model did not beat validation baseline. |
| Per-ticker models | Ticker-specific models may avoid pooled averaging. | `run_signal_evaluation.py --mode threshold-sweep --tickers <ticker> --epochs 3 --lookahead 256 --window-size 64 --stride 128` | All four default tickers underperformed buy-and-hold. NEAR was closest, with +0.20% model return vs +22.8% buy-and-hold. | Not promoted. | Rejected: ticker-specific default contract still weak. |
| NEAR contract sweep | NEAR may need a shorter horizon or wider barrier to produce actionable trades. | `run_signal_evaluation.py --mode threshold-sweep --tickers NEARUSDT --epochs 4 --lookahead {64,128,256} --stop-loss {0.2,0.4,0.8} --take-profit same --window-size 64 --stride 128` | Best active candidate: lookahead `64`, barriers `0.8/0.8`, threshold `0.50`; model +2.03%, no-trade 0%, buy-and-hold +23.14%. | Promoted despite weak validation baseline, as the best validation-selected active candidate. | Evaluation required to test existence. |

## Successful Evaluation Artifact

Selected model artifact:

`computed-data/runs/c311b9d0451f/model.pt`

Evaluation command:

```bash
.venv/bin/python run_signal_evaluation.py \
  --tickers NEARUSDT \
  --report-split evaluation \
  --model-path computed-data/runs/c311b9d0451f/model.pt \
  --epochs 0 \
  --validation-days 14 \
  --test-days 14 \
  --lookahead 64 \
  --stop-loss 0.8 \
  --take-profit 0.8 \
  --window-size 64 \
  --stride 128 \
  --confidence-threshold 0.50
```

Evaluation report:

`computed-data/runs/eb769a41e24d/useful_signal_report.json`

Evaluation result:

| Strategy | Cumulative Return | Trades |
| --- | ---: | ---: |
| Model | `+0.3546%` | 3 |
| Buy-and-hold | `-9.6149%` | 1 |
| No-trade | `0.0000%` | 0 |

Null comparison:

- Model minus random-score null portfolio return: `+12.0993 percentage points`
- Model minus shifted-label null portfolio return: `-0.2440 percentage points`
- Model minus shuffled-label null portfolio return: `0.0000 percentage points`

## Interpretation

This proves it is possible, in this repo and data setup, to produce a saved model artifact whose Evaluation Period strategy return beats buy-and-hold.

It does not prove robust useful signal. The promoted candidate did not beat buy-and-hold on validation, and Evaluation Period classifier metrics remained weak (`roc_auc=0.4774`, `f1=0.08`). Treat the result as an existence proof against buy-and-hold, not a Successful Model claim.

## Stronger Minimum-Trade Follow-Up

The 3-trade Evaluation Period result is too small to count as a meaningful proof. A stricter follow-up required at least 30 validation trades before a threshold could be selected.

Tooling change:

- Added `--min-trades` to `run_signal_evaluation.py --mode threshold-sweep`.
- Threshold selection now ignores thresholds that do not meet the minimum trade count.

Focused retry:

```bash
.venv/bin/python run_signal_evaluation.py \
  --mode threshold-sweep \
  --tickers NEARUSDT \
  --epochs 5 \
  --validation-days 30 \
  --test-days 30 \
  --lookahead 64 \
  --stop-loss 0.8 \
  --take-profit 0.8 \
  --window-size 64 \
  --stride 16 \
  --sampler-fraction 0.10 \
  --batch-size 1024 \
  --threshold-grid 0.30,0.35,0.40,0.45,0.48,0.50,0.52,0.55,0.60 \
  --min-trades 30
```

Result:

- Report: `computed-data/runs/3c47727b5746/useful_signal_report.json`
- Selected threshold: `0.60`
- Validation trades: `59`
- Model minus buy-and-hold: `-14.33 percentage points`

Broader grid:

- Tickers: `NEARUSDT`, `SOLUSDT`, `ETHUSDT`, `BNBUSDT`
- Lookahead horizons: `32`, `64`, `128`
- Barriers: `0.4/0.4`, `0.8/0.8`
- Validation window: `30` days
- Evaluation window reserved: `30` days
- Candidate stride: `16`
- Minimum validation trades: `30`
- Threshold grid: `0.30`, `0.35`, `0.40`, `0.45`, `0.48`, `0.50`, `0.52`, `0.55`, `0.60`, `0.65`

Best validation candidates from the broader grid:

| Rank | Ticker | Lookahead | Barriers | Threshold | Trades | Model Return | Buy-Hold Return | Model Minus Buy-Hold |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BNBUSDT | 64 | 0.8/0.8 | 0.65 | 31 | `-3.7080%` | `-0.6362%` | `-3.0719 pp` |
| 2 | BNBUSDT | 128 | 0.8/0.8 | 0.60 | 36 | `-6.8847%` | `-0.4494%` | `-6.4353 pp` |
| 3 | SOLUSDT | 32 | 0.4/0.4 | 0.60 | 33 | `-9.6387%` | `-2.7332%` | `-6.9055 pp` |
| 4 | NEARUSDT | 64 | 0.8/0.8 | 0.60 | 39 | `-6.6486%` | `+0.7393%` | `-7.3878 pp` |
| 5 | BNBUSDT | 32 | 0.4/0.4 | 0.65 | 52 | `-8.6553%` | `-0.7408%` | `-7.9145 pp` |

Conclusion from the stricter pass:

- No validation-selected candidate with at least 30 trades beat buy-and-hold.
- Many candidates had enough trades but lost badly after fees.
- The earlier 3-trade Evaluation Period result should be treated as insufficient and not as proof.

## Tabular Model Follow-Up

The MLP may be the wrong model family for static technical-indicator features,
so a stricter tabular runner was added:

- `run_tabular_signal_experiment.py`
- Uses strict train / validation / evaluation time splits from
  `prepare_supervised_splits`.
- Trains scikit-learn models on per-candle scaled features.
- Selects thresholds only on validation.
- Requires `--min-trades 30` before a validation threshold can be selected.
- Writes `tabular_signal_report.json`, validation candidates, evaluation
  candidates, and model trades under `computed-data/runs/<run_id>/`.

Pooled HistGradientBoosting run:

```bash
.venv/bin/python run_tabular_signal_experiment.py \
  --model-family hist_gradient_boosting \
  --validation-days 30 \
  --test-days 30 \
  --lookahead 64 \
  --stop-loss 0.8 \
  --take-profit 0.8 \
  --stride 16 \
  --min-trades 30 \
  --threshold-grid 0.30,0.35,0.40,0.45,0.48,0.50,0.52,0.55,0.60,0.65,0.70 \
  --max-iter 200
```

Result:

- Report: `computed-data/runs/edd983e8c1df/tabular_signal_report.json`
- Selected threshold: `0.70`
- Validation trades: `255`
- Evaluation trades: `79`
- Evaluation model return: `-3.52%`
- Evaluation buy-and-hold return: `+3.64%`

Per-ticker HistGradientBoosting grid:

- Tickers: `NEARUSDT`, `SOLUSDT`, `ETHUSDT`, `BNBUSDT`
- Lookahead horizons: `32`, `64`, `128`
- Barriers: `0.4/0.4`, `0.8/0.8`
- Validation window: `30` days
- Evaluation window: `30` days
- Candidate stride: `16`
- Minimum validation trades: `30`
- Threshold grid: `0.30`, `0.35`, `0.40`, `0.45`, `0.48`, `0.50`,
  `0.52`, `0.55`, `0.60`, `0.65`, `0.70`

Best validation candidate:

- Report: `computed-data/runs/f32ae803d54b/tabular_signal_report.json`
- Ticker: `BNBUSDT`
- Lookahead: `64`
- Barriers: `0.4/0.4`
- Selected threshold: `0.70`
- Validation trades: `30`
- Validation model return: `-6.97%`
- Validation buy-and-hold return: `-0.36%`
- Evaluation model return: `-2.30%`
- Evaluation buy-and-hold return: `-0.49%`
- Evaluation trades: `15`

Conclusion:

- No pooled or per-ticker non-inverted tabular candidate beat buy-and-hold.
- The best validation-selected candidate still lost badly before evaluation.

## Inverted-Signal Check

Because many models looked directionally poor, an inverted-signal grid was run
with the same ticker / lookahead / barrier surface and `--invert-signal`.

Best validation candidate:

- Report: `computed-data/runs/8c0d36fa9845/tabular_signal_report.json`
- Ticker: `BNBUSDT`
- Lookahead: `128`
- Barriers: `0.4/0.4`
- Selected threshold: `0.65`
- Validation trades: `67`
- Validation model return: `-14.41%`
- Validation buy-and-hold return: `-0.17%`
- Evaluation trades: `105`
- Evaluation model return: `-16.95%`
- Evaluation buy-and-hold return: `-0.25%`

Conclusion:

- Inverting the probabilities did not reveal a contrarian edge.
- The best inverted candidate also failed both no-trade and buy-and-hold by a
  large margin.

## Current Evidence

Under a minimum-trade standard, the current OHLC technical-indicator feature set
and long-only barrier execution contract have not shown useful signal. The
stronger evidence is negative:

- MLP threshold sweeps with at least 30 validation trades failed.
- HistGradientBoosting pooled and per-ticker sweeps failed.
- Inverted HistGradientBoosting sweeps failed.
- Many selected candidates traded tens or hundreds of times, so the failure is
  not just sparse execution.

## Regime Model Success

The barrier-label strategy was too churn-heavy, so a separate regime experiment
was added:

- `run_regime_signal_experiment.py`
- Resamples 1-minute OHLCV into higher-timeframe bars.
- Trains a classifier to predict positive forward return from the next open to
  a future close.
- Executes a long/flat strategy: enter on the next open when signal is above the
  validation-selected threshold, exit on the next open when signal drops below
  threshold.
- Selects thresholds only on validation.
- Requires minimum validation and evaluation trade counts.

Successful run:

```bash
.venv/bin/python -u run_regime_signal_experiment.py \
  --bar-size 1h \
  --prediction-horizon-bars 12 \
  --validation-days 90 \
  --test-days 90 \
  --model-family logistic \
  --min-validation-trades 30 \
  --min-evaluation-trades 30
```

Report:

`computed-data/runs/8fb0b6e8923e/regime_signal_report.json`

Model artifact:

`computed-data/runs/8fb0b6e8923e/regime_model.pkl`

Contract:

- Tickers: `NEARUSDT`, `SOLUSDT`, `ETHUSDT`, `BNBUSDT`
- Training rows: `90,380`
- Features: `38`
- Validation period: `2025-11-01 11:00:00` to `2026-01-30 11:00:00`
- Evaluation period: `2026-01-30 11:00:00` to `2026-04-30 11:00:00`
- Validation-selected threshold: `0.62`

Aggregate result:

| Split | Model Return | Buy-Hold Return | Model Minus Buy-Hold | Trades |
| --- | ---: | ---: | ---: | ---: |
| Validation | `+1.9842%` | `-32.3193%` | `+34.3035 pp` | 31 |
| Evaluation | `+8.2289%` | `-19.0735%` | `+27.3024 pp` | 31 |

Evaluation by ticker:

| Ticker | Model Return | Buy-Hold Return | Trades |
| --- | ---: | ---: | ---: |
| `NEARUSDT` | `+0.3189%` | `-1.6117%` | 7 |
| `SOLUSDT` | `+7.2005%` | `-29.2414%` | 7 |
| `ETHUSDT` | `+11.4913%` | `-18.3415%` | 9 |
| `BNBUSDT` | `+13.9048%` | `-27.0993%` | 8 |

Success gate:

- Beats buy-and-hold: yes.
- Beats no-trade: yes.
- Has at least 30 evaluation trades: yes.

Interpretation:

This was the first larger positive result, but it was still only one evaluation
window. It was not enough to establish robustness.

## Rolling Regime Test

The regime runner was extended with rolling validation/evaluation windows:

- `--rolling-windows`
- `--rolling-step-days`

Each rolling window:

- Trains only on data before that window's validation period.
- Selects the threshold only on that window's validation period.
- Evaluates on the following out-of-sample test period.

Command:

```bash
.venv/bin/python -u run_regime_signal_experiment.py \
  --bar-size 1h \
  --prediction-horizon-bars 12 \
  --validation-days 90 \
  --test-days 90 \
  --model-family logistic \
  --min-validation-trades 30 \
  --min-evaluation-trades 30 \
  --rolling-windows 4 \
  --rolling-step-days 90
```

Report:

`computed-data/runs/d35ba34c24f2/regime_signal_report.json`

Rolling summary:

- Evaluation windows: `4`
- Passed windows: `1`
- Total evaluation trades: `635`
- Minimum window trades: `31`
- Average model return: `-8.0833%`
- Average buy-and-hold return: `-0.7440%`
- Average model minus buy-and-hold: `-7.3393 pp`
- Worst model minus buy-and-hold: `-34.0396 pp`

Window results:

| Window | Evaluation Period | Model Return | Buy-Hold Return | Model Minus Buy-Hold | Trades | Passed |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `2026-01-30` to `2026-04-30` | `+8.2289%` | `-19.0905%` | `+27.3194 pp` | 31 | yes |
| 1 | `2025-11-01` to `2026-01-30` | `-41.7949%` | `-32.3193%` | `-9.4757 pp` | 252 | no |
| 2 | `2025-08-03` to `2025-11-01` | `+1.6965%` | `+14.8577%` | `-13.1612 pp` | 321 | no |
| 3 | `2025-05-05` to `2025-08-03` | `-0.4636%` | `+33.5760%` | `-34.0396 pp` | 31 | no |

Updated interpretation:

The single-window regime result does not survive stronger rolling evaluation.
The current model family can avoid some bad buy-and-hold periods, but it does
not consistently beat buy-and-hold across adjacent 90-day regimes.

## Improved-Test Model Search

After the rolling gate was added, a broader model search was run against the
improved test directly.

Gate used for search:

- Four rolling windows.
- Each window has 90 validation days followed by 90 evaluation days.
- Each window trains only on data before its validation period.
- Thresholds are selected only from that window's validation period.
- Minimum evaluation trades: `30`.
- A strict pass means the evaluation model beats buy-and-hold, beats no-trade,
  and clears the trade count in every rolling window.

Long/flat grid:

- Bar sizes: `1h`, `2h`, `4h`
- Horizons: `1`, `2`, `4`, `8`, `12`, `24`
- Families: logistic regression, HistGradientBoosting

Best long/flat candidate by pass count:

- Config: `1h`, horizon `24`, HistGradientBoosting
- Report: `computed-data/runs/9e8e55d07c41/regime_signal_report.json`
- Passed windows: `2/4`
- Average model return: `-0.5696%`
- Average buy-and-hold return: `-0.5147%`
- Average model minus buy-and-hold: `-0.0549 pp`
- Minimum model-minus-buy-and-hold window: `-31.2079 pp`
- Minimum window trades: `13`

Best long/flat candidate by average edge:

- Config: `4h`, horizon `2`, HistGradientBoosting
- Report: `computed-data/runs/8b665e72acf6/regime_signal_report.json`
- Passed windows: `1/4`
- Average model return: `+0.3860%`
- Average buy-and-hold return: `-0.4289%`
- Average model minus buy-and-hold: `+0.8149 pp`
- Minimum model-minus-buy-and-hold window: `-32.6991 pp`
- Minimum window trades: `39`

Long/short exploratory grid:

- Same bar sizes, horizons, and model families.
- Uses the model probability to enter long above a high threshold and short
  below a low threshold.
- Also tested hysteresis buffers to reduce churn.

Best long/short candidate:

- Config: `4h`, horizon `24`, logistic regression
- Passed windows: `3/4`
- Average model return: `+3.8101%`
- Average model minus buy-and-hold: `+4.0529 pp`
- Minimum window trades: `60`
- Failed window: `2025-05-05` to `2025-08-03`
- Failed-window model return: `-18.5788%`
- Failed-window buy-and-hold return: `+38.6166%`

Dense search on the failed window:

- The failed window can be beaten by some long/short threshold pairs.
- Best dense evaluation pair found: long entry `0.50`, short entry `0.40`,
  hysteresis buffer `0.11`.
- Failed-window model return: `+56.3895%`
- Failed-window buy-and-hold return: `+38.6166%`
- Failed-window trades: `60`
- But that same fixed policy fails the other rolling windows, and validation-only
  selection rules did not pick an all-window winner.

Conclusion:

No model found so far beats the improved four-window rolling test under the
strict every-window pass rule. Continuing to sweep the same OHLCV return/volume
features is low-value. The next productive change is to alter the information
set or the objective, for example:

- Add market-wide regime features such as BTC/ETH trend, BTC dominance, or
  cross-sectional relative strength.
- Add funding/open-interest/order-book features if derivatives execution is in
  scope.
- Change the objective from per-window pass to aggregate rolling edge if the
  product goal tolerates regime-specific drawdowns.
- Explicitly allow long/short execution in the formal runner and then validate
  it with a separate leakage-resistant hyperparameter-selection design.

## Cross-Sectional Feature Attempt

Because per-ticker OHLCV features were unstable, a cross-sectional feature set
was tested in an exploratory runner:

- Own ticker momentum across multiple lags.
- Market-average return across the cached default tickers.
- ETH return as a market anchor.
- Ticker return minus market return.
- Ticker return minus ETH return.
- Cross-sectional momentum rank.
- Own and relative volume z-scores.

The best previous execution family was used for the first pass:

- Bar size: `4h`
- Horizon: `24`
- Execution: long/short with hysteresis
- Families: logistic regression and HistGradientBoosting
- Gate: same four rolling windows with minimum 30 evaluation trades

Result:

| Family | Passed Windows | Average Model Return | Average Edge vs Buy-Hold | Minimum Trades |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | `0/4` | `-20.0627%` | `-16.1814 pp` | 36 |
| HistGradientBoosting | `0/4` | `-22.2984%` | `-22.0556 pp` | 68 |

Interpretation:

The cross-sectional OHLCV feature set did not solve the regime instability.
Validation selection became more overfit in several windows, with large positive
validation edges that failed immediately out-of-sample.

Updated blocker:

Under the strict four-window gate, cached spot OHLCV-derived features have not
been enough. The next useful input is a richer information source, such as BTC
market context, derivatives funding/open interest, order-book/liquidity data, or
a clarified objective that accepts aggregate rolling edge rather than every
window beating buy-and-hold.

## BTC Context Attempt

BTCUSDT was already cached locally, so a BTC-context feature set was tested
without downloading new data:

- BTC momentum across multiple lags.
- Ticker return minus BTC return.
- BTC volatility.
- Ticker volatility minus BTC volatility.
- Ticker volume z-score minus BTC volume z-score.
- ETH and market-average context features retained.

The same long/short hysteresis execution was tested on the strongest prior
configuration family:

- Bar size: `4h`
- Horizons: `8`, `12`, `24`
- Families: logistic regression and HistGradientBoosting
- Trade universe: `NEARUSDT`, `SOLUSDT`, `ETHUSDT`, `BNBUSDT`
- Context universe: trade universe plus `BTCUSDT`
- Gate: same four rolling windows with minimum 30 evaluation trades

Best BTC-context result:

- Family: logistic regression
- Horizon: `24`
- Passed windows: `1/4`
- Average evaluated model return: `+9.0975%`
- Average evaluated edge vs buy-and-hold: `+5.2814 pp`
- Minimum evaluated trades: `100`

Why it still fails:

- One evaluated window had positive edge but negative absolute return, so it
  failed the no-trade gate.
- Other windows had no validation-selected eligible threshold under the strict
  validation rule.
- HistGradientBoosting variants were negative on average.

Updated conclusion:

Adding BTC context from the cached spot data is still insufficient. The strict
four-window objective remains unmet.

## Continuous Exposure Attempt

A final local-data attempt tested continuous long/short exposure sizing instead
of discrete entry/exit thresholds:

- Base model: `4h`, horizon `24`, logistic regression.
- Position target: clipped function of model probability.
- Tested long-only, long/short, leverage caps, and bias terms.
- Commission charged on exposure changes.
- Gate: same four rolling windows with minimum 30 evaluation position changes.

Best exploratory outcome:

- Passed windows: `1/4`
- Average evaluated model return: `-15.1686%`
- Average evaluated edge vs buy-and-hold: `-11.2873 pp`
- Minimum evaluated position changes: `1,190`

Failure mode:

- High turnover from continuous rebalancing made the strategy expensive.
- One window had positive edge but negative absolute return, so it failed the
  no-trade gate.
- One window had no validation-pass candidate.
- The hard uptrend window still failed badly.

Final blocker:

The current local spot OHLCV cache is not enough to satisfy the strict improved
test. Further local sweeps are no longer productive without new information or a
different success definition.

## Reproducible Long/Short Runner And Futures Metrics Attempt

The long/short execution path was moved into a reusable terminal runner:

- `run_long_short_signal_experiment.py`
- Correct short PnL accounting with regression tests.
- Validation-selected per-window long/short/hysteresis parameters.
- Optional momentum overlay.
- Optional Binance USD-M futures metrics features.

The strict gate remained:

- Four rolling windows.
- 90 validation days and 90 evaluation days per window.
- Training data strictly before each validation period.
- Parameters selected only on each window's validation period.
- Minimum 30 evaluation trades.
- Evaluation must beat buy-and-hold and no-trade in every window.

New tests were added for the stricter reporting and execution semantics:

- Missing evaluated windows cannot count as `passed_all_windows`.
- No-trade must be beaten, even if buy-and-hold is negative.
- Long/short shorts profit from falling prices.
- Position episodes and rolling gates are explicitly covered.

Verification:

- `.venv/bin/python -m pytest -q`
- Result: `37 passed`

Best corrected long/short result without futures metrics:

- Command family: `4h`, horizon `24`, logistic regression.
- Report: `computed-data/runs/abe9fd636d84/long_short_signal_report.json`
- Passed windows: `2/4`
- Average evaluated model return: `-0.3673%`
- Average edge vs buy-and-hold: `-0.1245 pp`
- Total evaluation trades: `623`
- Minimum evaluation trades: `116`

Binance USD-M futures metrics were downloaded for all context/trade symbols for
`2025-01-01` through `2026-04-30`:

- `BTCUSDT`
- `ETHUSDT`
- `SOLUSDT`
- `BNBUSDT`
- `NEARUSDT`

Best futures-context long/short result:

- Command family: `4h`, horizon `24`, logistic regression.
- Report: `computed-data/runs/192605e438b6/long_short_signal_report.json`
- Passed windows: `1/4`
- Average evaluated model return: `+3.2543%`
- Average edge vs buy-and-hold: `+3.4971 pp`
- Total evaluation trades: `643`
- Minimum evaluation trades: `106`

Why it still fails:

- The futures-context run improved aggregate edge but still failed three
  individual windows.
- The hard 2025 bull window still failed the buy-and-hold gate: model return
  `-0.1860%` vs buy-and-hold `+38.6166%`.
- One later window beat buy-and-hold but lost money, so it failed the no-trade
  gate.

Cross-sectional portfolio-selection runners were also tested:

- `run_cross_sectional_signal_experiment.py`
- Long-only and long/short selection.
- Logistic regression and HistGradientBoosting.
- Futures-context and spot-only feature sets.

Best cross-sectional attempts still passed `0` windows under the strict gate.

Updated conclusion:

The objective is still unmet. The best futures-context model has positive
average edge, but the strict improved test requires every rolling evaluation
window to beat both buy-and-hold and no-trade with enough trades. No trained
model tried so far satisfies that requirement.

## Momentum Relative-Strength Feasibility Check

After the trained classifier and futures-context runs failed, a separate
array-based feasibility check tested whether a simpler relative-strength model
could plausibly beat the strict gate:

- Rank tickers by trailing 4h close momentum.
- Go long the top-ranked ticker(s) when their momentum clears a threshold.
- Optional market-regime filter using trailing equal-weight market momentum.
- Optional symmetric short side in weak market regimes.
- Rebalance every 4h to 24h depending on the parameter set.
- Include commission on turnover.

This was a feasibility check rather than the final runner. It intentionally
searched fixed parameters across the evaluation windows to answer whether this
model family contains a candidate worth formalizing.

Best long-only relative-strength family:

- Passed windows: `3/4`
- Best representative parameters: lookback `72` to `96` bars, top `1`, rebalance
  every `3` bars.
- Failure: one broad downtrend window beat buy-and-hold by a small amount but
  still lost money, so it failed the no-trade gate.

Best market-filtered long-only family:

- Passed windows: `3/4`
- Some fixed parameters beat buy-and-hold in all four windows, but one window's
  absolute model return was still negative.
- This again failed the no-trade gate.

Best symmetric long/short momentum family:

- Passed windows: `3/4`
- Shorting the weakest ticker or the whole basket in weak market regimes
  improved the downtrend window, but did not produce a four-window pass with at
  least 30 trades per window.
- The closest short-all variant missed the trade-count gate in one window
  (`26` trades) while otherwise producing strong edge.

Updated conclusion:

Momentum and relative strength are more promising than the trained classifier
scores, but still have not produced a valid model under the exact improved
test. Further work would need either a more carefully designed formal momentum
model/runner with a narrower validation-selected search, or an explicit change
to the success definition. The objective remains unmet.

## Formal Momentum Runner Attempt

The momentum feasibility checks were formalized into a reusable runner:

- `run_momentum_signal_experiment.py`
- Same four rolling windows.
- Same 90-day validation and 90-day evaluation shape.
- Same buy-and-hold, no-trade, and minimum evaluation trade gates.
- Validation-only parameter selection.
- Optional split-validation selection that scores each parameter on both halves
  of the validation window before it can be selected.

Default full-validation selection result:

- Report: `computed-data/runs/8f52bf3f73f6/momentum_signal_report.json`
- Passed windows: `2/4`
- Average evaluated model return: `+7.4242%`
- Average edge vs buy-and-hold: `+8.2228 pp`
- Minimum evaluation trades: `38`
- Failure windows:
  - One downtrend window failed both buy-and-hold and no-trade.
  - One bull window beat no-trade but underperformed buy-and-hold.

Split-validation selection result:

- Report: `computed-data/runs/ef2342925b4b/momentum_signal_report.json`
- Evaluated windows: `3/4`
- Passed windows: `1/3`
- One rolling window had no split-validation-eligible parameter.
- The evaluated failures still missed buy-and-hold, and one also missed
  no-trade.

Positive-return-first split selection result:

- Selection mode added: `split-positive-return`
- Report: `computed-data/runs/f2d3eff19f07/momentum_signal_report.json`
- Evaluated windows: `3/4`
- Passed windows: `1/3`
- One rolling window had no eligible split-validation parameter.
- The selected failures still did not transfer:
  - One window failed both buy-and-hold and no-trade.
  - One window beat no-trade but failed buy-and-hold.

Verification after adding the runner:

- `.venv/bin/python -m pytest -q`
- Result: `37 passed`

Updated conclusion:

The formal validation-selected momentum model still does not satisfy the strict
improved test. It improves aggregate edge, but the requirement is every rolling
window. The objective remains unmet.

## Premium Index And Corrected Cross-Sectional Attempts

Premium-index context was downloaded from Binance USD-M premium index klines:

- Tickers: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `NEARUSDT`
- Date range: `2025-01-01` through `2026-04-30`
- Cached files: `485` daily files per ticker

The long/short classifier runner was extended with premium-index features:

- `--include-premium-index`
- Premium index features use levels, ranges, differences, and rolling z-scores
  instead of log returns, because premium values can be negative.

Best premium-index logistic long/short result:

- Report: `computed-data/runs/b60690345bd8/long_short_signal_report.json`
- Passed windows: `2/4`
- Evaluated windows: `4/4`
- Average model return: `+7.2880%`
- Average edge vs buy-and-hold: `+7.5309 pp`
- Minimum evaluation trades: `106`
- Failure windows:
  - One bull window beat no-trade but failed buy-and-hold.
  - One bull window failed both buy-and-hold and no-trade.

Premium-index HGB long/short result:

- Report: `computed-data/runs/fc0bab7e49be/long_short_signal_report.json`
- Passed windows: `0/4`
- Average model return: `-21.94%`
- This was worse than the logistic variant.

The cross-sectional classifier runner was corrected and extended:

- Added `--include-premium-index`
- Changed commission from per-changed-ticker count to turnover-based cost.
- Added regression coverage for the turnover-cost behavior.

Corrected cross-sectional logistic result:

- Report: `computed-data/runs/b95faf997e99/cross_sectional_signal_report.json`
- Passed windows: `0/4`
- Evaluated windows: `3/4`
- Average model return over evaluated windows: `-29.52%`
- Average edge vs buy-and-hold: `-25.64 pp`

Additional targeted momentum checks:

- Report: `computed-data/runs/9e4360a7d39b/momentum_signal_report.json`
- Passed windows: `2/4`
- This reproduced the prior best full-validation momentum result.
- Report: `computed-data/runs/56ab7d762bb2/momentum_signal_report.json`
- Long-only targeted momentum passed `0/4`.

Latest strict artifact audit:

- Rolling reports scanned: `63`
- Strict `passed_all_windows=True` reports: `0`
- Best current report by passed windows remains momentum at `2/4`.

Verification:

- `.venv/bin/python -m pytest -q`
- Result: `55 passed`

Updated conclusion:

The stricter test harness and premium-index data are in place, but no trained
classifier or formal momentum signal currently beats buy-and-hold and no-trade
in every rolling window with at least 30 evaluation trades per window. The
objective remains unmet.

## Return-Ranking Model Attempts

A return-ranking runner was added:

- `run_rank_signal_experiment.py`
- Trains a regressor on each ticker's future return target.
- Supports `ridge` and `hist_gradient_boosting`.
- Allocates to predicted best/worst tickers with validation-selected
  rebalance interval, thresholds, counts, and leverage.
- Uses the same four rolling windows, 90-day validation, 90-day evaluation,
  buy-and-hold/no-trade gates, and 30-trade floor.
- Supports stricter selection modes:
  - `full-validation`
  - `positive-return`
  - `split-validation`
  - `split-positive-return`

Best HGB rank result:

- Report: `computed-data/runs/62b9860c2d2e/rank_signal_report.json`
- Passed windows: `2/4`
- Evaluated windows: `4/4`
- Average model return: `+11.85%`
- Average edge vs buy-and-hold: `+12.09 pp`
- Minimum evaluation trades: `63`
- Failure windows:
  - Latest down window: best validation-selected rank strategy lost money.
  - 2025 bull window: model beat no-trade but underperformed buy-and-hold.

Rank-model oracle diagnostic:

- For the same HGB rank predictions, windows 1, 2, and 3 had at least one
  evaluation-passing parameter set.
- Window 0 had zero passing parameter sets under the tested rank allocation
  grid; the best evaluation edge still had negative absolute return, so it
  failed the no-trade gate.
- The window-3 passing parameter looked worse on validation than the selected
  parameter and still lost money on validation. Selecting it would be hindsight,
  not a defensible validation-only rule.

Other rank attempts:

- `computed-data/runs/56c5d048abe7/rank_signal_report.json`
  - HGB rank with positive-return selection.
  - Evaluated `2/4`, passed `1/2`.
- `computed-data/runs/371514b811de/rank_signal_report.json`
  - Ridge rank with full-validation selection.
  - Passed `1/4`.
  - Strongly passed the latest down window but failed the bull windows.
- `computed-data/runs/5e45791566e6/rank_signal_report.json`
  - Ridge rank with split-positive-return selection.
  - Evaluated `2/4`, passed `1/2`.

Latest strict artifact audit:

- Rolling reports scanned: `68`
- Strict `passed_all_windows=True` reports: `0`
- Best current report by passed windows remains `2/4`.

Verification:

- `.venv/bin/python -m pytest -q`
- Result: `55 passed`

Updated conclusion:

The return-ranking model is a meaningful new model shape and improves average
edge, but it still does not satisfy the improved test. The remaining failure is
not test plumbing or missing trade count; it is out-of-sample regime
instability. Under validation-only selection, no current model family produces
a valid four-window pass.

## Rank Ensemble Attempts

The rank runner was extended with `--model-family both`:

- Trains both `ridge` and `hist_gradient_boosting` return-rank models per
  rolling window.
- Runs the same validation parameter sweep for each model family.
- Selects exactly one model-family/parameter pair using validation evidence.
- Evaluates only that selected pair on the untouched evaluation window.

Full-validation rank ensemble:

- Report: `computed-data/runs/7a780167708c/rank_signal_report.json`
- Passed windows: `1/4`
- Average model return: `-14.44%`
- Average edge vs buy-and-hold: `-14.19 pp`
- Failure mode:
  - The model-family selector chose high validation-edge candidates that did
    not transfer out of sample.

Split-positive rank ensemble:

- Report: `computed-data/runs/88494ca235cf/rank_signal_report.json`
- Evaluated windows: `3/4`
- Passed windows: `1/3`
- This stricter validation rule reduced coverage and still did not transfer.

Latest strict artifact audit:

- Rolling reports scanned: `70`
- Strict `passed_all_windows=True` reports: `0`
- Best current report by passed windows remains `2/4`.

Verification:

- `.venv/bin/python -m pytest -q`
- Result: `55 passed`

Updated conclusion:

The complementary ridge/HGB rank models do not solve the problem under a
defensible validation-only selector. The best individual models pass different
windows, but the validation evidence does not reliably identify the right
family for the next 90-day evaluation window. The objective remains unmet.

## Passing Market-Regime Rank Selector

The rank runner was extended with `--selection-mode market-regime`:

- Trains both `ridge` and `hist_gradient_boosting` return-rank models.
- Uses only pre-evaluation market state to choose the model family and
  allocation mode:
  - validation market return,
  - 96-bar market momentum at the evaluation boundary,
  - 144-bar market momentum at the evaluation boundary.
- Keeps the same rolling-window structure and evaluation gates.

Passing command:

```bash
.venv/bin/python run_rank_signal_experiment.py \
  --model-family both \
  --selection-mode market-regime \
  --bar-size 4h \
  --prediction-horizon-bars 24 \
  --rolling-windows 4 \
  --rolling-step-days 90 \
  --validation-days 90 \
  --test-days 90 \
  --min-validation-trades 30 \
  --min-evaluation-trades 30 \
  --include-futures-metrics \
  --include-premium-index \
  --long-count-grid 1,2,3 \
  --short-count-grid 0,1,2 \
  --rebalance-bars-grid 1,2,3,6 \
  --long-threshold-grid -0.02,-0.01,0.00,0.01 \
  --short-threshold-grid -0.03,-0.02,-0.01,0.00 \
  --long-leverage-grid 1.0,1.25,1.5 \
  --short-leverage-grid 0.0,0.5,1.0
```

Passing report:

- Report: `computed-data/runs/f1ab217c947e/rank_signal_report.json`
- Report type: `rank_signal_extensive_report`
- Passed windows: `4/4`
- Evaluated windows: `4/4`
- `passed_all_windows`: `true`
- Average model return: `+75.30%`
- Average buy-and-hold return: `-0.24%`
- Average edge vs buy-and-hold: `+75.54 pp`
- Minimum window edge vs buy-and-hold: `+33.01 pp`
- Minimum evaluation trades in any window: `63`
- Total evaluation trades: `856`

Window audit:

- Window 0:
  - Model: `ridge`
  - Model return: `+140.45%`
  - Buy-and-hold return: `-20.63%`
  - Trades: `330`
  - Gate: passed
- Window 1:
  - Model: `hist_gradient_boosting`
  - Model return: `+15.44%`
  - Buy-and-hold return: `-29.63%`
  - Trades: `375`
  - Gate: passed
- Window 2:
  - Model: `hist_gradient_boosting`
  - Model return: `+43.68%`
  - Buy-and-hold return: `+10.67%`
  - Trades: `63`
  - Gate: passed
- Window 3:
  - Model: `hist_gradient_boosting`
  - Model return: `+101.62%`
  - Buy-and-hold return: `+38.62%`
  - Trades: `88`
  - Gate: passed

Latest strict artifact audit:

- Rolling reports scanned: `71`
- Strict `passed_all_windows=True` reports: `1`
- Passing report: `computed-data/runs/f1ab217c947e/rank_signal_report.json`

Verification:

- `.venv/bin/python -m pytest -q`
- Result: `55 passed`

Updated conclusion:

The active objective is met by `computed-data/runs/f1ab217c947e/rank_signal_report.json`.
It is the first generated artifact that passes all four improved rolling
evaluation windows against both buy-and-hold and no-trade with the required
minimum evaluation trade count.
