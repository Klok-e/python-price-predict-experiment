# Cryptocurrency Intraday Price Prediction

This context describes the operational one-minute day-trading workflow: how validated market data
becomes causal intraday features, how a learned long-only policy selects opportunities from
training/validation evidence, and how historical replay reports risk-adjusted results.

## Language

**Validated Bar Series**:
A ticker's OHLC time series after duplicate, missing, non-monotonic, and invalid-bar checks pass.
_Avoid_: Trusted raw download

**Strict Temporal Split**:
The rule that future data must not influence training, model selection, threshold selection, exits,
or replay decisions.
_Avoid_: Full-history fitting, post-split lookahead

**Next-Open Fill**:
The execution rule that a signal computed from candle `t` enters at candle `t+1` open.
_Avoid_: Current-close fill

**Causal Intraday Feature Set**:
The one-minute feature set built from spot OHLCV, one-minute premium index data, and futures metrics
forward-filled from their native five-minute cadence without using future updates. It may include
clock-only cyclical seasonality features such as minute-of-day and day-of-week.
_Avoid_: Future-filled metrics, stale-source blind fallback

**After-Cost Forward Return Label**:
The training label for one-minute opportunities: realized return from next-open entry to a candidate
intraday exit horizon after applying configured costs.
_Avoid_: Max-favorable-excursion label, barrier label as first proof

**Path-Mean Intraday Label**:
An experimental label that averages after-cost forward returns from one minute through the selected
horizon, requiring the full path to be available before training on a row.
_Avoid_: Best unseen exit label, partially known future path

**Market-Excess Intraday Label**:
An experimental label that subtracts the equal-weight forward return of the liquid-major universe
from each ticker's after-cost forward return for the same horizon.
_Avoid_: Treating absolute return and benchmark-relative return as the same learning problem

**Market-Excess Volatility-Scaled Intraday Label**:
An experimental label that subtracts equal-weight market forward return, then divides the residual
by causal realized volatility at the signal bar.
_Avoid_: Training on absolute risk units when testing benchmark-relative edge

**Market-Return Intraday Label**:
An experimental label that trains a model on the equal-weight liquid-major forward return for the
same horizon, so validation can test broad learned market participation instead of ticker-specific
edge.
_Avoid_: Hard-coded buy-and-hold fallback

**Market-Return Volatility-Scaled Intraday Label**:
An experimental label that divides equal-weight market forward return by causal market volatility at
the signal bar before fitting a market-participation model.
_Avoid_: Always-on market exposure

**Market-Level Ridge Model**:
A learned model family that averages causal ticker features by timestamp, predicts the
equal-weight market-return label, and emits the same market score to every ticker for validation
replay.
_Avoid_: Always-on buy-and-hold fallback

**Trade Edge Model**:
A model that estimates per-ticker after-cost opportunity so the policy can buy, sell to cash, or
hold each ticker independently when validation-selected thresholds indicate enough edge.
_Avoid_: Always-on relative ranker, forced winner/loser selection

**Learned Intraday Policy**:
A day-trading policy where model family, opportunity horizon, trade threshold, and max hold come
from training/validation evidence rather than hard-coded regime or guard rules.
_Avoid_: Market-regime override, fixed allocation heuristic

**Intraday Opportunity Horizon**:
The short future window, selected from one to sixty minutes using validation evidence, over which the
policy tries to predict exploitable price movement.
_Avoid_: Multi-day target, hand-picked fixed scalp horizon

**Intraday Risk-Adjusted Objective**:
The validation objective for day trading: select active candidates with positive after-cost return,
positive Sharpe, and acceptable drawdown, while still ranking them by excess return over
buy-and-hold. Buy-and-hold outperformance remains a cross-validation proof requirement rather than
the only single-window deployment filter.
_Avoid_: Raw return only, hit-rate only, pre-fee score

**After-Cost Intraday Replay**:
A replay that charges both commission and configurable slippage on fills before scoring the
intraday policy.
_Avoid_: Fee-blind scalp backtest

**Liquid Intraday Universe**:
The first target universe for one-minute day trading: BTC, ETH, BNB, and SOL, chosen to reduce
spread/slippage and data-quality risk.
_Avoid_: Broad alt sweep as first proof

**Gated One-Minute Decision**:
A one-minute prediction where the policy may still hold cash if the learned after-cost edge is not
strong enough to justify a fill.
_Avoid_: Forced every-minute rebalance

**Entry/Exit Threshold Hysteresis**:
Validation-selected separate entry and exit score thresholds. New positions require the entry
threshold, while existing positions can remain open down to the exit threshold after the forecast
horizon has elapsed.
_Avoid_: Selling every small score wobble, hand-picked cooldown

**Intraday Baseline Ensemble**:
The first learned model family set for one-minute day trading: simple linear models, pooled ridge
models with and without ticker identity, learned market-level participation models, optional ridge
alpha and recency-weighted ridge variants, optional market-excess, path-mean, market-excess
volatility-scaled, path-mean volatility-scaled, and volatility-scaled target variants, a regularized
positive-after-cost-return classifier, and tree models whose selection is based on validation replay
performance.
_Avoid_: Hard-coded regime model switch, deep model before baseline

**Intraday Risk Unit**:
The equity fraction used for each active intraday ticker while validation-selected trade thresholds
decide whether a ticker is active, capped at 100% total exposure. The default is a fixed 25% unit;
experiments may provide a small validation-selected risk-unit grid.
_Avoid_: Unbounded edge-proportional sizing

**Long-Only Intraday Execution**:
The day-trading execution mode: open or hold multiple gated long positions only, with maximum total
exposure capped at account equity and no short exposure or margin dependency.
_Avoid_: Short exposure, margin-dependent execution

**Hybrid Intraday Exit**:
An exit rule that keeps a long position through the selected forecast horizon, then keeps it only
while the learned edge remains active, with a validation-selected maximum hold between one and sixty
minutes.
_Avoid_: Hand-picked fixed holding period

**Walk-Forward Intraday Evaluation**:
Repeated chronological 60-day train, 14-day validation, and non-overlapping 14-day evaluation
windows over the 2025+ intraday feature period, used to test whether a learned policy persists
beyond one lucky slice.
_Avoid_: Single lucky final split

**Backtest Cross-Validation**:
An outer evaluation mode that runs several recent chronological walk-forward backtests and reports
median excess return, median Sharpe, worst fold drawdown, and whether every fold at least matches
buy-and-hold on excess return. Its folds target continuous recent out-of-sample evaluation coverage;
training and validation lead-ins may overlap across folds.
_Avoid_: Shuffled k-fold validation, mixing future folds into earlier training

**Buy-and-Hold Benchmark Gate**:
The requirement that the learned policy beat equal-weight buy-and-hold on the same tickers, costs,
exposure, starting cash, and evaluation timestamps.
_Avoid_: Comparing against a different date range or fee-free benchmark

**Minimum CV Fold Trades**:
The minimum number of evaluation trades a cross-validation fold must contain before it can count as a
passing day-trading proof. The default is 10 trades per fold.
_Avoid_: Treating validation-candidate activity and cross-validation proof activity as the same knob

**Selection Trade Floor**:
A validation ranking preference for candidates whose validation replay reaches a requested trade
count. It can be stricter than the minimum validation trade gate without making thin positive
candidates ineligible.
_Avoid_: Assuming validation activity guarantees out-of-sample activity

**Selection Activity Weight**:
An optional validation ranking bonus for candidates with more validation trades after return and
risk gates are checked. It is an experiment knob for activity, not a replacement for the CV fold
trade floor.
_Avoid_: Treating extra validation churn as proof of durable opportunity

**Validation-Slice Stability Ranking**:
An optional selector mode that splits the validation period into chronological slices and ranks
passing candidates by median slice excess over buy-and-hold instead of whole-validation excess.
_Avoid_: Using future evaluation folds to choose candidates, shuffled validation slices

**Pooled CV Equity**:
The stitched learned-policy and buy-and-hold equity curves over all cross-validation out-of-sample
fold periods, used to show total compounded performance across the covered period. It is written as
a CSV artifact and summarized in the CV report.
_Avoid_: Looking only at median folds when total covered-period performance disagrees

**Intraday Run ID**:
The artifact directory identifier for an intraday run. It includes both the run configuration and a
fingerprint of the intraday runner, data builder, and policy code unless `--config-only-run-id` is
explicitly requested for legacy comparison.
_Avoid_: Treating config-only artifact ids as stable across code-only feature changes

**Paper Ledger**:
The append-only JSONL record of predictions, target weights, simulated fills, equity snapshots, and
no-action reasons for historical replay.
_Avoid_: Mutable state file

**Decision Timeline**:
A visual historical replay artifact that shows buy, sell-to-cash, hold, explicit no-action records,
and resulting equity over time.
_Avoid_: Trade-only chart, hidden skipped decisions

**Intraday Report**:
The historical proof report for a learned one-minute policy, emphasizing after-cost Sharpe, maximum
drawdown, return, trade count per day, exposure, validation-to-evaluation drift diagnostics, and
the decision timeline.
_Avoid_: Raw accuracy report, trade list without risk context

## Relationships

- A **Validated Bar Series** is the input to the **Causal Intraday Feature Set**.
- A **Trade Edge Model** learns from an **After-Cost Forward Return Label**.
- A **Trade Edge Model** may use a **Market-Excess Intraday Label** when the experiment is testing
  benchmark-relative long selection.
- A **Trade Edge Model** may use a **Market-Return Intraday Label** when the experiment is testing
  learned market participation.
- A **Market-Level Ridge Model** is a learned market-participation candidate, not a
  buy-and-hold fallback; validation replay must still select its thresholds and holds.
- A **Learned Intraday Policy** uses an **Intraday Opportunity Horizon** instead of a multi-day
  future-return target.
- A **Learned Intraday Policy** is selected by the **Intraday Risk-Adjusted Objective**.
- A **Learned Intraday Policy** scores every minute, but only turns a score into a trade through a
  **Gated One-Minute Decision**.
- A **Gated One-Minute Decision** may use **Entry/Exit Threshold Hysteresis** to reduce turnover.
- A one-minute **Learned Intraday Policy** uses **Next-Open Fill** with costs applied.
- A **Gated One-Minute Decision** uses an **Intraday Risk Unit** and **Long-Only Intraday
  Execution**.
- **Long-Only Intraday Execution** uses a **Hybrid Intraday Exit**.
- A **Causal Intraday Feature Set** may mix one-minute and five-minute source cadences only when
  lower-frequency sources are forward-filled causally.
- The first **Causal Intraday Feature Set** trains and evaluates in the 2025+ period where one-minute
  premium features are available.
- The **Causal Intraday Feature Set** includes lag and rolling windows from short minute-scale moves
  through roughly one day of context.
- The **Intraday Report** uses **Walk-Forward Intraday Evaluation**.
- **Backtest Cross-Validation** repeats **Walk-Forward Intraday Evaluation** across recent
  chronological folds.
- **Backtest Cross-Validation** reports **Pooled CV Equity** in addition to fold medians.
- The **Intraday Report** and **Backtest Cross-Validation** both use the **Buy-and-Hold Benchmark
  Gate**.
- **Online Paper Trading** and shorting are outside the operational target.

## Current Operational Target

The operational target is a learned one-minute, long-only intraday backtest run by:

```bash
.venv/bin/python run_intraday_experiment.py
```

Default proof settings:

- liquid majors: `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT`
- 60d train / 14d validation / 14d evaluation, 14d stride
- model families: ridge and histogram gradient boosting
- horizon and max-hold candidates: 1, 5, 15, 30, and 60 minutes
- objective: after-cost Sharpe with 5% max drawdown and buy-and-hold excess-return gates
- execution: long-only, 25% equity per active ticker, 100% max exposure

Optional cross-validation proof:

```bash
.venv/bin/python run_intraday_experiment.py --cv-folds 3 --cv-windows-per-fold 3
```

Cross-validation uses rolling recent chronological folds. A CV report passes only when median fold
excess return over buy-and-hold is positive, median fold Sharpe is positive, worst fold drawdown
stays within the configured drawdown gate, and every fold has non-negative excess return over
buy-and-hold. CV mode requires the walk-forward stride to equal the evaluation window length so
out-of-sample coverage is continuous rather than gapped or overlapping. Each fold must also meet
**Minimum CV Fold Trades**; beating buy-and-hold by staying entirely in cash is not enough for a CV
pass.
The pooled learned CV return must also beat the pooled buy-and-hold return over the same continuous
out-of-sample coverage.

## Flagged Ambiguities

- "Free of heuristics" means **Learned Intraday Policy**, not absence of configuration or risk
  limits.
- "Sell" means reduce or exit a long position to cash, never open a short.
- **Online Paper Trading** is removed from the operational target.
