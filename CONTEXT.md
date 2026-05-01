# Cryptocurrency Price Prediction

This context describes the experiment's trading-language model: how market data becomes supervised labels, model signals, and backtest outcomes.

## Language

**Barrier Label**:
A supervised long-entry outcome that is positive only when the take-profit barrier is reached before the stop-loss barrier within the lookahead horizon.
_Avoid_: Future-close count, favorable-close majority

**Take-Profit Barrier**:
The price threshold above entry that defines a successful long trade candidate.
_Avoid_: TP count

**Stop-Loss Barrier**:
The price threshold below entry that defines a failed long trade candidate.
_Avoid_: SL count

**Lookahead Horizon**:
The fixed future time window used to decide whether a barrier label is positive or negative.
_Avoid_: Future sample count

**Trading Contract**:
The canonical experiment definition for timeframe, lookahead horizon, stop-loss, take-profit, fees, and same-candle barrier handling.
_Avoid_: Scattered defaults, strategy-specific constants

**Validated Bar Series**:
A ticker's OHLC time series after duplicate, missing, non-monotonic, and invalid-bar checks pass.
_Avoid_: Trusted raw download

**Training Period**:
The historical time span used to fit preprocessing state and train model weights.
_Avoid_: In-sample data

**Evaluation Period**:
The future time span used to evaluate supervised metrics and backtest outcomes after training.
_Avoid_: Test slice, validation tail

**Validation Period**:
The chronological holdout used for model selection, threshold tuning, and provisional strategy choices before final evaluation.
_Avoid_: Final test, reporting period

**Strict Temporal Split**:
The rule that future data must not influence training labels, preprocessing state, or model selection.
_Avoid_: Full-history preprocessing, post-split lookahead

**Model Signal**:
The strategy-facing probability that a candidate long entry has a positive barrier label.
_Avoid_: Raw score, mixed logit

**Trading Baseline**:
A simple reference strategy, such as buy-and-hold or no-trade, that a learned strategy must beat after fees.
_Avoid_: Sanity check only

**Successful Model**:
A model whose held-out backtests beat the trading baselines after fees across the selected tickers.
_Avoid_: High AUC alone, good validation loss

**Equal-Capital Portfolio**:
A multi-ticker backtest aggregation where each ticker receives the same starting capital and total starting cash is the sum of all sleeves.
_Avoid_: Raw equity sum with single-sleeve cash

**Pooled Model**:
A single model trained across multiple tickers to learn reusable price-action patterns.
_Avoid_: Implicit per-ticker model

**Single-Ticker OHLC Feature Set**:
Causal features derived only from one ticker's own OHLC history.
_Avoid_: Market context, volume-first feature expansion

**Benchmark Model**:
The simplest model used to test whether the corrected pipeline contains tradeable signal.
_Avoid_: First serious LSTM

**Experiment Config**:
The reproducible definition of tickers, date ranges, trading contract, feature set, split policy, sampling policy, model settings, and random seed.
_Avoid_: Ad hoc run settings

**Config-Keyed Artifact**:
A processed-data, model, metric, or backtest output whose path or metadata is tied to the experiment config that produced it.
_Avoid_: Reused generic cache

**Balanced Window Sampling**:
The training rule that ticker identity and barrier-label class should have comparable influence when sampling windows.
_Avoid_: Dominant ticker sampling, global class balance only

**Candidate Entry**:
A timestamp at which the strategy may open a long position if the model signal passes the entry rule.
_Avoid_: Every row as independent trade

**Cooldown-Aware Sampling**:
The rule that candidate entries should be spaced or filtered so heavily overlapping future trade paths do not masquerade as independent examples.
_Avoid_: Dense duplicate windows

**Next-Open Entry**:
The execution rule that a signal computed from candle `t` enters at candle `t+1` open.
_Avoid_: Current-close fill, implicit engine fill

**Fee-Aware Barrier**:
A stop-loss or take-profit threshold chosen with round-trip trading costs in mind.
_Avoid_: Gross-only barrier

**Expected-Value Entry Rule**:
The rule that opens a long position only when the model signal exceeds the break-even probability implied by barriers and trading costs, plus any required edge.
_Avoid_: Arbitrary confidence threshold

## Relationships

- A **Trading Contract** defines the **Lookahead Horizon**, **Take-Profit Barrier**, and **Stop-Loss Barrier** used by both labeling and backtesting.
- A **Barrier Label** is generated only from a **Validated Bar Series**.
- A **Trading Contract** uses **Fee-Aware Barrier** values.
- A **Barrier Label** is evaluated over exactly one **Lookahead Horizon**.
- A positive **Barrier Label** requires the **Take-Profit Barrier** to be reached before the **Stop-Loss Barrier**.
- A negative **Barrier Label** includes the **Stop-Loss Barrier** being reached first, neither barrier being reached, or both barriers being reached in the same candle.
- A **Strict Temporal Split** separates the **Training Period**, **Validation Period**, and **Evaluation Period** before fitting scalers or assigning labels that depend on future bars.
- A **Model Signal** estimates the probability of a positive **Barrier Label** under the active **Trading Contract**.
- A **Successful Model** beats the **Trading Baseline** on the **Evaluation Period**.
- Multi-ticker success is reported through an **Equal-Capital Portfolio** and per-ticker results.
- A **Pooled Model** must be reported per ticker so aggregate results do not hide ticker-specific failures.
- A **Pooled Model** uses **Balanced Window Sampling** during training.
- A **Pooled Model** initially uses the **Single-Ticker OHLC Feature Set**.
- The initial **Benchmark Model** is a simple feed-forward model.
- A **Config-Keyed Artifact** is produced from exactly one **Experiment Config**.
- A **Candidate Entry** receives one **Model Signal** and one **Barrier Label** under the active **Trading Contract**.
- **Cooldown-Aware Sampling** selects **Candidate Entry** timestamps for training and evaluation.
- A **Candidate Entry** uses **Next-Open Entry** timing, and its barriers are anchored to the next-open fill price.
- A **Model Signal** becomes a trade only through the **Expected-Value Entry Rule**.

## Example dialogue

> **Dev:** "Should a sample be positive if the next 256 closes contain more closes above take-profit than below stop-loss?"
> **Domain expert:** "No. A positive **Barrier Label** means the **Take-Profit Barrier** is reached before the **Stop-Loss Barrier** within the **Lookahead Horizon**."

## Flagged ambiguities

- "Profitable prediction" previously mixed future-close counts with executable trade outcomes; resolved: the canonical supervised target is the **Barrier Label**.
- Label-generation and backtest parameters were previously allowed to drift; resolved: both must share one **Trading Contract**.
- Raw Binance bars were resolved to require a **Validated Bar Series** gate before labeling and backtesting.
- Full-history scaling and boundary-crossing labels previously risked temporal leakage; resolved: use a **Strict Temporal Split**.
- Model outputs previously mixed logits and probabilities; resolved: strategy-facing **Model Signal** values are probabilities.
- "Working model" was previously undefined; resolved: success requires beating **Trading Baseline** strategies after fees in held-out backtests.
- Aggregate reporting was resolved as an **Equal-Capital Portfolio**.
- The model scope was resolved as a **Pooled Model** across tickers rather than separate per-ticker models.
- Initial feature scope was resolved as the **Single-Ticker OHLC Feature Set** until labels, splits, and execution semantics are corrected.
- Initial model complexity was resolved as a simple feed-forward **Benchmark Model** before LSTM work.
- Processed data, models, and backtest outputs were resolved as **Config-Keyed Artifact** outputs.
- Pooled training sampling was resolved as **Balanced Window Sampling** across tickers and barrier-label classes.
- Candidate-entry cadence was resolved as **Cooldown-Aware Sampling** rather than dense overlapping windows.
- Signal-to-entry timing was resolved as **Next-Open Entry**.
- Stop-loss and take-profit thresholds were resolved as **Fee-Aware Barrier** values rather than gross-only price moves.
- Entry thresholds were resolved as an **Expected-Value Entry Rule** rather than arbitrary confidence thresholds.
- Probability calibration is intentionally deferred; until it exists, the **Expected-Value Entry Rule** is provisional because raw sigmoid outputs may be miscalibrated.
- Model selection and threshold tuning were resolved to use a **Validation Period**, with final reporting reserved for the **Evaluation Period**.
