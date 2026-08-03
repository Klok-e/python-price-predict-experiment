# Profit-Maximizing Crypto Trading

This context describes a causal, cost-aware cryptocurrency portfolio policy and the evidence needed
to call it profitable.

## Language

### Objective and Risk

**Compounded Net Return**:
The multiplicative change in portfolio equity after transaction costs and funding. It is the primary
optimization target.
_Avoid_: Gross return, prediction score, model accuracy

**Net Log Growth**:
The additive form of compounded net return used to evaluate consecutive portfolio decisions.
_Avoid_: Raw forward-return label, classification accuracy

**Maximum Drawdown**:
The largest peak-to-trough decline in Marked Equity over a contiguous period.
_Avoid_: Worst trade, volatility

**Marked Equity**:
Portfolio equity after valuing open positions at each risk-observation price and applying due
funding cashflows.
_Avoid_: Decision-only equity, realized cash balance

**Drawdown Limit**:
The maximum acceptable Maximum Drawdown within one evidence run; a breach triggers its Risk Stop
and fails the policy regardless of profit.
_Avoid_: Drawdown target, soft risk preference

**Risk Stop**:
The terminal transition after a Drawdown Limit breach: target a flat portfolio at the next eligible
fill, then permit no further decisions in that run.
_Avoid_: Report-only breach, temporary pause, hindsight stop

**Profitable Policy**:
A Policy Protocol that completes Forward Paper Proof with positive Compounded Net Return and no
Drawdown Limit breach.
_Avoid_: Profitable model, accurate predictor, backtest winner, holdout winner

### Portfolio

**Trading Policy**:
A causal mapping from observable Market State and the Current Portfolio to Target Weights.
_Avoid_: Price predictor, signal classifier, strategy heuristic

**Policy Protocol**:
The fixed features, architecture, objective, constraints, training procedure, and retraining schedule
that produce a Trading Policy.
_Avoid_: Current model weights, run configuration variant

**Fitted Policy**:
The learned parameters produced by one scheduled execution of a Policy Protocol. Routine retraining
may replace a Fitted Policy without changing its Policy Protocol.
_Avoid_: Policy revision, permanent model

**Policy Revision**:
Any change to a Policy Protocol. A Policy Revision creates a new proof candidate and resets its
Proof Clock.
_Avoid_: Scheduled retraining, new fitted weights

**Market State**:
All market information observable when a portfolio decision is made, including causal history and
explicit source-availability indicators.
_Avoid_: Feature row, future-filled data

**Optional Market Input**:
A Market State observation whose absence is represented explicitly and does not invalidate an
otherwise executable decision interval.
_Avoid_: Required fill data, silently imputed source

**Execution-Complete Interval**:
A decision interval containing every price and funding cashflow required to reproduce portfolio
execution. An incomplete interval is invalid rather than filled or assumed costless.
_Avoid_: Best-effort interval, interpolated execution

**Current Portfolio**:
The signed instrument weights and cash held immediately before a portfolio decision.
_Avoid_: Previous signal, model state

**Target Weight**:
The signed fraction of equity the Trading Policy wants allocated to one instrument after execution.
Positive is long, negative is short, and zero is flat.
_Avoid_: Buy probability, class label

**Gross Exposure**:
The sum of absolute Target Weights across instruments.
_Avoid_: Net exposure, invested capital

**No Leverage**:
The constraint that Gross Exposure cannot exceed portfolio equity. It does not prohibit short
perpetual positions.
_Avoid_: Spot-only, long-only, no derivatives

**Cash Weight**:
The zero-yield fraction of equity left unallocated when the Trading Policy does not find enough net
edge.
_Avoid_: No-action error, forced allocation

**Flat Start**:
The initial state of an independent evidence run: full cash and zero instrument exposure.
_Avoid_: Inherited position, warm portfolio

**Policy Handoff**:
Replacement of one Fitted Policy by the next scheduled Fitted Policy while preserving the Current
Portfolio. It is not a forced liquidation or a Policy Revision.
_Avoid_: Weekly flattening, proof reset

**Trading Universe**:
The fixed set of liquid perpetual instruments the Trading Policy may hold.
_Avoid_: Candidate scanner, dynamic altcoin universe

**Common Trading Start**:
The earliest timestamp when every instrument in the Trading Universe has Execution-Complete
Intervals. Evidence never changes portfolio shape by starting an instrument later.
_Avoid_: Partial-universe training, listing backfill

### Decisions and Execution

**UTC Market Time**:
The timezone-aware clock used for bars, funding, folds, retraining, artifacts, and paper records.
_Avoid_: Local time, timezone-naive timestamp

**Decision Bar**:
A fully closed market interval that anchors one portfolio decision.
_Avoid_: Current candle, partial bar

**Signal Time**:
The instant after a Decision Bar closes when its Market State becomes eligible for a decision.
_Avoid_: Fill time, bar open

**Decision Latency**:
The required delay between Signal Time and the earliest eligible fill.
_Avoid_: Instant fill, next-close delay

**Reference Price**:
The first eligible observed market price after Decision Latency, before the all-in Transaction Cost
adjustment.
_Avoid_: Signal-bar close, interpolated fill

**Effective Fill**:
The Reference Price moved against the trade direction by the all-in Transaction Cost assumption.
_Avoid_: Raw quote, fee-free fill

**Portfolio Change**:
The signed difference between Current Portfolio weights and Target Weights that must be executed.
_Avoid_: Signal change, prediction update

**Qualifying Portfolio Change**:
A decision-time Portfolio Change whose total executed turnover across all instruments is large
enough to execute and count once toward Forward Paper Proof. Smaller desired changes accumulate
against the Current Portfolio rather than creating fills.
_Avoid_: Prediction update, micro-fill, activity-only trade

**Transaction Cost**:
The all-in fee, spread, and slippage adjustment charged once against executed Portfolio Change.
_Avoid_: Fixed label deduction, fee-free fill

**Funding**:
The timestamped perpetual payment earned or paid by an open position.
_Avoid_: Trading fee, estimated carry ignored by replay

**Causal Replay**:
Historical execution of a Trading Policy using only Market State available at each Signal Time and
the same timing, cost, funding, and portfolio constraints intended for paper trading.
_Avoid_: Backtest shortcut, current-close replay

### Evidence

**Walk-Forward Fold**:
A chronological training period followed by a disjoint validation period evaluated prequentially,
with every initial fit and scheduled refit derived only from observations already revealed.
_Avoid_: Random fold, shuffled cross-validation

**Purged Boundary**:
A split boundary with crossing labels and training sequences removed.
_Avoid_: Overlapping split, embargo-free boundary

**Development Evidence**:
Results used to choose or revise a Trading Policy.
_Avoid_: Final proof, untouched result

**Validated Policy Protocol**:
A Policy Protocol with positive Compounded Net Return across Walk-Forward Folds and no Drawdown
Limit breach. It is eligible for Historical Holdout evaluation but is not yet a Profitable Policy.
_Avoid_: Winning model, proven strategy

**Historical Holdout**:
A period excluded from every choice until a Policy Protocol is frozen. Its revealed prefix may be
used by scheduled retraining, but no decision may use its future or revise the Policy Protocol.
_Avoid_: Validation fold, reusable test set

**Prequential Evaluation**:
Sequential evaluation that predicts each next interval before revealing it, while permitting the
unchanged Policy Protocol to retrain later from observations already revealed.
_Avoid_: Frozen-weights replay, future-aware refit

**Holdout-Passing Protocol**:
A Validated Policy Protocol with positive Compounded Net Return and no Drawdown Limit breach on its
Historical Holdout. It is eligible for Forward Paper Proof but is not yet a Profitable Policy.
_Avoid_: Proven strategy, reusable holdout winner

**Consumed Holdout**:
A former Historical Holdout whose results have been observed and may only be treated as Development
Evidence afterward.
_Avoid_: Retested holdout, final test after tuning

**Forward Paper Proof**:
Live-market simulation of a frozen policy protocol using contemporaneous public data and no real
orders.
_Avoid_: Historical replay, exchange testnet fill proof

**Proof Clock**:
The required uninterrupted duration and activity of Forward Paper Proof for one Policy Protocol.
_Avoid_: Combined runs, inherited proof
