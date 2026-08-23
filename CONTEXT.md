# Profit-Maximizing Crypto Trading

This context describes a causal, cost-aware cryptocurrency portfolio policy and the simulated
evidence used to evaluate it.

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
The maximum acceptable Maximum Drawdown within one Paper Account or evidence run; a breach triggers
its Risk Stop regardless of profit.
_Avoid_: Drawdown target, soft risk preference

**Risk Stop**:
The terminal transition after a Drawdown Limit breach: target a flat portfolio at the next eligible
fill, then permit no further decisions for that Paper Account until Manual Reset.
_Avoid_: Report-only breach, temporary pause, hindsight stop

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
Any change to a Policy Protocol. A Policy Revision creates a distinct strategy identity in account
history.
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
The initial state of a Paper Account or independent evidence run: full cash and zero instrument
exposure.
_Avoid_: Inherited position, warm portfolio

**Policy Handoff**:
Replacement of one Fitted Policy by the next scheduled Fitted Policy while preserving the Current
Portfolio. It is not a forced liquidation or a Policy Revision.
_Avoid_: Weekly flattening, account reset

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

**Missed Execution**:
A scheduled Portfolio Change that expires because no fresh executable quote was observed within its
allowed fill window. It changes neither positions nor cash and remains visible in account history.
_Avoid_: Rejected signal, zero-turnover decision, delayed fill

**Portfolio Change**:
The signed difference between Current Portfolio weights and Target Weights that must be executed.
_Avoid_: Signal change, prediction update

**Decision Record**:
The durable account history linking one Market State and Current Portfolio to the resulting Target
Weights, execution outcome, model identity, and any applicable constraints or interventions.
_Avoid_: Trade row, chart annotation, model explanation

**Model Attribution**:
A post-hoc estimate of which observed inputs and time regions most influenced a Fitted Policy's
Target Weights. It is influence evidence, not a causal explanation.
_Avoid_: Trade reason, model intent, proven cause

**Executable Portfolio Change**:
A decision-time Portfolio Change whose total turnover across all instruments is large enough to
execute. Smaller desired changes accumulate against the Current Portfolio rather than creating
fills.
_Avoid_: Prediction update, micro-fill, qualifying change

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
_Avoid_: Real-money result, proof of profitability

**Validated Policy Protocol**:
A Policy Protocol with positive Compounded Net Return across Walk-Forward Folds and no Drawdown
Limit breach. It is eligible for Historical Holdout evaluation but remains Development Evidence.
_Avoid_: Winning model, proven strategy

**Historical Holdout**:
A period excluded from every choice until a Policy Protocol is frozen. Its revealed prefix may be
used by scheduled retraining, but no decision may use its future or revise the Policy Protocol.
_Avoid_: Validation fold, reusable test set

**Prequential Evaluation**:
Sequential evaluation that predicts each next interval before revealing it, while permitting the
unchanged Policy Protocol to retrain later from observations already revealed.
_Avoid_: Frozen-weights replay, future-aware refit

**Consumed Holdout**:
A former Historical Holdout whose results have been observed and may only be treated as Development
Evidence afterward.
_Avoid_: Retested holdout, final test after tuning

**Paper Account**:
A persistent simulated account whose cash, open positions, and history survive operator restarts
until Manual Reset. Its results are Development Evidence, not proof of real-world profitability.
_Avoid_: Evidence run, service process, real-money account

**Operating Window**:
A period during which the paper-trading operator is running and may make decisions and execute
Portfolio Changes for a Paper Account.
_Avoid_: Paper account, proof interval, browser session

**Manual Reset**:
The explicit flattening and archival of one Paper Account followed by creation of a new Flat Start;
stopping or restarting the operator never performs it implicitly.
_Avoid_: Automatic recovery, service restart, history deletion

**Operator Intervention**:
A human-requested pause, flatten, resume, or Manual Reset recorded separately from decisions and
Portfolio Changes produced by the Trading Policy.
_Avoid_: Policy decision, manual strategy trade, unexplained account mutation
