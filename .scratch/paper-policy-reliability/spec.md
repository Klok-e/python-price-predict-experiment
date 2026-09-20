# Paper Policy Reliability and Evaluation

Design confirmed. Implementation and local verification authorized; live service deployment and activation remain separate.

## Agreed scope and sequence

Correct operational reliability and decision audit semantics first, then evaluate the Trading
Policy against passive comparisons before changing its training budget or architecture.

Include investigation and correction of invalid Operating Window timestamps, failed notifications,
and Model Attribution recovery failures in the reliability phase, preserving historical records.
Defer dashboard performance optimization unless measurements show that it interferes with timely
decisions or execution.

## Failed scheduled fitting

Continue trading with the last successful Fitted Policy when scheduled fitting fails, and show
that model's age alongside fitting failure status. Retry after successive delays of 5, 15, and
30 minutes, then every 60 minutes until successful; persist the retry schedule across restarts
and permit only one fitting attempt at a time.

## Execution eligibility and Policy Revision

Apply [ADR-0006](../../docs/adr/0006-signal-time-execution-eligibility.md) across training, historical
replay, and the Paper Account:

- Determine the 1% turnover eligibility threshold at Signal Time.
- Complete below-threshold decisions immediately without pending execution or Missed Execution.
- Retain eligibility for qualifying decisions even when turnover later falls below the threshold;
  calculate quantities at fresh fill prices to reach recorded Target Weights, subject to risk
  controls and execution expiry.
- Require a model trained and evaluated under the revised rule before activation, while the old
  policy continues running.
- Preserve the account, positions, history, and original semantics of existing results; record
  the Policy Revision boundary and report performance separately on either side.
- At activation, withhold new decisions until pending execution resolves under the old rules,
  continuing market observation and risk controls.
- Retain positive compounded net return across validation folds and no fold exceeding the
  Drawdown Limit as the validation gate; benchmark outperformance is diagnostic evidence.

## Forward comparison

Initialize the primary Hold Benchmark with the account's equity and positions at the revision
boundary. Hold its quantities unchanged and include subsequent Funding; compare with the account
after Transaction Costs to measure what subsequent policy decisions add without hindsight
exposure selection.
