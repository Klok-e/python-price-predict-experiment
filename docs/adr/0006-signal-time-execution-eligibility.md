---
status: accepted
---

# Determine Execution Eligibility at Signal Time

Determine minimum-turnover eligibility once at Signal Time across training, historical replay,
and the Paper Account so that Decision Records describe the same behavior evaluated and executed.
Below-threshold decisions complete immediately without scheduling execution; qualifying decisions
retain eligibility despite subsequent turnover changes, with quantities calculated at fresh fill
prices to reach the recorded Target Weights, subject to risk controls and execution expiry.
Treat this shared rule as a Policy Revision and preserve existing results under their original
semantics, because rechecking the threshold at fill time can turn recorded below-threshold decisions
into executions and makes historical comparisons ambiguous.

Continue the existing Paper Account with its positions and history intact when adopting the
revision, recording the Policy Revision boundary and reporting performance separately before and
after it so account continuity does not merge evidence from different policy semantics.
Require a Fitted Policy trained and evaluated under the revised eligibility rule before activation;
keep the old policy running while that model is prepared.
Retain the existing validation gate: positive compounded net return across the Walk-Forward Folds
and no fold exceeding the Drawdown Limit; report exposure-matched passive comparisons as
Development Evidence without making benchmark outperformance an activation requirement.
Use a Hold Benchmark initialized from the account's equity and positions at the revision boundary
as the primary forward comparison, retaining its quantities and including subsequent Funding to
measure what later policy decisions add after Transaction Costs without hindsight exposure sizing.
Before activation, withhold new policy decisions while any pending execution completes or expires
under the old rules; continue market observation and risk controls throughout the transition.
