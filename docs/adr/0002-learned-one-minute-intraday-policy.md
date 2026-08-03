# Learned One-Minute Intraday Policy

Status: accepted

The repository will hard-cut from the guarded market-regime rank-selector workflow to a learned
one-minute intraday policy. The goal is day trading: score every one-minute point, look for small
long-only opportunities with no short exposure, and select horizons, thresholds, model family, and
max hold from training/validation evidence instead of hard-coded regime or guard allocation rules.

## Consequences

- The old guarded market-regime winner is no longer the operational target.
- The old guarded market-regime replay implementation is deleted during the cutover rather than kept
  as an archived reference.
- The cutover uses historical intraday replay and reports only; online paper trading is removed from
  the operational surface.
- The policy uses liquid majors, one-minute spot/premium data, causally forward-filled five-minute
  futures metrics, fee plus slippage replay, fixed capped long-only risk units, and hybrid exits.
- Validation selects active candidates with positive after-cost return, positive Sharpe, sufficient
  validation trades, and a 5% maximum drawdown gate. Candidates are still ranked by excess return
  over equal-weight buy-and-hold, but single-window buy-and-hold outperformance is not required for
  deployment because the rolling CV layer is the proof gate for buy-and-hold excess.
- Replay holds a selected position through its forecast horizon before score-based exits can close
  it, so a multi-minute label is not evaluated as one-minute churn. The validation-selected maximum
  hold remains the hard cap.
- Replay supports validation-selected entry/exit threshold hysteresis: a new long requires the entry
  threshold, while an existing long can remain open down to the selected exit threshold after the
  forecast horizon. This is a turnover-control surface selected from validation evidence, not a
  hand-picked cooldown.
- The model-family namespace supports `_market_excess` variants that train on ticker forward return
  minus equal-weight market forward return. These variants are experimental candidates and are not
  default proof settings unless validation/CV evidence selects them.
- The model-family namespace supports `_market_return` variants that train on equal-weight market
  forward return. These variants are experimental candidates for learned market participation and
  are not hard-coded buy-and-hold fallbacks.
- The model-family namespace supports `market_ridge`, which trains on equal-weight market forward
  return using averaged causal ticker features and emits one shared market score to every ticker.
  Validation replay still selects thresholds and holds, so this is a learned participation
  candidate rather than an always-on buy-and-hold fallback.
- The model-family namespace supports `market_ridge_vol_scaled`, which trains the same shared market
  score on equal-weight market forward return divided by causal market volatility. This is an
  experimental learned participation target, not an always-on market exposure rule.
- The model-family namespace supports `ridge_shared`, which trains one pooled ridge model after
  dropping ordinal `ticker_id`. This is an experimental generalization candidate rather than a
  per-ticker specialization.
- The model-family namespace supports `ridge_decay_<days>` variants that train ridge with
  exponentially decayed sample weights. These variants are experimental recency-adaptation
  candidates and are not default proof settings unless validation/CV evidence selects them.
- The model-family namespace supports `_vol_scaled` variants that train on after-cost forward return
  divided by causal realized volatility at the signal bar. These variants are experimental
  candidates for risk-adjusted edge and are not default proof settings unless validation/CV evidence
  selects them.
- The model-family namespace supports `_path_mean` variants that train on the average after-cost
  forward return from one minute through the selected horizon. `_path_mean_vol_scaled` additionally
  divides that label by causal realized volatility. These are experimental path-alignment targets,
  not max-favorable-excursion labels.
- The model-family namespace supports `_market_excess_vol_scaled` variants that train on
  equal-weight market-excess forward return divided by causal realized volatility at the signal bar.
  These variants are experimental candidates for benchmark-relative risk-adjusted edge and are not
  default proof settings unless validation/CV evidence selects them.
- The CLI supports an optional validation-selected risk-unit grid. The default proof setting remains
  the fixed 25% risk unit unless an experiment explicitly supplies a grid.
- The CLI supports an optional selection trade floor. This prefers validation candidates whose replay
  reaches the requested trade count while keeping the minimum validation trade gate separate.
- The CLI supports an optional selection activity weight. This adds a validation-trade-count bonus
  to candidate ranking, but CV fold activity remains the proof gate.
- The CLI supports optional validation-slice stability ranking. Passing candidates can be ranked by
  median excess return across chronological validation slices, but this remains experimental unless
  rolling CV evidence selects it.
- The model-family namespace supports `huber` and `ridge_alpha_<N>` as explicit robust/regularized
  linear probes. They are non-default unless selected by validation evidence.
- The feature set includes causal cyclical time features for minute-of-day and day-of-week
  seasonality. These use only the timestamp of the closed signal bar.
- Backtest reports include selected validation-to-evaluation drift diagnostics for excess return,
  return, Sharpe, drawdown, and trade count so unstable selectors are visible in the artifact.
- Top-level run ids include both configuration and a fingerprint of the intraday runner, data
  builder, and policy code by default. `--config-only-run-id` is reserved for intentional legacy
  comparisons.
- Rolling backtest cross-validation is a proof layer over complete walk-forward backtests. It passes
  only when median fold excess return is positive, median fold Sharpe is positive, worst fold
  drawdown stays within the configured gate, and every fold has non-negative excess return over
  buy-and-hold. CV folds target continuous recent out-of-sample evaluation coverage; training and
  validation lead-ins may overlap across folds. CV mode rejects configurations where the
  walk-forward stride differs from the evaluation window length, because those produce gapped or
  overlapping out-of-sample coverage. A CV pass also requires every fold to meet a separate
  minimum-CV-fold-trades setting; staying entirely in cash is reported but does not prove the
  day-trading objective. CV reports pooled learned and buy-and-hold equity over the continuous
  out-of-sample coverage, and the pooled learned return must beat pooled buy-and-hold for `passed_cv`
  to be true.
