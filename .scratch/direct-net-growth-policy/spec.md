# Direct Net-Growth Portfolio Policy Hard Cut

Status: ready-for-agent

## Problem Statement

The repository is optimized around a learned one-minute, long-only price-prediction experiment that
has not demonstrated durable profit after costs. Its main implementation is a large model-family,
target, threshold, horizon, and hold-time search surface whose retained result fails rolling proof.
The repository also carries duplicate download paths, obsolete workflow documentation, nearly five
gigabytes of old run artifacts, a long narrative experiment history, and tests for configuration
surfaces that will be removed.

The user does not care whether profit comes from price prediction, classification, ranking, or a
hand-written rule. The desired outcome is one reproducible systematic Trading Policy that maximizes
Compounded Net Return after costs and funding, while never accepting more than 20% Maximum
Drawdown. Statistical prediction quality and buy-and-hold outperformance are not goals. Historical
results alone are insufficient: the result must survive causal validation, one initial Historical
Holdout evaluation, and Forward Paper Proof before it may be called a Profitable Policy.

## Solution

Hard-cut the repository to a single 15-minute direct portfolio-policy workflow for BTCUSDT,
ETHUSDT, BNBUSDT, and SOLUSDT USD-M perpetuals. The Trading Policy observes causal Binance-native
Market State and Current Portfolio, then emits continuous long, short, or cash Target Weights.
Train against portfolio Net Log Growth after an all-in turnover cost and actual funding, subject to
No Leverage, concentration constraints, and an operational Drawdown Limit.

Use one linear policy as a sanity baseline and a deliberately small causal temporal-convolution
search surface. Select through twelve purged, prequential 90-day Walk-Forward Folds. Evaluate the
frozen Policy Protocol once on the May-July 2026 Historical Holdout. Run the selected protocol in
public-data-only paper mode for at least 60 days and 100 Qualifying Portfolio Changes. Routine
weekly retraining produces a new Fitted Policy without resetting proof; a Policy Revision resets
the Proof Clock.

Expose the workflow through one small CLI backed by one deep workflow interface. Keep historical
and live market-data adapters behind the same seam, keep execution and risk simulation pure, and
write only reproducibility manifests, reports, equity, trades, and model artifacts. Remove the old
one-minute implementation and every compatibility surface that exists only for it.

## User Stories

1. As a quantitative researcher, I want the system to optimize Compounded Net Return, so that model
   training matches the actual goal of making money.
2. As a quantitative researcher, I want transaction costs included during training, so that turnover
   is discouraged before model selection.
3. As a quantitative researcher, I want actual historical funding applied, so that perpetual carry
   is part of measured profit.
4. As a quantitative researcher, I want the Trading Policy to emit portfolio weights directly, so
   that no proxy classification or fixed-horizon target controls execution.
5. As a quantitative researcher, I want the Current Portfolio included in each decision, so that the
   policy can reason about the cost of changing existing exposure.
6. As a quantitative researcher, I want continuous Target Weights, so that the policy can express
   long, short, cash, and partial conviction without threshold grids.
7. As a capital owner, I want Gross Exposure capped at 100% of equity, so that the policy uses No
   Leverage.
8. As a capital owner, I want absolute exposure to one ticker capped at 50% of equity, so that one
   instrument cannot consume the whole portfolio.
9. As a capital owner, I want unallocated equity kept as zero-yield cash, so that reported profit is
   attributable to trading.
10. As a capital owner, I want Maximum Drawdown capped at 20%, so that return is never accepted with
    intolerable loss.
11. As a capital owner, I want a Drawdown Limit breach to trigger a causal Risk Stop, so that the
    system exits rather than merely reporting a violation afterward.
12. As a capital owner, I want the Risk Stop to flatten at the next eligible Effective Fill, so that
    emergency execution uses the same realistic timing and cost contract as normal execution.
13. As a researcher, I want equity marked every minute, so that a 15-minute decision cadence cannot
    conceal an intrabar drawdown breach.
14. As a researcher, I want decisions based only on fully closed 15-minute bars, so that partial
    candles never leak into Market State.
15. As a researcher, I want a 60-second Decision Latency before fills, so that replay does not assume
    impossible computation and execution at Signal Time.
16. As a researcher, I want historical Reference Price taken from the delayed one-minute open, so
    that fills can be reproduced from stored data.
17. As a paper operator, I want paper Reference Price taken from contemporaneous midpoint data, so
    that live simulation uses observable market state.
18. As a researcher, I want Effective Fill moved 0.07% against trade direction, so that fee, spread,
    and slippage are charged once under one contract.
19. As a researcher, I want bid/ask spread recorded diagnostically but not charged twice, so that
    paper and replay remain comparable.
20. As a researcher, I want missing optional inputs represented with explicit masks, so that shorter
    feature histories do not discard otherwise useful market history.
21. As a researcher, I want missing execution prices or funding cashflows to invalidate data, so that
    required cashflows are never interpolated or treated as free.
22. As a researcher, I want the fixed Trading Universe available from Common Trading Start, so that
    portfolio shape does not change across history.
23. As a researcher, I want maximum available Binance history from 2020 onward, so that the model has
    substantially more samples than the former 2025-only feature intersection.
24. As a researcher, I want 15-minute decision data with 15-minute, one-hour, four-hour, and one-day
    context, so that the policy gets more samples than a four-hour strategy without returning to
    one-minute noise.
25. As a researcher, I want only Binance-native inputs in the first protocol, so that every feature
    is reproducible from one market-data source.
26. As a researcher, I want spot and perpetual OHLCV, funding, open interest, basis or premium,
    taker activity, and trade count available as Market State, so that both price and derivatives
    context can inform decisions.
27. As a researcher, I want market-wide and instrument-relative features, so that the policy can
    distinguish common crypto movement from cross-instrument opportunity.
28. As a researcher, I want one fixed causal feature set, so that arbitrary indicator and lag grids
    cannot overfit validation.
29. As a researcher, I want causal rolling robust normalization, so that scale changes and outliers
    are handled without fitting on future observations.
30. As a researcher, I want a linear direct-policy baseline, so that nonlinear complexity must beat
    a transparent comparator.
31. As a researcher, I want a small causal temporal-convolution policy, so that sequence information
    is modeled without a transformer, reinforcement-learning stack, or recurrent hidden-state
    dependency.
32. As a researcher, I want only widths 32 and 64 and receptive fields of one and seven days, so that
    architecture search remains predeclared and small.
33. As a researcher, I want three random seeds per temporal-convolution configuration, so that a
    lucky initialization cannot choose architecture.
34. As a researcher, I want median seed return to select temporal-convolution architecture, so that
    selection rewards stability.
35. As a researcher, I want the selected three seeds averaged into one policy ensemble, so that the
    deployed candidate is less dependent on one fit.
36. As a researcher, I want the actual ensemble replayed against the linear policy, so that candidate
    selection scores what will run rather than a proxy seed statistic.
37. As a researcher, I want the winning candidate to have positive compounded validation return, so
    that an unprofitable policy cannot unlock the Historical Holdout.
38. As a researcher, I want every validation fold within the Drawdown Limit, so that aggregate return
    cannot hide one unacceptable regime.
39. As a researcher, I want twelve non-overlapping 90-day Walk-Forward Folds, so that model choice is
    tested across several chronological regimes.
40. As a researcher, I want expanding training history beginning at Common Trading Start, so that all
    available past information can train each Fitted Policy.
41. As a researcher, I want Purged Boundaries, so that labels and training sequences never cross from
    training into scored time.
42. As a researcher, I want prequential weekly retraining inside validation, so that validation
    matches intended deployment.
43. As a researcher, I want all fitted scalers and parameters derived only from revealed history, so
    that every scored decision is out of sample.
44. As a researcher, I want the May-July 2026 period hidden until a Policy Protocol is frozen, so that
    the initial Historical Holdout is genuine.
45. As a researcher, I want the Historical Holdout evaluated prequentially, so that scheduled weekly
    retraining is tested without future awareness.
46. As a researcher, I want the holdout command locked after the initial evaluation, so that repeated
    attempts cannot masquerade as untouched proof.
47. As a researcher, I want an observed failed holdout converted to Development Evidence, so that its
    data can inform later research without retaining false holdout status.
48. As a researcher, I want continued development allowed after a failed holdout, so that I can seek
    a better policy without waiting for another historical quarter.
49. As a researcher, I want only fresh Forward Paper Proof to validate a post-holdout revision, so
    that reused historical evidence is not called final proof.
50. As a paper operator, I want the protocol retrained every Sunday UTC using only revealed data, so
    that the policy adapts on a fixed schedule.
51. As a paper operator, I want Policy Handoff to preserve Current Portfolio, so that weekly
    retraining does not force costly liquidation.
52. As a paper operator, I want scheduled Fitted Policy replacement not to reset Proof Clock, so that
    the frozen Policy Protocol can prove its intended adaptive behavior.
53. As a paper operator, I want any Policy Revision to reset Proof Clock, so that proof never combines
    materially different systems.
54. As a paper operator, I want Forward Paper Proof to last 60 days and include 100 Qualifying
    Portfolio Changes, whichever takes longer, so that both time and activity are observed.
55. As a paper operator, I want one Qualifying Portfolio Change counted per decision timestamp, so
    that simultaneous orders do not inflate activity.
56. As a paper operator, I want desired turnover below 1% of equity accumulated rather than executed,
    so that continuous output does not create meaningless micro-fills.
57. As a paper operator, I want paper proof to start from a Flat Start, so that inherited positions do
    not contaminate evidence.
58. As a paper operator, I want paper proof to fail on non-positive return or Drawdown Limit breach,
    so that a policy must satisfy both objective and risk.
59. As a paper operator, I want a failed paper run to inform development, so that observed weaknesses
    can guide a Policy Revision.
60. As a paper operator, I want a Policy Revision after paper failure to restart the complete proof
    requirement, so that revised behavior earns fresh evidence.
61. As a security-conscious operator, I want paper mode to use only public market data, so that the
    first implementation cannot place real orders or require secrets.
62. As a maintainer, I want one CLI with data-sync, validate, holdout, and paper commands, so that the
    operational workflow has one discoverable entry point.
63. As a maintainer, I want one checked-in configuration containing all strategy decisions, so that
    CLI flags cannot silently create untracked policy variants.
64. As a maintainer, I want CLI overrides limited to paths and compute device, so that a run cannot
    bypass the frozen Policy Protocol.
65. As a maintainer, I want every run identified by configuration, code, data, and model hashes, so
    that evidence is reproducible.
66. As a maintainer, I want minimal reports, equity, trades, manifest, and model artifacts, so that
    run output remains useful and bounded.
67. As a maintainer, I want charts generated on demand, so that stored HTML timelines do not inflate
    artifacts.
68. As a maintainer, I want one deep workflow interface, so that callers and tests do not coordinate
    data, training, replay, proof, and artifact details themselves.
69. As a maintainer, I want historical and live market-data adapters behind one canonical dataset
    seam, so that replay and paper consume identical domain structures.
70. As a maintainer, I want pure execution and risk simulation, so that costs, fills, funding,
    Marked Equity, and Risk Stop behavior can be tested deterministically.
71. As a maintainer, I want the old one-minute policy and model-search surfaces deleted, so that there
    is only one operational target.
72. As a maintainer, I want duplicate download scripts replaced by data-sync, so that one download
    contract owns Binance data.
73. As a maintainer, I want obsolete reports, ledgers, timelines, docs, tests, and helpers removed, so
    that old concepts cannot leak back into the new design.
74. As a maintainer, I want old run artifacts deleted while raw market data remains, so that nearly
    five gigabytes are reclaimed without redownloading useful inputs.
75. As a maintainer, I want the long experiment history condensed into decisions and rejected
    approaches, so that failed work remains learnable without dominating the repository.
76. As a maintainer, I want tracked editor metadata removed, so that repository state contains no
    personal IDE configuration.
77. As a maintainer, I want repository agent skills retained, so that review, TDD, research, and
    domain workflows remain available.
78. As a maintainer, I want a locked Python 3.13 environment managed by uv, so that Torch and all
    other dependencies resolve reproducibly.
79. As a maintainer, I want only interesting behavioral tests, so that the suite protects causal and
    financial contracts rather than parser defaults.
80. As a future live-trading operator, I want real order placement explicitly absent until Forward
    Paper Proof passes, so that unproven code cannot risk capital.

## Implementation Decisions

- This is a hard cut. Removed commands, configurations, artifacts, module interfaces, and domain
  concepts receive no backward-compatible aliases or migration layer.
- The product is one Python package with a thin CLI and one deep workflow module. The workflow
  interface owns data synchronization, validation, holdout evaluation, and paper operation. Callers
  supply only the fixed configuration plus path or device choices and receive result objects.
- The CLI exposes exactly four operational commands: data-sync, validate, holdout, and paper.
- One checked-in TOML configuration records policy-defining constants. Strategy, risk, feature,
  model, evidence, universe, timing, and cost decisions are not CLI overrides.
- The market-data module presents one canonical time-indexed dataset interface. A historical adapter
  reads cached/archive data; a live adapter reads public contemporaneous data. These are real
  adapters at one seam because replay and paper vary while downstream behavior must remain shared.
- The canonical dataset uses timezone-aware UTC exclusively. Decision Bars align to Binance
  15-minute UTC boundaries. Funding retains its actual UTC event timestamp.
- The Trading Universe is fixed to BTCUSDT, ETHUSDT, BNBUSDT, and SOLUSDT USD-M perpetuals.
- Historical synchronization requests maximum available Binance-native data from 2020 onward:
  perpetual OHLCV, spot OHLCV, funding, futures/open-interest metrics, premium or basis, taker
  activity, and trade count.
- Common Trading Start is the first timestamp at which all four perpetuals have execution-complete
  coverage. Training never uses a smaller universe before that time.
- Optional Market Inputs preserve rows through explicit source-availability masks. Required
  execution Reference Prices and scheduled funding cashflows cannot be missing, interpolated, or
  defaulted to zero; incomplete execution data fails validation and must be repaired.
- Spot/perpetual data is resampled causally into closed 15-minute Decision Bars. Market State includes
  fixed 15-minute, one-hour, four-hour, and one-day context.
- The fixed Market State includes returns, realized volatility, candle range and location, volume,
  taker imbalance, perpetual-spot basis or premium, funding, open-interest level and change,
  cross-asset market and relative returns, clock cycles, and missingness masks.
- Continuous inputs use causal rolling robust normalization. Normalization parameters at one Signal
  Time use only observations available at or before that time. Normalization choices are fixed policy
  configuration, not a search grid.
- Inputs explicitly exclude news, social sentiment, LLM signals, and macroeconomic feeds.
- The policy module receives Market State and Current Portfolio and returns four continuous Target
  Weights. It does not expose predictions, classes, thresholds, horizons, or hold-time choices to the
  workflow.
- Weight projection is part of the policy interface contract: absolute weight per ticker is at most
  0.50, Gross Exposure is at most 1.00, both signs are allowed, and unused exposure becomes Cash
  Weight. The projection is deterministic.
- The initial equity for every independent validation, holdout, and paper evidence run is $10,000
  under a Flat Start. Cash earns zero yield.
- A decision observes a fully closed 15-minute bar. Signal Time follows that close; a fill is not
  eligible until 60 seconds later.
- Historical Reference Price is the next eligible one-minute open. Paper Reference Price is the
  contemporaneous bid/ask midpoint at the same delayed time. Paper records bid and ask for audit.
- Effective Fill moves Reference Price 0.07% against trade direction. This is one all-in fee, spread,
  and slippage assumption and is never charged again as a second spread or fee line. Funding remains
  a separate timestamped cashflow.
- Desired aggregate turnover below 1% of current equity is not executed. It accumulates as distance
  between Current Portfolio and later Target Weights. At or above the threshold, the full resulting
  Portfolio Change executes and counts once at that decision timestamp.
- Marked Equity is recomputed at one-minute frequency from perpetual prices and due funding, while
  ordinary policy decisions remain 15-minute.
- Drawdown Limit is 20% from the high-water mark of the current evidence run. A causal breach fails
  the run, schedules a flat Target Weight at the next 60-second-delayed Effective Fill, and terminates
  further policy decisions. Realized exit latency may make final drawdown worse than 20%; it remains
  a failure.
- Training optimizes contiguous portfolio Net Log Growth rather than a fixed-horizon return label.
  Each step includes held-position return, executed turnover cost, and due funding.
- The training loss is negative cumulative Net Log Growth plus an augmented-Lagrangian penalty for
  drawdown above 20% on 90-day training episodes. The constraint multiplier is updated by the
  training procedure rather than selected as a model hyperparameter. Sharpe is diagnostic only.
- The linear direct policy is the sanity baseline. It uses the same Market State, Current Portfolio,
  projection, loss, execution, and evidence contracts as nonlinear candidates.
- The nonlinear candidate is a small causal temporal-convolution policy. The complete architecture
  search contains widths 32 and 64 crossed with one-day and seven-day receptive fields. No
  transformer, recurrent network, reinforcement-learning stack, Optuna study, or open model zoo is
  included.
- Each temporal-convolution architecture trains under exactly three fixed seeds. Median seed
  Compounded Net Return chooses the stable temporal-convolution architecture. The chosen three
  Fitted Policies produce one candidate by averaging their Target Weights before deterministic
  projection and execution.
- The actual three-seed ensemble and actual linear policy compete on validation Compounded Net
  Return. A candidate is eligible only when aggregate return is positive and no Walk-Forward Fold
  breaches the Drawdown Limit. Highest eligible compounded return wins.
- Validation contains twelve non-overlapping 90-day Walk-Forward Folds ending before the Historical
  Holdout on 2026-04-30. Training expands from Common Trading Start.
- Every fold starts flat. Its initial fit uses only pre-fold data. Policy Protocol retraining occurs
  weekly on Sunday UTC and may use only fold observations already revealed. Fitted scaler and model
  state are never shared backward or across unrevealed time.
- Purged Boundaries remove every target/reward transition and training sequence that would cross a
  scored split boundary.
- Validation concatenates or compounds only genuinely scored fold returns. It records per-fold and
  aggregate return, drawdown, turnover, costs, funding, exposure, Risk Stop state, and diagnostics.
- A non-positive best validation return or any per-fold Drawdown Limit breach leaves validation
  failed and prevents initial holdout evaluation.
- The initial Historical Holdout is 2026-05-01 through 2026-07-31 UTC. Its data may be synchronized
  but cannot inform feature, architecture, objective, risk, or search decisions before the Policy
  Protocol is frozen.
- Holdout evaluation is prequential. The initial Fitted Policy uses data through 2026-04-30;
  scheduled Sunday refits may use only the holdout prefix already revealed. This does not revise the
  protocol.
- Holdout state is immutable and single-use for the frozen protocol/period/data identity. A repeated
  holdout invocation cannot perform new selection or create a new supposedly untouched result; it
  returns or points to the recorded result.
- A positive holdout return with no Drawdown Limit breach produces a Holdout-Passing Protocol. A
  failed holdout becomes a Consumed Holdout and Development Evidence forever.
- After a failed holdout, development may continue against all prior folds plus the consumed period
  until another Validated Policy Protocol exists. The consumed period is never relabeled as a
  Historical Holdout. Forward Paper Proof is the only fresh final evidence for such revisions.
- Forward Paper Proof starts flat, consumes public data only, and runs the complete frozen Policy
  Protocol including Sunday retraining, weight projection, minimum turnover, fills, funding,
  one-minute risk marks, and Risk Stop.
- Forward Paper Proof must run for at least 60 days and record at least 100 Qualifying Portfolio
  Changes; completion occurs only when both requirements are met. It passes only with positive
  Compounded Net Return and no Drawdown Limit breach.
- One Qualifying Portfolio Change is one decision timestamp with aggregate executed turnover of at
  least 1% of equity, regardless of order count.
- Routine weekly retraining changes Fitted Policy weights but does not reset Proof Clock. Any change
  to features, architecture, objective, constraints, training procedure, retraining schedule,
  execution, costs, evidence rules, or fixed configuration is a Policy Revision and starts a new
  Proof Clock.
- A failed paper run becomes Development Evidence. Any resulting Policy Revision must repeat the
  complete 60-day and 100-change Forward Paper Proof.
- Paper mode has no private-key, account-authentication, or real-order adapter. Real-money execution
  remains absent until a separate post-proof decision.
- The artifacts module writes only an immutable manifest, structured report, equity series, trade
  records, and frozen linear or ensemble model artifacts. The manifest hashes configuration, code,
  canonical data snapshot, and model identity.
- Stored JSONL decision ledgers and Plotly HTML timelines are removed. Any visualization is generated
  on demand from equity and trades.
- The project migrates from requirements files to a Python 3.13 uv project with a committed universal
  lockfile. Torch is CPU-only. Runtime, test, and development dependencies are declared separately.
- The source tree becomes one focused package with modules for CLI, workflow, market data, policy,
  simulation, and artifacts. The generic utility bag and independent runner-script pattern are
  removed.
- The old learned one-minute runner, data builder, policy module, model/target grids, old CLI tests,
  duplicate downloaders, legacy experiment helpers, old artifact writers, and obsolete concepts are
  deleted rather than wrapped.
- Existing raw market data is retained. All 132 old generated run directories, currently about 4.9
  GB, are permanently deleted after durable lessons are extracted.
- The experiment log is replaced by a concise objective/constraint header, a compact table of prior
  rejected approaches, and append-only new experiment entries containing exact configuration hash,
  code hash, data hash, metrics, and verdict.
- Superseded architecture documentation and obsolete scratch issues are deleted. The accepted
  direct-net-growth ADR, domain glossary, project README, agent instructions, agent process docs, and
  repository-local agent skills remain.
- Tracked IDE metadata is deleted and ignored. Disposable Python and test caches remain ignored.
- README describes only the new installation, data-sync, validate, holdout, paper, artifacts, and
  proof-state workflow. It does not advertise removed commands.

## Testing Decisions

- Tests exercise externally observable financial and evidence behavior. They do not assert private
  helper structure, parser defaults, every configuration field, or implementation details of Torch
  layers.
- The highest test seam is the deep workflow interface. End-to-end tests invoke workflow operations
  with small deterministic market-data adapters and inspect result objects and artifacts. CLI tests
  are limited to command routing and rejection of forbidden strategy overrides.
- The canonical market-data interface is tested once with both historical and live fake adapters to
  prove they emit the same dataset contract. Adapter-specific parsing tests exist only for genuinely
  different source behavior.
- The simulation interface is a pure behavioral seam for fills, funding, costs, Current Portfolio,
  Marked Equity, Qualifying Portfolio Changes, and Risk Stop. It receives deterministic policy
  outputs and market data and returns a replay result without filesystem or network effects.
- The policy interface is tested for deterministic projection, Gross Exposure, per-ticker
  concentration, both long and short weights, cash remainder, and ensemble-before-projection
  behavior. Tests do not inspect layer internals.
- A causal-data test proves a Decision Bar contains only closed information and every multi-timeframe
  feature at Signal Time is unchanged when future rows are mutated.
- A resampling test proves 15-minute, one-hour, four-hour, and one-day features align to UTC close
  boundaries without partial-bar leakage.
- A missing-data test proves Optional Market Inputs receive masks while a missing required Reference
  Price or funding cashflow fails canonical dataset validation.
- A Common Trading Start test proves no smaller historical universe is emitted before all four
  instruments are executable.
- An execution test proves Signal Time, 60-second latency, historical one-minute Reference Price,
  and adverse 0.07% Effective Fill for long and short changes.
- A paper-fill test proves midpoint reference and diagnostic bid/ask do not double-charge spread.
- A funding test proves signed funding is applied to the correct held position at its UTC timestamp.
- A turnover test proves sub-1% desired changes accumulate without fills and crossing 1% executes one
  portfolio-level qualifying event.
- A continuity test proves weekly Policy Handoff preserves Current Portfolio and does not create a
  forced flatten or Proof Clock reset.
- A Flat Start test proves each independent fold, holdout, and paper proof begins in cash.
- A risk test proves equity marks every minute, Drawdown Limit uses high-water Marked Equity, breach
  schedules flatten after latency, exit costs apply, and no later decisions occur.
- A causality test proves weekly fold and holdout refits can use revealed prefixes but cannot use any
  future scored interval.
- A purge test proves no reward transition or receptive-field sequence crosses a Walk-Forward Fold or
  Historical Holdout boundary.
- A candidate-selection test proves unstable seed success cannot select architecture, the actual
  ensemble competes with linear, non-positive return is ineligible, and any fold drawdown breach
  rejects the candidate.
- A holdout-state test proves validation failure keeps holdout locked, one successful validation
  unlocks exactly one initial evaluation, repeat invocation cannot produce another untouched result,
  and a failed result becomes Consumed Holdout.
- A proof-state test proves scheduled Fitted Policy retraining preserves proof identity, Policy
  Revision resets Proof Clock, and failed paper evidence cannot be combined with a revision.
- A paper-completion test proves both 60 elapsed days and 100 Qualifying Portfolio Changes are
  required, positive net return is required, and Drawdown Limit breach fails immediately.
- A reproducibility test proves equal configuration, code, data, and seed identities yield equal
  manifests and deterministic result hashes.
- One synthetic end-to-end workflow test supplies a causal learnable market pattern and proves the
  direct policy can train, validate, execute after costs, remain within constraints, and emit the
  minimal artifact set.
- Existing tests for next-open causality, OHLC validation, timestamp conversion, long-only weight
  caps, walk-forward folds, and run hashing are prior art. Retain or rewrite only the behavioral
  insight that still matches this spec; delete old target-family, threshold, parser-default, and
  artifact-shape tests.
- Full verification runs through the locked uv environment on CPU and includes format/lint checks,
  unit and workflow tests, plus a tiny deterministic training smoke test. Full historical training is
  an experiment, not a unit-test prerequisite.

## Out of Scope

- Real-money order placement, authenticated Binance endpoints, API secrets, account reconciliation,
  exchange order lifecycle, and production trading deployment.
- Leverage above 1.0 Gross Exposure, margin optimization, liquidation modeling for leveraged
  positions, options, dated futures, or non-Binance venues.
- Dynamic ticker discovery, broad altcoin scanning, instruments outside BTCUSDT, ETHUSDT, BNBUSDT,
  and SOLUSDT, or changing portfolio shape before instrument listing.
- News, social sentiment, LLM-derived signals, macroeconomic feeds, or paid external datasets.
- Transformers, recurrent models, reinforcement learning, architecture search beyond the declared
  linear and temporal-convolution candidates, or automatic hyperparameter optimization.
- Fixed-horizon price labels, direction classifiers, cross-sectional rank targets, threshold grids,
  hold-time grids, buy-and-hold proof gates, Sharpe optimization, or prediction-accuracy objectives.
- Compatibility with old one-minute, rank-selector, guarded-regime, historical paper, or online paper
  commands and artifacts.
- Persisted interactive charts, minute-by-minute decision ledgers, dashboards, web interfaces, or
  alerting beyond command output and structured artifacts.
- Claiming a Consumed Holdout as final proof or combining paper evidence from different Policy
  Protocols.
- Stablecoin yield, collateral rewards, lending, staking, or profit sources outside instrument PnL
  and perpetual funding.
- Proving live profitability. A Profitable Policy in this context means completed Forward Paper
  Proof; live-capital authorization is a later decision.

## Further Notes

- Domain vocabulary is defined by the project glossary. Implementation and tickets must use Trading
  Policy, Policy Protocol, Fitted Policy, Policy Revision, Target Weight, Qualifying Portfolio Change,
  Causal Replay, Historical Holdout, Consumed Holdout, Prequential Evaluation, Forward Paper Proof,
  Proof Clock, and Risk Stop exactly as defined there.
- The accepted architecture decision mandates direct Net Log Growth, continuous weights, causal
  15-minute operation, purged evidence, Historical Holdout, Forward Paper Proof, and no compatibility
  path.
- The workflow interface is the primary test seam accepted during design. Market-data adapters and
  pure simulation are supporting seams because behavior genuinely varies there or requires isolated
  financial verification.
- No prototype was required. Technical feasibility is straightforward; whether Market State contains
  profitable information is answered by validation, holdout, and paper experiments rather than a
  throwaway implementation.
- Every actual model experiment must be recorded in the condensed experiment log. Infrastructure
  tests and purely mechanical cleanup are not trading experiments.
- Old generated run deletion is intentional and permanent. Raw data remains locally cached.
