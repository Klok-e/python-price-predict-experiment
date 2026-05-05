# Cryptocurrency Price Prediction

This context describes the operational rank-selector workflow: how market data becomes return-rank
features, how the market-regime selector chooses the active model/allocation policy, how the guarded
paper replay policy can override that selection from validation-only evidence, and how historical
replay and online paper trading record decisions.

## Language

**Validated Bar Series**:
A ticker's OHLC time series after duplicate, missing, non-monotonic, and invalid-bar checks pass.
_Avoid_: Trusted raw download

**Return-Rank Model**:
A regression model that scores each ticker by expected future return so the portfolio can rank long
and short allocation candidates.
_Avoid_: Direction-only classifier, static winner model

**Rank Dataset**:
The aligned per-ticker dataset containing resampled OHLCV bars, causal base features,
market-relative features, optional futures metrics, optional premium index features, and future-return
targets.
_Avoid_: Per-script feature copy

**Market-Regime Selector**:
The active policy selector that trains both `ridge` and `hist_gradient_boosting`, then uses
validation-period market return and boundary momentum to choose the model family and allocation
parameters before evaluation or paper trading.
_Avoid_: Best single static model

**Guarded Market-Regime Policy**:
The paper replay/forward policy that starts from the **Market-Regime Selector**, then applies
validation-only regime guards such as broad long exposure in strong bull regimes or no-trade
allocation after weak validation evidence.
_Avoid_: Post-window hindsight switch

**Training Period**:
The historical time span used to fit rank-model weights.
_Avoid_: In-sample data

**Validation Period**:
The chronological holdout used for model-family and allocation-policy selection before final
evaluation or a forward paper decision.
_Avoid_: Final test, reporting period

**Evaluation Period**:
The future time span used to score the selected policy after validation choices are fixed.
_Avoid_: Validation tail

**Final Holdout Objective**:
The primary selector objective: beat buy-and-hold on the most recent evaluation window, currently the
last 90 days, without training or selecting parameters on that window.
_Avoid_: Average rolling result as the primary goal

**Strict Temporal Split**:
The rule that future data must not influence training, model selection, allocation selection, or paper
decisions.
_Avoid_: Full-history fitting, post-split lookahead

**Next-Open Fill**:
The execution rule that a signal computed from candle `t` enters at candle `t+1` open.
_Avoid_: Current-close fill

**Closed Signal Bar**:
The completed candle used for an online paper prediction. Online paper trading must not use a partial
current candle as signal evidence.
_Avoid_: Latest cached partial bar

**Trading Baseline**:
A reference portfolio, currently buy-and-hold or no-trade, that the selected rank policy must beat
after fees.
_Avoid_: Sanity check only

**Online Paper Trading**:
The live simulation workflow that repeatedly fills pending paper orders, marks open positions to
market, and appends new paper decisions from the latest **Closed Signal Bar**.
_Avoid_: Historical replay loop

**Paper Ledger**:
The append-only JSONL record of strategy selections, predictions, target weights, simulated orders,
fills, equity snapshots, and no-action reasons for historical replay or forward paper trading.
_Avoid_: Mutable paper state file

**Cash/Units Paper State**:
The online paper account state where cash and per-ticker units are authoritative, while weights,
exposures, equity, and PnL are derived from current mark prices.
_Avoid_: Weight-only account state

**Pending Paper Order**:
A simulated target-weight order created from a **Closed Signal Bar** and waiting for next-open data
before it can become an online paper fill.
_Avoid_: Real exchange order

**No-Action Record**:
A ledger entry that records why no simulated order/fill was produced, including stale data,
missing next-open fill data, not-yet-reselect, not-yet-rebalance, validation-ineligible, or
already-processed decisions.
_Avoid_: Silent skip

## Relationships

- A **Validated Bar Series** is the input to the **Rank Dataset**.
- A **Rank Dataset** uses strict index alignment between features and future-return targets.
- A **Return-Rank Model** is trained only on the **Training Period**.
- The **Market-Regime Selector** makes choices only from pre-evaluation **Validation Period**
  evidence.
- The **Guarded Market-Regime Policy** may override allocation parameters, but only from
  pre-evaluation **Validation Period** evidence.
- A selected policy is evaluated on the **Evaluation Period** against **Trading Baseline** returns.
- The **Final Holdout Objective** is primary; older rolling windows are robustness evidence.
- Paper replay and **Online Paper Trading** use **Next-Open Fill** semantics and fee-aware turnover
  costs.
- Online paper trading creates **Pending Paper Order** records before the next-open fill exists.
- A **Paper Ledger** is append-only and reduced into historical replay state or **Cash/Units Paper
  State**.
- Duplicate paper-forward decisions are blocked by idempotency keys and append zero new records.
- Data quality, stale data, and missing next-open data produce **No-Action Record** entries instead
  of implicit trades.
- A daemon poll with no new **Closed Signal Bar** is not a **No-Action Record**; it is no ledger event.

## Current Winner

The active winner is:

`computed-data/runs/f1ab217c947e/rank_signal_report.json`

The active historical paper replay winner is:

`computed-data/runs/434a6f82d8ce/paper_replay_report.json`

That report uses:

- `model_family=both`
- `selector_policy=guarded-market-regime`
- `include_futures_metrics=true`
- `include_premium_index=true`
- rolling train/validation/evaluation windows

The report fields that matter operationally are the final holdout model cumulative return,
buy-and-hold cumulative return, model minus buy-and-hold, model minus no-trade, trade count, and
rolling pass/fail gates as robustness context.

For historical replay, the operational gates are the replay aggregate and the non-overlapping 90-day
windows: model cumulative return, buy-and-hold cumulative return, model minus buy-and-hold, model
minus no-trade, trade count, and the paper ledger artifacts.

## Flagged Ambiguities

- "This model selector" means the **Market-Regime Selector**, not one frozen model family.
- The old MLP/barrier-label workflow is removed from the operational surface.
- Legacy trade-list metrics based on `EntryTime`, `ExitTime`, and `ReturnPct` are removed; surviving
  evaluation is report-aggregate and **Paper Ledger** state based.
