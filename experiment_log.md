# Experiment Log

## Objective and acceptance contract

- Primary objective: maximize Compounded Net Return after all-in turnover cost and actual funding.
- Hard risk constraint: every evidence run must remain within the 20% Drawdown Limit.
- Final proof: at least 60 days and 100 Qualifying Portfolio Changes of fresh Forward Paper Proof,
  with positive net return and no Risk Stop.
- Statistical prediction quality and buy-and-hold outperformance are diagnostics, not proof gates.

## Rejected approaches

| Approach | Durable lesson | Verdict |
| --- | --- | --- |
| One-minute fixed-horizon price prediction | Large target, model, threshold, horizon, and hold grids produced fragile selection surfaces; retained results failed rolling proof after costs. | Rejected |
| Long-only threshold execution | Proxy predictions and forced directional exposure did not optimize the portfolio objective and could not express short or cash conviction. | Rejected |
| Rank and regime selection | Some isolated holdouts looked strong, but results were not stable across chronological regimes and did not establish fresh paper profitability. | Rejected |
| Buy-and-hold excess gates | Beating a benchmark is not the capital objective; a losing strategy can still beat a worse benchmark. | Rejected |
| Stored ledgers and HTML timelines | High-volume artifacts obscured the reproducible evidence contract and consumed unnecessary storage. | Rejected |

## New experiment entry format

Append each actual model experiment below. Infrastructure tests are not experiments.

### YYYY-MM-DD - Short name

- Configuration hash:
- Code hash:
- Data hash:
- Model hash:
- Walk-Forward Compounded Net Return:
- Maximum Drawdown by fold:
- Turnover / costs / funding:
- Historical Holdout or Forward Paper Proof state:
- Verdict:
- Notes:
