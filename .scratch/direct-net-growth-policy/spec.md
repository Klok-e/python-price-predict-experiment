# Direct Net-Growth Portfolio Policy Hard Cut (Superseded)

Status: wontfix

This specification produced the causal, cost-aware Trading Policy and Historical Holdout workflows
that remain as Development Evidence. Its proof-oriented paper operator and delivery gate are no
longer active requirements.

The persistent Paper Account specification at
`.scratch/paper-trading-dashboard/spec.md` supersedes paper operation, lifecycle, metrics, fitting,
and service delivery. ADR-0005 records the hard-cut decision: Paper Account results are Development
Evidence, not proof of real-world profitability, and no proof-clock compatibility path remains.

The retained scope from this effort is the fixed Trading Universe, direct Target Weights, causal
Market State, transaction costs and funding, No Leverage, concentration and Drawdown Limit
semantics, scheduled fitting, validation, and the single-use Historical Holdout.
