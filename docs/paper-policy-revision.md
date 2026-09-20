# Preparing and activating the Signal-Time Policy Revision

Preparation and verification do not activate the revision. Keep the incumbent service running from
its unchanged checkout while preparing in an isolated checkout. Never run two operators against
the same Paper Account database.

## Prepare the candidate

Use the incumbent checkpoint to retain its architecture and the configured training budget. Use a
copy of the public dataset and prior evidence state. The configured development period includes
already consumed Historical Holdout data; preparation checks that prior evidence exists and does
not run another holdout or architecture search.

```bash
uv run netgrowth prepare-revision \
  --config policy.toml --device cuda \
  --data-directory computed-data/dataset \
  --checkpoint computed-data/incumbent/model.pt \
  --evidence-state computed-data/prior-evidence-state.json \
  --output-directory computed-data/revision
```

Each completed fold is cached by Policy Protocol, dataset identity, incumbent checkpoint identity,
and fold index. An interrupted preparation resumes completed folds. `validation.json` records all
twelve chronological folds; `revision.json` and its immutable checkpoint are produced only when
compounded fold returns are positive and every fold stays within the 20% Drawdown Limit. Failed
validation retains its evidence and produces no new activation bundle. Check the bundle's
`training_observed_at`: it is the dataset cutoff, distinct from `fitted_at`, the actual fitting time.
Refresh stale source data and prepare again before a later production activation.

## Approved deployment procedure

1. Confirm local checks and the candidate validation report. Retain the incumbent checkout,
   checkpoint, account database, WAL-safe backup, and service configuration for recovery.
2. Stop the incumbent service before switching its executable checkout. Preserve the existing
   operational directory; do not use a new database or Manual Reset.
3. Start the revised executable with the same config, data directory, operational directory, host,
   and device. The checked-in unit reads
   `computed-data/paper-dashboard/active-revision.json` with `--revision-bundle`. Install the
   validated bundle there, retaining its checksum-matched checkpoint and using an absolute
   checkpoint path. The bundle and checkpoint must be readable by the service user.
4. Startup validates the bundle and copies its checkpoint into the operational checkpoint store.
   A staged revision survives restart without the external bundle. Pending legacy execution drains
   under its recorded old eligibility rule. No new decisions occur during the drain; market marks,
   funding, expiry, and risk controls continue. Startup model preparation also defers new
   decisions and activation while leaving market marks and dashboard reads available. Activation
   waits for preparation to finish and a fresh observed mark.
5. Verify one Policy Revision boundary, unchanged account identity, preserved positions/history,
   the candidate model identity, and an initialized Hold Benchmark with identical boundary equity
   and quantities. Confirm subsequent Decision Records report Signal-Time eligibility and History
   separates the revision segments. System shows the known model fit time and retry schedule.
6. Once activation is recorded, keep the revised executable. An executable rollback across the
   revision boundary requires a reviewed recovery procedure; never silently relabel the new state
   as the old protocol or overwrite the database with an older snapshot.

The Hold Benchmark retains its boundary quantities, includes subsequent funding, and makes no
trades. Account-minus-Hold performance therefore includes the costs and results of subsequent
policy decisions. It is a forward diagnostic, not a validation gate.

## Recovery behavior

Scheduled fit failures retain the active model and retry after 5, 15, 30, then successive 60-minute
delays. The cycle, count, and deadline survive restart, and only one fit runs at a time. Desktop
notification delivery runs outside market advancement and failed delivery has a durable retry
schedule. System distinguishes outstanding retries from historical failures.

Each new Decision Record atomically retains its exact prepared attribution input and model
identity. Recovery uses this retained input, not a refetched or corrected market history. Successful
attribution releases the large payload while retaining identity metadata; failures retain it for
inspection. Legacy decisions without retained inputs report attribution unavailable explicitly.
Historical invalid Operating Window timestamps remain intact and are annotated as invalid;
newly closed windows cannot end before their start. Schema migration creates a backup first.
