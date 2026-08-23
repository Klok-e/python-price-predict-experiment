# Persistent Paper Trading Dashboard

Status: ready-for-agent

## Problem Statement

The user wants to operate the existing Trading Policy for several hours when the workstation is on
and quickly understand what the simulated account is doing. The current paper operator is designed
as an uninterrupted Forward Paper Proof: it has no browser UI, treats continuity gaps as proof
failures, persists proof-oriented CSV artifacts, and requires a 60-day/100-change gate that the user
has explicitly abandoned. It does not provide a convenient live account view, durable manually
controlled lifecycle, model explanations, or browsable history.

The replacement must remain strictly paper-only while behaving like a persistent account. A crash,
service restart, or workstation shutdown must not silently reset cash or positions. Open positions
remain economically exposed during downtime, while the policy makes no decisions and performs no
trades when the service is off. The user needs trustworthy performance and risk metrics, explicit
distinction between policy behavior and human controls, chart-linked explanations of when trades
happened and what influenced them, and permanent local history that is easy to inspect.

## Solution

Hard-cut the Forward Paper Proof runtime into one persistent local Paper Account operated by an
enabled lingering user systemd service. The service starts whenever the workstation is on and trades
without requiring a browser to be open. It preserves account state through process and machine
restarts until Manual Reset, records every Operating Window and gap, reconstructs objectively
recoverable one-minute marks and funding after downtime, never replays missed decisions, and never
invents an unavailable historical executable quote.

Run the autonomous trader, background GPU fitting, control API, persistence, and dashboard in one
Python process. Use a single SQLite WAL writer in an ignored generated-data directory inside the
repository for append-only account history, durable control state, recovery snapshots, model
identity, and archived Paper Accounts. Serve a localhost-only, build-free browser dashboard with
Live, History, and System views. Make exact Decision Records and execution outcomes authoritative;
supplement each decision with an asynchronous, clearly approximate Model Attribution.

Preserve the existing Trading Policy contract: fully closed 15-minute Decision Bars, 60-second
Decision Latency, fresh public Binance USD-M bid/ask Reference Prices, a 0.07% adverse all-in
Transaction Cost, actual published funding, one-minute marking, No Leverage, concentration limits,
and a 20% Drawdown Limit. Keep historical validation and Historical Holdout artifacts as Development
Evidence and bootstrap material, but remove the Proof Clock, proof pass/fail gate, proof counters,
and every claim that paper results establish real-world profitability.

## User Stories

1. As a paper operator, I want the trader to start automatically when my workstation is on, so that
   I do not have to remember to launch it each evening.
2. As a paper operator, I want trading to continue when the browser is closed, so that viewing the
   dashboard is not part of execution timing.
3. As a paper operator, I want a stable localhost URL, so that I can bookmark and open the dashboard
   whenever I want to check it.
4. As a security-conscious operator, I want the dashboard available only on this machine, so that
   account controls are not exposed to the LAN or internet.
5. As a paper operator, I want the service to run without root privileges, so that the experiment
   cannot mutate system state unnecessarily.
6. As a paper operator, I want the browser to open only when I choose, so that boot does not create
   an unwanted window.
7. As a paper operator, I want one Paper Account to persist until I explicitly reset it, so that
   crashes and shutdowns do not destroy the experiment I am watching.
8. As a paper operator, I want the initial Paper Account to start flat with $10,000, so that its
   performance has a clear baseline.
9. As a paper operator, I want cash, positions, pending state, and history restored after a process
   crash, so that systemd recovery continues rather than restarts the account.
10. As a paper operator, I want cash, positions, pending state, and history restored after a machine
    reboot, so that an evening-only workstation can maintain a long-lived account.
11. As a paper operator, I want every service-up period recorded as an Operating Window, so that I
    can distinguish active trading time from downtime.
12. As a paper operator, I want downtime shown as a shaded graph region, so that missing live
    observation periods are visually obvious.
13. As a capital owner, I want open positions to remain exposed while the service is off, so that
    the Paper Account does not receive an unrealistic free pause from market movement.
14. As a capital owner, I want published funding during downtime applied to open positions, so that
    perpetual carry remains economically realistic.
15. As a capital owner, I want recoverable one-minute marks during downtime reconstructed and
    labeled, so that equity and drawdown include the path the positions actually experienced.
16. As a paper operator, I want no decisions replayed for downtime, so that the policy never acts on
    information it did not observe live.
17. As a paper operator, I want no fills invented during downtime, so that unavailable historical
    bid/ask snapshots are not presented as observed execution.
18. As a paper operator, I want a pending Portfolio Change to survive a short crash, so that a brief
    restart does not needlessly lose a valid decision.
19. As a paper operator, I want a pending Portfolio Change filled only when a fresh quote arrives
    within two minutes of its scheduled fill, so that stale signals are not executed much later.
20. As a paper operator, I want an overdue pending Portfolio Change recorded as a Missed Execution,
    so that a non-fill cannot disappear from account history.
21. As a paper operator, I want the account to retain its Current Portfolio after a Missed Execution,
    so that recovery does not fabricate a position change.
22. As a paper operator, I want Binance data outages to leave the dashboard available, so that I can
    see what is wrong instead of finding a dead service.
23. As a paper operator, I want stale data status and observation age displayed prominently, so that
    I do not mistake old metrics for live metrics.
24. As a paper operator, I want new decisions and fills suspended while data is stale, so that the
    service never trades without fresh required inputs.
25. As a paper operator, I want data retries to happen in process with bounded backoff, so that a
    routine provider outage does not create a systemd crash loop.
26. As a paper operator, I want recovery from a data outage to continue the same Paper Account, so
    that transient availability never causes an implicit reset.
27. As a researcher, I want every policy decision based on a fully closed 15-minute Decision Bar, so
    that partial candles cannot influence Target Weights.
28. As a researcher, I want normal policy fills delayed by 60 seconds, so that the paper runtime
    retains the Policy Protocol's execution assumption.
29. As a researcher, I want fresh public Binance USD-M bid/ask quotes used as Reference Prices, so
    that fills are tied to contemporaneously observable market data.
30. As a capital owner, I want Effective Fill moved 0.07% against the trade direction, so that fees,
    spread, and slippage remain charged once under the established contract.
31. As a capital owner, I want actual funding payments recorded separately from trading costs, so
    that I can see the distinct sources of account return.
32. As a security-conscious operator, I want no exchange account, API key, or testnet order
    integration, so that the dashboard cannot place real orders or expose credentials.
33. As a paper operator, I want desired turnover below the configured minimum to accumulate rather
    than execute, so that the policy does not create meaningless micro-fills.
34. As a paper operator, I want an Executable Portfolio Change clearly distinguished from a no-trade
    decision, so that activity metrics describe actual account mutations.
35. As a capital owner, I want account equity marked every minute, so that 15-minute decisions do
    not hide intrabar risk.
36. As a capital owner, I want the existing No Leverage and per-instrument concentration limits
    enforced, so that dashboard operation cannot exceed the tested risk contract.
37. As a capital owner, I want a 20% Drawdown Limit to trigger Risk Stop, so that unacceptable loss
    causes terminal flattening rather than a warning only.
38. As a capital owner, I want Risk Stop to prevent every later policy decision for that Paper
    Account, so that a normal Resume cannot bypass it.
39. As a paper operator, I want a Risk-Stopped account to require Manual Reset, so that returning to
    trading is explicit and historically visible.
40. As a paper operator, I want Pause to stop new decisions and trades while retaining positions, so
    that pausing does not quietly change exposure.
41. As a paper operator, I want Flatten and Pause to close positions at the next fresh quote, so that
    I have a deliberate safety action distinct from an ordinary pause.
42. As a paper operator, I want operator-requested flattening to skip policy Decision Latency, so
    that a safety or lifecycle action is executed as soon as a fresh quote is available.
43. As a paper operator, I want normal Transaction Cost applied to operator flattening, so that
    intervention does not receive free execution.
44. As a researcher, I want every pause, resume, flatten, and reset recorded as an Operator
    Intervention, so that human actions are never attributed to the Trading Policy.
45. As a paper operator, I want Pause and Resume to be immediate and idempotent, so that repeated
    clicks cannot enqueue duplicate actions.
46. As a paper operator, I want Flatten and Pause to show current exposure and estimated cost before
    confirmation, so that I understand the mutation I am authorizing.
47. As a paper operator, I want Manual Reset to flatten open positions before archival, so that
    positions cannot disappear from history without an Effective Fill.
48. As a paper operator, I want an account shown as Reset Pending until flattening succeeds, so that
    a network outage cannot masquerade as a completed reset.
49. As a paper operator, I want Manual Reset to require a second clear confirmation, so that I do
    not end the active account accidentally.
50. As a paper operator, I want Manual Reset to archive rather than delete the old account, so that
    all prior history remains inspectable.
51. As a paper operator, I want the new account created only after the old account is flat and
    archived, so that there is exactly one unambiguous active Paper Account.
52. As a paper operator, I want Paused, Reset Pending, Risk Stopped, and Migration Required states to
    survive restarts, so that a reboot never implies Resume.
53. As a paper operator, I want Resume available only for an ordinary paused account, so that it
    cannot bypass risk, reset, or migration requirements.
54. As a researcher, I want weekly fitting to begin in the first Operating Window after its deadline,
    so that an offline workstation does not permanently miss scheduled adaptation.
55. As a paper operator, I want weekly fitting to run in the background on the GPU, so that market
    observation and existing-policy trading continue while training runs.
56. As a paper operator, I want the current Fitted Policy to remain active until a replacement is
    complete, so that a partial training run cannot affect decisions.
57. As a researcher, I want Policy Handoff to be atomic and preserve Current Portfolio, so that
    weekly fitting does not force liquidation or create mixed model state.
58. As a paper operator, I want a fitting failure shown without resetting or stopping the account,
    so that the last valid Fitted Policy can continue.
59. As a researcher, I want every Fitted Policy identity and Policy Handoff stored in history, so
    that past decisions can be attributed to the exact model that made them.
60. As a researcher, I want a compatible Policy Revision to continue the Paper Account with a
    visible revision boundary, so that reset remains a manual choice.
61. As a researcher, I want performance segmented by Policy Protocol across compatible revisions,
    so that combined account return does not hide which protocol produced it.
62. As a capital owner, I want an incompatible Policy Revision to enter Migration Required instead
    of transforming positions automatically, so that universe or accounting changes cannot corrupt
    the account.
63. As a paper operator, I want every 15-minute decision stored as a Decision Record, so that I can
    inspect both trading and no-trade outcomes.
64. As a paper operator, I want each Decision Record to include Market State identity, Current
    Portfolio, Target Weights, model identity, turnover, thresholds, timing, and outcome, so that the
    record is auditable.
65. As a paper operator, I want signal time and fill time represented separately, so that the chart
    does not imply instant execution.
66. As a paper operator, I want a decision linked to every resulting fill or Missed Execution, so
    that I can trace an account mutation back to its source.
67. As a paper operator, I want decisions that remained below the turnover threshold explained, so
    that no-trade behavior is visible rather than absent.
68. As a paper operator, I want decisions that kept effectively unchanged exposure explained, so
    that I can distinguish policy conviction from an operational failure.
69. As a paper operator, I want a Model Attribution for every decision, so that I can inspect what
    inputs most influenced each Target Weight.
70. As a researcher, I want Model Attribution clearly labeled as approximate influence evidence, so
    that it is never mistaken for a causal rule or model intent.
71. As a researcher, I want attribution grouped into momentum, volatility/range, flow/activity,
    derivatives positioning/carry, availability, Current Portfolio, and time regions, so that a TCN
    explanation is readable.
72. As a researcher, I want Integrated Gradients measured against a neutral normalized-market
    baseline, so that positive and negative influences have a consistent reference.
73. As a paper operator, I want attribution computed asynchronously after a decision, so that an
    explanation can never delay its scheduled fill.
74. As a paper operator, I want the dashboard to show Explanation Pending when necessary, so that
    asynchronous work is explicit rather than appearing broken.
75. As a researcher, I want top signed influences, attribution parameters, input hash, and model
    hash retained, so that the summary is reproducible.
76. As a researcher, I want full attribution recomputable from canonical input and immutable model
    identity, so that the database need not store a huge raw tensor for every decision.
77. As a paper operator, I want a Live view with account status, risk, positions, charts, pending
    actions, and recent activity, so that the current situation is easy to scan.
78. As a paper operator, I want a History view with decisions, fills, funding, gaps, interventions,
    handoffs, and archived accounts, so that I can investigate past behavior.
79. As a paper operator, I want a System view with data freshness, errors, fitting, model identity,
    Operating Windows, backups, and service diagnostics, so that operational health is separate from
    investment performance.
80. As a paper operator, I want to select one ticker at a time on the main price chart, so that four
    incompatible price scales do not obscure trades.
81. As a paper operator, I want candlesticks annotated with signal, fill, Missed Execution, funding,
    and Operator Intervention markers, so that meaningful events appear where they occurred.
82. As a paper operator, I want Current and Target Weight beneath the selected price chart, so that
    position intent is visible alongside price.
83. As a paper operator, I want the price crosshair synchronized with equity, benchmark, drawdown,
    and exposure panels, so that market and account state can be inspected at one time.
84. As a paper operator, I want current-window, 24-hour, 7-day, and full-account chart ranges, so
    that both recent behavior and long history are accessible.
85. As a paper operator, I want clicking an event marker to open its exact Decision Record and Model
    Attribution, so that the graph is an entry point into the audit trail.
86. As a paper operator in Kyiv, I want times displayed in Europe/Kiev by default, so that evening
    operation is easy to read.
87. As a researcher, I want UTC shown in details and used for storage and calculation, so that every
    event remains unambiguous and reproducible.
88. As a capital owner, I want starting equity, current equity, net P&L, and Compounded Net Return,
    so that I can see whether the Paper Account made money.
89. As a capital owner, I want gross trading P&L, Transaction Cost, funding paid or received, and
    turnover shown separately, so that I can understand return composition.
90. As a capital owner, I want current drawdown, Maximum Drawdown, high-water equity, and the 20%
    limit shown together, so that current and historical risk are obvious.
91. As a capital owner, I want gross exposure, net exposure, Cash Weight, and concentration shown,
    so that portfolio risk is visible beyond P&L.
92. As a paper operator, I want side, quantity, mark, notional, Current Weight, Target Weight, average
    entry, and unrealized P&L for every instrument, so that open positions are understandable.
93. As a researcher, I want average-cost position accounting with explicit reversal handling, so
    that realized and unrealized P&L are consistent across continuous rebalancing.
94. As a researcher, I want Marked Equity to remain authoritative over position-level P&L summaries,
    so that a presentation convention cannot change account truth.
95. As a paper operator, I want counts of decisions, Executable Portfolio Changes, fills, Missed
    Executions, and Operator Interventions, so that activity is transparent.
96. As a paper operator, I want active model identity, last and next fit, fitting progress, failures,
    and handoff history, so that I know which policy is operating.
97. As a capital owner, I want a passive equal-weight-long benchmark started with 25% in each ticker,
    so that the policy is compared with simply holding the same Trading Universe.
98. As a capital owner, I want benchmark Transaction Cost, funding, and downtime exposure applied,
    so that the comparison is not artificially favorable or unfavorable.
99. As a capital owner, I want cash shown as a zero-return baseline, so that positive and negative
    performance are immediately interpretable.
100. As a researcher, I do not want prediction accuracy or arbitrary per-fill win rate shown, so that
    continuous portfolio rebalancing is not reduced to misleading classification metrics.
101. As a paper operator, I want every one-minute account mark and every material event retained, so
    that account history is complete.
102. As a paper operator, I want long chart ranges downsampled only when rendered, so that storage
    remains exact while the UI stays responsive.
103. As a paper operator, I want every archived Paper Account selectable and comparable, so that a
    Manual Reset does not make earlier experience inaccessible.
104. As a maintainer, I want operational state stored in an ignored generated-data directory inside
    the repository, so that all project data remains colocated without entering Git history.
105. As a maintainer, I want one SQLite WAL writer, so that decisions, fills, interventions, and
    recovery state have one atomic ordering authority.
106. As a maintainer, I want recovery snapshots alongside append-only history, so that restart is
    fast without discarding the event audit trail.
107. As a maintainer, I want an online backup before schema migration, so that an upgrade cannot
    irreversibly corrupt the only database.
108. As a paper operator, I want one rotating backup per active day with seven retained, so that
    recent software or database corruption is locally recoverable.
109. As a paper operator, I want financial charts and persistence updated once per minute, so that
    dashboard resolution matches the risk-marking contract.
110. As a paper operator, I want clocks and countdowns to update every second, so that the UI still
    feels live without storing tick-level account state.
111. As a paper operator, I want desktop notifications for executed Portfolio Changes, so that I
    notice real account activity without watching the page.
112. As a paper operator, I want desktop notifications for Risk Stop and completed Reset, so that
    terminal lifecycle changes are hard to miss.
113. As a paper operator, I want a desktop notification after five minutes of stale data, so that a
    meaningful outage receives attention without immediate noise.
114. As a paper operator, I want desktop notifications for fitting failure and incompatible Policy
    Revision, so that maintenance problems are actionable.
115. As a paper operator, I do not want notifications for every routine no-trade decision, so that
    the notification channel remains useful.
116. As a maintainer, I want a build-free frontend served by the Python service, so that the project
    needs no Node toolchain or separate deployment artifact.
117. As a maintainer, I want chart assets bundled locally, so that the dashboard does not depend on a
    third-party CDN during operation.
118. As a maintainer, I want browser controls and the trader in one process, so that UI state cannot
    race a separate CSV-reading daemon.
119. As a maintainer, I want state-changing browser requests same-origin protected and idempotent, so
    that a local dashboard cannot be mutated accidentally by duplicate or cross-site requests.
120. As a maintainer, I want the obsolete paper-proof command, service, state, counters, and docs
    removed, so that proof semantics cannot leak into the Paper Account implementation.
121. As a researcher, I want historical validation and Consumed Holdout artifacts retained as
    Development Evidence, so that prior model work remains useful without claiming proof.
122. As a paper operator, I want compatible active legacy state imported at cutover when safe, so
    that a valid existing virtual portfolio need not be discarded.
123. As a paper operator, I want incompatible legacy state preserved as an auditable artifact and a
    new Flat Start created, so that migration never guesses at account truth.
124. As a maintainer, I want the old and new services prevented from running concurrently, so that
    one market observation cannot be processed twice by competing operators.
125. As a maintainer, I want deterministic tests to drive time and market data without sleeping, so
    that crash, outage, and timing behavior is reliable and fast to verify.
126. As a maintainer, I want one high-level application test seam, so that browser/API behavior,
    persistence, trading, and recovery are verified as one observable system.
127. As a paper operator, I want delivery verified by one live 15-minute decision and explanation,
    so that the installed dashboard proves its real data path without waiting days.
128. As a researcher, I do not want a live fill required for delivery, so that legitimate no-trade
    policy output is not treated as implementation failure.
129. As a paper operator, I want no new multi-day completion gate, so that I can begin evaluating the
    dashboard immediately during normal evening operation.

## Implementation Decisions

- This is a hard cut. Replace the proof-oriented paper command with a serve command and replace the
  Forward Paper Proof user unit with a paper-dashboard user unit. Remove the Proof Clock, proof
  pass/fail state, minimum-day and minimum-change gates, proof-specific artifacts, and compatibility
  aliases. Keep data synchronization, validation, and Historical Holdout workflows as Development
  Evidence workflows.
- The new service runs as an enabled lingering user systemd service, never as root. It starts at boot,
  restarts after process failure, invokes the locked project environment without dependency syncing,
  uses the ROCm device for fitting, and does not launch a browser.
- Bind the HTTP server only to `127.0.0.1` on a stable port, initially `8765`. No login is required.
  State-changing operations use POST semantics, same-origin/CSRF protection, durable idempotency, and
  optimistic state validation so duplicate clicks or stale pages cannot enqueue duplicate mutations.
- Implement one paper-dashboard application as the deep operational interface. It owns the minute
  loop, Paper Account state machine, market recovery, execution, background fitting and attribution,
  persistence, read models, controls, notifications, and dashboard/API snapshots. The CLI, systemd
  runner, and browser remain thin adapters.
- Preserve the existing canonical market-data adapter and pure execution/risk simulation seams where
  they still match the accepted behavior. Remove proof-specific orchestration rather than wrapping it
  in a compatibility layer.
- Run the trader, API, and UI in one FastAPI/uvicorn process. Use one background application loop and
  one database writer; browser requests submit serialized commands to that owner rather than mutating
  account tables independently.
- Use SQLite in WAL mode as the operational source of truth. Store it, fitted-policy checkpoints,
  and backups under an explicitly ignored generated-data directory inside the repository. Keep the
  locally bundled chart assets with tracked application source. Add only operational paths to Git
  ignore rules.
- Persist an append-only, monotonically ordered event ledger plus transactional recovery snapshots
  and query projections. The ledger covers Paper Accounts, Operating Windows, observed and
  reconstructed marks, decisions, pending changes, fills, Missed Executions, funding, interventions,
  risk transitions, data status, fitting, attributions, revisions, handoffs, notifications, and
  backups. Projections may be rebuilt from ledger plus versioned snapshots.
- There is exactly one active Paper Account. A new account starts with $10,000 cash and zero
  positions. Manual Reset closes and archives it only after any open exposure has been filled flat,
  then creates the next Flat Start. Archived accounts and their event histories are immutable.
- Treat Trading, Paused, Reset Pending, Risk Stopped, and Migration Required as durable lifecycle
  states. Data Stale, Fitting, Attribution Pending, and notification health are operational overlays,
  not alternate account identities. Process or machine restart preserves every lifecycle state.
- Pause cancels any unfilled policy change as an Operator Intervention, retains current exposure, and
  prevents new decisions and policy fills. Resume is valid only from ordinary Paused and resumes at
  the next naturally due Decision Bar; it never replays missed decisions.
- Flatten and Pause cancels pending policy execution, records an intervention, and uses the next fresh
  bid/ask immediately without policy Decision Latency. After all fills succeed it remains Paused and
  flat. Manual Reset uses the same flattening semantics but advances to account archival and a new
  Flat Start after its separate confirmation.
- Risk Stop retains the established Drawdown Limit and policy execution contract. A breach schedules
  terminal flattening, blocks all later decisions for the account, and cannot be bypassed by Resume.
  Manual Reset is the only route from Risk Stopped to a new Trading account.
- Preserve the fixed Trading Universe, fully closed 15-minute Decision Bars, one-minute marks,
  60-second normal Decision Latency, minimum-turnover execution threshold, 0.07% adverse all-in cost,
  funding, No Leverage, concentration, and drawdown behavior from the current Policy Protocol.
- A pending policy change persists across restart. It may execute at the first fresh quote only when
  that quote is no more than two minutes later than the scheduled fill time. Otherwise it becomes a
  durable Missed Execution, changes neither cash nor positions, and awaits the next natural decision.
- Market-data unavailability never terminates the web service. Enter Data Stale, expose observation
  age and error, prevent new decisions/fills, and retry in process with bounded backoff. Notify after
  five minutes. Systemd restart is reserved for process failure.
- On restart or data recovery, backfill objectively reconstructible one-minute valuation marks from
  public closed market bars and all published funding events for retained positions. Mark these rows
  as reconstructed and place them inside an explicit downtime gap. Do not reconstruct policy
  decisions, Target Weights, bid/ask executions, or Operator Interventions.
- Run scheduled weekly fitting in a background worker on the first Operating Window after the weekly
  deadline. Continue observing and trading with the current Fitted Policy. Atomically persist the new
  checkpoint and Policy Handoff only after fitting succeeds. A failure retains the prior model,
  records diagnostics, and sends a desktop notification.
- A compatible Policy Revision preserves the Paper Account and positions, records a revision
  boundary, and produces per-protocol metric segments. Compatibility requires at least the same
  Trading Universe, account currency, and position/execution/risk semantics. An incompatible revision
  enters Migration Required and performs no automated position transformation or account reset.
- Persist a Decision Record for every scheduled policy decision, including no-trade decisions. Store
  signal/model/input identity, Current Portfolio, raw and constrained Target Weights, projected
  turnover, threshold result, pending/fill timing, actual Reference and Effective Fill, costs,
  constraints, and final outcome. Link every instrument fill or Missed Execution back to its decision.
- Compute Model Attribution asynchronously after the decision is durable and execution scheduling is
  complete. Explanation work has lower priority than observation, execution, controls, and fitting
  handoff and may display Attribution Pending without delaying a fill.
- Use Integrated Gradients against a neutral normalized-market baseline. Aggregate signed influence
  by ticker, feature family, and temporal region for momentum, volatility/range, flow/activity,
  derivatives positioning/carry, availability, and Current Portfolio. Store the top signed influences,
  method parameters, input identity, and model identity. Do not store the full raw attribution tensor;
  retain enough immutable identity to recompute it from canonical data and a model checkpoint.
- Explicitly label Model Attribution as post-hoc influence evidence, not a causal explanation, model
  intent, or a deterministic trading rule. Exact Decision Records and fills remain authoritative.
- Track average-cost basis per ticker. Same-direction increases update weighted average entry;
  reductions realize P&L against that average; reversals close the old side and open the residual new
  side at the fill. Keep Transaction Cost and Funding separate. Marked Equity is the authoritative
  account total and must reconcile with cash, marked positions, costs, and funding.
- Maintain two passive comparison projections from each account reset: zero-yield cash and an initial
  25%-per-ticker long allocation across the Trading Universe. The equal-weight benchmark is passive,
  incurs the same initial cost assumption, carries actual funding, and remains exposed during downtime.
- Persist all one-minute account marks and material events indefinitely. Retain exact data in storage;
  choose query-time downsampling based on pixel width and range for current window, 24 hours, seven
  days, and full account. Never downsample or discard Decision Records, fills, funding, gaps,
  interventions, or model events.
- Create an online database backup before each schema migration and no more than one daily backup
  while active. Retain the seven newest daily backups. Surface backup age and failures in System.
- Serve a build-free dashboard shell with small plain-browser scripts and a locally bundled financial
  chart library. Do not introduce a Node toolchain, SPA framework, external CDN dependency, or a
  second dashboard process.
- Live contains service/account status, freshness, fitting status, next decision/pending fill, account
  and risk cards, position table, selected-ticker candlesticks, Current/Target Weight, synchronized
  equity/benchmark/drawdown/exposure panels, and recent activity. Lifecycle controls stay visible here.
- History provides filterable account and event timelines, archived-account selection/comparison,
  event-to-decision/fill links, and full Decision Record/Model Attribution detail. System provides
  feed errors/freshness, model/protocol and revision status, fitting progress, Operating Windows,
  backups, notifications, and service diagnostics.
- Use one selected ticker price scale at a time. Render signal, fill, Missed Execution, funding, and
  Operator Intervention markers and synchronize the crosshair with portfolio panels. Clicking a marker
  resolves to the durable account event, never to a chart-only annotation.
- Store and calculate every timestamp in timezone-aware UTC. Display Europe/Kiev by default and expose
  UTC in detailed views and tooltips.
- Financial state and persisted chart data update once per minute. Browser clocks, freshness age, and
  countdowns may update once per second without creating tick-level account history.
- Send desktop notifications for executed Portfolio Changes, Risk Stop, completed Manual Reset, data
  stale beyond five minutes, fitting failure, and incompatible Policy Revision. Do not notify for
  routine decisions or ordinary no-trade outcomes. Every notification source remains visible in the
  dashboard even if desktop delivery is unavailable.
- Perform a single explicit cutover from the legacy proof operator. Stop and disable the old unit before
  enabling the new unit. At the cutover instant, import active virtual state only if its protocol,
  universe, accounting, model checkpoint, and history are compatible with the new schema; otherwise
  preserve it as legacy Development Evidence and create a Flat Start. Do not retain an ongoing legacy
  state reader after cutover.
- Remove obsolete proof vocabulary, counters, reports, tests, configuration fields, and operating
  instructions. Rename remaining activity metrics from proof-oriented qualifying changes to
  Executable Portfolio Changes. Update the experiment log with the cutover and live acceptance result.

## Testing Decisions

- The primary new test seam is the complete in-process ASGI application. Construct it with an injected
  deterministic clock, fake public market feed, fake policy fitting/attribution backend, desktop
  notification recorder, and temporary SQLite database. Drive the same HTTP read/control operations
  as the browser, advance time without sleeping, recreate the application for recovery tests, and
  assert only externally visible account snapshots and durable history.
- Good high-level tests assert account state, event ordering, rendered/API-visible data, durable
  recovery, and observable side effects. They do not assert private coroutine structure, SQL query
  text, CSS class names, implementation-specific thread counts, or incidental serialization details.
- Preserve existing pure simulation tests as the authoritative lower seam for exact fills, delayed
  execution, costs, funding, Marked Equity, exposure, drawdown, Risk Stop, and incremental state.
  Extend those tests only when the accepted financial contract changes.
- Reuse the existing fake market-data adapters, fake policy backend, incremental workflow restart
  scenarios, parallel-refit barrier patterns, CLI routing tests, and end-to-end direct-policy fixture
  as prior art. Prefer adapting these behavioral fixtures to the new application seam over duplicating
  proof-specific test helpers.
- Test Flat Start, one-active-account enforcement, Manual Reset flatten/archive/create ordering,
  immutable archived history, and account comparison behavior through the application seam.
- Test process restart and machine-equivalent restart by closing and recreating the application over
  the same database. Assert preservation of Trading, Paused, Reset Pending, Risk Stopped, Migration
  Required, open positions, pending execution, Fitted Policy, and event identity.
- Test downtime with long and short positions. Assert reconstructed one-minute marks and funding affect
  equity/drawdown, the gap is visible and labeled, and no decision, Target Weight, or fill is created for
  the unavailable Operating Window.
- Test a pending fill recovered inside the two-minute window and a pending fill recovered outside it.
  The former uses only a fresh observed quote; the latter produces one Missed Execution and no account
  mutation.
- Test transient and extended market-data outages. Assert Data Stale, fresh-data age, suspended
  decisions/fills, available UI/control reads, bounded retry behavior, one five-minute notification,
  recovery backfill, and continuation of the same account without process termination.
- Test Pause cancellation of pending policy work, exposure persistence while paused, immediate Resume
  eligibility, idempotent repeats, and absence of replayed decisions after Resume.
- Test Flatten and Pause and Manual Reset with long, short, mixed, already-flat, stale-data, and partial
  failure scenarios. Assert immediate-next-fresh-quote timing, normal costs, confirmation and idempotency,
  durable Reset Pending, account archival only after flat, and correct intervention attribution.
- Test Drawdown Limit breach, delayed terminal Risk Stop fill under the policy contract, persistence of
  Risk Stopped, and rejection of Resume until Manual Reset.
- Test average-cost accounting for add, partial reduction, full close, long-to-short reversal,
  short-to-long reversal, transaction costs, funding, and reconciliation to Marked Equity.
- Test cash and passive equal-weight-long benchmarks from the same reset instant, including initial
  costs, drift, funding, downtime marks, and separation from policy account state.
- Test background fitting with a controllable barrier. Assert observations, account marks, controls,
  and existing-model decisions continue while fitting blocks; handoff is atomic on success; failure
  preserves the old model and records/announces the error.
- Test compatible and incompatible Policy Revisions. Assert compatible revision segmentation without
  account reset and incompatible transition to Migration Required without transformed positions or
  later policy decisions.
- Test every decision outcome: executable target, below-threshold target, unchanged target, Risk Stop
  exclusion, pending execution, successful fills across multiple tickers, and Missed Execution. Assert
  complete Decision Record linkage and exact model/input identity.
- Test that attribution is enqueued only after the decision and pending execution are durable, cannot
  delay fill processing, exposes Attribution Pending, produces signed grouped influences, records
  reproducibility identity, and clearly labels its approximate status. Numerical unit tests should
  validate Integrated Gradients completeness on small deterministic models without coupling high-level
  tests to exact values from a trained TCN.
- Test one-minute history retention, UTC ordering, Europe/Kiev presentation including daylight-saving
  transitions, gap shading data, long-range render-only downsampling, and preservation of all material
  event markers.
- Test Live, History, and System responses through the ASGI client, including all accepted metric
  groups, selected-ticker chart data, cross-panel timestamps, event detail resolution, archived account
  selection, service diagnostics, and stale/fitting/risk states. A small real-browser smoke may verify
  chart boot and control wiring without making DOM layout the behavioral contract.
- Test state-changing HTTP requests for same-origin/CSRF rejection, stale version rejection, durable
  idempotency, duplicate-click behavior, invalid-state controls, and no mutation through read methods.
- Test daily backup cadence, seven-backup rotation, pre-migration backup, online backup consistency,
  and visible backup failure without stopping trading.
- Test notification selection and deduplication. Executed changes and actionable terminal/failure states
  notify; routine decisions and no-trade outcomes do not; failure to reach the desktop notification bus
  remains visible but does not affect trading.
- Test cutover against representative compatible and incompatible legacy states. Assert the old and new
  operators cannot run concurrently, compatible state imports exactly once, incompatible state is
  preserved without guessing, and no ongoing proof compatibility path remains.
- Keep CLI parser/routing tests thin: the new serve command constructs and runs the one application;
  removed proof commands and proof-only options are rejected rather than aliased.
- Verify the user systemd unit structurally, including user scope, network ordering, direct locked
  environment invocation, restart policy, working directory, ROCm environment, localhost service, and
  absence of dependency syncing or browser launch.
- Run lock verification, formatting, lint, strict type checking, and the complete test suite. Then enable
  and start the service, verify the localhost dashboard through the installed unit, verify ROCm-backed
  operation, and retain one real closed 15-minute Decision Record with Model Attribution as live
  acceptance evidence. A live fill is optional because a legitimate policy decision may not execute.

## Out of Scope

- Real-money order placement, exchange credentials, Binance account access, and Binance testnet orders.
- Forward Paper Proof, Proof Clock, 60-day or 100-change gates, paper pass/fail certification, or any
  claim that the Paper Account proves real-world profitability.
- Manual per-instrument buy, sell, quantity, limit, stop, or Target Weight controls. Human actions are
  limited to Pause, Resume, Flatten and Pause, and Manual Reset.
- LAN or internet exposure, multi-user accounts, authentication, TLS termination, remote control, or a
  dedicated mobile application.
- Tick-level equity, tick history, order-book capture, websocket quote archiving, or sub-minute risk
  calculation. Exact fill quotes remain stored for material executions.
- Retroactive policy decisions, reconstructed historical bid/ask fills, execution of arbitrarily stale
  pending targets, or automatic account reset after gaps.
- Generic prediction accuracy, classification scores, arbitrary trade win rate, or a claim that Model
  Attribution is a causal explanation.
- A Node build, React/Vue frontend, separately deployed SPA, external CDN assets, or a second dashboard
  process.
- Dashboard export or import. Operational SQLite and its rotating local backups are the only persistence
  products in this version.
- Automatic deletion of archived Paper Accounts, event history, model checkpoints required by history,
  or backups inside their seven-day retention window.
- Cloud/off-machine backup, protection from physical disk loss, or disaster recovery beyond local
  rotating SQLite backups.
- Automatic migration of incompatible Trading Universes, account currencies, execution semantics, or
  risk contracts.
- Automatic browser launch at boot.
- A live trade as a release gate or any long-duration unattended acceptance period.

## Further Notes

- ADR-0004 fixes the single-process FastAPI/SQLite service boundary. ADR-0005 supersedes the Forward
  Paper Proof requirement in ADR-0003 while preserving the direct Net-Growth policy and historical
  Development Evidence decisions.
- The glossary now uses Paper Account, Operating Window, Manual Reset, Operator Intervention, Decision
  Record, Model Attribution, Missed Execution, and Executable Portfolio Change. Implementation and UI
  text must use these terms rather than reviving proof-specific synonyms.
- The current proof service remains untouched until implementation cutover. Cutover must inspect its
  exact live state after stopping it, because account contents may have changed since this spec was
  written.
- The implementation completion boundary is behavioral and operational, not profitable performance:
  deterministic verification, a running enabled ROCm service, a working localhost UI, and one real
  decision/explanation are sufficient.
