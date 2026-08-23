---
status: accepted
---

# Single-Process Paper-Trading Dashboard

Run the autonomous paper trader, GPU fitting, control API, and browser dashboard in one Python
systemd service, with a single SQLite WAL writer providing append-only account history and recovery
snapshots. This boundary keeps decisions, fills, interventions, and UI state atomically ordered;
separate trading and dashboard processes or CSV-derived state would introduce synchronization and
recovery ambiguity.
