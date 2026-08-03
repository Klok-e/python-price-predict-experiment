# 01 — Hard-cut repository cleanup

**What to build:** Remove every obsolete one-minute prediction, model-grid, downloader, report,
test, documentation, and development surface so the repository contains only a minimal,
reproducible foundation for the approved Direct Net-Growth Portfolio Policy goal. Preserve raw
market data, the domain model, accepted architecture decision, approved spec, issue tracker, and
repository agent tooling.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] Delete the old one-minute runner, data and policy implementation, duplicate download paths,
      generic experiment helpers, trivial or obsolete tests, superseded documentation, and all
      compatibility surfaces.
- [x] Replace the long experiment narrative with a concise objective and constraints section plus a
      durable summary of rejected approaches that prevents repeating failed research.
- [x] Permanently delete obsolete generated run artifacts while retaining the raw market-data cache.
- [x] Delete obsolete scratch work and tracked IDE metadata; keep repository agent skills, their
      lock, agent process documentation, domain glossary, accepted ADR, spec, and these tickets.
- [x] Establish a minimal Python 3.13 uv project, universal dependency lock, focused package shell,
      fixed TOML configuration, and one four-command CLI surface ready for the goal run.
- [x] Ensure strategy-defining values cannot be overridden through the CLI; only paths and compute
      device may vary.
- [x] Update repository guidance to describe only the approved target and remove every reference to
      removed operational workflows.
- [x] Leave the cleaned repository installable, formatting-clean, test-clean, and free of imports or
      executable references to deleted code.
