# Online Paper Trading Removed

Status: superseded by ADR 0002

The previous cash/units online paper trading path is no longer operational. The repository has
hard-cut to historical learned one-minute intraday backtesting, so live daemon, append, pending
order, and online paper state semantics are removed rather than maintained as a parallel path.
