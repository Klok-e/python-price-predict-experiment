from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from utils.util import open_time_to_datetime


DEFAULT_CASH = 1_000_000


def jsonable(value: Any):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def parse_linear_arch(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_stats(split):
    rows = sum(len(ticker[0]) for ticker in split)
    labels = sum(len(ticker[2]) for ticker in split)
    positives = sum(float(ticker[2]["Label"].sum()) for ticker in split)
    tickers = {}
    for scaled, original, labels_df, _, ticker in split:
        tickers[ticker] = {
            "rows": len(scaled),
            "labels": len(labels_df),
            "positives": float(labels_df["Label"].sum()) if len(labels_df) else 0.0,
            "positive_rate": float(labels_df["Label"].mean()) if len(labels_df) else 0.0,
            "start": original.index.min().isoformat() if len(original) else None,
            "end": original.index.max().isoformat() if len(original) else None,
        }

    return {
        "rows": rows,
        "labels": labels,
        "positives": positives,
        "positive_rate": positives / labels if labels else 0.0,
        "tickers": tickers,
    }


def raw_data_stats(raw_tickers):
    stats = {}
    for df, ticker in raw_tickers:
        index = open_time_to_datetime(df["Open time"]) if "Open time" in df.columns else pd.to_datetime(df.index)
        stats[ticker] = {
            "rows": len(df),
            "start": index.min().isoformat() if len(index) else None,
            "end": index.max().isoformat() if len(index) else None,
        }
    return stats


def git_state():
    def run_git(args):
        try:
            result = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = run_git(["status", "--short"]) or ""
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "changed_files": [line[3:] for line in status.splitlines() if len(line) >= 4],
    }
