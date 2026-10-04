"""Simple latency summaries used by diagnostics and benchmarking."""

import numpy as np


def summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99")}
    return {
        "mean": float(np.mean(samples)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
    }
