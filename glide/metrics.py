"""Metrics for completed GLIDE inference requests."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List


def _value(request: Any, name: str, default: Any = None) -> Any:
    if isinstance(request, dict):
        return request.get(name, default)
    return getattr(request, name, default)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def compute_metrics(requests: Iterable[Any]) -> Dict[str, float]:
    items = list(requests)
    latencies = [float(_value(item, 'latency_ms', 0.0) or 0.0) for item in items]
    computes = [float(_value(item, 'emulated_compute_ms', 0.0) or 0.0) for item in items]
    memories = [float(_value(item, 'memory_mb', 0.0) or 0.0) for item in items]
    starts = [float(_value(item, 'arrival_time')) for item in items if _value(item, 'arrival_time') is not None]
    ends = [float(_value(item, 'end_time')) for item in items if _value(item, 'end_time') is not None]
    total_duration = max(ends) - min(starts) if starts and ends else 0.0
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    slowdowns = [latency / compute for latency, compute in zip(latencies, computes) if compute > 0.0]
    total_slowdown = sum(slowdowns)
    slowdown_fairness = (
        total_slowdown ** 2 / (len(slowdowns) * sum(value * value for value in slowdowns))
        if slowdowns and total_slowdown else 1.0
    )
    waits = [
        max(0.0, (float(_value(item, 'start_time')) - float(_value(item, 'arrival_time'))) * 1000.0)
        for item in items
        if _value(item, 'start_time') is not None and _value(item, 'arrival_time') is not None
    ]
    wait_outliers = sum(
        1 for item in items
        if _value(item, 'start_time') is not None
        and _value(item, 'arrival_time') is not None
        and float(_value(item, 'emulated_compute_ms', 0.0) or 0.0) > 0.0
        and float(_value(item, 'start_time')) - float(_value(item, 'arrival_time'))
        > 3.0 * float(_value(item, 'emulated_compute_ms', 0.0) or 0.0) / 1000.0
    )
    return {
        'throughput': len(items) / total_duration if total_duration > 0 else 0.0,
        'avg_latency_ms': avg,
        'p50_latency_ms': _percentile(latencies, 0.50),
        'p95_latency_ms': _percentile(latencies, 0.95),
        'p99_latency_ms': _percentile(latencies, 0.99),
        'max_latency_ms': max(latencies) if latencies else 0.0,
        'min_latency_ms': min(latencies) if latencies else 0.0,
        'max_wait_ms': max(waits) if waits else 0.0,
        'slowdown_fairness_index': slowdown_fairness,
        'service_relative_wait_outlier_count': wait_outliers,
        'compute_utilisation': (sum(computes) / (total_duration * 1000.0)) if total_duration > 0 else 0.0,
        'memory_peak_mb': max(memories) if memories else 0.0,
    }


def compare_schedulers(results_dict: Dict[str, Iterable[Any]]) -> Dict[str, Dict[str, float]]:
    return {name: compute_metrics(requests) for name, requests in results_dict.items()}


def print_comparison_table(comparison_dict: Dict[str, Dict[str, float]]) -> None:
    names = list(comparison_dict)
    print('Metric                 | ' + ' | '.join(f'{name.upper():<7}' for name in names))
    print('-' * (24 + 10 * len(names)))
    rows = [
        ('Avg Latency(ms)', 'avg_latency_ms', '.2f'),
        ('P50 Latency(ms)', 'p50_latency_ms', '.2f'),
        ('P95 Latency(ms)', 'p95_latency_ms', '.2f'),
        ('P99 Latency(ms)', 'p99_latency_ms', '.2f'),
        ('Max Wait(ms)', 'max_wait_ms', '.2f'),
        ('Throughput(r/s)', 'throughput', '.2f'),
        ('Slowdown Fairness', 'slowdown_fairness_index', '.4f'),
        ('Wait Outliers', 'service_relative_wait_outlier_count', '.0f'),
    ]
    for label, key, fmt in rows:
        print(f'{label:<22} | ' + ' | '.join(f'{comparison_dict[name].get(key, 0):{fmt}}' for name in names))


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
