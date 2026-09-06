"""Compare FIFO, SJF, and HASP schedulers on the same workload."""

import math
import os
import sys
from typing import Dict, List, Tuple

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glide.engine import InferenceEngine


WORKLOAD: List[Tuple[str, int]] = [
    ('resnet18', 1),
    ('vgg16', 8),
    ('mobilenet_v2', 32),
    ('resnet50', 16),
    ('alexnet', 8),
    ('densenet121', 4),
    ('shufflenet_v2_x1_0', 32),
    ('wide_resnet50_2', 2),
    ('squeezenet1_0', 16),
    ('resnet34', 8),
]


def _jain_fairness(values: List[float]) -> float:
    if not values:
        return 1.0
    total = sum(values)
    total_sq = sum(value * value for value in values)
    if total_sq == 0.0:
        return 1.0
    return (total * total) / (len(values) * total_sq)


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    p95_index = max(0, int(math.ceil(len(sorted_values) * 0.95)) - 1)
    return sorted_values[p95_index]


def _run_scheduler(scheduler_name: str) -> Dict[str, float]:
    engine = InferenceEngine(
        gpu='Tesla_M40',
        model='resnet18',
        scheduler=scheduler_name,
    )

    for model_name, batch_size in WORKLOAD:
        engine.submit_request(model_name=model_name, batch_size=batch_size)

    completed_requests = engine.run_queue()
    latencies = [request.latency_ms for request in completed_requests if request.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _p95(latencies)
    fairness = _jain_fairness(latencies)

    starvation_count = 0
    scheduler = engine.scheduler
    get_starvation_count = getattr(scheduler, 'get_starvation_count', None)
    if callable(get_starvation_count):
        starvation_count = int(get_starvation_count())

    return {
        'avg_latency_ms': avg_latency,
        'p95_latency_ms': p95_latency,
        'fairness_index': fairness,
        'starvation_count': float(starvation_count),
    }


def main() -> None:
    print('=' * 90)
    print('GLIDE Scheduler Comparison (10-request mixed workload)')
    print('=' * 90)

    results = {
        'FIFO': _run_scheduler('fifo'),
        'SJF': _run_scheduler('sjf'),
        'HASP': _run_scheduler('hasp'),
    }

    print('Scheduler | Avg Latency | P95 Latency | Fairness | Starvation')
    print('-' * 90)
    for scheduler_name in ('FIFO', 'SJF', 'HASP'):
        row = results[scheduler_name]
        print(
            f"{scheduler_name:<9} | "
            f"{row['avg_latency_ms']:.3f} ms | "
            f"{row['p95_latency_ms']:.3f} ms | "
            f"{row['fairness_index']:.4f} | "
            f"{int(row['starvation_count'])}"
        )
    print('=' * 90)


if __name__ == '__main__':
    main()
