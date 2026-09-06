"""Run repeatable scheduler experiments over generated workloads."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .engine import InferenceEngine
    from .metrics import compare_schedulers, print_comparison_table
    from .workload import WorkloadGenerator
except ImportError:
    from glide.engine import InferenceEngine
    from glide.metrics import compare_schedulers, print_comparison_table
    from glide.workload import WorkloadGenerator


def run_experiment(workload_config: Dict[str, Any], output_dir: str) -> Dict[str, Dict[str, float]]:
    generator = WorkloadGenerator(
        workload_config['models'],
        workload_config.get('arrival_mode', 'poisson'),
        workload_config.get('rate', 1.0),
        workload_config.get('duration', 30.0),
        workload_config.get('seed', 42),
    )
    workload = generator.generate()
    results = {}
    queue_histories = {}
    previous_skip_sleep = InferenceEngine._skip_sleep
    InferenceEngine._skip_sleep = True
    try:
        for scheduler_name in ('fifo', 'sjf', 'hasp'):
            engine = InferenceEngine(
                gpu=workload_config.get('gpu', 'Tesla_M40'),
                model=workload_config['models'][0],
                scheduler=scheduler_name,
            )
            base_time = time.time() - workload_config.get('duration', 30.0)
            for item in workload:
                request_id = engine.submit_request(
                    model_name=item['model_name'],
                    batch_size=item['batch_size'],
                    priority=item['priority'],
                )
                request = engine.request_queue[-1]
                request.arrival_time = base_time + item['arrival_time']
            results[scheduler_name] = engine.run_queue()
            queue_histories[scheduler_name] = engine.queue_history
    finally:
        InferenceEngine._skip_sleep = previous_skip_sleep

    comparison = compare_schedulers(results)
    for scheduler_name, history in queue_histories.items():
        comparison[scheduler_name]['queue_history'] = history
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as handle:
        json.dump(comparison, handle, indent=2)
    print(f"\nExperiment: {workload_config.get('name', 'unnamed')}")
    print_comparison_table(comparison)
    return comparison


EXPERIMENTS = [
    {'name': 'single_model_uniform', 'models': ['resnet50'], 'arrival_mode': 'uniform', 'rate': 2.0, 'duration': 30.0},
    {'name': 'single_model_poisson', 'models': ['resnet50'], 'arrival_mode': 'poisson', 'rate': 2.0, 'duration': 30.0},
    {'name': 'multi_model_uniform', 'models': ['resnet18', 'resnet50', 'vgg16'], 'arrival_mode': 'uniform', 'rate': 2.0, 'duration': 30.0},
    {'name': 'multi_model_bursty', 'models': ['resnet18', 'resnet50', 'vgg16'], 'arrival_mode': 'bursty', 'rate': 2.0, 'duration': 30.0},
]


if __name__ == '__main__':
    for config in EXPERIMENTS:
        run_experiment(config, os.path.join('experiment_results', config['name']))
