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
    from .report import build_report, export_report, summarize_trials
    from .workload import WorkloadGenerator
except ImportError:
    from glide.engine import InferenceEngine
    from glide.metrics import compare_schedulers, print_comparison_table
    from glide.report import build_report, export_report, summarize_trials
    from glide.workload import WorkloadGenerator


def run_experiment(workload_config: Dict[str, Any], output_dir: str) -> Dict[str, Dict[str, float]]:
    generator = WorkloadGenerator(
        workload_config['models'],
        workload_config.get('arrival_mode', 'poisson'),
        workload_config.get('rate', 1.0),
        workload_config.get('duration', 30.0),
        workload_config.get('seed', 42),
        workload_config.get('batch_sizes', (1,)),
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
    export_report(
        build_report(comparison, {
            **workload_config,
            'schedulers': ['fifo', 'sjf', 'hasp'],
            'seed': workload_config.get('seed', 42),
        }),
        output_dir,
    )
    print(f"\nExperiment: {workload_config.get('name', 'unnamed')}")
    print_comparison_table(comparison)
    return comparison


def run_batching_experiment(
    workload_config: Dict[str, Any],
    output_dir: str,
    batchers: tuple[str, ...] = ('static', 'dynamic', 'continuous'),
) -> Dict[str, Any]:
    """Compare all schedulers under each batching policy using one trace."""
    generator = WorkloadGenerator(
        workload_config['models'], workload_config.get('arrival_mode', 'poisson'),
        workload_config.get('rate', 1.0), workload_config.get('duration', 30.0),
        workload_config.get('seed', 42),
        workload_config.get('batch_sizes', (1,)),
    )
    workload = generator.generate()
    previous = InferenceEngine._skip_sleep
    InferenceEngine._skip_sleep = True
    results: Dict[str, Any] = {}
    try:
        for batcher in batchers:
            results[batcher] = {}
            for scheduler_name in ('fifo', 'sjf', 'hasp'):
                engine = InferenceEngine(
                    gpu=workload_config.get('gpu', 'Tesla_M40'),
                    model=workload_config['models'][0],
                    scheduler=scheduler_name,
                    batcher=batcher,
                    batch_size=workload_config.get('batch_size', 4),
                    batch_timeout_s=workload_config.get('batch_timeout_s', 0.05),
                )
                base_time = time.time() - workload_config.get('duration', 30.0)
                for item in workload:
                    request_id = engine.submit_request(
                        model_name=item['model_name'], batch_size=item['batch_size'],
                        priority=item['priority'],
                    )
                    request = next(item for item in engine.request_queue if item.request_id == request_id)
                    request.arrival_time = base_time + item['arrival_time']
                completed = engine.run_queue()
                results[batcher][scheduler_name] = compare_schedulers({scheduler_name: completed})[scheduler_name]
    finally:
        InferenceEngine._skip_sleep = previous
    os.makedirs(output_dir, exist_ok=True)
    export_report(build_report(results, {**workload_config, 'batchers': list(batchers)}), output_dir)
    return results


def run_repeated_experiment(
    workload_config: Dict[str, Any], output_dir: str, trials: int = 5,
) -> Dict[str, Any]:
    """Run fixed-seed-offset trials and report uncertainty for each scheduler."""
    if trials < 1:
        raise ValueError('trials must be at least 1')
    trial_results = []
    for index in range(trials):
        config = {**workload_config, 'seed': workload_config.get('seed', 42) + index}
        trial_results.append(run_experiment(config, os.path.join(output_dir, f'trial_{index + 1}')))
    summary = {
        scheduler: summarize_trials([trial[scheduler] for trial in trial_results])
        for scheduler in ('fifo', 'sjf', 'hasp')
    }
    report = build_report({'trials': trial_results, 'summary': summary}, {
        **workload_config, 'trials': trials, 'seed_policy': 'base_seed + trial_index',
    })
    export_report(report, output_dir)
    return report


def run_hasp_ablation(workload_config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Compare HASP with variants that disable one affinity component."""
    from .scheduler import HASPScheduler
    generator = WorkloadGenerator(
        workload_config['models'], workload_config.get('arrival_mode', 'poisson'),
        workload_config.get('rate', 1.0), workload_config.get('duration', 30.0),
        workload_config.get('seed', 42),
        workload_config.get('batch_sizes', (1,)),
    )
    trace = generator.generate()
    variants = {
        'hasp': {},
        'without_aging': {'use_aging': False},
        'without_memory_affinity': {'use_memory_affinity': False},
        'without_compute_affinity': {'use_compute_affinity': False},
    }
    output: Dict[str, Any] = {}
    previous = InferenceEngine._skip_sleep
    InferenceEngine._skip_sleep = True
    try:
        for name, options in variants.items():
            engine = InferenceEngine(
                gpu=workload_config.get('gpu', 'Tesla_M40'),
                model=workload_config['models'][0], scheduler='hasp',
            )
            engine.scheduler = HASPScheduler(engine.gpu, **options)
            base_time = time.time() - workload_config.get('duration', 30.0)
            for item in trace:
                request_id = engine.submit_request(item['model_name'], item['batch_size'], item['priority'])
                request = next(item for item in engine.request_queue if item.request_id == request_id)
                request.arrival_time = base_time + item['arrival_time']
            output[name] = compare_schedulers({name: engine.run_queue()})[name]
    finally:
        InferenceEngine._skip_sleep = previous
    export_report(build_report(output, {**workload_config, 'ablation_variants': list(variants)}), output_dir)
    return output


EXPERIMENTS = [
    {'name': 'low_load', 'models': ['resnet50'], 'arrival_mode': 'poisson', 'rate': 0.5, 'duration': 30.0},
    {'name': 'medium_load', 'models': ['resnet50'], 'arrival_mode': 'poisson', 'rate': 2.0, 'duration': 30.0},
    {'name': 'high_load', 'models': ['resnet50'], 'arrival_mode': 'poisson', 'rate': 8.0, 'duration': 30.0},
    {'name': 'mixed_batch_sizes', 'models': ['resnet18', 'resnet50'], 'arrival_mode': 'poisson', 'rate': 3.0, 'duration': 30.0, 'batch_sizes': [1, 8, 16, 32]},
    {'name': 'multi_model_bursty', 'models': ['resnet18', 'resnet50', 'vgg16'], 'arrival_mode': 'bursty', 'rate': 2.0, 'duration': 30.0, 'batch_sizes': [1, 8, 32]},
]


if __name__ == '__main__':
    for config in EXPERIMENTS:
        run_experiment(config, os.path.join('experiment_results', config['name']))
