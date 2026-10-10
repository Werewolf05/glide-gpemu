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
    from .engine import InferenceEngine, InferenceRequest, get_profiled_compute_time, get_profiled_memory
    from .metrics import compare_schedulers, print_comparison_table
    from .report import build_report, export_report, summarize_trials
    from .scheduler import HASPScheduler, create_scheduler
    from .workload import WorkloadGenerator
except ImportError:
    from glide.engine import InferenceEngine, InferenceRequest, get_profiled_compute_time, get_profiled_memory
    from glide.metrics import compare_schedulers, print_comparison_table
    from glide.report import build_report, export_report, summarize_trials
    from glide.scheduler import HASPScheduler, create_scheduler
    from glide.workload import WorkloadGenerator


def _simulate_scheduler_trace(
    workload: list[Dict[str, Any]], gpu: str, scheduler_name: str,
    hasp_aging_threshold_s: float = 5.0,
) -> list[InferenceRequest]:
    """Run one scheduler over a shared trace with virtual single-request service."""
    future = sorted(enumerate(workload), key=lambda item: (item[1]['arrival_time'], item[0]))
    queued: list[InferenceRequest] = []
    completed: list[InferenceRequest] = []
    scheduler = (
        HASPScheduler(gpu, aging_threshold=hasp_aging_threshold_s)
        if scheduler_name.lower() == 'hasp'
        else create_scheduler(scheduler_name, gpu)
    )
    clock = 0.0
    next_request = 0

    while next_request < len(future) or queued:
        if not queued and next_request < len(future):
            clock = max(clock, float(future[next_request][1]['arrival_time']))

        while next_request < len(future) and float(future[next_request][1]['arrival_time']) <= clock:
            index, item = future[next_request]
            model = item['model_name']
            batch_size = int(item['batch_size'])
            queued.append(InferenceRequest(
                request_id=str(index),
                model_name=model,
                batch_size=batch_size,
                arrival_time=float(item['arrival_time']),
                priority=int(item.get('priority', 0)),
                emulated_compute_ms=get_profiled_compute_time(gpu, model, batch_size),
                memory_mb=get_profiled_memory(gpu, model, batch_size),
            ))
            next_request += 1

        request = scheduler.select_next(queued, clock)
        if request is None:
            continue
        queued.remove(request)
        request.start_time = clock
        compute_ms = request.emulated_compute_ms or request.batch_size * 2.0
        request.emulated_compute_ms = compute_ms
        request.end_time = clock + compute_ms / 1000.0
        request.latency_ms = (request.end_time - request.arrival_time) * 1000.0
        request.status = 'completed'
        scheduler.record_completion(request)
        completed.append(request)
        clock = request.end_time

    return completed


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
    gpu = workload_config.get('gpu', 'Tesla_M40')
    hasp_aging_threshold_s = workload_config.get('hasp_aging_threshold_s', 5.0)
    results = {
        scheduler_name: _simulate_scheduler_trace(
            workload, gpu, scheduler_name, hasp_aging_threshold_s,
        )
        for scheduler_name in ('fifo', 'sjf', 'hasp')
    }
    comparison = compare_schedulers(results)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as handle:
        json.dump(comparison, handle, indent=2)
    export_report(
        build_report(comparison, {
            **workload_config,
            'gpu': gpu,
            'request_count': len(workload),
            'hasp_aging_threshold_s': hasp_aging_threshold_s,
            'schedulers': ['fifo', 'sjf', 'hasp'],
            'seed': workload_config.get('seed', 42),
            'simulation': 'single-server, non-preemptive, virtual service time',
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
