"""One-command GLIDE Phase 3 demonstration."""

from __future__ import annotations

import os
import sys
import json

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glide.engine import InferenceEngine
from glide.experiment import run_experiment
from glide.metrics import print_comparison_table
from glide.profiler import profile_and_save
from glide.test_scheduler import _run_scheduler


def main() -> None:
    print('=' * 72)
    print('GLIDE GPU Layer-Level Inference and Dispatching Emulator DEMO')
    print('=' * 72)

    print('\n[1/4] Profiling resnet18...')
    profile_and_save('resnet18', 'Tesla_M40', batch_size=32, num_runs=3)
    print('Profiler complete.')

    print('\n[2/4] Running V100/resnet50 engine test...')
    InferenceEngine._skip_sleep = True
    engine = InferenceEngine(gpu='Tesla_V100-PCIE-32GB', model='resnet50', scheduler='fifo')
    for _ in range(5):
        engine.submit_request(batch_size=32)
    completed = engine.run_queue()
    engine.write_to_dashboard()
    print(f'Engine complete: {len(completed)} requests.')

    print('\n[3/4] Comparing schedulers...')
    scheduler_results = {name: _run_scheduler(name) for name in ('fifo', 'sjf', 'hasp')}
    print_comparison_table({name: {'avg_latency_ms': row['avg_latency_ms'], 'p95_latency_ms': row['p95_latency_ms'], 'jains_fairness_index': row['fairness_index'], 'starvation_count': row['starvation_count'], 'throughput': 0.0} for name, row in scheduler_results.items()})

    print('\n[4/4] Running multi-model Poisson experiment...')
    comparison = run_experiment({
        'name': 'multi_model_poisson',
        'models': ['resnet18', 'resnet50', 'vgg16'],
        'arrival_mode': 'poisson',
        'rate': 2.0,
        'duration': 30.0,
    }, 'experiment_results')
    with open(os.path.join('glide', 'experiment_results.json'), 'w', encoding='utf-8') as handle:
        json.dump(comparison, handle, indent=2)
    print('\nFinal summary:')
    print_comparison_table(comparison)
    print('Dashboard metrics written to glide/glide_metrics.json')


if __name__ == '__main__':
    main()
