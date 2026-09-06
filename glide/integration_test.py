"""End-to-end GLIDE integration verification."""

from __future__ import annotations

import os
import sys

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glide import database
from glide.dashboard_server import app
from glide.engine import InferenceEngine
from glide.experiment import run_experiment
from glide.metrics import compute_metrics
from glide.model_analyser import analyse_model
from glide.profiler import profile_and_save
from glide.scheduler import HASPScheduler
from glide.workload import WorkloadGenerator


def check(name, callback) -> bool:
    try:
        callback()
        print(f'PASS | {name}')
        return True
    except Exception as exc:
        print(f'FAIL | {name} | {exc}')
        return False


def main() -> int:
    InferenceEngine._skip_sleep = True
    checks = [
        ('module imports', lambda: (WorkloadGenerator, compute_metrics, HASPScheduler)),
        ('profiler resnet18', lambda: profile_and_save('resnet18', 'Tesla_M40', 32, num_runs=1)),
        ('all schedulers', lambda: [InferenceEngine(model='resnet18', scheduler=name).run_queue() for name in ('fifo', 'sjf', 'hasp')]),
        ('model analyser', lambda: analyse_model('resnet18', 'Tesla_M40')),
        ('one experiment', lambda: run_experiment({'models': ['resnet18'], 'arrival_mode': 'uniform', 'rate': 1, 'duration': 2}, 'integration_results')),
        ('dashboard server', lambda: app.test_client().get('/').status_code == 200),
    ]
    passed = sum(check(name, callback) for name, callback in checks)
    print(f'SUMMARY | {passed}/{len(checks)} passed')
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
