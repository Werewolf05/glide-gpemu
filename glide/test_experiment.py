"""Tests for reproducible scheduler experiment simulation."""

import unittest

from glide.experiment import _simulate_scheduler_trace
from glide.workload import WorkloadGenerator


class SchedulerExperimentTests(unittest.TestCase):
    def test_policies_choose_from_the_same_ready_queue(self) -> None:
        trace = [
            {'arrival_time': 0.0, 'model_name': 'vgg16', 'batch_size': 8, 'priority': 0},
            {'arrival_time': 0.0, 'model_name': 'mobilenet_v2', 'batch_size': 1, 'priority': 0},
            {'arrival_time': 0.0, 'model_name': 'resnet18', 'batch_size': 1, 'priority': 0},
        ]

        fifo = _simulate_scheduler_trace(trace, 'Tesla_M40', 'fifo')
        sjf = _simulate_scheduler_trace(trace, 'Tesla_M40', 'sjf')
        hasp = _simulate_scheduler_trace(trace, 'Tesla_M40', 'hasp')

        self.assertEqual(fifo[0].model_name, 'vgg16')
        self.assertEqual(sjf[0].model_name, 'resnet18')
        self.assertEqual(hasp[0].model_name, 'resnet18')
        self.assertGreater(fifo[-1].end_time, fifo[0].end_time)
        self.assertAlmostEqual(fifo[-1].end_time, sjf[-1].end_time)

    def test_hasp_aging_threshold_changes_high_load_order(self) -> None:
        trace = WorkloadGenerator(
            ['resnet18', 'resnet50', 'vgg16', 'mobilenet_v2', 'densenet121', 'alexnet'],
            arrival_mode='poisson',
            rate=40.0,
            duration=3.0,
            seed=42,
            batch_sizes=[1, 8, 16, 32],
        ).generate()

        default = _simulate_scheduler_trace(trace, 'Tesla_M40', 'hasp', 5.0)
        short_aging = _simulate_scheduler_trace(trace, 'Tesla_M40', 'hasp', 0.05)

        self.assertNotEqual(
            [request.request_id for request in default],
            [request.request_id for request in short_aging],
        )


if __name__ == '__main__':
    unittest.main()