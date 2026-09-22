"""Focused regression tests for GLIDE's research-facing features."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from glide import database
from glide.batching import DynamicBatcher, StaticBatcher
from glide.model_analyser import decompose_model
from glide.profiler import get_layer_config
from glide.report import build_report, export_report, summarize_trials


class GlideFeatureTests(unittest.TestCase):
    def test_database_metadata_and_decomposition(self) -> None:
        import torch
        from torch import nn

        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, 'layers.sqlite')
            database.init_db(db_path)
            model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten())
            database.record_layer_cost(
                'TestGPU', 'TinyModel', 'Conv2d',
                get_layer_config(model[0]), 2.0, 1.0,
                input_shape=[1, 3, 8, 8], batch_size=1,
                precision='float32', warmup_runs=2, measured_runs=3,
                db_path=db_path,
            )
            profile = database.query_layer_cost(
                'TestGPU', 'TinyModel', 'Conv2d',
                get_layer_config(model[0]), db_path=db_path,
            )
            self.assertEqual(profile['parallelism_degree'], 4)
            self.assertEqual(profile['batch_size'], 1)
            result = decompose_model(model, 'TestGPU', 'TinyModel', (1, 3, 8, 8), db_path)
            self.assertEqual(result['layer_count'], 3)
            self.assertEqual(result['profiled_layer_count'], 1)
            self.assertEqual(result['missing_profile_count'], 2)
            self.assertAlmostEqual(result['total_compute_ms'], 2.0)

    def test_batchers_have_distinct_dispatch_contracts(self) -> None:
        class Request:
            def __init__(self, arrival_time: float) -> None:
                self.arrival_time = arrival_time

        queue = [Request(0.0), Request(0.0)]
        self.assertEqual(len(StaticBatcher(2).select_batch(queue)), 2)
        self.assertEqual(DynamicBatcher(2).select_batch(queue), queue)

    def test_report_exports_all_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(
                {'scenario': {'schedulers': {'fifo': {'avg_latency_ms': 1.0}}}},
                {'seed': 42, 'gpu': 'TestGPU'},
            )
            paths = export_report(report, directory)
            self.assertEqual(set(paths), {'json', 'csv', 'html'})
            self.assertTrue(all(os.path.exists(path) for path in paths.values()))
            with open(paths['json'], encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['config']['seed'], 42)

    def test_trial_summary_includes_confidence_interval(self) -> None:
        summary = summarize_trials([
            {'avg_latency_ms': 10.0},
            {'avg_latency_ms': 12.0},
            {'avg_latency_ms': 11.0},
        ])
        self.assertAlmostEqual(summary['avg_latency_ms']['mean'], 11.0)
        self.assertLess(summary['avg_latency_ms']['ci95_low'], 11.0)
        self.assertGreater(summary['avg_latency_ms']['ci95_high'], 11.0)


if __name__ == '__main__':
    unittest.main()
