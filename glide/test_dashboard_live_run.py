"""Regression tests for live dashboard run status handling."""

import os
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

from glide import dashboard_server


class DashboardLiveRunTests(unittest.TestCase):
    def test_metrics_preserve_terminal_status_after_batch_completion(self) -> None:
        base_payload = {
            'batches': [],
            'status': 'waiting',
            'selected_model': 'resnet18',
            'selected_gpu': 'Tesla_M40',
            'profile_stats': {},
        }
        profile = {
            'compute_time_s': 0.05,
            'memory_peak_gb': 0.5,
            'estimated_bandwidth_gbps': 1.0,
        }

        with (
            patch.object(dashboard_server, '_safe_metrics_payload', return_value=base_payload),
            patch.object(dashboard_server, '_load_selected_model', return_value='resnet18'),
            patch.object(dashboard_server, '_load_selected_gpu', return_value='Tesla_M40'),
            patch.object(dashboard_server, '_load_selected_scheduler', return_value='fifo'),
            patch.object(dashboard_server, '_get_profile_stats', return_value=profile),
            patch.object(dashboard_server, 'get_utilization', return_value={'gpu_util': 40, 'compute_util': 20}),
            patch.object(dashboard_server, '_calculate_dynamic_utilization', return_value={'gpu_util': 30, 'compute_util': 20}),
        ):
            for status in ('completed', 'stopped', 'failed'):
                with self.subTest(status=status):
                    result = dashboard_server._enrich_metrics({
                        'status': status,
                        'model': 'resnet18',
                        'batches': [{'compute_time': 0.05, 'memory_gb': 0.2}],
                    })
                    self.assertEqual(result['status'], status)
                    self.assertEqual(result['total_batches'], 1)

    def test_metrics_use_saved_scheduler_after_previous_run_finishes(self) -> None:
        base_payload = {
            'batches': [],
            'status': 'waiting',
            'selected_model': 'resnet18',
            'selected_gpu': 'Tesla_M40',
            'profile_stats': {},
        }

        with (
            patch.object(dashboard_server, '_safe_metrics_payload', return_value=base_payload),
            patch.object(dashboard_server, '_load_selected_model', return_value='resnet18'),
            patch.object(dashboard_server, '_load_selected_gpu', return_value='Tesla_M40'),
            patch.object(dashboard_server, '_load_selected_scheduler', return_value='hasp'),
            patch.object(dashboard_server, 'get_utilization', return_value={'gpu_util': 40, 'compute_util': 20}),
        ):
            result = dashboard_server._enrich_metrics({
                'status': 'completed',
                'scheduler_name': 'fifo',
                'model': 'resnet18',
                'batches': [],
            })

        self.assertEqual(result['scheduler_name'], 'hasp')

    def test_stop_waits_for_worker_before_accepting_restart(self) -> None:
        worker_started = threading.Event()
        release_worker = threading.Event()
        previous_thread = dashboard_server._RUN_THREAD
        dashboard_server._RUN_THREAD = None
        dashboard_server._RUN_STOP.clear()

        def blocking_worker(*_args: object) -> None:
            worker_started.set()
            release_worker.wait(timeout=2)

        try:
            with tempfile.TemporaryDirectory() as directory:
                with (
                    patch.object(dashboard_server, 'GLIDE_DIR', directory),
                    patch.object(dashboard_server, 'METRICS_PATH', os.path.join(directory, 'metrics.json')),
                    patch.object(dashboard_server, 'SELECTED_MODEL_PATH', os.path.join(directory, 'model.json')),
                    patch.object(dashboard_server, 'SELECTED_GPU_PATH', os.path.join(directory, 'gpu.json')),
                    patch.object(dashboard_server, 'SELECTED_SCHEDULER_PATH', os.path.join(directory, 'scheduler.json')),
                    patch.object(dashboard_server, '_run_live_engine', side_effect=blocking_worker),
                ):
                    with open(dashboard_server.METRICS_PATH, 'w', encoding='utf-8') as handle:
                        handle.write('{"status":"completed","batches":[{"compute_time":99}]}')
                    with open(os.path.splitext(dashboard_server.METRICS_PATH)[0] + '.ndjson', 'w', encoding='utf-8') as handle:
                        handle.write('{"compute_time":99}\n')
                    client = dashboard_server.app.test_client()
                    started = client.post('/api/start_run')
                    self.assertEqual(started.status_code, 202)
                    self.assertTrue(worker_started.wait(timeout=1))
                    with open(os.path.splitext(dashboard_server.METRICS_PATH)[0] + '.ndjson', encoding='utf-8') as handle:
                        self.assertEqual(handle.read(), '')
                    with open(dashboard_server.METRICS_PATH, encoding='utf-8') as handle:
                        self.assertEqual(json.load(handle)['status'], 'starting')

                    stopped = client.post('/api/stop_run')
                    self.assertEqual(stopped.status_code, 202)
                    self.assertEqual(stopped.get_json()['status'], 'stopping')

                    rejected = client.post('/api/start_run')
                    self.assertEqual(rejected.status_code, 409)

                    release_worker.set()
                    dashboard_server._RUN_THREAD.join(timeout=1)
                    restarted = client.post('/api/start_run')
                    self.assertEqual(restarted.status_code, 202)
                    dashboard_server._RUN_THREAD.join(timeout=1)
        finally:
            release_worker.set()
            if dashboard_server._RUN_THREAD is not None:
                dashboard_server._RUN_THREAD.join(timeout=1)
            dashboard_server._RUN_THREAD = previous_thread
            dashboard_server._RUN_STOP.clear()


if __name__ == '__main__':
    unittest.main()