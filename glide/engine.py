"""
GLIDE Inference Emulation Engine

Simulates GPU inference without real hardware using GPEmu's profiled timing data.
Manages request queues, calculates latency, and integrates with the dashboard.
"""

import csv
import dataclasses
import fcntl
import json
import math
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional


GLIDE_DIR = os.path.dirname(os.path.abspath(__file__))
GPEMU_ROOT = os.path.dirname(GLIDE_DIR)
PROFILED_DATA_ROOT = os.path.join(GPEMU_ROOT, 'profiled_data')
SELECTED_GPU_PATH = os.path.join(GLIDE_DIR, 'selected_gpu.json')
SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')
DASHBOARD_METRICS_PATH = os.path.join(GLIDE_DIR, 'glide_metrics.json')


@dataclasses.dataclass
class InferenceRequest:
    """Represents a single inference request."""

    request_id: str
    model_name: str
    batch_size: int
    arrival_time: float
    priority: int = 0
    status: str = 'queued'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    emulated_compute_ms: Optional[float] = None
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _normalize_gpu(gpu_name: Optional[str]) -> str:
    if not gpu_name:
        return 'Tesla_M40'
    return gpu_name.replace(' ', '_').replace('(', '').replace(')', '')


def _normalize_model(model_name: Optional[str]) -> str:
    if not model_name:
        return 'resnet18'
    return model_name.strip().lower()


def _interpolate_value(batch_sizes: List[int], values: List[float], target_batch: int) -> Optional[float]:
    if not batch_sizes or not values:
        return None

    if len(batch_sizes) == 1:
        return values[0]

    sorted_pairs = sorted(zip(batch_sizes, values))
    batch_sizes_sorted = [batch for batch, _ in sorted_pairs]
    values_sorted = [value for _, value in sorted_pairs]

    if target_batch <= batch_sizes_sorted[0]:
        return values_sorted[0]
    if target_batch >= batch_sizes_sorted[-1]:
        return values_sorted[-1]

    for index in range(len(batch_sizes_sorted) - 1):
        batch_low = batch_sizes_sorted[index]
        batch_high = batch_sizes_sorted[index + 1]
        value_low = values_sorted[index]
        value_high = values_sorted[index + 1]

        if batch_low <= target_batch <= batch_high:
            if batch_high == batch_low:
                return value_low
            ratio = (target_batch - batch_low) / (batch_high - batch_low)
            return value_low + ratio * (value_high - value_low)

    return values_sorted[-1]


def get_profiled_compute_time(gpu: str, model: str, batch_size: int) -> Optional[float]:
    gpu_norm = _normalize_gpu(gpu)
    model_norm = _normalize_model(model)

    csv_path = os.path.join(
        PROFILED_DATA_ROOT,
        'time',
        'compute',
        'forward',
        gpu_norm,
        model_norm,
        'time_by_batch_size.csv',
    )

    if not os.path.exists(csv_path):
        return None

    try:
        batch_sizes: List[int] = []
        times_ms: List[float] = []

        with open(csv_path, 'r', encoding='utf-8') as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                try:
                    batch = int(float(row.get('Batch_Size', row.get('batch_size', 0))))
                    time_seconds = float(row.get('Time_In_SECONDS', row.get('time_s', 0)))
                except (TypeError, ValueError):
                    continue
                batch_sizes.append(batch)
                times_ms.append(time_seconds * 1000.0)

        if batch_sizes:
            return _interpolate_value(batch_sizes, times_ms, batch_size)
    except Exception:
        pass

    return None


def get_profiled_memory(gpu: str, model: str, batch_size: int) -> Optional[float]:
    gpu_norm = _normalize_gpu(gpu)
    model_norm = _normalize_model(model)

    csv_path = os.path.join(
        PROFILED_DATA_ROOT,
        'memory',
        gpu_norm,
        model_norm,
        'memory.csv',
    )

    if not os.path.exists(csv_path):
        return None

    try:
        batch_sizes: List[int] = []
        peak_values: List[float] = []

        with open(csv_path, 'r', encoding='utf-8') as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                try:
                    batch = int(float(row.get('batch_size', row.get('Batch_Size', 0))))
                    peak_mb = float(row.get('peak', row.get('Peak', 0)))
                except (TypeError, ValueError):
                    continue
                batch_sizes.append(batch)
                peak_values.append(peak_mb)

        if batch_sizes:
            return _interpolate_value(batch_sizes, peak_values, batch_size)
    except Exception:
        pass

    return None


class InferenceEngine:
    """GPU inference emulation engine using profiled timing/memory data."""

    _skip_sleep = False  # Set to True for testing to avoid actual time delays

    def __init__(self, gpu: Optional[str] = None, model: Optional[str] = None):
        self.gpu = gpu or self._load_selected_gpu()
        self.model = model or self._load_selected_model()
        self.request_queue: List[InferenceRequest] = []
        self.completed_requests: List[InferenceRequest] = []
        self.current_memory_mb = 0.0
        self.is_running = False

    def _load_selected_gpu(self) -> str:
        if not os.path.exists(SELECTED_GPU_PATH):
            return 'Tesla_M40'
        try:
            with open(SELECTED_GPU_PATH, 'r', encoding='utf-8') as file_handle:
                data = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return 'Tesla_M40'
        if isinstance(data, dict):
            return data.get('gpu', 'Tesla_M40')
        return 'Tesla_M40'

    def _load_selected_model(self) -> str:
        if not os.path.exists(SELECTED_MODEL_PATH):
            return 'resnet18'
        try:
            with open(SELECTED_MODEL_PATH, 'r', encoding='utf-8') as file_handle:
                data = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return 'resnet18'
        if isinstance(data, dict):
            return data.get('model', 'resnet18')
        return 'resnet18'

    def submit_request(self, model_name: Optional[str] = None, batch_size: int = 32, priority: int = 0) -> str:
        model = model_name or self.model
        request_id = str(uuid.uuid4())[:8]
        request = InferenceRequest(
            request_id=request_id,
            model_name=model,
            batch_size=batch_size,
            arrival_time=time.time(),
            priority=priority,
        )
        self.request_queue.append(request)
        return request_id

    def process_next(self) -> Optional[InferenceRequest]:
        if not self.request_queue:
            return None

        self.request_queue.sort(key=lambda request: (-request.priority, request.arrival_time))
        request = self.request_queue.pop(0)

        request.status = 'processing'
        request.start_time = time.time()

        compute_time_ms = get_profiled_compute_time(self.gpu, request.model_name, request.batch_size)
        if compute_time_ms is None:
            compute_time_ms = request.batch_size * 2.0
        request.emulated_compute_ms = compute_time_ms
        if not self._skip_sleep:
            time.sleep(compute_time_ms / 1000.0)

        memory_mb = get_profiled_memory(self.gpu, request.model_name, request.batch_size)
        if memory_mb is not None:
            self.current_memory_mb = memory_mb
            request.memory_mb = memory_mb

        request.end_time = time.time()
        request.status = 'completed'
        request.latency_ms = (request.end_time - request.arrival_time) * 1000.0
        self.completed_requests.append(request)
        return request

    def run_queue(self) -> List[InferenceRequest]:
        self.is_running = True
        try:
            while self.request_queue:
                self.process_next()
        finally:
            self.is_running = False
        return self.completed_requests

    def get_queue_status(self) -> Dict[str, Any]:
        queue_length = len(self.request_queue)
        completed_count = len(self.completed_requests)

        avg_latency_ms = 0.0
        p95_latency_ms = 0.0
        if self.completed_requests:
            latencies = [request.latency_ms for request in self.completed_requests if request.latency_ms is not None]
            if latencies:
                avg_latency_ms = sum(latencies) / len(latencies)
                latencies_sorted = sorted(latencies)
                p95_index = max(0, int(math.ceil(len(latencies_sorted) * 0.95)) - 1)
                p95_latency_ms = latencies_sorted[p95_index]

        return {
            'queue_length': queue_length,
            'completed_count': completed_count,
            'avg_latency_ms': round(avg_latency_ms, 2),
            'p95_latency_ms': round(p95_latency_ms, 2),
            'current_memory_mb': round(self.current_memory_mb, 2),
            'is_running': self.is_running,
        }

    def get_results(self) -> List[Dict[str, Any]]:
        return [request.to_dict() for request in self.completed_requests]

    def reset(self) -> None:
        self.request_queue.clear()
        self.completed_requests.clear()
        self.current_memory_mb = 0.0
        self.is_running = False

    def write_to_dashboard(self) -> None:
        payload = {
            'status': 'completed',
            'engine_results': {
                'gpu': self.gpu,
                'model': self.model,
                'queue_status': self.get_queue_status(),
                'completed_requests': self.get_results(),
            },
            'timestamp': time.time(),
        }

        try:
            os.makedirs(os.path.dirname(DASHBOARD_METRICS_PATH), exist_ok=True)
            with open(DASHBOARD_METRICS_PATH, 'a+', encoding='utf-8') as file_handle:
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
                try:
                    file_handle.seek(0)
                    raw = file_handle.read().strip()
                    data: Dict[str, Any] = {}
                    if raw:
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            data = {}
                    data.update(payload)
                    file_handle.seek(0)
                    file_handle.truncate()
                    json.dump(data, file_handle)
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                finally:
                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            print(f'[WARNING] Failed to write dashboard metrics: {exc}')



def run_comprehensive_test() -> None:
    """Test all 216 GPU/model combinations and report results."""
    gpus = [
        'NVIDIA_A100-SXM4-40GB',
        'Quadro_RTX_6000',
        'Tesla_K80',
        'Tesla_M40',
        'Tesla_P100-PCIE-16GB',
        'Tesla_V100-PCIE-32GB',
    ]

    models = [
        'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
        'googlenet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
        'squeezenet1_0', 'squeezenet1_1',
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
        'wide_resnet50_2', 'wide_resnet101_2',
    ]

    print("=" * 100)
    print("GLIDE Comprehensive Test: All GPU/Model Combinations")
    print("=" * 100)
    print(f"Testing {len(gpus)} GPUs × {len(models)} models = {len(gpus) * len(models)} combinations\n")

    # Enable skip_sleep for all engines
    InferenceEngine._skip_sleep = True

    results = []
    total = 0
    passed = 0
    partial = 0
    failed = 0

    for gpu in gpus:
        for model in models:
            total += 1
            try:
                # Create engine and submit single request
                engine = InferenceEngine(gpu=gpu, model=model)
                request_id = engine.submit_request(batch_size=32)
                completed = engine.run_queue()

                if completed:
                    req = completed[0]
                    compute_ms = req.emulated_compute_ms or 0.0
                    memory_mb = engine.current_memory_mb or 0.0
                    latency_ms = req.latency_ms or 0.0

                    # Determine status
                    if compute_ms > 0 and memory_mb > 0:
                        status = "PASS"
                        passed += 1
                    elif compute_ms > 0:
                        status = "PARTIAL"
                        partial += 1
                    else:
                        status = "FAIL"
                        failed += 1

                    results.append({
                        'gpu': gpu,
                        'model': model,
                        'compute_ms': compute_ms,
                        'memory_mb': memory_mb,
                        'latency_ms': latency_ms,
                        'status': status,
                    })
                else:
                    status = "FAIL"
                    failed += 1
                    results.append({
                        'gpu': gpu,
                        'model': model,
                        'compute_ms': 0.0,
                        'memory_mb': 0.0,
                        'latency_ms': 0.0,
                        'status': status,
                    })
            except Exception as exc:
                failed += 1
                results.append({
                    'gpu': gpu,
                    'model': model,
                    'compute_ms': 0.0,
                    'memory_mb': 0.0,
                    'latency_ms': 0.0,
                    'status': f"ERROR: {str(exc)[:30]}",
                })

    # Print results table
    print(f"{'GPU':<25} | {'Model':<25} | {'Compute(ms)':<12} | {'Memory(MB)':<12} | {'Status':<10}")
    print("-" * 100)
    for result in results:
        print(
            f"{result['gpu']:<25} | {result['model']:<25} | "
            f"{result['compute_ms']:>10.2f}  | {result['memory_mb']:>10.2f}  | {result['status']:<10}"
        )

    # Print summary
    print("=" * 100)
    print(f"SUMMARY: Total={total}, PASS={passed}, PARTIAL={partial}, FAIL={failed}")
    print(f"Pass Rate: {(passed/total*100):.1f}% | Partial Rate: {(partial/total*100):.1f}% | Fail Rate: {(failed/total*100):.1f}%")
    print("=" * 100)

    # Reset skip_sleep
    InferenceEngine._skip_sleep = False


if __name__ == '__main__':
    # Check for --test-all flag
    if '--test-all' in sys.argv:
        run_comprehensive_test()
    else:
        print('=' * 60)
        print('GLIDE Inference Engine Test')
        print('=' * 60)

        engine = InferenceEngine(gpu='Tesla_V100-PCIE-32GB', model='resnet50')
        print(f'\n[ENGINE] GPU: {engine.gpu}')
        print(f'[ENGINE] Model: {engine.model}')

        print('\n[SUBMIT] Submitting 5 inference requests...')
        for index in range(5):
            request_id = engine.submit_request(batch_size=32)
            print(f'  Request {index + 1}: {request_id}')

        print('\n[PROCESS] Running queue...')
        completed_requests = engine.run_queue()

        print(f'\n[RESULTS] Completed {len(completed_requests)} requests:')
        print(f"{'Request ID':<12} {'Latency (ms)':<15} {'Emulated Compute (ms)':<22} {'Memory(MB)':<12}")
        print('-' * 66)
        for request in completed_requests:
            print(
                f'{request.request_id:<12} {request.latency_ms:<15.2f} '
                f'{request.emulated_compute_ms:<22.2f} {((request.memory_mb or 0.0)):<12.2f}'
            )

        print('\n[STATS]')
        status = engine.get_queue_status()
        print(f"  Queue Length: {status['queue_length']}")
        print(f"  Completed: {status['completed_count']}")
        print(f"  Avg Latency: {status['avg_latency_ms']:.2f} ms")
        print(f"  P95 Latency: {status['p95_latency_ms']:.2f} ms")
        print(f"  Memory Used: {status['current_memory_mb']:.2f} MB")

        print('\n[DASHBOARD] Writing results to dashboard...')
        engine.write_to_dashboard()
        print('  Done!')

        print('\n' + '=' * 60)
