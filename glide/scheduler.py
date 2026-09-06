"""Scheduling policies for GLIDE inference requests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import InferenceRequest


def _estimate_compute_ms(model_name: str, batch_size: int) -> float:
    name = model_name.lower()
    if name.startswith('vgg'):
        base = 45.0
    elif name.startswith('resnet') or name.startswith('resnext'):
        base = 28.0
    elif name.startswith('densenet'):
        base = 34.0
    elif name.startswith('wide_resnet'):
        base = 36.0
    elif name.startswith('squeezenet') or name.startswith('shufflenet') or name.startswith('mnasnet') or name.startswith('mobilenet'):
        base = 12.0
    elif name.startswith('alexnet') or name.startswith('googlenet'):
        base = 18.0
    else:
        base = 20.0
    return max(1.0, base * max(1, batch_size) / 8.0)


class Scheduler(ABC):
    name: str = 'base'

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu
        self._selected_count = 0

    @abstractmethod
    def select_next(self, queue: List['InferenceRequest'], current_time: float) -> Optional['InferenceRequest']:
        raise NotImplementedError

    def record_completion(self, request: 'InferenceRequest') -> None:
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {'name': self.name, 'selected_count': self._selected_count}


class FIFOScheduler(Scheduler):
    name = 'fifo'

    def select_next(self, queue: List['InferenceRequest'], current_time: float) -> Optional['InferenceRequest']:
        if not queue:
            return None
        self._selected_count += 1
        return min(queue, key=lambda request: request.arrival_time)


class SJFScheduler(Scheduler):
    name = 'sjf'

    def _lookup_compute_ms(self, request: 'InferenceRequest') -> float:
        from .engine import get_profiled_compute_time

        compute_ms = request.emulated_compute_ms
        if compute_ms is None:
            compute_ms = get_profiled_compute_time(self.gpu, request.model_name, request.batch_size)
        if compute_ms is None:
            compute_ms = _estimate_compute_ms(request.model_name, request.batch_size)
        return compute_ms

    def select_next(self, queue: List['InferenceRequest'], current_time: float) -> Optional['InferenceRequest']:
        if not queue:
            return None
        self._selected_count += 1
        return min(queue, key=self._lookup_compute_ms)


class HASPScheduler(Scheduler):
    name = 'hasp'

    def __init__(self, gpu: str, aging_threshold: float = 5.0, memory_threshold_mb: float = 2048.0) -> None:
        super().__init__(gpu)
        self.aging_threshold = aging_threshold
        self.memory_threshold_mb = memory_threshold_mb
        self._starvation_count = 0
        self._starved_request_ids = set()
        self._latencies_ms: List[float] = []

    def _lookup_compute_ms(self, request: 'InferenceRequest') -> float:
        from .engine import get_profiled_compute_time

        compute_ms = request.emulated_compute_ms
        if compute_ms is None:
            compute_ms = get_profiled_compute_time(self.gpu, request.model_name, request.batch_size)
        if compute_ms is None:
            compute_ms = _estimate_compute_ms(request.model_name, request.batch_size)
        return max(compute_ms, 0.001)

    def _lookup_memory_mb(self, request: 'InferenceRequest') -> float:
        from .engine import get_profiled_memory

        memory_mb = request.memory_mb
        if memory_mb is None:
            memory_mb = get_profiled_memory(self.gpu, request.model_name, request.batch_size)
        if memory_mb is None:
            memory_mb = 0.0
        return memory_mb

    def _affinity_score(self, request: 'InferenceRequest', current_time: float) -> float:
        compute_ms = self._lookup_compute_ms(request)
        memory_mb = self._lookup_memory_mb(request)
        memory_fit_score = 1.0 if memory_mb <= self.memory_threshold_mb else 0.5
        wait_seconds = max(0.0, current_time - request.arrival_time)
        age_boost = wait_seconds / self.aging_threshold
        return (1.0 / compute_ms) * memory_fit_score * (1.0 + age_boost)

    def select_next(self, queue: List['InferenceRequest'], current_time: float) -> Optional['InferenceRequest']:
        if not queue:
            return None

        for request in queue:
            wait_seconds = max(0.0, current_time - request.arrival_time)
            if wait_seconds > (2.0 * self.aging_threshold) and request.request_id not in self._starved_request_ids:
                self._starved_request_ids.add(request.request_id)
                self._starvation_count += 1

        self._selected_count += 1
        return max(queue, key=lambda request: self._affinity_score(request, current_time))

    def record_completion(self, request: 'InferenceRequest') -> None:
        if request.latency_ms is not None:
            self._latencies_ms.append(request.latency_ms)

    def get_starvation_count(self) -> int:
        return self._starvation_count

    def get_fairness_index(self) -> float:
        if not self._latencies_ms:
            return 1.0
        total = sum(self._latencies_ms)
        squares = sum(latency * latency for latency in self._latencies_ms)
        if squares == 0.0:
            return 1.0
        return (total * total) / (len(self._latencies_ms) * squares)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'selected_count': self._selected_count,
            'aging_threshold': self.aging_threshold,
            'memory_threshold_mb': self.memory_threshold_mb,
            'starvation_count': self.get_starvation_count(),
            'fairness_index': self.get_fairness_index(),
        }


def create_scheduler(name: str, gpu: str) -> Scheduler:
    scheduler_name = name.lower()
    if scheduler_name == 'fifo':
        return FIFOScheduler(gpu=gpu)
    if scheduler_name == 'sjf':
        return SJFScheduler(gpu=gpu)
    if scheduler_name == 'hasp':
        return HASPScheduler(gpu=gpu)
    raise ValueError(f"Unsupported scheduler '{name}'. Expected one of: fifo, sjf, hasp.")