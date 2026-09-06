"""Deterministic workload trace generation for GLIDE experiments."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Sequence


class WorkloadGenerator:
    def __init__(
        self,
        model_list: Sequence[str],
        arrival_mode: str = 'poisson',
        rate: float = 1.0,
        duration: float = 30.0,
        seed: int = 42,
    ) -> None:
        if not model_list:
            raise ValueError('model_list must contain at least one model')
        if rate <= 0 or duration < 0:
            raise ValueError('rate must be positive and duration must be non-negative')
        if arrival_mode not in {'uniform', 'poisson', 'bursty'}:
            raise ValueError("arrival_mode must be 'uniform', 'poisson', or 'bursty'")
        self.model_list = list(model_list)
        self.arrival_mode = arrival_mode
        self.rate = rate
        self.duration = duration
        self.seed = seed

    def generate(self) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        trace: List[Dict[str, Any]] = []
        arrival_time = 0.0
        index = 0
        while arrival_time < self.duration:
            if self.arrival_mode == 'uniform':
                interval = 1.0 / self.rate
            elif self.arrival_mode == 'poisson':
                interval = rng.expovariate(self.rate)
            else:
                phase = int(arrival_time // 5.0) % 2
                active_rate = self.rate * (5.0 if phase == 0 else 0.2)
                interval = rng.expovariate(active_rate)

            if trace and arrival_time + interval >= self.duration:
                break
            if not trace and arrival_time >= self.duration:
                break
            if trace or arrival_time == 0.0:
                arrival_time += interval if trace or interval > 0 else 0.0
            else:
                arrival_time = 0.0

            if arrival_time >= self.duration:
                break
            trace.append({
                'arrival_time': round(arrival_time, 6),
                'model_name': self.model_list[index % len(self.model_list)],
                'batch_size': 1,
                'priority': 0,
            })
            index += 1
        return trace

    def save_trace(self, path: str) -> List[Dict[str, Any]]:
        trace = self.generate()
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(trace, handle, indent=2)
        return trace

    @staticmethod
    def load_trace(path: str) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as handle:
            trace = json.load(handle)
        if not isinstance(trace, list):
            raise ValueError('Trace JSON must contain a list')
        return trace


def single_model_trace(model: str, rate: float, duration: float) -> List[Dict[str, Any]]:
    return WorkloadGenerator([model], 'poisson', rate, duration).generate()


def multi_model_trace(models: Sequence[str], rate: float, duration: float) -> List[Dict[str, Any]]:
    return WorkloadGenerator(models, 'poisson', rate, duration).generate()


def burst_trace(models: Sequence[str], duration: float) -> List[Dict[str, Any]]:
    return WorkloadGenerator(models, 'bursty', 1.0, duration).generate()
