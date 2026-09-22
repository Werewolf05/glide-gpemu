"""Batch formation policies for the GLIDE inference engine."""

from __future__ import annotations

import time
from typing import Any, List


class Batcher:
    """Base interface for selecting requests to dispatch together."""

    name = 'base'

    def __init__(self, batch_size: int = 1, timeout_s: float = 0.05) -> None:
        if batch_size < 1:
            raise ValueError('batch_size must be at least 1')
        if timeout_s < 0:
            raise ValueError('timeout_s must be non-negative')
        self.batch_size = batch_size
        self.timeout_s = timeout_s

    def select_batch(self, queue: List[Any], current_time: float | None = None,
                     flush: bool = False) -> List[Any]:
        raise NotImplementedError

    def get_batch(self, queue: List[Any], current_time: float | None = None,
                  flush: bool = False) -> List[Any]:
        return self.select_batch(queue, current_time, flush)


class StaticBatcher(Batcher):
    """Dispatch only complete fixed-size batches, unless the queue is flushed."""

    name = 'static'

    def select_batch(self, queue: List[Any], current_time: float | None = None,
                     flush: bool = False) -> List[Any]:
        if len(queue) >= self.batch_size:
            return list(queue[:self.batch_size])
        return list(queue) if flush else []


class DynamicBatcher(Batcher):
    """Dispatch when full or when the oldest queued request reaches the timeout."""

    name = 'dynamic'

    def select_batch(self, queue: List[Any], current_time: float | None = None,
                     flush: bool = False) -> List[Any]:
        if not queue:
            return []
        if len(queue) >= self.batch_size or flush:
            return list(queue[:self.batch_size])
        now = time.time() if current_time is None else current_time
        oldest = min(float(getattr(request, 'arrival_time', now)) for request in queue)
        if now - oldest >= self.timeout_s:
            return list(queue)
        return []


class ContinuousBatcher(Batcher):
    """Keep dispatching newly arrived work in bounded micro-batches."""

    name = 'continuous'

    def select_batch(self, queue: List[Any], current_time: float | None = None,
                     flush: bool = False) -> List[Any]:
        return list(queue[:self.batch_size])


def create_batcher(name: str | None, batch_size: int = 1,
                   timeout_s: float = 0.05) -> Batcher:
    policies = {
        'static': StaticBatcher,
        'dynamic': DynamicBatcher,
        'continuous': ContinuousBatcher,
    }
    normalized = (name or 'continuous').lower()
    if normalized not in policies:
        raise ValueError(f'Unknown batcher: {name}')
    return policies[normalized](batch_size=batch_size, timeout_s=timeout_s)
