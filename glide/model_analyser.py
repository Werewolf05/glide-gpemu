"""Layer-complexity analysis backed by the GLIDE profiling database."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict

from . import database


def analyse_model(model_name: str, gpu_name: str) -> Dict[str, Any]:
    rows = database.get_slowest_layers(gpu=gpu_name, model=model_name, limit=10000)
    totals = defaultdict(float)
    total_time = sum(float(row.get('compute_cost_ms') or 0.0) for row in rows)
    total_memory = sum(float(row.get('memory_cost_mb') or 0.0) for row in rows)
    for row in rows:
        totals[row['layer_type']] += float(row.get('compute_cost_ms') or 0.0)

    distribution = {
        layer_type: {
            'compute_ms': value,
            'percent': (value / total_time * 100.0) if total_time else 0.0,
        }
        for layer_type, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    }
    peak_memory = max((float(row.get('memory_cost_mb') or 0.0) for row in rows), default=0.0)
    return {
        'model': model_name,
        'gpu': gpu_name,
        'layer_count': len(rows),
        'total_compute_ms': total_time,
        'layer_time_distribution': distribution,
        'bottlenecks': rows[:3],
        'memory_peak_mb': peak_memory,
        'memory_total_mb': total_memory,
        'memory_pressure_score': min(1.0, peak_memory / 8192.0),
    }


def print_analysis(analysis: Dict[str, Any]) -> None:
    print(f"Model complexity: {analysis['model']} on {analysis['gpu']}")
    print(f"Layers: {analysis['layer_count']} | Total compute: {analysis['total_compute_ms']:.2f} ms")
    print('\nLayer time distribution:')
    for layer_type, values in analysis['layer_time_distribution'].items():
        bars = '█' * max(1, round(values['percent'] / 3))
        print(f"{layer_type:<12} {bars:<34} {values['percent']:.1f}%")
    print('\nBottleneck layers:')
    for index, row in enumerate(analysis['bottlenecks'], 1):
        print(f"{index}. {row['layer_type']} {row['compute_cost_ms']:.2f} ms {json.dumps(row['config'], sort_keys=True)}")
    print(f"\nMemory peak: {analysis['memory_peak_mb']:.2f} MB")
    print(f"Memory pressure: {analysis['memory_pressure_score']:.3f}")


if __name__ == '__main__':
    import sys
    print_analysis(analyse_model(sys.argv[1], sys.argv[2]))
