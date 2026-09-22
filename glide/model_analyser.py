"""Model decomposition and layer-complexity analysis for GLIDE."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from . import database
from .profiler import get_layer_config


def decompose_model(
    model: Any,
    gpu_name: str,
    model_name: Optional[str] = None,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    db_path: str = database.LAYER_DB_PATH,
) -> Dict[str, Any]:
    """Estimate an arbitrary PyTorch model from independently profiled layers.

    The model is executed once only to collect input/output shapes. Costs always
    come from the profiling database; an absent profile is reported explicitly.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError('PyTorch is required for model decomposition') from exc

    model_label = model_name or model.__class__.__name__.lower()
    captured: Dict[str, Dict[str, Any]] = {}

    def capture(name: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            first = inputs[0] if inputs else None
            captured[name] = {
                'input_shape': list(first.shape) if isinstance(first, torch.Tensor) else None,
                'output_shape': list(output.shape) if isinstance(output, torch.Tensor) else None,
            }
        return hook

    hooks = [
        (name, module.register_forward_hook(capture(name)))
        for name, module in model.named_modules()
        if name
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape))
    finally:
        for _, hook in hooks:
            hook.remove()
        model.train(was_training)

    layers = []
    missing = []
    for name, module in model.named_modules():
        if not name or name not in captured:
            continue
        layer_type = type(module).__name__
        config = get_layer_config(module)
        profile = database.query_layer_cost(
            gpu_name, model_label, layer_type, config, db_path=db_path
        )
        item = {
            'name': name,
            'layer_type': layer_type,
            'config': config,
            **captured[name],
            'profiled': profile is not None,
            'compute_cost_ms': profile['compute_cost_ms'] if profile else None,
            'memory_cost_mb': profile['memory_cost_mb'] if profile else None,
            'parallelism_degree': profile['parallelism_degree'] if profile else None,
        }
        layers.append(item)
        if profile is None:
            missing.append(item)

    profiled = [layer for layer in layers if layer['profiled']]
    return {
        'model': model_label,
        'gpu': gpu_name,
        'input_shape': list(input_shape),
        'layer_count': len(layers),
        'profiled_layer_count': len(profiled),
        'missing_profile_count': len(missing),
        'total_compute_ms': sum(layer['compute_cost_ms'] or 0.0 for layer in profiled),
        'total_memory_mb': sum(layer['memory_cost_mb'] or 0.0 for layer in profiled),
        'layers': layers,
        'missing_profiles': missing,
    }


def analyse_model(model_name: str, gpu_name: str, db_path: str = database.LAYER_DB_PATH) -> Dict[str, Any]:
    rows = database.get_slowest_layers(gpu=gpu_name, model=model_name, limit=10000, db_path=db_path)
    totals = defaultdict(float)
    total_time = sum(float(row.get('compute_cost_ms') or 0.0) for row in rows)
    total_memory = sum(float(row.get('memory_cost_mb') or 0.0) for row in rows)
    for row in rows:
        totals[row['layer_type']] += float(row.get('compute_cost_ms') or 0.0)
    distribution = {
        layer_type: {'compute_ms': value, 'percent': (value / total_time * 100.0) if total_time else 0.0}
        for layer_type, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    }
    peak_memory = max((float(row.get('memory_cost_mb') or 0.0) for row in rows), default=0.0)
    return {
        'model': model_name, 'gpu': gpu_name, 'layer_count': len(rows),
        'total_compute_ms': total_time, 'layer_time_distribution': distribution,
        'bottlenecks': rows[:3], 'memory_peak_mb': peak_memory,
        'memory_total_mb': total_memory, 'memory_pressure_score': min(1.0, peak_memory / 8192.0),
    }


def print_analysis(analysis: Dict[str, Any]) -> None:
    print(f"Model complexity: {analysis['model']} on {analysis['gpu']}")
    print(f"Layers: {analysis['layer_count']} | Total compute: {analysis['total_compute_ms']:.2f} ms")
    for layer_type, values in analysis.get('layer_time_distribution', {}).items():
        print(f"{layer_type:<16} {values['percent']:>6.1f}%")
    for index, row in enumerate(analysis.get('bottlenecks', []), 1):
        print(f"{index}. {row['layer_type']} {row['compute_cost_ms']:.2f} ms {json.dumps(row['config'], sort_keys=True)}")


if __name__ == '__main__':
    import sys
    print_analysis(analyse_model(sys.argv[1], sys.argv[2]))
