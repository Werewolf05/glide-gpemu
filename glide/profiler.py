"""Layer-wise profiler for GLIDE.

Profiles each layer in isolation using real inputs captured from a single
forward pass, then stores the results in the layer database.
"""

import time
from typing import Any, Dict, List

from . import database


def get_layer_config(module: Any) -> Dict[str, Any]:
    """Extract a configuration dictionary for a layer."""
    import torch.nn as nn

    if isinstance(module, nn.Conv2d):
        return {
            'in': module.in_channels,
            'out': module.out_channels,
            'k': module.kernel_size[0],
            'stride': module.stride[0],
            'padding': module.padding[0],
        }
    if isinstance(module, nn.Conv1d):
        return {
            'in': module.in_channels,
            'out': module.out_channels,
            'k': module.kernel_size[0],
            'stride': module.stride[0],
            'padding': module.padding[0],
        }
    if isinstance(module, nn.Linear):
        return {
            'in': module.in_features,
            'out': module.out_features,
        }
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return {
            'num_features': module.num_features,
        }
    if isinstance(module, nn.LayerNorm):
        return {
            'normalized_shape': list(module.normalized_shape),
        }
    if isinstance(module, (nn.ReLU, nn.ReLU6)):
        return {
            'inplace': module.inplace,
        }
    if isinstance(module, nn.MaxPool2d):
        kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        stride = module.stride if isinstance(module.stride, int) else (module.stride[0] if module.stride else 1)
        return {
            'k': kernel_size,
            'stride': stride,
        }
    if isinstance(module, nn.AvgPool2d):
        kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        stride = module.stride if isinstance(module.stride, int) else (module.stride[0] if module.stride else 1)
        return {
            'k': kernel_size,
            'stride': stride,
        }
    if isinstance(module, nn.AdaptiveAvgPool2d):
        output_size = module.output_size
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        return {
            'output_size': list(output_size),
        }
    if isinstance(module, nn.Dropout):
        return {
            'p': module.p,
        }
    return {
        'type': type(module).__name__,
    }


def _output_memory_mb(output: Any) -> float:
    import torch

    if isinstance(output, torch.Tensor):
        return (output.element_size() * output.nelement()) / (1024.0 * 1024.0)

    if isinstance(output, (tuple, list)):
        memory_mb = 0.0
        for item in output:
            if isinstance(item, torch.Tensor):
                memory_mb += (item.element_size() * item.nelement()) / (1024.0 * 1024.0)
        return memory_mb

    return 0.0


def profile_model(
    model_name: str,
    batch_size: int = 32,
    device: str = 'cpu',
    num_warmup: int = 3,
    num_runs: int = 7,
) -> List[Dict[str, Any]]:
    """Profile a model's layers with isolated per-layer timing."""
    import torch
    import torchvision.models as models

    model = getattr(models, model_name)(pretrained=False)
    model.eval()
    model = model.to(device)

    skip_types = ('Sequential', 'ModuleList', 'ModuleDict', 'BasicBlock', 'Bottleneck')

    captured_inputs: Dict[str, torch.Tensor] = {}
    capture_handles: List[Any] = []

    def make_capture_hook(layer_name: str):
        def hook(module: Any, input: Any, output: Any) -> None:
            if layer_name in captured_inputs or len(input) == 0:
                return
            first_input = input[0]
            if isinstance(first_input, torch.Tensor):
                captured_inputs[layer_name] = first_input.detach().clone()

        return hook

    for name, module in model.named_modules():
        if name == '':
            continue
        if type(module).__name__ in skip_types:
            continue
        capture_handles.append(module.register_forward_hook(make_capture_hook(name)))

    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        model(dummy)

    for handle in capture_handles:
        handle.remove()

    results: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if name == '':
            continue
        if type(module).__name__ in skip_types:
            continue
        if name not in captured_inputs:
            continue

        inp = captured_inputs[name]
        config = get_layer_config(module)

        with torch.no_grad():
            for _ in range(num_warmup):
                try:
                    module(inp)
                except Exception:
                    break

        times: List[float] = []
        with torch.no_grad():
            for _ in range(num_runs):
                try:
                    start = time.perf_counter()
                    output = module(inp)
                    end = time.perf_counter()
                except Exception:
                    break
                times.append((end - start) * 1000.0)

        if not times:
            continue

        try:
            with torch.no_grad():
                output = module(inp)
            memory_mb = _output_memory_mb(output)
        except Exception:
            memory_mb = 0.0

        results.append(
            {
                'layer_name': name,
                'layer_type': type(module).__name__,
                'config': config,
                'compute_time_ms': sum(times) / len(times),
                'memory_mb': memory_mb,
            }
        )

    return results


def profile_and_save(
    model_name: str,
    batch_size: int = 32,
    device: str = 'cpu',
    num_runs: int = 7,
) -> List[Dict[str, Any]]:
    """Profile a model and save results to the database."""
    results = profile_model(
        model_name,
        batch_size=batch_size,
        device=device,
        num_runs=num_runs,
    )

    database.init_db()
    for result in results:
        database.record_layer_cost(
            layer_type=result['layer_type'],
            config=result['config'],
            compute_cost_ms=result['compute_time_ms'],
            memory_cost_mb=result['memory_mb'],
        )

    return results


__all__ = ['get_layer_config', 'profile_model', 'profile_and_save']


if __name__ == '__main__':
    print('=' * 80)
    print('GLIDE Layer Profiler')
    print('=' * 80)

    model_name = 'resnet18'
    batch_size = 32
    num_runs = 3

    print(f'\n[PROFILER] Profiling {model_name} with batch_size={batch_size}, num_runs={num_runs}')
    results = profile_and_save(
        model_name=model_name,
        batch_size=batch_size,
        device='cpu',
        num_runs=num_runs,
    )

    print(f'\n[RESULTS] Layer profiles for {model_name}:')
    print(f"{'Layer Name':<28} | {'Layer Type':<18} | {'Compute(ms)':<12} | {'Memory(MB)':<12}")
    print('-' * 80)
    for result in sorted(results, key=lambda item: item['compute_time_ms'], reverse=True):
        print(
            f"{result['layer_name']:<28} | {result['layer_type']:<18} | "
            f"{result['compute_time_ms']:>10.3f}  | {result['memory_mb']:>10.3f}"
        )

    print(f'\n[DATABASE] Top 5 slowest layers:')
    slowest = database.get_slowest_layers(limit=5)
    if slowest:
        print(f"{'Layer Type':<20} | {'Config Summary':<30} | {'Compute(ms)':<12}")
        print('-' * 70)
        for result in slowest:
            config = result['config']
            config_str = ', '.join([f'{key}={value}' for key, value in list(config.items())[:2]])
            print(f"{result['layer_type']:<20} | {config_str:<30} | {result['compute_cost_ms']:>10.3f}")
    else:
        print('(No results in database yet)')

    print('\n' + '=' * 80)
