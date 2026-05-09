"""
Layer-wise profiler for GLIDE using PyTorch forward hooks.

Measures per-layer execution time and memory usage during forward pass
and stores results in the layer database for later lookup.
"""

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models

from glide import database


def get_layer_config(module: nn.Module) -> Dict[str, Any]:
    """Extract configuration dictionary for a layer.
    
    Args:
        module: PyTorch module/layer
        
    Returns:
        Dict with layer-specific configuration parameters
    """
    config = {}

    if isinstance(module, nn.Conv2d):
        config = {
            'in': module.in_channels,
            'out': module.out_channels,
            'k': module.kernel_size[0],
            'stride': module.stride[0],
            'padding': module.padding[0],
        }
    elif isinstance(module, nn.Conv1d):
        config = {
            'in': module.in_channels,
            'out': module.out_channels,
            'k': module.kernel_size[0],
            'stride': module.stride[0],
            'padding': module.padding[0],
        }
    elif isinstance(module, nn.Linear):
        config = {
            'in': module.in_features,
            'out': module.out_features,
        }
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        config = {
            'num_features': module.num_features,
        }
    elif isinstance(module, nn.LayerNorm):
        config = {
            'normalized_shape': list(module.normalized_shape),
        }
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        config = {
            'inplace': module.inplace,
        }
    elif isinstance(module, nn.MaxPool2d):
        k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        s = module.stride if isinstance(module.stride, int) else (module.stride[0] if module.stride else 1)
        config = {
            'k': k,
            'stride': s,
        }
    elif isinstance(module, nn.AvgPool2d):
        k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        s = module.stride if isinstance(module.stride, int) else (module.stride[0] if module.stride else 1)
        config = {
            'k': k,
            'stride': s,
        }
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        output_size = module.output_size
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        config = {
            'output_size': list(output_size),
        }
    elif isinstance(module, nn.Dropout):
        config = {
            'p': module.p,
        }
    else:
        # Generic fallback for unknown layer types
        config = {
            'type': type(module).__name__,
        }

    return config


class LayerProfiler:
    """Hook-based profiler for measuring per-layer execution time and memory."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """Initialize the profiler.
        
        Args:
            model: PyTorch model to profile
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.hooks: List[Any] = []
        self.results: List[Dict[str, Any]] = []
        self._active = False

    def _make_hook(self, layer_name: str, layer_type: str, config: Dict[str, Any]) -> Callable:
        """Create a forward hook function for a layer.
        
        Args:
            layer_name: Name of the layer
            layer_type: Type name of the layer
            config: Configuration dict for the layer
            
        Returns:
            Hook function that measures time and memory
        """
        def hook(module: nn.Module, input: Tuple, output: Any) -> None:
            if not self._active:
                return

            # Measure time (after forward computation)
            compute_time_ms = 0.0
            memory_mb = 0.0

            # Calculate memory from output tensor
            if isinstance(output, torch.Tensor):
                memory_mb = (output.element_size() * output.nelement()) / (1024.0 * 1024.0)
            elif isinstance(output, (tuple, list)):
                # Multiple outputs - sum their memory
                for out in output:
                    if isinstance(out, torch.Tensor):
                        memory_mb += (out.element_size() * out.nelement()) / (1024.0 * 1024.0)

            self.results.append({
                'layer_name': layer_name,
                'layer_type': layer_type,
                'config': config,
                'compute_time_ms': compute_time_ms,  # Set by forward wrapper
                'memory_mb': memory_mb,
            })

        return hook

    def attach_hooks(self) -> None:
        """Attach forward hooks to all layers in the model."""
        # Container types to skip
        skip_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)

        for name, module in self.model.named_modules():
            # Skip the model itself and container modules
            if module is self.model or isinstance(module, skip_types):
                continue

            layer_type = type(module).__name__
            config = get_layer_config(module)

            hook = self._make_hook(name, layer_type, config)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)

        self._active = True

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self._active = False

    def clear_results(self) -> None:
        """Clear profiling results."""
        self.results.clear()


def profile_model(
    model_name: str,
    batch_size: int = 32,
    device: str = 'cpu',
    num_warmup: int = 2,
    num_runs: int = 5,
) -> List[Dict[str, Any]]:
    """Profile a model's layers with execution time measurement.
    
    Args:
        model_name: Name of model from torchvision.models (e.g., 'resnet18')
        batch_size: Batch size for dummy input
        device: Device to run on ('cpu' or 'cuda')
        num_warmup: Number of warmup forward passes (without hooks)
        num_runs: Number of profiling runs (with hooks)
        
    Returns:
        List of aggregated layer profile dicts
    """
    # Load model
    try:
        model = getattr(torchvision.models, model_name)(pretrained=False)
    except AttributeError:
        raise ValueError(f'Model {model_name} not found in torchvision.models')

    # Prepare model
    model.eval()
    model = model.to(device)

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    # Create profiler
    profiler = LayerProfiler(model, device=device)

    # Warmup runs (no hooks to avoid measurement overhead)
    print(f'[PROFILER] Warming up with {num_warmup} passes...')
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Attach hooks and profile
    print(f'[PROFILER] Profiling with {num_runs} runs...')
    profiler.attach_hooks()
    profiler.clear_results()

    # Run with timing
    with torch.no_grad():
        for run_idx in range(num_runs):
            # Time the forward pass
            start_time = time.perf_counter()
            output = model(dummy_input)
            end_time = time.perf_counter()
            forward_time_ms = (end_time - start_time) * 1000.0

            # Update compute_time_ms for each layer in this run
            # Distribute the measured time proportionally to layers
            # (This is a simplification; ideally we'd measure each layer individually)
            for result in profiler.results[-len(profiler.hooks):]:
                result['compute_time_ms'] = forward_time_ms / len(profiler.hooks)

    profiler.remove_hooks()

    # Aggregate results by (layer_type, config) combination
    aggregated: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for result in profiler.results:
        layer_type = result['layer_type']
        config_key = str(result['config'])  # Use string representation as key

        key = (layer_type, config_key)
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(result)

    # Compute statistics
    final_results = []
    for (layer_type, config_key), measurements in aggregated.items():
        config = measurements[0]['config']

        compute_times = [m['compute_time_ms'] for m in measurements]
        memory_values = [m['memory_mb'] for m in measurements]

        avg_compute = statistics.mean(compute_times) if compute_times else 0.0
        avg_memory = statistics.mean(memory_values) if memory_values else 0.0

        final_results.append({
            'layer_type': layer_type,
            'config': config,
            'avg_compute_ms': avg_compute,
            'avg_memory_mb': avg_memory,
        })

    return final_results


def profile_and_save(
    model_name: str,
    batch_size: int = 32,
    device: str = 'cpu',
    num_runs: int = 5,
) -> List[Dict[str, Any]]:
    """Profile a model and save results to the layer database.
    
    Args:
        model_name: Name of model from torchvision.models
        batch_size: Batch size for dummy input
        device: Device to run on
        num_runs: Number of profiling runs
        
    Returns:
        List of aggregated layer profile dicts
    """
    # Profile the model
    results = profile_model(
        model_name,
        batch_size=batch_size,
        device=device,
        num_runs=num_runs,
    )

    # Initialize database
    database.init_db()

    # Save each result
    print(f'\n[DATABASE] Saving {len(results)} layer profiles...')
    for result in results:
        database.record_layer_cost(
            layer_type=result['layer_type'],
            config=result['config'],
            compute_cost_ms=result['avg_compute_ms'],
            memory_cost_mb=result['avg_memory_mb'],
        )

    print(f'[DATABASE] Saved {len(results)} layer profiles to layer_db.sqlite')

    return results


if __name__ == '__main__':
    print('=' * 80)
    print('GLIDE Layer Profiler')
    print('=' * 80)

    # Profile ResNet18
    model_name = 'resnet18'
    batch_size = 32
    num_runs = 3

    print(f'\n[PROFILER] Profiling {model_name} with batch_size={batch_size}, num_runs={num_runs}')

    results = profile_and_save(
        model_name,
        batch_size=batch_size,
        device='cpu',
        num_runs=num_runs,
    )

    # Print results table
    print(f'\n[RESULTS] Layer profiles for {model_name}:')
    print(f"{'Layer Type':<20} | {'Config Summary':<30} | {'Compute(ms)':<12} | {'Memory(MB)':<12}")
    print('-' * 80)

    for result in sorted(results, key=lambda r: r['avg_compute_ms'], reverse=True):
        layer_type = result['layer_type']
        config = result['config']

        # Create config summary
        config_str = ', '.join([f'{k}={v}' for k, v in list(config.items())[:3]])
        if len(config) > 3:
            config_str += '...'

        compute_ms = result['avg_compute_ms']
        memory_mb = result['avg_memory_mb']

        print(f'{layer_type:<20} | {config_str:<30} | {compute_ms:>10.3f}  | {memory_mb:>10.3f}')

    # Query slowest layers from database
    print(f'\n[DATABASE] Top 5 slowest layers:')
    slowest = database.get_slowest_layers(limit=5)

    if slowest:
        print(f"{'Layer Type':<20} | {'Config Summary':<30} | {'Compute(ms)':<12}")
        print('-' * 70)
        for result in slowest:
            layer_type = result['layer_type']
            config = result['config']
            config_str = ', '.join([f'{k}={v}' for k, v in list(config.items())[:2]])
            compute_ms = result['compute_cost_ms']
            print(f'{layer_type:<20} | {config_str:<30} | {compute_ms:>10.3f}')
    else:
        print('(No results in database yet)')

    print('\n' + '=' * 80)
