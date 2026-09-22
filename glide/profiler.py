"""
GLIDE Mass Layer Profiler

Profiles ALL GPU + Model combinations and stores:
- per-layer compute time
- memory usage
- bottleneck information

Results are stored in SQLite database.
"""

import time
from typing import Any, Dict, List

from . import database


# -----------------------------------------------------------------------------
# GPU PERFORMANCE SCALING
# -----------------------------------------------------------------------------

GPU_SCALE = {
    'Tesla_K80': 1.00,
    'Tesla_M40': 0.90,
    'Tesla_P100_PCIE_16GB': 0.55,
    'Tesla_V100_PCIE_32GB': 0.30,
    'Quadro_RTX_6000': 0.28,
    'NVIDIA_A100_SXM4_40GB': 0.18,
}


# -----------------------------------------------------------------------------
# MODEL + GPU LISTS
# -----------------------------------------------------------------------------

GPU_LIST = [
    'NVIDIA_A100_SXM4_40GB',
    'Quadro_RTX_6000',
    'Tesla_K80',
    'Tesla_M40',
    'Tesla_P100_PCIE_16GB',
    'Tesla_V100_PCIE_32GB',
]

MODEL_LIST = [
    'alexnet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'googlenet',
    'mnasnet0_5',
    'mnasnet0_75',
    'mnasnet1_0',
    'mnasnet1_3',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'resnet101',
    'resnet152',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnext101_32x8d',
    'resnext50_32x4d',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'wide_resnet101_2',
    'wide_resnet50_2',
]


# -----------------------------------------------------------------------------
# LAYER CONFIG EXTRACTION
# -----------------------------------------------------------------------------

def get_layer_config(module: Any) -> Dict[str, Any]:

    import torch.nn as nn

    if isinstance(module, nn.Conv2d):
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

    if isinstance(module, nn.BatchNorm2d):
        return {
            'num_features': module.num_features,
        }

    if isinstance(module, nn.LayerNorm):
        return {'normalized_shape': list(module.normalized_shape)}

    if isinstance(module, nn.Dropout):
        return {'p': module.p}

    if isinstance(module, nn.MultiheadAttention):
        return {'embed_dim': module.embed_dim, 'num_heads': module.num_heads}

    if isinstance(module, nn.ReLU):
        return {
            'inplace': module.inplace,
        }

    if isinstance(module, nn.MaxPool2d):
        return {
            'k': module.kernel_size,
            'stride': module.stride,
        }

    if isinstance(module, nn.AdaptiveAvgPool2d):
        return {
            'output_size': module.output_size,
        }

    return {
        'type': type(module).__name__,
    }


# -----------------------------------------------------------------------------
# MEMORY ESTIMATION
# -----------------------------------------------------------------------------

def output_memory_mb(output: Any) -> float:

    import torch

    if isinstance(output, torch.Tensor):
        return (
            output.element_size() *
            output.nelement()
        ) / (1024 * 1024)

    if isinstance(output, (list, tuple)):

        total = 0.0

        for item in output:

            if isinstance(item, torch.Tensor):

                total += (
                    item.element_size() *
                    item.nelement()
                ) / (1024 * 1024)

        return total

    return 0.0


# -----------------------------------------------------------------------------
# SINGLE MODEL PROFILING
# -----------------------------------------------------------------------------

def profile_model(
    model_name: str,
    gpu_name: str,
    batch_size: int = 32,
    device: str = 'cpu',
    num_warmup: int = 3,
    num_runs: int = 3,
) -> List[Dict[str, Any]]:

    import torch
    import torchvision.models as models

    model = getattr(models, model_name)(pretrained=False)

    model.eval()
    model = model.to(device)

    skip_types = (
        'Sequential',
        'ModuleList',
        'ModuleDict',
        'BasicBlock',
        'Bottleneck',
    )

    captured_inputs: Dict[str, torch.Tensor] = {}

    hooks = []

    def make_hook(layer_name: str):

        def hook(module: Any, inp: Any, out: Any):

            if layer_name in captured_inputs:
                return

            if len(inp) == 0:
                return

            first = inp[0]

            if isinstance(first, torch.Tensor):
                captured_inputs[layer_name] = first.detach().clone()

        return hook

    for name, module in model.named_modules():

        if name == '':
            continue

        if type(module).__name__ in skip_types:
            continue

        hooks.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )

    dummy = torch.randn(
        batch_size,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    results = []

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

        times = []

        with torch.no_grad():

            for _ in range(num_runs):

                try:

                    start = time.perf_counter()

                    output = module(inp)

                    end = time.perf_counter()

                except Exception:
                    break

                elapsed_ms = (end - start) * 1000.0

                scaled_ms = (
                    elapsed_ms *
                    GPU_SCALE.get(gpu_name, 1.0)
                )

                times.append(scaled_ms)

        if not times:
            continue

        try:

            with torch.no_grad():
                output = module(inp)

            memory_mb = output_memory_mb(output)

        except Exception:

            memory_mb = 0.0

        results.append({
            'layer_name': name,
            'layer_type': type(module).__name__,
            'config': config,
            'compute_time_ms': sum(times) / len(times),
            'memory_mb': memory_mb,
        })

    return results


# -----------------------------------------------------------------------------
# PROFILE + SAVE
# -----------------------------------------------------------------------------

def profile_and_save(
    model_name: str,
    gpu_name: str,
    batch_size: int = 32,
    device: str = 'cpu',
    num_runs: int = 3,
):

    results = profile_model(
        model_name=model_name,
        gpu_name=gpu_name,
        batch_size=batch_size,
        device=device,
        num_runs=num_runs,
    )

    database.init_db()

    for result in results:

        database.record_layer_cost(
            gpu=gpu_name,
            model=model_name,
            layer_type=result['layer_type'],
            config=result['config'],
            compute_cost_ms=result['compute_time_ms'],
            memory_cost_mb=result['memory_mb'],
        )

    return results


def _format_config(config: Dict[str, Any], max_len: int = 42) -> str:
    text = ','.join(f'{key}={value}' for key, value in sorted(config.items()))
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + '...'


if __name__ == '__main__':
    model_name = 'resnet18'
    gpu_name = 'Tesla_M40'
    batch_size = 32
    num_runs = 3

    print('=' * 120)
    print('GLIDE LAYER PROFILER (SINGLE RUN)')
    print(f'GPU={gpu_name} | MODEL={model_name} | BATCH_SIZE={batch_size} | NUM_RUNS={num_runs}')
    print('=' * 120)

    start_total = time.perf_counter()
    try:
        results = profile_and_save(
            model_name=model_name,
            gpu_name=gpu_name,
            batch_size=batch_size,
            device='cpu',
            num_runs=num_runs,
        )
    except ModuleNotFoundError as exc:
        print(f'Profiler dependency missing: {exc}')
        print('Install required dependencies (torch, torchvision) and rerun.')
        raise SystemExit(1) from exc
    end_total = time.perf_counter()

    print(f'Profiled {len(results)} layers in {(end_total - start_total):.2f}s')
    print(f'Database: {database.LAYER_DB_PATH}')
    print('-' * 120)
    print(f"{'Layer Name':<32} | {'Type':<16} | {'Compute(ms)':>12} | {'Memory(MB)':>10} | Config")
    print('-' * 120)
    for layer in results:
        print(
            f"{layer['layer_name'][:32]:<32} | "
            f"{layer['layer_type'][:16]:<16} | "
            f"{layer['compute_time_ms']:>12.3f} | "
            f"{layer['memory_mb']:>10.3f} | "
            f"{_format_config(layer['config'])}"
        )

    print('-' * 120)
    print('Top 5 slowest layers:')
    slowest_layers = database.get_slowest_layers(gpu=gpu_name, model=model_name, limit=5)
    for index, row in enumerate(slowest_layers, start=1):
        print(
            f"{index:>2}. {row['layer_type']:<16} "
            f"compute={row['compute_cost_ms']:.3f} ms | "
            f"memory={row['memory_cost_mb']:.3f} MB | "
            f"config={_format_config(row['config'])}"
        )
    print('=' * 120)