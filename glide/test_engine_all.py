"""Run full GLIDE engine coverage across all GPU/model/batch combinations."""

import os
import sys

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glide.engine import InferenceEngine


GPUS = [
    'NVIDIA_A100-SXM4-40GB',
    'Quadro_RTX_6000',
    'Tesla_K80',
    'Tesla_M40',
    'Tesla_P100-PCIE-16GB',
    'Tesla_V100-PCIE-32GB',
]

MODELS = [
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
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
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
    'wide_resnet50_2',
    'wide_resnet101_2',
]

BATCH_SIZES = [1, 8, 16, 32]


def main() -> None:
    InferenceEngine._skip_sleep = True

    total = len(GPUS) * len(MODELS) * len(BATCH_SIZES)
    passed = 0
    failed = 0

    print('=' * 140)
    print('GLIDE ENGINE ALL-COMBINATION TEST')
    print(f'{len(GPUS)} GPUs x {len(MODELS)} models x {len(BATCH_SIZES)} batch sizes = {total} runs')
    print('=' * 140)
    print('GPU | Model | BatchSize | Compute(ms) | Memory(MB) | Latency(ms) | PASS/FAIL')
    print('-' * 140)

    for gpu in GPUS:
        for model in MODELS:
            for batch_size in BATCH_SIZES:
                status = 'FAIL'
                compute_ms = 0.0
                memory_mb = 0.0
                latency_ms = 0.0

                try:
                    engine = InferenceEngine(gpu=gpu, model=model, scheduler='fifo')
                    engine.submit_request(batch_size=batch_size)
                    completed = engine.run_queue()
                    if completed:
                        request = completed[0]
                        compute_ms = request.emulated_compute_ms or 0.0
                        memory_mb = request.memory_mb or 0.0
                        latency_ms = request.latency_ms or 0.0
                        if compute_ms > 0.0:
                            status = 'PASS'
                except Exception:
                    status = 'FAIL'

                if status == 'PASS':
                    passed += 1
                else:
                    failed += 1

                print(
                    f'{gpu} | {model} | {batch_size} | '
                    f'{compute_ms:.3f} | {memory_mb:.3f} | {latency_ms:.3f} | {status}'
                )

    print('-' * 140)
    print(f'SUMMARY: Total={total} PASS={passed} FAIL={failed}')
    print('=' * 140)


if __name__ == '__main__':
    main()
