"""
GLIDE Full Inference Engine Test
Runs all 6 GPUs × 36 models × 4 batch sizes = 864 combinations.
"""

import time
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

BATCH_SIZES = [1, 8, 16, 32]


def main():
    InferenceEngine._skip_sleep = True

    total = len(GPUS) * len(MODELS) * len(BATCH_SIZES)

    print('=' * 100)
    print('GLIDE FULL INFERENCE EMULATION TEST')
    print(f'{len(GPUS)} GPUs  x  {len(MODELS)} Models  x  {len(BATCH_SIZES)} Batch Sizes  =  {total} runs')
    print('=' * 100)

    run_number = 1
    passed = 0
    partial = 0
    failed = 0

    for gpu in GPUS:
        print(f'\n{"=" * 100}')
        print(f'GPU: {gpu}')
        print('=' * 100)

        for model in MODELS:
            for batch_size in BATCH_SIZES:
                try:
                    engine = InferenceEngine(gpu=gpu, model=model)
                    engine.submit_request(batch_size=batch_size)
                    completed = engine.run_queue()

                    if completed:
                        req = completed[0]
                        compute_ms = req.emulated_compute_ms or 0.0
                        memory_mb = engine.current_memory_mb or 0.0
                        latency_ms = req.latency_ms or 0.0

                        if compute_ms > 0 and memory_mb > 0:
                            status = 'PASS'
                            passed += 1
                        elif compute_ms > 0:
                            status = 'PARTIAL'
                            partial += 1
                        else:
                            status = 'FAIL'
                            failed += 1

                        print(
                            f'[{run_number:>4}/{total}] {gpu:<25} | {model:<22} | batch={batch_size:<3} '
                            f'| compute={compute_ms:>8.2f}ms | mem={memory_mb:>7.0f}MB '
                            f'| latency={latency_ms:>6.2f}ms | {status}'
                        )
                    else:
                        failed += 1
                        print(f'[{run_number:>4}/{total}] {gpu:<25} | {model:<22} | batch={batch_size:<3} | FAIL (no output)')

                except Exception as exc:
                    failed += 1
                    print(f'[{run_number:>4}/{total}] {gpu:<25} | {model:<22} | batch={batch_size:<3} | ERROR: {exc}')

                run_number += 1

    print(f'\n{"=" * 100}')
    print('DONE')
    print(f'Total={total}  PASS={passed}  PARTIAL={partial}  FAIL={failed}')
    print(f'Pass Rate: {passed/total*100:.1f}%  |  Partial: {partial/total*100:.1f}%  |  Fail: {failed/total*100:.1f}%')
    print('=' * 100)


if __name__ == '__main__':
    main()