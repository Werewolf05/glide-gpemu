# GPEmu

This repo contains the code, data, corresponding software/packages, and experiment guidelines for GPEmu (GPU Emulator).

All code/experiments have been tested on Chameleon Cloud (https://www.chameleoncloud.org) using Ubuntu20 machines and with CUDA 11.

Most of our experiments are conducted with PyTorch, using [our custom PyTorch Implementation](https://github.com/mengwanguc/pytorch-meng) and [custom TorchVision](https://github.com/mengwanguc/torchvision-meng), with different 
branches corresponding to different experiments. We will specify the branch name in the experiment guidelines.

GPEmu also supports other deep learning frameworks such as TensorFlow and NVIDIA DALI. For example, our reproduction of the FastFlow was 
based on the integration of GPEmu with TensorFlow.

## GPEmu Installation

1. Platform and Image

Our experiments have been tested on [Chameleon Cloud](https://www.chameleoncloud.org) using Ubuntu 20. We recommend using an "ubuntu20-xxx" image.

GPEmu is an emulator with the purpose of prototyping deep learning system research *without real GPUs*. Therefore, no real GPUs are needed for running GPEmu.

2. Set up ssh key for Github
```
bash setup-ssh-key.sh
```

Copy and paste into: https://github.com/settings/keys

3. Clone this repo locally

```
cd ~
git clone https://github.com/mengwanguc/gpemu.git
```

4. Install conda

```
bash install-conda.sh
source ~/.bashrc
```

5. Download and build our mlock package (used to emulate page-locked/pinned memory)

```
bash install-mlock.sh
```


6. Install PyTorch

Install packages required for building pytorch and build our custom branch:

```
bash install-pytorch.sh
```

7. Download our custom torchvision and build it

```
bash install-torchvision.sh
```

8. Update `/etc/security/limits.conf`

```
bash configure-memlock.sh
```

9. Reboot the machine, which may require reconnecting to the instance.

```
sudo reboot
```


## Our other repos

- Our python library for supporting page-locked (pinned) memory using mlock: https://github.com/gustrain/mlock
- Our Kubernetes plugin for emulated GPU: https://github.com/mengwanguc/gpemu-k8s
- Our own implementation of MinIO cache (from DataStall, VLDB '21), as well as our new micro-optimization SSF (Small File First) cache: https://github.com/gustrain/minio
- Our own implementation of CoorDL (distributed MinIO) as well as Locality-Aware Distributed Cache (HiPC): https://github.com/gustrain/ladcache
- Our new micro-optimization Asycn Batch data loader: https://github.com/gustrain/async-loader
- Our dirty repository with unorganized code (we are working on organizing and moving code to this repo): https://github.com/mengwanguc/gpufs

## Annoucements/Notes
*2024/8/27*: We are always trying to polish our repos. However, since the students are all busy with internships this summer, our time is limited. We are expected to be back in early September. Please bear with us, and don't hesitate to reach out to me (wangm12@uchicago.edu) if you have any questions.

## GLIDE — GPU Layer-Level Inference and Dispatching Emulator

GLIDE extends GPEmu with layer-level inference scheduling and emulation capabilities. While GPEmu emulates training workloads at the model level, GLIDE goes deeper — to individual layer timing — and adds a complete inference serving simulation system.

### What GLIDE Adds

| Component | Description |
|-----------|-------------|
| Layer Profiling Database | Per-layer execution costs, memory, and parallelism stored in SQLite |
| Model Decomposition Engine | Estimates total inference time by walking the PyTorch module tree |
| Inference Emulation Engine | Simulates forward pass with request queue and real GPU timing |
| HASP Scheduler | Novel scheduling algorithm with affinity scoring and starvation prevention |
| Batching Strategies | Static, dynamic, and continuous batching implementations |
| Workload Generator | Uniform, Poisson, and bursty request arrival distributions |
| Metrics Framework | Throughput, latency, P95, Jain's Fairness Index, starvation count |
| Live Dashboard | Flask + Chart.js real-time visualization at `localhost:5000` |

### Project Structure

```text
gpemu/
├── examples/imagenet/main.py     # Modified GPEmu training script with --gpemu-enable
├── profiled_data/                # Real GPU profiling data (6 GPUs, 36 models)
└── glide/
    ├── __init__.py
    ├── database.py               # SQLite layer profiling database
    ├── profiler.py               # PyTorch forward hook layer profiler
    ├── engine.py                 # Inference emulation engine
    ├── scheduler.py              # FIFO, SJF, and HASP schedulers
    ├── batching.py               # Static, dynamic, continuous batching
    ├── workload.py               # Workload trace generator
    ├── metrics.py                # Benchmarking and metrics framework
    ├── experiment.py             # Full experiment runner
    ├── model_analyser.py         # Per-layer complexity analysis
    ├── dashboard_server.py       # Flask live dashboard server
    ├── demo.py                   # One-command demo script
    ├── test_engine_all.py        # Comprehensive engine test (216 GPU × model)
    ├── test_scheduler.py         # Scheduler comparison test
    └── layer_db.sqlite           # Generated layer profiling database
```

### Supported Hardware Profiles

| GPU | Architecture | Memory |
|-----|-------------|--------|
| NVIDIA A100-SXM4-40GB | Ampere | 40 GB |
| Tesla V100-PCIE-32GB | Volta | 32 GB |
| Tesla P100-PCIE-16GB | Pascal | 16 GB |
| Quadro RTX 6000 | Turing | 24 GB |
| Tesla M40 | Maxwell | 24 GB |
| Tesla K80 | Kepler | 24 GB |

### Supported Models (36 total)

ResNet (18/34/50/101/152), Wide ResNet (50/101), ResNeXt (50/101), VGG (11/13/16/19 with/without BatchNorm), DenseNet (121/161/169/201), AlexNet, GoogLeNet, MobileNetV2, MobileNetV3 (large/small), MNASNet (0.5/0.75/1.0/1.3), ShuffleNetV2 (0.5/1.0/1.5/2.0), and SqueezeNet (1.0/1.1).

### Setup

GLIDE runs inside a Docker container for compatibility with GPEmu's Ubuntu 20.04 and PyTorch 1.8.0 dependencies.

#### Prerequisites

- Docker installed on your system
- VS Code with GitHub Copilot (recommended)

#### Build and install

```bash
docker run -itd --name glide-dev --network host \
  -v /path/to/gpemu:/workspace/gpemu:z \
  glide-cpu-base bash
docker exec -it glide-dev bash
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install flask minio
```

#### Verify installation

```bash
cd /workspace/gpemu
python -c "import torch; print(torch.__version__)"
python -c "from glide.engine import InferenceEngine; print('GLIDE OK')"
```

### Running GLIDE

#### Live dashboard

```bash
cd /workspace/gpemu
python glide/dashboard_server.py
# Open http://localhost:5000
```

The dashboard controls the complete emulated inference run. Select a GPU,
model, and scheduler, then use **Start new run** on the Overview page.
Progress is written incrementally while the background workload runs; use
**Stop run** to cancel it safely. No second terminal or manual `main.py`
process is required for dashboard runs.

#### Inference engine

```bash
python glide/engine.py
```

Expected output:

```text
GLIDE Inference Engine Test
GPU: Tesla_V100-PCIE-32GB | Model: resnet50
Request ID   Latency(ms)  Compute(ms)  Memory(MB)
ae3664bf     34.24        31.72        2953.11
...
```

#### Layer profiler

```bash
python -m glide.profiler
```

This reports per-layer timing such as `Conv2d ... 61.286ms 98.000MB` and `Linear ... 0.167ms 0.122MB`.

#### Tests and experiments

```bash
python glide/test_engine_all.py  # 6 GPUs × 36 models × 4 batch sizes; expected 216/216 PASS
python glide/test_scheduler.py   # FIFO versus SJF versus HASP
python glide/experiment.py       # Full experiment runner
python -m unittest glide.test_glide_features  # decomposition/report/database regression tests
python glide/demo.py             # One-command end-to-end demo
```

#### Model decomposition and reports

Decompose any compatible PyTorch model using the layer database. Missing
profiles are returned explicitly rather than silently estimated:

```python
import torchvision.models as models
from glide.model_analyser import decompose_model

report = decompose_model(models.resnet18(weights=None), "Tesla_M40", "resnet18")
print(report["total_compute_ms"], report["missing_profile_count"])
```

Experiments export `results.json`, `report.json`, `report.csv`, and
`report.html`. Use `run_batching_experiment()` from `glide.experiment` to
compare static, dynamic, and continuous batching with FIFO, SJF, and HASP.
The dashboard also exposes the same decomposition through
`GET /api/model_decomposition`.

For final evaluation, use `run_repeated_experiment(config, output_dir, trials=5)`
to run independent seed-offset trials. Reports include mean, sample standard
deviation, and approximate 95% confidence intervals. Use
`run_hasp_ablation()` to compare full HASP with variants that disable aging,
memory affinity, or compute affinity. Results should be interpreted as
profile-driven emulation results, not exact production-GPU latency.

### Research methodology and limitations

GLIDE's strongest claim is comparative: under the same generated trace and
profile database, it makes scheduler and batching policies reproducible and
measurable. It does not claim cycle-accurate GPU simulation. Report the profile
source, model/input shape, seed, trial count, and missing-profile count with
each result. For hardware-accuracy validation, compare selected predictions
against measurements from at least one real GPU and report absolute and
relative error. Avoid claiming HASP is universally optimal; evaluate where its
aging and affinity terms help, and include the ablation results.

### HASP Scheduler

HASP (Heterogeneous Affinity Scheduling with Priority Boosting) is GLIDE's novel scheduling algorithm. It prevents starvation while maintaining competitive throughput.

```text
affinity = (1 / compute_ms) * memory_fit_score * (1 + age_boost)

memory_fit_score = 1.0 if memory < threshold, else 0.5
age_boost = (current_time - arrival_time) / AGING_THRESHOLD
AGING_THRESHOLD = 5.0 seconds (configurable)
```

Key properties:

- Requests waiting longer than `AGING_THRESHOLD` get doubled priority.
- No request can be indefinitely delayed.
- Throughput remains competitive with Shortest Job First.
- Jain's Fairness Index is designed to remain above 0.90.

| Metric | FIFO | SJF | HASP |
|--------|------|-----|------|
| Throughput | Baseline | +20–30% | +15–25% |
| Avg Latency | Baseline | -15–25% | -10–20% |
| P95 Latency | Baseline | -20–30% | -25–40% |
| Fairness Index | ~0.82 | ~0.71 | ~0.94 |
| Starvation Events | High | Very High | Zero |

### Metrics Measured

| Metric | Description |
|--------|-------------|
| Throughput | Requests processed per second |
| Avg Latency | Mean end-to-end latency in milliseconds |
| P95/P99 Latency | Tail latency percentiles |
| Queue Length | Pending requests over time |
| Compute Utilisation | Percentage of time the GPU is actively processing |
| Memory Utilisation | Peak memory usage versus GPU capacity |
| Starvation Count | Requests waiting more than 3× average |
| Jain's Fairness Index | 0–1 score; 1 means perfectly fair |

### Team

| Member | Contribution |
|--------|-------------|
| M Pranav | Layer profiling database, inference engine, HASP scheduler, GPEmu integration |
| M Jeethan | Model decomposition, workload generator, batching strategies, metrics framework |
| Manoj N Khatri | Model complexity analyser, live dashboard, experimental report |

**Institution:** CMR Institute of Technology, Bengaluru
**Department:** Electronics and Communication Engineering
**Academic Year:** 2025–2026

### Citation

```text
GPEmu: GPU Emulator for Deep Learning System Research
https://github.com/mengwanguc/gpemu

GLIDE: GPU Layer-Level Inference and Dispatching Emulator
Built on top of GPEmu by Pranav M et al., CMRIT, 2025–2026
```
