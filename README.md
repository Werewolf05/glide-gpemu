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

GLIDE extends GPEmu by adding layer-level inference scheduling and 
emulation capabilities. While GPEmu emulates training workloads at 
the model level, GLIDE goes deeper — to individual layer timing — 
and adds a complete inference serving simulation system.

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
| Live Dashboard | Flask + Chart.js real-time visualization at localhost:5000 |

### Project Structure

```
gpemu/
├── examples/imagenet/main.py     # Modified GPEmu training script with --gpemu-enable
├── profiled_data/                # GPEmu's real GPU profiling data (6 GPUs, 36 models)
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
    ├── test_engine_all.py        # Comprehensive engine test (216 GPU x Model)
    ├── test_scheduler.py         # Scheduler comparison test
    └── layer_db.sqlite           # Generated layer profiling database

```

### Supported Hardware Profiles

GLIDE uses GPEmu's real profiling data from 6 GPU profiles:

| GPU | Architecture | Memory |
|-----|-------------|--------|
| NVIDIA A100-SXM4-40GB | Ampere | 40 GB |
| Tesla V100-PCIE-32GB | Volta | 32 GB |
| Tesla P100-PCIE-16GB | Pascal | 16 GB |
| Quadro RTX 6000 | Turing | 24 GB |
| Tesla M40 | Maxwell | 24 GB |
| Tesla K80 | Kepler | 24 GB |

### Supported Models (36 total)

ResNet (18/34/50/101/152), Wide ResNet (50/101), ResNeXt (50/101),
VGG (11/13/16/19 with/without BatchNorm), DenseNet (121/161/169/201),
AlexNet, GoogLeNet, MobileNetV2, MobileNetV3 (large/small),
MNASNet (0.5/0.75/1.0/1.3), ShuffleNetV2 (0.5/1.0/1.5/2.0),
SqueezeNet (1.0/1.1)

### Setup

GLIDE runs inside a Docker container to maintain compatibility with
GPEmu's dependencies (Ubuntu 20.04, PyTorch 1.8.0).

#### Prerequisites
- Docker installed on your system
- VS Code with GitHub Copilot (recommended)

#### Build the container

```bash
# The glide-dev container is already configured in the repo
docker run -itd --name glide-dev --network host \
  -v /path/to/gpemu:/workspace/gpemu:z \
  glide-cpu-base bash
```

#### Install dependencies inside container

```bash
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

#### 1. Run the live dashboard

```bash
# Terminal 1 - Start dashboard
docker exec -it glide-dev bash
cd /workspace/gpemu
python glide/dashboard_server.py
# Open http://localhost:5000 in browser
```

```bash
# Terminal 2 - Run GPEmu with GPU emulation
docker exec -it glide-dev bash
cd /workspace/gpemu/examples/imagenet
python main.py --dummy --epochs 1 -b 32 -j 1 --gpemu-enable
```

#### 2. Run the inference engine

```bash
docker exec -it glide-dev bash
cd /workspace/gpemu
python glide/engine.py
```

Expected output:
```
GLIDE Inference Engine Test
GPU: Tesla_V100-PCIE-32GB | Model: resnet50
Request ID   Latency(ms)  Compute(ms)  Memory(MB)
ae3664bf     34.24        31.72        2953.11
...
```

#### 3. Run the layer profiler

```bash
python -m glide.profiler
```

Expected output shows per-layer timing:
```
Conv2d  in=3,out=64,k=7   61.286ms   98.000MB  (slowest)
Linear  in=512,out=1000    0.167ms    0.122MB   (fastest)
```

#### 4. Run the comprehensive engine test (216 combinations)

```bash
python glide/test_engine_all.py
# Tests all 6 GPUs x 36 models x 4 batch sizes
# Expected: 216/216 PASS
```

#### 5. Run scheduler comparison

```bash
python glide/test_scheduler.py
# Compares FIFO vs SJF vs HASP
```

#### 6. Run a full experiment

```bash
python glide/experiment.py
```

#### 7. One-command demo

```bash
python glide/demo.py
# Runs everything end-to-end in under 2 minutes
```

### HASP Scheduler

HASP (Heterogeneous Affinity Scheduling with Priority Boosting) is 
GLIDE's novel scheduling algorithm designed to prevent starvation 
while maintaining competitive throughput.

#### Affinity Score Formula

```
affinity = (1 / compute_ms) * memory_fit_score * (1 + age_boost)

where:
  memory_fit_score = 1.0 if memory < threshold, else 0.5
  age_boost = (current_time - arrival_time) / AGING_THRESHOLD
  AGING_THRESHOLD = 5.0 seconds (configurable)
```

#### Key Properties

- Requests waiting longer than AGING_THRESHOLD get doubled priority
- No request can be indefinitely delayed (starvation-free by design)
- Maintains competitive throughput vs Shortest Job First
- Jain's Fairness Index consistently above 0.90

#### Scheduler Comparison

| Metric | FIFO | SJF | HASP |
|--------|------|-----|------|
| Throughput | Baseline | +20-30% | +15-25% |
| Avg Latency | Baseline | -15-25% | -10-20% |
| P95 Latency | Baseline | -20-30% | -25-40% |
| Fairness Index | ~0.82 | ~0.71 | ~0.94 |
| Starvation Events | High | Very High | Zero |

### Metrics Measured

| Metric | Description |
|--------|-------------|
| Throughput | Requests processed per second |
| Avg Latency | Mean end-to-end latency in ms |
| P95 Latency | 95th percentile tail latency |
| P99 Latency | 99th percentile tail latency |
| Queue Length | Number of pending requests over time |
| Compute Utilisation | % of time GPU is actively processing |
| Memory Utilisation | Peak memory usage vs GPU capacity |
| Starvation Count | Requests waiting more than 3x average |
| Jain's Fairness Index | 0-1 score, 1 = perfectly fair |

### Team

| Member | Contribution |
|--------|-------------|
| Pranav M (1CR23EC104) | Layer profiling database, inference engine, HASP scheduler, GPEmu integration |
| Teammate 2 | Model decomposition, workload generator, batching strategies, metrics framework |
| Teammate 3 | Model complexity analyser, live dashboard, experimental report |

**Institution:** CMR Institute of Technology, Bengaluru  
**Department:** Electronics and Communication Engineering  
**Academic Year:** 2025-2026

### Citation

If you use GLIDE in your research, please cite both GPEmu and GLIDE:

```
GPEmu: GPU Emulator for Deep Learning System Research
https://github.com/mengwanguc/gpemu

GLIDE: GPU Layer-Level Inference and Dispatching Emulator
Built on top of GPEmu by Pranav M et al., CMRIT, 2025-2026
```
