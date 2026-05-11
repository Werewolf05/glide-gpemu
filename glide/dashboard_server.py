import fcntl
import csv
import json
import os
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request


GLIDE_DIR = '/workspace/gpemu/glide'
METRICS_PATH = os.path.join(GLIDE_DIR, 'glide_metrics.json')
SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')
SELECTED_GPU_PATH = os.path.join(GLIDE_DIR, 'selected_gpu.json')
PROFILED_DATA_ROOT = '/workspace/gpemu/profiled_data'
FALLBACK_PROFILED_DATA_ROOT = '/home/pranav/gpemu/profiled_data'
DEFAULT_MODEL = 'resnet18'
DEFAULT_GPU = 'Tesla_M40'


def _ensure_write_dir() -> None:
  """Ensure a writable GLIDE_DIR exists. On permission errors, fall back to home path.

  Updates module-level path constants to point at the fallback when necessary.
  """
  global GLIDE_DIR, METRICS_PATH, SELECTED_MODEL_PATH, SELECTED_GPU_PATH, PROFILED_DATA_ROOT
  # Try to ensure the configured glide directory exists and is writable.
  try:
    os.makedirs(GLIDE_DIR, exist_ok=True)
  except PermissionError:
    # Can't create the configured GLIDE_DIR; fall back to a per-user config dir.
    user_fallback = os.path.expanduser('~/.gpemu')
    os.makedirs(user_fallback, exist_ok=True)
    GLIDE_DIR = user_fallback
    METRICS_PATH = os.path.join(GLIDE_DIR, 'glide_metrics.json')
    SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')
    SELECTED_GPU_PATH = os.path.join(GLIDE_DIR, 'selected_gpu.json')
    PROFILED_DATA_ROOT = FALLBACK_PROFILED_DATA_ROOT

  # If any of the selected files exist but are not writable by this user,
  # also switch to the per-user fallback directory to avoid PermissionError.
  for p in (SELECTED_MODEL_PATH, SELECTED_GPU_PATH, METRICS_PATH):
    if os.path.exists(p) and not os.access(p, os.W_OK):
      user_fallback = os.path.expanduser('~/.gpemu')
      os.makedirs(user_fallback, exist_ok=True)
      GLIDE_DIR = user_fallback
      METRICS_PATH = os.path.join(GLIDE_DIR, 'glide_metrics.json')
      SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')
      SELECTED_GPU_PATH = os.path.join(GLIDE_DIR, 'selected_gpu.json')
      PROFILED_DATA_ROOT = FALLBACK_PROFILED_DATA_ROOT
      break

MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    'resnet18': {
        'display_name': 'ResNet18',
        'description': 'A compact image classifier with residual skip connections. It balances speed and accuracy well for demos and baseline experiments.',
        'layers': '18',
        'use_case': 'Fast baseline image classification and education demos.'
    },
    'resnet50': {
        'display_name': 'ResNet50',
        'description': 'A deeper residual network with stronger representation power. It is slower than ResNet18 but usually more accurate.',
        'layers': '50',
        'use_case': 'General-purpose high-accuracy image classification.'
    },
    'alexnet': {
        'display_name': 'AlexNet',
        'description': 'A classic early deep learning CNN architecture. It is historically important and easy to explain to beginners.',
        'layers': '8',
        'use_case': 'Teaching core CNN concepts and historical comparisons.'
    },
    'vgg16': {
        'display_name': 'VGG16',
        'description': 'A very uniform deep CNN with repeated 3x3 convolutions. It is simple conceptually but heavier to compute.',
        'layers': '16',
        'use_case': 'Explain depth vs compute cost tradeoffs in CNNs.'
    },
    'squeezenet1_0': {
        'display_name': 'SqueezeNet',
        'description': 'A parameter-efficient CNN designed to be lightweight while keeping reasonable classification quality.',
        'layers': '18 (fire modules)',
        'use_case': 'Resource-constrained inference and efficiency demos.'
    }
}


GPU_MODEL_UTILIZATION: Dict[str, Dict[str, Dict[str, int]]] = {
    "NVIDIA_A100-SXM4-40GB": {
      "alexnet": {"gpu_util": 72, "compute_util": 45},
      "densenet121": {"gpu_util": 85, "compute_util": 58},
      "densenet161": {"gpu_util": 88, "compute_util": 61},
      "densenet169": {"gpu_util": 86, "compute_util": 59},
      "densenet201": {"gpu_util": 87, "compute_util": 60},
      "googlenet": {"gpu_util": 78, "compute_util": 50},
      "mnasnet0_5": {"gpu_util": 55, "compute_util": 30},
      "mnasnet0_75": {"gpu_util": 58, "compute_util": 33},
      "mnasnet1_0": {"gpu_util": 61, "compute_util": 35},
      "mnasnet1_3": {"gpu_util": 64, "compute_util": 38},
      "mobilenet_v2": {"gpu_util": 60, "compute_util": 34},
      "mobilenet_v3_large": {"gpu_util": 62, "compute_util": 36},
      "mobilenet_v3_small": {"gpu_util": 56, "compute_util": 31},
      "resnet18": {"gpu_util": 75, "compute_util": 48},
      "resnet34": {"gpu_util": 78, "compute_util": 51},
      "resnet50": {"gpu_util": 82, "compute_util": 55},
      "resnet101": {"gpu_util": 85, "compute_util": 57},
      "resnet152": {"gpu_util": 87, "compute_util": 59},
      "resnext50_32x4d": {"gpu_util": 83, "compute_util": 56},
      "resnext101_32x8d": {"gpu_util": 89, "compute_util": 63},
      "shufflenet_v2_x0_5": {"gpu_util": 50, "compute_util": 26},
      "shufflenet_v2_x1_0": {"gpu_util": 54, "compute_util": 29},
      "shufflenet_v2_x1_5": {"gpu_util": 57, "compute_util": 32},
      "shufflenet_v2_x2_0": {"gpu_util": 61, "compute_util": 35},
      "squeezenet1_0": {"gpu_util": 65, "compute_util": 38},
      "squeezenet1_1": {"gpu_util": 63, "compute_util": 36},
      "vgg11": {"gpu_util": 80, "compute_util": 55},
      "vgg11_bn": {"gpu_util": 81, "compute_util": 56},
      "vgg13": {"gpu_util": 82, "compute_util": 57},
      "vgg13_bn": {"gpu_util": 83, "compute_util": 57},
      "vgg16": {"gpu_util": 84, "compute_util": 58},
      "vgg16_bn": {"gpu_util": 85, "compute_util": 59},
      "vgg19": {"gpu_util": 86, "compute_util": 60},
      "vgg19_bn": {"gpu_util": 86, "compute_util": 60},
      "wide_resnet50_2": {"gpu_util": 84, "compute_util": 57},
      "wide_resnet101_2": {"gpu_util": 88, "compute_util": 62}
    },
    "Quadro_RTX_6000": {
      "alexnet": {"gpu_util": 68, "compute_util": 40},
      "densenet121": {"gpu_util": 78, "compute_util": 52},
      "densenet161": {"gpu_util": 80, "compute_util": 55},
      "densenet169": {"gpu_util": 79, "compute_util": 53},
      "densenet201": {"gpu_util": 80, "compute_util": 54},
      "googlenet": {"gpu_util": 72, "compute_util": 45},
      "mnasnet0_5": {"gpu_util": 50, "compute_util": 26},
      "mnasnet0_75": {"gpu_util": 53, "compute_util": 28},
      "mnasnet1_0": {"gpu_util": 56, "compute_util": 30},
      "mnasnet1_3": {"gpu_util": 59, "compute_util": 33},
      "mobilenet_v2": {"gpu_util": 55, "compute_util": 29},
      "mobilenet_v3_large": {"gpu_util": 57, "compute_util": 31},
      "mobilenet_v3_small": {"gpu_util": 51, "compute_util": 27},
      "resnet18": {"gpu_util": 70, "compute_util": 43},
      "resnet34": {"gpu_util": 73, "compute_util": 46},
      "resnet50": {"gpu_util": 76, "compute_util": 50},
      "resnet101": {"gpu_util": 78, "compute_util": 52},
      "resnet152": {"gpu_util": 80, "compute_util": 54},
      "resnext50_32x4d": {"gpu_util": 77, "compute_util": 51},
      "resnext101_32x8d": {"gpu_util": 82, "compute_util": 57},
      "shufflenet_v2_x0_5": {"gpu_util": 45, "compute_util": 22},
      "shufflenet_v2_x1_0": {"gpu_util": 49, "compute_util": 25},
      "shufflenet_v2_x1_5": {"gpu_util": 52, "compute_util": 27},
      "shufflenet_v2_x2_0": {"gpu_util": 56, "compute_util": 30},
      "squeezenet1_0": {"gpu_util": 60, "compute_util": 34},
      "squeezenet1_1": {"gpu_util": 58, "compute_util": 32},
      "vgg11": {"gpu_util": 74, "compute_util": 50},
      "vgg11_bn": {"gpu_util": 75, "compute_util": 51},
      "vgg13": {"gpu_util": 76, "compute_util": 52},
      "vgg13_bn": {"gpu_util": 77, "compute_util": 52},
      "vgg16": {"gpu_util": 78, "compute_util": 53},
      "vgg16_bn": {"gpu_util": 79, "compute_util": 54},
      "vgg19": {"gpu_util": 80, "compute_util": 55},
      "vgg19_bn": {"gpu_util": 80, "compute_util": 55},
      "wide_resnet50_2": {"gpu_util": 78, "compute_util": 52},
      "wide_resnet101_2": {"gpu_util": 82, "compute_util": 56}
    },
    "Tesla_K80": {
      "alexnet": {"gpu_util": 35, "compute_util": 18},
      "densenet121": {"gpu_util": 48, "compute_util": 28},
      "densenet161": {"gpu_util": 50, "compute_util": 30},
      "densenet169": {"gpu_util": 49, "compute_util": 29},
      "densenet201": {"gpu_util": 50, "compute_util": 30},
      "googlenet": {"gpu_util": 40, "compute_util": 22},
      "mnasnet0_5": {"gpu_util": 22, "compute_util": 10},
      "mnasnet0_75": {"gpu_util": 24, "compute_util": 11},
      "mnasnet1_0": {"gpu_util": 26, "compute_util": 12},
      "mnasnet1_3": {"gpu_util": 28, "compute_util": 14},
      "mobilenet_v2": {"gpu_util": 25, "compute_util": 11},
      "mobilenet_v3_large": {"gpu_util": 27, "compute_util": 13},
      "mobilenet_v3_small": {"gpu_util": 23, "compute_util": 10},
      "resnet18": {"gpu_util": 38, "compute_util": 20},
      "resnet34": {"gpu_util": 40, "compute_util": 22},
      "resnet50": {"gpu_util": 44, "compute_util": 25},
      "resnet101": {"gpu_util": 46, "compute_util": 27},
      "resnet152": {"gpu_util": 48, "compute_util": 28},
      "resnext50_32x4d": {"gpu_util": 45, "compute_util": 26},
      "resnext101_32x8d": {"gpu_util": 52, "compute_util": 32},
      "shufflenet_v2_x0_5": {"gpu_util": 18, "compute_util": 8},
      "shufflenet_v2_x1_0": {"gpu_util": 21, "compute_util": 10},
      "shufflenet_v2_x1_5": {"gpu_util": 23, "compute_util": 11},
      "shufflenet_v2_x2_0": {"gpu_util": 26, "compute_util": 12},
      "squeezenet1_0": {"gpu_util": 30, "compute_util": 15},
      "squeezenet1_1": {"gpu_util": 28, "compute_util": 14},
      "vgg11": {"gpu_util": 43, "compute_util": 26},
      "vgg11_bn": {"gpu_util": 44, "compute_util": 27},
      "vgg13": {"gpu_util": 45, "compute_util": 27},
      "vgg13_bn": {"gpu_util": 46, "compute_util": 28},
      "vgg16": {"gpu_util": 47, "compute_util": 28},
      "vgg16_bn": {"gpu_util": 48, "compute_util": 29},
      "vgg19": {"gpu_util": 49, "compute_util": 30},
      "vgg19_bn": {"gpu_util": 49, "compute_util": 30},
      "wide_resnet50_2": {"gpu_util": 47, "compute_util": 28},
      "wide_resnet101_2": {"gpu_util": 51, "compute_util": 31}
    },
    "Tesla_M40": {
      "alexnet": {"gpu_util": 42, "compute_util": 22},
      "densenet121": {"gpu_util": 55, "compute_util": 32},
      "densenet161": {"gpu_util": 58, "compute_util": 35},
      "densenet169": {"gpu_util": 56, "compute_util": 33},
      "densenet201": {"gpu_util": 57, "compute_util": 34},
      "googlenet": {"gpu_util": 48, "compute_util": 27},
      "mnasnet0_5": {"gpu_util": 28, "compute_util": 13},
      "mnasnet0_75": {"gpu_util": 30, "compute_util": 15},
      "mnasnet1_0": {"gpu_util": 33, "compute_util": 17},
      "mnasnet1_3": {"gpu_util": 35, "compute_util": 19},
      "mobilenet_v2": {"gpu_util": 31, "compute_util": 15},
      "mobilenet_v3_large": {"gpu_util": 33, "compute_util": 17},
      "mobilenet_v3_small": {"gpu_util": 29, "compute_util": 14},
      "resnet18": {"gpu_util": 45, "compute_util": 25},
      "resnet34": {"gpu_util": 48, "compute_util": 27},
      "resnet50": {"gpu_util": 52, "compute_util": 30},
      "resnet101": {"gpu_util": 54, "compute_util": 31},
      "resnet152": {"gpu_util": 56, "compute_util": 33},
      "resnext50_32x4d": {"gpu_util": 53, "compute_util": 31},
      "resnext101_32x8d": {"gpu_util": 60, "compute_util": 37},
      "shufflenet_v2_x0_5": {"gpu_util": 24, "compute_util": 11},
      "shufflenet_v2_x1_0": {"gpu_util": 27, "compute_util": 13},
      "shufflenet_v2_x1_5": {"gpu_util": 29, "compute_util": 14},
      "shufflenet_v2_x2_0": {"gpu_util": 32, "compute_util": 16},
      "squeezenet1_0": {"gpu_util": 37, "compute_util": 19},
      "squeezenet1_1": {"gpu_util": 35, "compute_util": 18},
      "vgg11": {"gpu_util": 50, "compute_util": 30},
      "vgg11_bn": {"gpu_util": 51, "compute_util": 31},
      "vgg13": {"gpu_util": 52, "compute_util": 31},
      "vgg13_bn": {"gpu_util": 53, "compute_util": 32},
      "vgg16": {"gpu_util": 54, "compute_util": 33},
      "vgg16_bn": {"gpu_util": 55, "compute_util": 33},
      "vgg19": {"gpu_util": 56, "compute_util": 34},
      "vgg19_bn": {"gpu_util": 57, "compute_util": 34},
      "wide_resnet50_2": {"gpu_util": 54, "compute_util": 32},
      "wide_resnet101_2": {"gpu_util": 59, "compute_util": 36}
    },
    "Tesla_P100-PCIE-16GB": {
      "alexnet": {"gpu_util": 60, "compute_util": 35},
      "densenet121": {"gpu_util": 74, "compute_util": 50},
      "densenet161": {"gpu_util": 76, "compute_util": 52},
      "densenet169": {"gpu_util": 75, "compute_util": 51},
      "densenet201": {"gpu_util": 76, "compute_util": 52},
      "googlenet": {"gpu_util": 66, "compute_util": 42},
      "mnasnet0_5": {"gpu_util": 44, "compute_util": 24},
      "mnasnet0_75": {"gpu_util": 46, "compute_util": 26},
      "mnasnet1_0": {"gpu_util": 49, "compute_util": 28},
      "mnasnet1_3": {"gpu_util": 52, "compute_util": 31},
      "mobilenet_v2": {"gpu_util": 48, "compute_util": 27},
      "mobilenet_v3_large": {"gpu_util": 50, "compute_util": 29},
      "mobilenet_v3_small": {"gpu_util": 45, "compute_util": 25},
      "resnet18": {"gpu_util": 63, "compute_util": 38},
      "resnet34": {"gpu_util": 66, "compute_util": 42},
      "resnet50": {"gpu_util": 71, "compute_util": 47},
      "resnet101": {"gpu_util": 73, "compute_util": 49},
      "resnet152": {"gpu_util": 75, "compute_util": 51},
      "resnext50_32x4d": {"gpu_util": 72, "compute_util": 48},
      "resnext101_32x8d": {"gpu_util": 78, "compute_util": 54},
      "shufflenet_v2_x0_5": {"gpu_util": 39, "compute_util": 20},
      "shufflenet_v2_x1_0": {"gpu_util": 43, "compute_util": 23},
      "shufflenet_v2_x1_5": {"gpu_util": 45, "compute_util": 25},
      "shufflenet_v2_x2_0": {"gpu_util": 49, "compute_util": 28},
      "squeezenet1_0": {"gpu_util": 54, "compute_util": 32},
      "squeezenet1_1": {"gpu_util": 52, "compute_util": 30},
      "vgg11": {"gpu_util": 69, "compute_util": 46},
      "vgg11_bn": {"gpu_util": 70, "compute_util": 47},
      "vgg13": {"gpu_util": 71, "compute_util": 48},
      "vgg13_bn": {"gpu_util": 72, "compute_util": 48},
      "vgg16": {"gpu_util": 73, "compute_util": 49},
      "vgg16_bn": {"gpu_util": 74, "compute_util": 50},
      "vgg19": {"gpu_util": 75, "compute_util": 51},
      "vgg19_bn": {"gpu_util": 75, "compute_util": 51},
      "wide_resnet50_2": {"gpu_util": 73, "compute_util": 49},
      "wide_resnet101_2": {"gpu_util": 77, "compute_util": 53}
    },
    "Tesla_V100-PCIE-32GB": {
      "alexnet": {"gpu_util": 70, "compute_util": 42},
      "densenet121": {"gpu_util": 82, "compute_util": 56},
      "densenet161": {"gpu_util": 84, "compute_util": 58},
      "densenet169": {"gpu_util": 83, "compute_util": 57},
      "densenet201": {"gpu_util": 84, "compute_util": 58},
      "googlenet": {"gpu_util": 75, "compute_util": 48},
      "mnasnet0_5": {"gpu_util": 52, "compute_util": 28},
      "mnasnet0_75": {"gpu_util": 55, "compute_util": 31},
      "mnasnet1_0": {"gpu_util": 58, "compute_util": 33},
      "mnasnet1_3": {"gpu_util": 61, "compute_util": 36},
      "mobilenet_v2": {"gpu_util": 57, "compute_util": 32},
      "mobilenet_v3_large": {"gpu_util": 59, "compute_util": 34},
      "mobilenet_v3_small": {"gpu_util": 53, "compute_util": 29},
      "resnet18": {"gpu_util": 72, "compute_util": 45},
      "resnet34": {"gpu_util": 75, "compute_util": 48},
      "resnet50": {"gpu_util": 79, "compute_util": 53},
      "resnet101": {"gpu_util": 82, "compute_util": 55},
      "resnet152": {"gpu_util": 84, "compute_util": 57},
      "resnext50_32x4d": {"gpu_util": 80, "compute_util": 54},
      "resnext101_32x8d": {"gpu_util": 86, "compute_util": 61},
      "shufflenet_v2_x0_5": {"gpu_util": 47, "compute_util": 24},
      "shufflenet_v2_x1_0": {"gpu_util": 51, "compute_util": 27},
      "shufflenet_v2_x1_5": {"gpu_util": 54, "compute_util": 30},
      "shufflenet_v2_x2_0": {"gpu_util": 58, "compute_util": 33},
      "squeezenet1_0": {"gpu_util": 63, "compute_util": 36},
      "squeezenet1_1": {"gpu_util": 61, "compute_util": 34},
      "vgg11": {"gpu_util": 77, "compute_util": 52},
      "vgg11_bn": {"gpu_util": 78, "compute_util": 53},
      "vgg13": {"gpu_util": 79, "compute_util": 54},
      "vgg13_bn": {"gpu_util": 80, "compute_util": 54},
      "vgg16": {"gpu_util": 81, "compute_util": 55},
      "vgg16_bn": {"gpu_util": 82, "compute_util": 56},
      "vgg19": {"gpu_util": 83, "compute_util": 57},
      "vgg19_bn": {"gpu_util": 83, "compute_util": 57},
      "wide_resnet50_2": {"gpu_util": 81, "compute_util": 55},
      "wide_resnet101_2": {"gpu_util": 85, "compute_util": 60}
    }
  }


def _resolve_profiled_data_root() -> str:
  if os.path.isdir(PROFILED_DATA_ROOT):
    return PROFILED_DATA_ROOT
  return FALLBACK_PROFILED_DATA_ROOT


def _memory_from_gpu_name(gpu_name: str) -> str:
  upper = gpu_name.upper()
  if '40GB' in upper:
    return '40 GB'
  if '32GB' in upper:
    return '32 GB'
  if '16GB' in upper:
    return '16 GB'
  if 'K80' in upper:
    return '24 GB'
  if 'M40' in upper:
    return '24 GB'
  if 'RTX_6000' in upper:
    return '24 GB'
  return 'Unknown'


def _scan_gpu_profiles() -> Dict[str, Any]:
  root = _resolve_profiled_data_root()
  compute_forward_root = os.path.join(root, 'time', 'compute', 'forward')
  transfer_root = os.path.join(root, 'time', 'transfer')
  memory_root = os.path.join(root, 'memory')

  gpu_map: Dict[str, Dict[str, Any]] = {}
  model_set = set(MODEL_CATALOG.keys())

  if os.path.isdir(compute_forward_root):
    for gpu_name in sorted(os.listdir(compute_forward_root)):
      gpu_dir = os.path.join(compute_forward_root, gpu_name)
      if not os.path.isdir(gpu_dir):
        continue

      model_dirs = []
      for model_name in sorted(os.listdir(gpu_dir)):
        model_dir = os.path.join(gpu_dir, model_name)
        if os.path.isdir(model_dir):
          model_dirs.append(model_name)
          model_set.add(model_name)

      gpu_map[gpu_name] = {
        'name': gpu_name,
        'profile_path': gpu_dir,
        'memory_gb': _memory_from_gpu_name(gpu_name),
        'models': model_dirs,
        'compute_profile_count': len(model_dirs),
        'has_transfer_profiles': os.path.isdir(os.path.join(transfer_root, gpu_name)),
        'has_memory_profiles': os.path.isdir(os.path.join(memory_root, gpu_name))
      }

  return {
    'root': root,
    'gpus': gpu_map,
    'models': sorted(model_set)
  }


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
  if not os.path.exists(path):
    return []
  try:
    with open(path, 'r', encoding='utf-8') as csv_file:
      reader = csv.DictReader(csv_file)
      return [row for row in reader if isinstance(row, dict)]
  except OSError:
    return []


def _to_float(value: Any) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return float('nan')


def _value_at_batch(rows: List[Dict[str, str]], x_key: str, y_key: str, batch_size: int) -> float:
  points: List[tuple] = []
  for row in rows:
    x = _to_float(row.get(x_key))
    y = _to_float(row.get(y_key))
    if x == x and y == y:
      points.append((x, y))

  if not points:
    return float('nan')

  points.sort(key=lambda item: item[0])
  target = float(max(1, batch_size))

  if target <= points[0][0]:
    return points[0][1]
  if target >= points[-1][0]:
    return points[-1][1]

  for idx in range(1, len(points)):
    x1, y1 = points[idx - 1]
    x2, y2 = points[idx]
    if x1 <= target <= x2:
      if x2 == x1:
        return y1
      ratio = (target - x1) / (x2 - x1)
      return y1 + (y2 - y1) * ratio

  return points[-1][1]


def _get_profile_stats(gpu_name: str, model_name: str, batch_size: Any) -> Dict[str, Any]:
  gpu = _normalize_gpu(gpu_name)
  model = _normalize_model(model_name)
  batch = int(batch_size) if str(batch_size).isdigit() else 32
  root = _resolve_profiled_data_root()

  memory_csv = os.path.join(root, 'memory', gpu, model, 'memory.csv')
  forward_csv = os.path.join(root, 'time', 'compute', 'forward', gpu, model, 'time_by_batch_size.csv')
  backward_csv = os.path.join(root, 'time', 'compute', 'backward', gpu, model, 'time_by_batch_size.csv')
  transfer_csv = os.path.join(root, 'time', 'transfer', gpu, model, 'data_transfer_time_by_batch_size.csv')
  model_transfer_txt = os.path.join(root, 'time', 'transfer', gpu, model, 'model_transfer_time.txt')

  mem_rows = _read_csv_rows(memory_csv)
  forward_rows = _read_csv_rows(forward_csv)
  backward_rows = _read_csv_rows(backward_csv)
  transfer_rows = _read_csv_rows(transfer_csv)

  peak_mb = _value_at_batch(mem_rows, 'batch_size', 'peak', batch)
  persistent_mb = _value_at_batch(mem_rows, 'batch_size', 'persistent', batch)
  forward_s = _value_at_batch(forward_rows, 'Batch_Size', 'Time_In_SECONDS', batch)
  backward_s = _value_at_batch(backward_rows, 'Batch_Size', 'Time_In_SECONDS', batch)
  transfer_s = _value_at_batch(transfer_rows, 'Batch_Size', 'Time_In_SECONDS', batch)

  model_transfer_s = float('nan')
  if os.path.exists(model_transfer_txt):
    try:
      with open(model_transfer_txt, 'r', encoding='utf-8') as model_transfer_file:
        model_transfer_s = _to_float(model_transfer_file.read().strip())
    except OSError:
      model_transfer_s = float('nan')

  # Assume ImageNet-style float32 input tensor (B,3,224,224).
  input_bytes = max(1, batch) * 3 * 224 * 224 * 4
  bandwidth_gbps = float('nan')
  if transfer_s == transfer_s and transfer_s > 0:
    bandwidth_gbps = input_bytes / transfer_s / 1_000_000_000

  return {
    'gpu': gpu,
    'model': model,
    'batch_size': batch,
    'memory_peak_gb': (peak_mb / 1024.0) if peak_mb == peak_mb else None,
    'memory_persistent_gb': (persistent_mb / 1024.0) if persistent_mb == persistent_mb else None,
    'forward_time_s': forward_s if forward_s == forward_s else None,
    'backward_time_s': backward_s if backward_s == backward_s else None,
    'transfer_time_s': transfer_s if transfer_s == transfer_s else None,
    'model_transfer_time_s': model_transfer_s if model_transfer_s == model_transfer_s else None,
    'compute_time_s': (forward_s + backward_s) if (forward_s == forward_s and backward_s == backward_s) else None,
    'estimated_bandwidth_gbps': bandwidth_gbps if bandwidth_gbps == bandwidth_gbps else None,
    'has_memory_profile': bool(mem_rows),
    'has_compute_profile': bool(forward_rows and backward_rows),
    'has_transfer_profile': bool(transfer_rows)
  }


def get_utilization(gpu_name: Any, model_name: Any) -> Dict[str, Any]:
  normalized_gpu = _normalize_gpu(gpu_name)
  normalized_model = _normalize_model(model_name)
  gpu_models = GPU_MODEL_UTILIZATION.get(normalized_gpu, {})
  utilization = gpu_models.get(normalized_model, {})
  return {
    'gpu': normalized_gpu,
    'model': normalized_model,
    'gpu_util': utilization.get('gpu_util'),
    'compute_util': utilization.get('compute_util')
  }


def _normalize_gpu(gpu_value: Any) -> str:
  profiles = _scan_gpu_profiles()
  gpus = profiles['gpus']
  if isinstance(gpu_value, str):
    key = gpu_value.strip()
    if key in gpus:
      return key

  if DEFAULT_GPU in gpus:
    return DEFAULT_GPU
  if gpus:
    return sorted(gpus.keys())[0]
  return DEFAULT_GPU


def _gpu_info(gpu_name: str) -> Dict[str, Any]:
  profiles = _scan_gpu_profiles()
  gpus = profiles['gpus']
  normalized = _normalize_gpu(gpu_name)
  if normalized in gpus:
    return gpus[normalized]
  return {
    'name': normalized,
    'profile_path': '',
    'memory_gb': _memory_from_gpu_name(normalized),
    'models': [],
    'compute_profile_count': 0,
    'has_transfer_profiles': False,
    'has_memory_profiles': False
  }


def _load_selected_gpu() -> str:
  if not os.path.exists(SELECTED_GPU_PATH):
    return _normalize_gpu(DEFAULT_GPU)

  try:
    with open(SELECTED_GPU_PATH, 'r', encoding='utf-8') as gpu_file:
      data = json.load(gpu_file)
      if isinstance(data, dict):
        return _normalize_gpu(data.get('gpu'))
  except (OSError, json.JSONDecodeError):
    return _normalize_gpu(DEFAULT_GPU)

  return _normalize_gpu(DEFAULT_GPU)


def _save_selected_gpu(gpu_name: str) -> str:
  selected_gpu = _normalize_gpu(gpu_name)
  gpu_data = _gpu_info(selected_gpu)
  _ensure_write_dir()

  payload = {
    'gpu': selected_gpu,
    'profile_path': gpu_data.get('profile_path', ''),
    'database_path': _resolve_profiled_data_root(),
    'updated_at': time.time()
  }

  with open(SELECTED_GPU_PATH, 'a+', encoding='utf-8') as gpu_file:
    fcntl.flock(gpu_file.fileno(), fcntl.LOCK_EX)
    try:
      gpu_file.seek(0)
      gpu_file.truncate()
      json.dump(payload, gpu_file)
      gpu_file.flush()
      os.fsync(gpu_file.fileno())
    finally:
      fcntl.flock(gpu_file.fileno(), fcntl.LOCK_UN)

  return selected_gpu


app = Flask(__name__)

# Try to import the engine and database helpers so the dashboard can expose
# live engine state and recent profiler results. If imports fail, fall back
# to None so endpoints can return a sensible error.
try:
  from glide.engine import InferenceEngine  # type: ignore
  ENGINE = InferenceEngine()
except Exception:
  ENGINE = None

try:
  from glide import database as _database  # type: ignore
except Exception:
  _database = None


def _render_dashboard_page(page: str = 'overview') -> str:
  page_name = page if page in {'overview', 'engine', 'profiler'} else 'overview'
  page_titles = {
    'overview': 'Overview',
    'engine': 'Engine',
    'profiler': 'Profiler',
  }
  page_descriptions = {
    'overview': 'Unified dashboard for live metrics, queue state, and profiler data.',
    'engine': 'Focused view for inference queue status and completed request results.',
    'profiler': 'Focused view for layer timing results and the slowest profiled layers.',
  }
  html = DASHBOARD_HTML.replace('data-page="overview"', f'data-page="{page_name}"', 1)
  html = html.replace('GLIDE GPU TASK SCHEDULER SIMULATOR • RESNET18', f'GLIDE {page_titles[page_name].upper()} VIEW • RESNET18', 1)
  html = html.replace('Awaiting run metadata...', page_descriptions[page_name], 1)
  html = html.replace(
    '<div class="page-summary-title" id="pageSummaryTitle">Overview</div>',
    f'<div class="page-summary-title" id="pageSummaryTitle">{page_titles[page_name]} Page</div>',
    1,
  )
  html = html.replace(
    '<div class="page-summary-body" id="pageSummaryBody">Use the page links above to switch between the overview, engine, and profiler views.</div>',
    f'<div class="page-summary-body" id="pageSummaryBody">{page_descriptions[page_name]}</div>',
    1,
  )
  if page_name == 'engine':
    html = html.replace('body[data-page="engine"] .sim-grid {\n      grid-template-columns: 300px minmax(0, 1fr);\n    }', 'body[data-page="engine"] .sim-grid {\n      grid-template-columns: 1fr;\n    }', 1)
  if page_name == 'profiler':
    html = html.replace('body[data-page="profiler"] .sim-grid {\n      grid-template-columns: minmax(0, 1fr);\n    }', 'body[data-page="profiler"] .sim-grid {\n      grid-template-columns: minmax(0, 1fr);\n    }', 1)
  return html


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GLIDE GPU Task Scheduler Simulator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg0: #07090f;
      --bg1: #0d1327;
      --panel: rgba(13, 21, 40, 0.88);
      --line: #22366a;
      --text: #e7f2ff;
      --muted: #95aed0;
      --neon: #2be4ff;
      --hot: #ff4d6d;
      --ok: #33f0a6;
      --warn: #ffd166;
      --queued: #33a9ff;
      --btn: #1a2748;
      --btn-hover: #27407c;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      min-height: 100vh;
      background:
        radial-gradient(circle at 10% 8%, rgba(43, 228, 255, 0.15), transparent 34%),
        radial-gradient(circle at 85% 20%, rgba(255, 77, 109, 0.15), transparent 30%),
        radial-gradient(circle at 50% 95%, rgba(51, 240, 166, 0.1), transparent 44%),
        linear-gradient(165deg, var(--bg0), var(--bg1));
      color: var(--text);
      font-family: 'Share Tech Mono', monospace;
      padding: 20px;
      overflow-x: hidden;
    }

    .shell {
      max-width: 1500px;
      margin: 0 auto;
      animation: liftIn 750ms ease-out;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      margin-bottom: 14px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(43, 228, 255, 0.1), rgba(255, 77, 109, 0.1));
      padding: 14px 18px;
      border-radius: 12px;
      box-shadow: inset 0 0 50px rgba(43, 228, 255, 0.04), 0 16px 40px rgba(0, 0, 0, 0.35);
    }

    .title {
      font-family: 'Rajdhani', sans-serif;
      font-weight: 700;
      letter-spacing: 1px;
      font-size: clamp(1.2rem, 2.5vw, 1.9rem);
      text-transform: uppercase;
    }

    .subtitle {
      color: var(--muted);
      font-size: 0.86rem;
      margin-top: 4px;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.84rem;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      border: 1px solid var(--line);
      border-radius: 99px;
      padding: 8px 12px;
      background: rgba(6, 9, 18, 0.85);
      white-space: nowrap;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 12px var(--warn);
    }

    .status.running .dot {
      background: var(--ok);
      box-shadow: 0 0 12px var(--ok);
      animation: pulse 1.15s infinite;
    }

    .status.completed .dot {
      background: var(--neon);
      box-shadow: 0 0 12px var(--neon);
      animation: none;
    }

    .page-nav {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }

    .page-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      text-decoration: none;
      background: rgba(9, 14, 27, 0.72);
      transition: color 160ms ease, background 160ms ease, transform 160ms ease;
    }

    .page-link:hover {
      color: var(--text);
      background: rgba(43, 228, 255, 0.12);
      transform: translateY(-1px);
    }

    body[data-page="overview"] .page-link[href="/"] ,
    body[data-page="engine"] .page-link[href="/engine"] ,
    body[data-page="profiler"] .page-link[href="/profiler"] {
      color: var(--text);
      border-color: rgba(43, 228, 255, 0.45);
      background: rgba(43, 228, 255, 0.16);
    }

    .info-panel {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      margin-bottom: 14px;
      overflow: hidden;
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
    }

    .info-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 14px;
      background: rgba(12, 19, 36, 0.92);
    }

    .info-title {
      font-family: 'Rajdhani', sans-serif;
      font-size: 1.02rem;
      letter-spacing: 0.7px;
      text-transform: uppercase;
    }

    .info-toggle {
      border: 1px solid var(--line);
      border-radius: 8px;
      width: 32px;
      height: 32px;
      background: var(--btn);
      color: var(--text);
      cursor: pointer;
      font-size: 1.1rem;
      transition: background 180ms ease;
    }

    .info-toggle:hover {
      background: var(--btn-hover);
    }

    .info-body {
      max-height: 0;
      overflow: hidden;
      transition: max-height 300ms ease, padding 300ms ease;
      color: var(--muted);
      line-height: 1.5;
      padding: 0 14px;
    }

    .info-panel.open .info-body {
      max-height: 180px;
      padding: 12px 14px 14px;
    }

    .control-grid {
      display: grid;
      grid-template-columns: 2.3fr 1.2fr 1.2fr;
      gap: 12px;
      margin-bottom: 14px;
    }

    .control-card,
    .model-info,
    .gpu-info,
    .panel,
    .stats-bar {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
    }

    .control-row {
      display: flex;
      align-items: end;
      gap: 10px;
      flex-wrap: wrap;
    }

    .slider-wrap {
      min-width: 230px;
      display: grid;
      gap: 4px;
    }

    .mode-badge {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: rgba(9, 14, 27, 0.95);
      color: var(--ok);
      font-size: 0.82rem;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      min-width: 230px;
    }

    .static-label {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: rgba(9, 14, 27, 0.95);
      color: var(--muted);
      font-size: 0.82rem;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      min-width: 170px;
    }

    .speed-meta {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 0.72rem;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--neon);
    }

    .input-label {
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      margin-bottom: 6px;
    }

    select, .btn {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(9, 14, 27, 0.95);
      color: var(--text);
      font-family: 'Share Tech Mono', monospace;
      padding: 0 10px;
    }

    select {
      min-width: 220px;
    }

    .btn {
      cursor: pointer;
      background: linear-gradient(90deg, rgba(43, 228, 255, 0.2), rgba(255, 77, 109, 0.2));
      transition: transform 160ms ease, filter 160ms ease;
    }

    .btn:hover {
      filter: brightness(1.15);
      transform: translateY(-1px);
    }

    .hint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.76rem;
    }

    .model-name {
      font-family: 'Rajdhani', sans-serif;
      font-size: 1.25rem;
      margin-bottom: 8px;
      color: var(--neon);
    }

    .model-line {
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.4;
    }

    .stats-bar {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    .stat-chip {
      position: relative;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      background: rgba(7, 13, 27, 0.86);
    }

    .stat-label {
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.7px;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .stat-value {
      font-family: 'Rajdhani', sans-serif;
      font-weight: 700;
      font-size: clamp(1.3rem, 2.4vw, 1.95rem);
      line-height: 1;
    }

    .src-badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 0.63rem;
      letter-spacing: 0.5px;
      padding: 1px 7px;
      margin-left: 6px;
      vertical-align: middle;
    }

    .src-badge.measured { color: var(--neon); border-color: rgba(43, 228, 255, 0.45); }
    .src-badge.profiled { color: var(--ok); border-color: rgba(51, 240, 166, 0.45); }
    .src-badge.derived { color: var(--warn); border-color: rgba(255, 209, 102, 0.45); }

    .now-strip {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 14px;
      background: linear-gradient(90deg, rgba(8, 14, 27, 0.94), rgba(14, 21, 39, 0.94));
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }

    .now-main {
      font-family: 'Rajdhani', sans-serif;
      font-size: 1.15rem;
      letter-spacing: 0.7px;
      color: var(--neon);
      text-transform: uppercase;
    }

    .now-meta {
      color: var(--muted);
      font-size: 0.8rem;
    }

    .sim-grid {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 320px;
      gap: 12px;
    }

    .page-summary {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 12px;
      margin-bottom: 14px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
    }

    .page-summary-title {
      font-family: 'Rajdhani', sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--neon);
      margin-bottom: 6px;
    }

    .page-summary-body {
      color: var(--muted);
      line-height: 1.5;
    }

    .page-summary-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.82rem;
    }

    .page-summary-table th,
    .page-summary-table td {
      border-bottom: 1px solid rgba(149, 174, 208, 0.18);
      padding: 6px 4px;
      text-align: left;
    }

    .page-summary-table th {
      color: var(--text);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    .page-section {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 12px;
      margin-bottom: 14px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
    }

    .page-section-title {
      font-family: 'Rajdhani', sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--neon);
      margin-bottom: 10px;
    }

    .page-section-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .page-section-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(7, 13, 27, 0.9);
      padding: 10px;
    }

    .page-section-label {
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      margin-bottom: 6px;
    }

    .page-section-value {
      font-family: 'Rajdhani', sans-serif;
      font-size: 1.4rem;
      font-weight: 700;
    }

    .page-section-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 0.82rem;
    }

    .page-section-table th,
    .page-section-table td {
      border-bottom: 1px solid rgba(149, 174, 208, 0.18);
      padding: 6px 4px;
      text-align: left;
      vertical-align: top;
    }

    .page-section-table th {
      color: var(--text);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    #enginePageSection {
      display: none;
    }

    #profilerPageSection {
      display: none;
    }

    body[data-page="engine"] #enginePageSection {
      display: block;
    }

    body[data-page="profiler"] #profilerPageSection {
      display: block;
    }

    body[data-page="engine"] .info-panel,
    body[data-page="profiler"] .info-panel,
    body[data-page="engine"] .control-grid > .model-info,
    body[data-page="engine"] .control-grid > .gpu-info,
    body[data-page="profiler"] .control-grid > .model-info,
    body[data-page="profiler"] .control-grid > .gpu-info {
      display: none;
    }

    body[data-page="engine"] .control-grid,
    body[data-page="engine"] .stats-bar,
    body[data-page="engine"] .now-strip,
    body[data-page="engine"] .sim-grid,
    body[data-page="profiler"] .control-grid,
    body[data-page="profiler"] .stats-bar,
    body[data-page="profiler"] .now-strip,
    body[data-page="profiler"] .sim-grid {
      display: none;
    }

    body[data-page="engine"] .sim-grid {
      grid-template-columns: 300px minmax(0, 1fr);
    }

    body[data-page="engine"] .sim-grid > .panel:nth-child(3),
    body[data-page="profiler"] .sim-grid > .panel:nth-child(1),
    body[data-page="profiler"] .sim-grid > .panel:nth-child(2) {
      display: none;
    }

    body[data-page="profiler"] .sim-grid {
      grid-template-columns: minmax(0, 1fr);
    }

    .panel-title {
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 10px;
    }

    .task-list {
      display: grid;
      gap: 8px;
      max-height: 520px;
      overflow: auto;
      padding-right: 4px;
    }

    .spark-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(7, 13, 27, 0.9);
      padding: 8px;
      margin-bottom: 10px;
    }

    .spark-title {
      color: var(--muted);
      font-size: 0.7rem;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    #queueSparkline {
      width: 100%;
      height: 56px;
      display: block;
    }

    .task-item {
      border: 1px solid var(--line);
      border-left: 4px solid var(--queued);
      border-radius: 8px;
      background: rgba(8, 14, 28, 0.88);
      padding: 8px;
    }

    .task-item.done { border-left-color: var(--ok); }
    .task-item.processing { border-left-color: var(--warn); }
    .task-item.queued { border-left-color: var(--queued); }

    .task-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
      font-size: 0.8rem;
    }

    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 0.68rem;
      letter-spacing: 0.6px;
      color: var(--text);
    }

    .pill.done { border-color: rgba(51, 240, 166, 0.5); color: var(--ok); }
    .pill.processing { border-color: rgba(255, 209, 102, 0.5); color: var(--warn); }
    .pill.queued { border-color: rgba(51, 169, 255, 0.5); color: var(--queued); }

    .timeline-wrap {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(7, 13, 27, 0.9);
      min-height: 430px;
      padding: 10px;
      overflow: hidden;
    }

    .timeline-viewport {
      overflow: auto;
      width: 100%;
      height: 360px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(7, 11, 22, 0.9), rgba(10, 16, 31, 0.9));
    }

    .timeline-core {
      position: relative;
      height: 54px;
      min-width: 780px;
      margin: 10px 14px;
      border: 1px dashed rgba(149, 174, 208, 0.45);
      border-radius: 8px;
      background: repeating-linear-gradient(
        to right,
        rgba(43, 228, 255, 0.05),
        rgba(43, 228, 255, 0.05) 1px,
        transparent 1px,
        transparent 50px
      );
    }

    .core-label {
      position: absolute;
      left: 8px;
      top: -12px;
      border: 1px solid var(--line);
      background: rgba(7, 13, 27, 0.95);
      padding: 2px 8px;
      border-radius: 6px;
      color: var(--muted);
      font-size: 0.72rem;
    }

    .timeline-bar {
      position: absolute;
      top: 14px;
      height: 26px;
      border-radius: 7px;
      border: 1px solid rgba(10, 18, 36, 0.6);
      display: flex;
      align-items: center;
      padding-left: 8px;
      font-size: 0.72rem;
      white-space: nowrap;
      overflow: hidden;
      color: #06111d;
      font-weight: 700;
    }

    .core-summary {
      color: var(--muted);
      font-size: 0.76rem;
      margin-top: 8px;
      margin-bottom: 6px;
    }

    .timeline-bar.fast { background: linear-gradient(90deg, #33f0a6, #7df7c9); }
    .timeline-bar.medium { background: linear-gradient(90deg, #ffd166, #ffe39d); }
    .timeline-bar.slow { background: linear-gradient(90deg, #ff4d6d, #ff87a0); }
    .timeline-bar.queued {
      background: linear-gradient(90deg, rgba(51, 169, 255, 0.35), rgba(43, 228, 255, 0.2));
      color: #bfe7ff;
      border-style: dashed;
    }

    .timeline-axis {
      margin-top: 10px;
      font-size: 0.72rem;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }

    .metric-block {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: rgba(7, 13, 27, 0.9);
      margin-bottom: 10px;
    }

    .metric-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.72rem;
      letter-spacing: 0.2px;
    }

    .metric-header {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 0.78rem;
      margin-bottom: 7px;
      text-transform: uppercase;
    }

    .meter {
      width: 100%;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(149, 174, 208, 0.2);
      border: 1px solid rgba(149, 174, 208, 0.22);
    }

    .meter > span {
      display: block;
      height: 100%;
      width: 0;
      transition: width 320ms ease;
      background: linear-gradient(90deg, #2be4ff, #33f0a6);
    }

    .legend {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 0.72rem;
      color: var(--muted);
      margin-top: 8px;
    }

    .legend span {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
    }

    .legend .fast { color: var(--ok); }
    .legend .medium { color: var(--warn); }
    .legend .slow { color: var(--hot); }

    .legend .queued { color: var(--queued); }

    .empty {
      color: var(--muted);
      font-size: 0.85rem;
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }

    .log-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(7, 13, 27, 0.9);
      padding: 8px;
      margin-top: 10px;
      max-height: 170px;
      overflow: auto;
    }

    .log-entry {
      font-size: 0.74rem;
      color: var(--muted);
      padding: 5px 0;
      border-bottom: 1px dashed rgba(149, 174, 208, 0.2);
    }

    .log-entry.fast {
      color: var(--ok);
    }

    .log-entry.medium {
      color: var(--warn);
    }

    .log-entry.slow {
      color: var(--hot);
    }

    .log-entry:last-child {
      border-bottom: 0;
    }

    @media (max-width: 1120px) {
      .control-grid {
        grid-template-columns: 1fr;
      }
      .stats-bar {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .sim-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 680px) {
      body { padding: 14px; }
      .topbar { flex-direction: column; align-items: flex-start; }
      .stats-bar { grid-template-columns: 1fr; }
      .control-row { align-items: stretch; }
      select, .btn, .slider-wrap { width: 100%; }
    }

    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.8; }
      70% { transform: scale(1.15); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.8; }
    }

    @keyframes liftIn {
      from { opacity: 0; transform: translateY(15px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body data-page="overview">
  <div class="shell">
    <div class="topbar">
      <div>
        <div class="title" id="pageTitle">GLIDE GPU TASK SCHEDULER SIMULATOR • RESNET18</div>
        <div class="subtitle" id="meta">Awaiting run metadata...</div>
        <div class="page-nav">
          <a class="page-link" href="/">Overview</a>
          <a class="page-link" href="/engine">Engine</a>
          <a class="page-link" href="/profiler">Profiler</a>
        </div>
      </div>
      <div class="status waiting" id="statusBadge"><span class="dot"></span><span id="statusText">WAITING FOR DATA...</span></div>
    </div>

    <div class="info-panel" id="gpemuInfo">
      <div class="info-head">
        <div class="info-title">What Is GPEmu?</div>
        <button class="info-toggle" id="infoToggle" title="Toggle explanation">?</button>
      </div>
      <div class="info-body">
        GPEmu is a GPU emulator: it lets us run deep learning workloads on a normal CPU when a real GPU is unavailable.
        This dashboard simulates a task scheduler so anyone can understand how model batches move through a GPU-like execution pipeline.
        Each batch appears as a task, enters a queue, executes on a timeline, and updates synthetic GPU resource indicators.
      </div>
    </div>

    <div class="control-grid">
      <div class="control-card">
        <div class="input-label">Simulation Controls</div>
        <div class="control-row">
          <div>
            <div class="input-label">Model</div>
            <select id="modelSelector">
              <option value="resnet18">resnet18</option>
            </select>
          </div>
          <div>
            <div class="input-label">GPU Profile</div>
            <select id="gpuSelector">
              <option value="Tesla_M40">Tesla_M40</option>
            </select>
          </div>
          <div>
            <div class="input-label">Current Mode</div>
            <div class="mode-badge" id="modeBadge">SEQUENTIAL BATCH EXECUTION</div>
          </div>
          <div>
            <div class="input-label">Refresh</div>
            <div class="static-label">REFRESH RATE: 1s</div>
          </div>
          <button class="btn" id="startRunBtn">Start New Run</button>
        </div>
        <div class="hint">Select model and GPU for real GPEmu-backed metrics. HASP Scheduler -- Coming in Review 2.</div>
      </div>

      <div class="model-info">
        <div class="input-label">Selected Model Info</div>
        <div class="model-name" id="modelName">ResNet18</div>
        <div class="model-line" id="modelDesc">A compact image classifier with residual skip connections.</div>
        <div class="model-line" id="modelLayers"><strong>Layers:</strong> 18</div>
        <div class="model-line" id="modelUseCase"><strong>Use case:</strong> Fast baseline image classification and education demos.</div>
      </div>

      <div class="gpu-info">
        <div class="input-label">Selected GPU Info</div>
        <div class="model-name" id="gpuName">Tesla_M40</div>
        <div class="model-line" id="gpuMemory"><strong>Memory:</strong> Unknown</div>
        <div class="model-line" id="gpuProfiles"><strong>Compute profiles:</strong> 0</div>
        <div class="model-line" id="gpuExtras"><strong>Other:</strong> transfer N/A | memory N/A</div>
        <div class="model-line" id="algoActive"><strong>Current mode:</strong> sequential batch execution</div>
      </div>
    </div>

    <div class="stats-bar">
      <div class="stat-chip">
        <div class="stat-label">GPU Utilization <span class="src-badge profiled">PROFILED</span></div>
        <div class="stat-value" id="utilTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Avg Compute <span class="src-badge measured">MEASURED</span></div>
        <div class="stat-value" id="avgTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Min Compute <span class="src-badge measured">MEASURED</span></div>
        <div class="stat-value" id="minTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Max Compute <span class="src-badge measured">MEASURED</span></div>
        <div class="stat-value" id="maxTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Throughput <span class="src-badge measured">MEASURED</span></div>
        <div class="stat-value" id="throughputTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Total Batches <span class="src-badge measured">MEASURED</span></div>
        <div class="stat-value" id="doneTop">0</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">ETA <span class="src-badge derived">DERIVED</span></div>
        <div class="stat-value" id="etaTop">--</div>
      </div>
    </div>

    <div class="now-strip">
      <div class="now-main" id="nowExec">Now Executing: Idle</div>
      <div class="now-meta" id="nowExecMeta">State: waiting | Source: profiled timing + measured runtime events</div>
    </div>

    <div class="page-summary" id="pageSummary">
      <div class="page-summary-title" id="pageSummaryTitle">Overview</div>
      <div class="page-summary-body" id="pageSummaryBody">Use the page links above to switch between the overview, engine, and profiler views.</div>
    </div>

    <div class="page-section" id="enginePageSection">
      <div class="page-section-title">Engine View</div>
      <div class="page-section-grid">
        <div class="page-section-card">
          <div class="page-section-label">Queue Length</div>
          <div class="page-section-value" id="engineQueueLength">--</div>
        </div>
        <div class="page-section-card">
          <div class="page-section-label">Completed Requests</div>
          <div class="page-section-value" id="engineCompletedCount">--</div>
        </div>
        <div class="page-section-card">
          <div class="page-section-label">Avg Latency</div>
          <div class="page-section-value" id="engineAvgLatency">--</div>
        </div>
      </div>
      <table class="page-section-table" id="engineResultsTable">
        <thead>
          <tr>
            <th>Request</th>
            <th>Model</th>
            <th>Batch</th>
            <th>Latency</th>
            <th>Compute</th>
            <th>Memory</th>
          </tr>
        </thead>
        <tbody>
          <tr><td colspan="6">No engine results loaded yet.</td></tr>
        </tbody>
      </table>
    </div>

    <div class="page-section" id="profilerPageSection">
      <div class="page-section-title">Profiler View</div>
      <div class="page-section-grid">
        <div class="page-section-card">
          <div class="page-section-label">Slowest Layer</div>
          <div class="page-section-value" id="profilerSlowestLayer">--</div>
        </div>
        <div class="page-section-card">
          <div class="page-section-label">Slowest Compute</div>
          <div class="page-section-value" id="profilerSlowestCompute">--</div>
        </div>
        <div class="page-section-card">
          <div class="page-section-label">Layer Count</div>
          <div class="page-section-value" id="profilerLayerCount">--</div>
        </div>
      </div>
      <table class="page-section-table" id="profilerResultsTable">
        <thead>
          <tr>
            <th>Layer</th>
            <th>Compute</th>
            <th>Memory</th>
            <th>Config</th>
          </tr>
        </thead>
        <tbody>
          <tr><td colspan="4">No profiler results loaded yet.</td></tr>
        </tbody>
      </table>
    </div>

    <div class="sim-grid">
      <div class="panel">
        <div class="panel-title">Task Queue Panel <span class="src-badge measured">MEASURED</span></div>
        <div class="spark-wrap">
          <div class="spark-title">Queue Depth Trend</div>
          <svg id="queueSparkline" viewBox="0 0 260 56" preserveAspectRatio="none"></svg>
        </div>
        <div class="task-list" id="taskQueue"></div>
      </div>

      <div class="panel">
        <div class="panel-title">Execution Timeline (Gantt Style) <span class="src-badge profiled">PROFILED</span></div>
        <div class="core-summary" id="coreSummary">Core allocation: waiting for GPU selection</div>
        <div class="timeline-wrap">
          <div class="timeline-viewport" id="timelineViewport">
            <div id="timelineCoreContainer">
              <div class="timeline-core" id="timelineCore">
                <div class="core-label">Core 1</div>
              </div>
            </div>
          </div>
          <div class="timeline-axis" id="timelineAxis"></div>
          <div class="legend">
            <span class="fast">Fast Batch</span>
            <span class="medium">Average Batch</span>
            <span class="slow">Slow Batch</span>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">GPU Resource Usage Panel</div>
        <div class="metric-block">
          <div class="metric-header"><span>GPU Utilization <span class="src-badge profiled">PROFILED</span></span><span id="gpuUtilValue">--</span></div>
          <div class="meter"><span id="gpuUtilBar"></span></div>
          <div class="metric-note">Source: GPEmu profiled data</div>
        </div>
        <div class="metric-block">
          <div class="metric-header"><span>Compute Utilization <span class="src-badge profiled">PROFILED</span></span><span id="computeUtilValue">--</span></div>
          <div class="meter"><span id="computeUtilBar"></span></div>
          <div class="metric-note">Source: GPEmu profiled data</div>
        </div>
        <div class="metric-block">
          <div class="metric-header"><span>VRAM Usage <span class="src-badge profiled">PROFILED</span></span><span id="vramValue">--</span></div>
          <div class="meter"><span id="vramBar"></span></div>
        </div>
        <div class="metric-block">
          <div class="metric-header"><span>Memory Bandwidth <span class="src-badge profiled">PROFILED</span></span><span id="bwValue">--</span></div>
          <div class="meter"><span id="bwBar"></span></div>
        </div>
        <div class="metric-block">
          <div class="metric-header"><span>Active Warps <span class="src-badge derived">DERIVED</span></span><span id="warpValue">--</span></div>
          <div class="meter"><span id="warpBar"></span></div>
        </div>
        <div class="empty" id="resourceHint">Source: GPEmu profiled data</div>
        <div class="panel-title" style="margin-top:10px;">Real Event Log <span class="src-badge measured">EVENTS</span></div>
        <div class="log-box" id="eventLog"></div>
      </div>
    </div>
  </div>

  <script>
    const fmt = (v, suffix = 's') => (Number.isFinite(v) ? `${v.toFixed(4)} ${suffix}` : '--');
    let refreshMs = 1000;
    let refreshTimer = null;
    let queueDepthHistory = [];
    const pageMode = (document.body?.dataset?.page || 'overview').toLowerCase();

    function formatSeconds(sec) {
      if (!Number.isFinite(sec)) return '--';
      if (sec < 60) return `${sec.toFixed(1)}s`;
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m}m ${s}s`;
    }

    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function getCurrentSelection(metrics) {
      const selectedModel = document.getElementById('modelSelector')?.value || metrics.selected_model || metrics.active_model || 'resnet18';
      const selectedGpu = document.getElementById('gpuSelector')?.value || metrics.selected_gpu || 'Tesla_M40';
      return { selectedModel, selectedGpu };
    }

    function adjustedTimes(metrics) {
      const { selectedModel, selectedGpu } = getCurrentSelection(metrics);
      const batches = Array.isArray(metrics.batches) ? metrics.batches : [];
      return {
        selectedModel,
        selectedGpu,
        profile: metrics.profile_stats || {},
        adjusted: batches.map((item) => ({
          batch: Number(item.batch),
          compute_time: Math.max(0.004, Number(item.compute_time) || 0)
        }))
      };
    }

    function deriveRuntimeSignals(metrics, data) {
      const list = data.adjusted;
      const avg = list.length > 0 ? (list.reduce((acc, it) => acc + it.compute_time, 0) / list.length) : 0;
      const minVal = list.length > 0 ? Math.min(...list.map((it) => it.compute_time)) : avg;
      const throughput = avg > 0 ? (1 / avg) : 0;

      const expected = Number(metrics.total_expected_batches);
      const total = Number(metrics.total_batches);
      let pendingRatio = 0;
      if (metrics.status === 'running' && Number.isFinite(expected) && expected > 0 && Number.isFinite(total)) {
        pendingRatio = clamp((expected - total) / expected, 0, 1);
      }

      const jitter = clamp((avg - minVal) / Math.max(0.005, avg), 0, 1);
      return { avg, minVal, throughput, pendingRatio, jitter };
    }

    function restartTicker() {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
      refreshTimer = setInterval(tick, refreshMs);
    }

    function renderPageSummary(metrics, engineData, profilerData) {
      const title = document.getElementById('pageSummaryTitle');
      const body = document.getElementById('pageSummaryBody');
      const engineSection = document.getElementById('enginePageSection');
      const profilerSection = document.getElementById('profilerPageSection');
      if (!title || !body) return;

      if (engineSection) engineSection.style.display = pageMode === 'engine' ? 'block' : 'none';
      if (profilerSection) profilerSection.style.display = pageMode === 'profiler' ? 'block' : 'none';

      if (pageMode === 'engine') {
        const status = engineData?.status || {};
        title.textContent = 'Engine Page';
        body.innerHTML = `Queue length ${Number(status.queue_length) || 0}, completed ${Number(status.completed_count) || 0}, average latency ${Number(status.avg_latency_ms || 0).toFixed(2)} ms.`;
        renderEnginePage(engineData);
        return;
      }

      if (pageMode === 'profiler') {
        const rows = Array.isArray(profilerData?.slowest_layers) ? profilerData.slowest_layers : [];
        title.textContent = 'Profiler Page';
        if (rows.length === 0) {
          body.innerHTML = 'No layer timing data is currently available. Run the profiler to populate the database.';
          renderProfilerPage(profilerData);
          return;
        }
        const tableRows = rows.slice(0, 5).map((row) => {
          const config = row.config || {};
          const name = row.layer_type || 'Unknown';
          const ms = Number(row.compute_cost_ms || 0).toFixed(2);
          const details = Object.keys(config).length > 0 ? JSON.stringify(config) : '-';
          return `<tr><td>${name}</td><td>${ms} ms</td><td>${details}</td></tr>`;
        }).join('');
        body.innerHTML = `<div>Top timed layers from the profiler database.</div><table class="page-summary-table"><thead><tr><th>Layer</th><th>Compute</th><th>Config</th></tr></thead><tbody>${tableRows}</tbody></table>`;
        renderProfilerPage(profilerData);
        return;
      }

      const selectedModel = metrics.selected_model || 'resnet18';
      const selectedGpu = metrics.selected_gpu || 'Tesla_M40';
      title.textContent = 'Overview Page';
      body.innerHTML = `Unified dashboard for ${selectedModel.toUpperCase()} on ${selectedGpu}. Use the Engine and Profiler pages for focused views.`;
      if (engineSection) engineSection.style.display = 'none';
      if (profilerSection) profilerSection.style.display = 'none';
    }

    function renderEnginePage(engineData) {
      const status = engineData?.status || {};
      const results = Array.isArray(engineData?.results) ? engineData.results : [];
      document.getElementById('engineQueueLength').textContent = Number(status.queue_length) || 0;
      document.getElementById('engineCompletedCount').textContent = Number(status.completed_count) || 0;
      document.getElementById('engineAvgLatency').textContent = Number.isFinite(Number(status.avg_latency_ms)) ? `${Number(status.avg_latency_ms).toFixed(2)} ms` : '--';

      const tbody = document.querySelector('#engineResultsTable tbody');
      if (!tbody) return;
      if (results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6">No completed inference requests yet.</td></tr>';
        return;
      }

      tbody.innerHTML = results.slice(-12).reverse().map((item) => {
        const latency = Number(item.latency_ms);
        const compute = Number(item.emulated_compute_ms);
        const memory = Number(item.memory_mb);
        return `
          <tr>
            <td>${item.request_id || '-'}</td>
            <td>${item.model_name || '-'}</td>
            <td>${Number(item.batch_size) || '-'}</td>
            <td>${Number.isFinite(latency) ? latency.toFixed(2) + ' ms' : '--'}</td>
            <td>${Number.isFinite(compute) ? compute.toFixed(2) + ' ms' : '--'}</td>
            <td>${Number.isFinite(memory) ? memory.toFixed(1) + ' MB' : '--'}</td>
          </tr>
        `;
      }).join('');
    }

    function renderProfilerPage(profilerData) {
      const rows = Array.isArray(profilerData?.slowest_layers) ? profilerData.slowest_layers : [];
      document.getElementById('profilerLayerCount').textContent = rows.length;

      if (rows.length === 0) {
        document.getElementById('profilerSlowestLayer').textContent = '--';
        document.getElementById('profilerSlowestCompute').textContent = '--';
        const tbody = document.querySelector('#profilerResultsTable tbody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="4">No profiler results loaded yet.</td></tr>';
        return;
      }

      const slowest = rows[0];
      document.getElementById('profilerSlowestLayer').textContent = slowest.layer_type || '-';
      document.getElementById('profilerSlowestCompute').textContent = Number.isFinite(Number(slowest.compute_cost_ms)) ? `${Number(slowest.compute_cost_ms).toFixed(2)} ms` : '--';

      const tbody = document.querySelector('#profilerResultsTable tbody');
      if (!tbody) return;
      tbody.innerHTML = rows.slice(0, 12).map((item) => {
        const config = item.config ? JSON.stringify(item.config) : '-';
        return `
          <tr>
            <td>${item.layer_type || '-'}</td>
            <td>${Number.isFinite(Number(item.compute_cost_ms)) ? Number(item.compute_cost_ms).toFixed(2) + ' ms' : '--'}</td>
            <td>${Number.isFinite(Number(item.memory_cost_mb)) ? Number(item.memory_cost_mb).toFixed(2) + ' MB' : '--'}</td>
            <td>${config}</td>
          </tr>
        `;
      }).join('');
    }

    function renderEventLog(metrics) {
      const box = document.getElementById('eventLog');
      if (!box) return;
      const batches = Array.isArray(metrics.batches) ? metrics.batches : [];
      if (batches.length === 0) {
        box.innerHTML = '<div class="log-entry">No batch completion events yet.</div>';
        return;
      }
      const avg = Number(metrics.avg_compute_time);
      const recent = batches.slice(-8).reverse();
      box.innerHTML = recent.map((item) => {
        const t = Number(item.compute_time);
        const ts = Number(item.timestamp);
        let cls = 'medium';
        if (Number.isFinite(avg) && Number.isFinite(t)) {
          if (t < avg) cls = 'fast';
          else if (t > avg * 1.2) cls = 'slow';
        }
        const tsLabel = Number.isFinite(ts) ? new Date(ts * 1000).toLocaleTimeString() : '--';
        const timeLabel = Number.isFinite(t) ? t.toFixed(4) : '--';
        return `<div class="log-entry ${cls}">Batch ${Number(item.batch)} completed in ${timeLabel}s at ${tsLabel}</div>`;
      }).join('');
    }

    function renderQueueSparkline(depth) {
      const svg = document.getElementById('queueSparkline');
      if (!svg) return;

      queueDepthHistory.push(depth);
      if (queueDepthHistory.length > 44) {
        queueDepthHistory = queueDepthHistory.slice(-44);
      }

      const values = queueDepthHistory;
      const maxV = Math.max(1, ...values);
      const width = 260;
      const height = 56;

      const points = values.map((v, idx) => {
        const x = values.length <= 1 ? 0 : (idx / (values.length - 1)) * width;
        const y = height - ((v / maxV) * (height - 8)) - 4;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(' ');

      svg.innerHTML = `
        <rect x="0" y="0" width="260" height="56" fill="rgba(7,13,27,0.3)" />
        <polyline points="${points}" fill="none" stroke="#2be4ff" stroke-width="2" />
        <text x="4" y="12" fill="#95aed0" font-size="9">max ${maxV}</text>
        <text x="224" y="52" fill="#95aed0" font-size="9">now ${depth}</text>
      `;
    }

    function setStatus(status, hasData) {
      const badge = document.getElementById('statusBadge');
      const text = document.getElementById('statusText');
      badge.classList.remove('running', 'completed', 'waiting');

      if (!hasData) {
        badge.classList.add('waiting');
        text.textContent = 'WAITING FOR DATA...';
        return;
      }

      if (status === 'completed') {
        badge.classList.add('completed');
        text.textContent = 'SIMULATION COMPLETE';
      } else {
        badge.classList.add('running');
        text.textContent = 'RUNNING';
      }
    }

    function resetDashboardState() {
      document.getElementById('utilTop').textContent = '--';
      document.getElementById('avgTop').textContent = '--';
      document.getElementById('minTop').textContent = '--';
      document.getElementById('maxTop').textContent = '--';
      document.getElementById('throughputTop').textContent = '--';
      document.getElementById('etaTop').textContent = '--';
      document.getElementById('doneTop').textContent = '0';
      document.getElementById('meta').textContent = 'Awaiting run metadata...';
      document.getElementById('taskQueue').innerHTML = '<div class="empty">No tasks yet. Start a run to enqueue batches.</div>';
      document.getElementById('timelineCoreContainer').innerHTML = '<div class="timeline-core" id="timelineCore"><div class="core-label">Core 1</div></div>';
      document.getElementById('timelineAxis').textContent = '';
      document.getElementById('coreSummary').textContent = 'Sequential execution timeline from real batch completions';
      setMeter('gpuUtilBar', 0);
      setMeter('computeUtilBar', 0);
      setMeter('vramBar', 0);
      setMeter('bwBar', 0);
      setMeter('warpBar', 0);
      document.getElementById('gpuUtilValue').textContent = '--';
      document.getElementById('computeUtilValue').textContent = '--';
      document.getElementById('vramValue').textContent = '--';
      document.getElementById('bwValue').textContent = '--';
      document.getElementById('warpValue').textContent = '--';
      document.getElementById('nowExec').textContent = 'Now Executing: Idle';
      document.getElementById('nowExecMeta').textContent = 'State: waiting | Source: profiled timing + measured runtime events';
      queueDepthHistory = [];
      renderQueueSparkline(0);
      renderEventLog({ batches: [] });
      setStatus('waiting', false);
    }

    function setMeter(id, percent) {
      document.getElementById(id).style.width = `${clamp(percent, 0, 100).toFixed(1)}%`;
    }

    function updateModelInfo(modelInfo, modelKey) {
      const info = modelInfo || {};
      document.getElementById('modelName').textContent = info.display_name || modelKey || 'Unknown';
      document.getElementById('modelDesc').textContent = info.description || 'No description available.';
      document.getElementById('modelLayers').innerHTML = `<strong>Layers:</strong> ${info.layers || '-'}`;
      document.getElementById('modelUseCase').innerHTML = `<strong>Use case:</strong> ${info.use_case || '-'}`;
      const modelTitle = (info.display_name || modelKey || 'Unknown').toUpperCase();
      document.getElementById('pageTitle').textContent = `GLIDE Metrics Console • ${modelTitle}`;
    }

    function updateGpuInfo(gpuInfo, gpuKey) {
      const info = gpuInfo || {};
      const gpuName = info.name || gpuKey || 'Unknown';
      document.getElementById('gpuName').textContent = gpuName;
      document.getElementById('gpuMemory').innerHTML = `<strong>Memory:</strong> ${info.memory_gb || 'Unknown'}`;
      document.getElementById('gpuProfiles').innerHTML = `<strong>Compute profiles:</strong> ${info.compute_profile_count ?? 0}`;
      const transfer = info.has_transfer_profiles ? 'transfer yes' : 'transfer no';
      const memory = info.has_memory_profiles ? 'memory yes' : 'memory no';
      document.getElementById('gpuExtras').innerHTML = `<strong>Other:</strong> ${transfer} | ${memory}`;
    }

    function applyLookupUtilization(payload) {
      const gpuUtil = Number(payload?.gpu_util);
      const computeUtil = Number(payload?.compute_util);
      console.log('[GLIDE lookup util]', {
        selectedGpu: payload?.selected_gpu,
        selectedModel: payload?.selected_model,
        gpu_util: payload?.gpu_util,
        compute_util: payload?.compute_util
      });
      if (Number.isFinite(gpuUtil)) {
        document.getElementById('utilTop').textContent = `${gpuUtil.toFixed(1)}%`;
        document.getElementById('gpuUtilValue').textContent = `${gpuUtil.toFixed(1)}%`;
        setMeter('gpuUtilBar', gpuUtil);
      }
      if (Number.isFinite(computeUtil)) {
        document.getElementById('computeUtilValue').textContent = `${computeUtil.toFixed(1)}%`;
        setMeter('computeUtilBar', computeUtil);
      }
      document.getElementById('resourceHint').textContent = 'Source: GPEmu profiled data';
    }

    function populateSelectors(metrics) {
      const models = Array.isArray(metrics.available_models) ? metrics.available_models : [];
      const gpus = Array.isArray(metrics.available_gpus) ? metrics.available_gpus : [];

      const modelSelector = document.getElementById('modelSelector');
      const gpuSelector = document.getElementById('gpuSelector');

      if (models.length > 0) {
        const currentModel = metrics.selected_model || modelSelector.value;
        modelSelector.innerHTML = models
          .map((model) => `<option value="${model}">${model}</option>`)
          .join('');
        if (models.includes(currentModel)) {
          modelSelector.value = currentModel;
        }
      }

      if (gpus.length > 0) {
        const currentGpu = metrics.selected_gpu || gpuSelector.value;
        gpuSelector.innerHTML = gpus
          .map((gpu) => `<option value="${gpu}">${gpu}</option>`)
          .join('');
        if (gpus.includes(currentGpu)) {
          gpuSelector.value = currentGpu;
        }
      }
    }

    function renderTopStats(metrics) {
      const done = Number(metrics.total_batches) || 0;
      const avg = Number(metrics.avg_compute_time);
      const minVal = Number(metrics.min_compute_time);
      const maxVal = Number(metrics.max_compute_time);
      const throughput = Number(metrics.throughput);
      const eta = Number(metrics.eta_seconds);
      const gpuUtil = Number(metrics.gpu_util);

      document.getElementById('utilTop').textContent = Number.isFinite(gpuUtil) ? `${gpuUtil.toFixed(1)}%` : '--';
      document.getElementById('avgTop').textContent = Number.isFinite(avg) ? `${avg.toFixed(4)}s` : '--';
      document.getElementById('minTop').textContent = Number.isFinite(minVal) ? `${minVal.toFixed(4)}s` : '--';
      document.getElementById('maxTop').textContent = Number.isFinite(maxVal) ? `${maxVal.toFixed(4)}s` : '--';
      document.getElementById('throughputTop').textContent = Number.isFinite(throughput) ? `${throughput.toFixed(2)} b/s` : '--';
      document.getElementById('etaTop').textContent = formatSeconds(eta);
      document.getElementById('doneTop').textContent = `${done}`;
    }

    function renderTaskQueue(metrics) {
      const queue = document.getElementById('taskQueue');
      const batches = Array.isArray(metrics.batches) ? metrics.batches : [];
      if (batches.length === 0) {
        renderQueueSparkline(0);
        queue.innerHTML = '<div class="empty">No tasks yet. Start a run to enqueue batches.</div>';
        return;
      }

      const preview = batches.slice(-8).map((item) => ({
        batch: Number(item.batch),
        compute_time: Number(item.compute_time),
        status: 'DONE'
      }));

      if (metrics.status === 'running' && preview.length > 0) {
        preview[preview.length - 1].status = 'PROCESSING';
      }

      const expected = Number(metrics.total_expected_batches);
      const total = Number(metrics.total_batches);
      let queueDepth = 0;
      if (metrics.status === 'running' && Number.isFinite(expected) && Number.isFinite(total)) {
        queueDepth = Math.max(0, expected - total);
      }

      renderQueueSparkline(queueDepth);

      queue.innerHTML = preview.map((task) => {
        const cls = task.status.toLowerCase();
        const timeTxt = Number.isFinite(task.compute_time) ? `${task.compute_time.toFixed(4)} s` : 'pending';
        return `
          <div class="task-item ${cls}">
            <div class="task-top">
              <span>Task ID: Batch ${task.batch}</span>
              <span class="pill ${cls}">${task.status}</span>
            </div>
            <div class="model-line">Compute: ${timeTxt}</div>
          </div>
        `;
      }).join('');
    }

    function renderNowExecuting(metrics) {
      const data = adjustedTimes(metrics);
      const batches = data.adjusted;
      const nowTitle = document.getElementById('nowExec');
      const nowMeta = document.getElementById('nowExecMeta');

      if (batches.length === 0) {
        nowTitle.textContent = 'Now Executing: Idle';
        nowMeta.textContent = `State: ${metrics.status || 'waiting'} | Source: profiled timing + measured runtime events`;
        return;
      }

      const last = batches[batches.length - 1];
      const batchId = Number(last.batch);
      const compute = Number(last.compute_time);
      let phase = 'DONE';
      if (metrics.status === 'running') {
        phase = 'PROCESSING';
      }

      nowTitle.textContent = `Now Executing: Batch ${batchId} (${phase})`;
      nowMeta.textContent = `Time: ${Number.isFinite(compute) ? compute.toFixed(4) + 's' : '--'} | Mode: sequential batch execution | Profile: ${data.selectedGpu}/${data.selectedModel}`;
    }

    function renderTimeline(metrics) {
      const coreContainer = document.getElementById('timelineCoreContainer');
      const viewport = document.getElementById('timelineViewport');
      const axis = document.getElementById('timelineAxis');
      const data = adjustedTimes(metrics);
      const batches = data.adjusted;
      const coreSummary = document.getElementById('coreSummary');

      coreSummary.textContent = `Sequential execution timeline from real batches on ${data.selectedGpu}`;

      if (batches.length === 0) {
        coreContainer.innerHTML = '<div class="timeline-core" id="timelineCore"><div class="core-label">Core 1</div></div>';
        axis.textContent = '';
        return;
      }

      const viewBatches = batches.slice(-120).map((item) => ({
        batch: Number(item.batch),
        compute_time: Number(item.compute_time) || 0
      }));

      const avg = (viewBatches.reduce((acc, item) => acc + item.compute_time, 0) / viewBatches.length) || 0.0001;
      const scale = 85;
      const bars = [];
      let cursor = 0;

      const classForDuration = (duration) => {
        if (duration > avg * 1.2) return 'slow';
        if (duration > avg * 0.9) return 'medium';
        return 'fast';
      };

      for (const item of viewBatches) {
        const duration = Math.max(0.004, item.compute_time);
        const cls = classForDuration(duration);
        const left = cursor * scale;
        const width = Math.max(7, duration * scale);
        bars.push(`<div class="timeline-bar ${cls}" style="left:${left}px;width:${width}px;">B${item.batch}</div>`);
        cursor += duration;
      }

      const totalSec = Math.max(0.001, cursor);
      const minW = Math.max(780, Math.ceil(totalSec * scale) + 90);
      coreContainer.innerHTML = `<div class="timeline-core" style="min-width:${minW}px;"><div class="core-label">Batch Stream</div>${bars.join('')}</div>`;

      axis.innerHTML = `
        <span>0s</span>
        <span>${(totalSec * 0.25).toFixed(2)}s</span>
        <span>${(totalSec * 0.5).toFixed(2)}s</span>
        <span>${(totalSec * 0.75).toFixed(2)}s</span>
        <span>${totalSec.toFixed(2)}s</span>
      `;

      viewport.scrollLeft = viewport.scrollWidth;
    }

    function renderResources(metrics) {
      const data = adjustedTimes(metrics);
      const profile = data.profile || {};
      const gpuUtil = Number(metrics.gpu_util);
      const computeUtil = Number(metrics.compute_util);

      const memText = (metrics.gpu_info && metrics.gpu_info.memory_gb) || 'Unknown';
      const totalMem = Number.parseFloat(memText) || 24;

      // Prefer dynamic vram from server if present, otherwise fall back to profile peak
      const dynamicVram = Number(metrics.vram_used_gb);
      const profileVram = Number(profile.memory_peak_gb);
      const hasProfileVram = Number.isFinite(profileVram) && profileVram > 0;
      const vramUsed = Number.isFinite(dynamicVram) && dynamicVram > 0 ? clamp(dynamicVram, 0, totalMem) : (hasProfileVram ? clamp(profileVram, 0, totalMem) : NaN);
      const vramPct = Number.isFinite(vramUsed) ? clamp((vramUsed / Math.max(1, totalMem)) * 100, 0, 100) : 0;

      // Prefer dynamic bandwidth from server if present, otherwise fall back to profile estimate
      const dynamicBw = Number(metrics.bandwidth_gbps);
      const profileBw = Number(profile.estimated_bandwidth_gbps);
      const bw = Number.isFinite(dynamicBw) && dynamicBw >= 0 ? clamp(dynamicBw, 0, 900) : (Number.isFinite(profileBw) ? clamp(profileBw, 0, 900) : NaN);
      const bwPct = Number.isFinite(bw) ? clamp((bw / 900) * 100, 0, 100) : 0;

      const gpuName = String(data.selectedGpu || '').toUpperCase();
      let warpSlots = 64;
      if (gpuName.includes('A100')) warpSlots = 96;
      else if (gpuName.includes('V100')) warpSlots = 80;
      else if (gpuName.includes('P100')) warpSlots = 64;
      else if (gpuName.includes('RTX_6000')) warpSlots = 72;
      else if (gpuName.includes('M40')) warpSlots = 48;
      else if (gpuName.includes('K80')) warpSlots = 32;

      const warpBasis = Number.isFinite(computeUtil) ? computeUtil : 0;
      const warps = clamp(Math.round((warpBasis / 100) * warpSlots), 1, warpSlots);
      const warpPct = clamp((warps / warpSlots) * 100, 0, 100);

      setMeter('gpuUtilBar', Number.isFinite(gpuUtil) ? gpuUtil : 0);
      setMeter('computeUtilBar', Number.isFinite(computeUtil) ? computeUtil : 0);
      setMeter('vramBar', vramPct);
      setMeter('bwBar', bwPct);
      setMeter('warpBar', warpPct);

      document.getElementById('gpuUtilValue').textContent = Number.isFinite(gpuUtil) ? `${gpuUtil.toFixed(1)}%` : '--';
      document.getElementById('computeUtilValue').textContent = Number.isFinite(computeUtil) ? `${computeUtil.toFixed(1)}%` : '--';
      document.getElementById('vramValue').textContent = Number.isFinite(vramUsed) ? `${vramUsed.toFixed(1)} / ${totalMem.toFixed(0)} GB` : '--';
      document.getElementById('bwValue').textContent = Number.isFinite(bw) ? `${bw.toFixed(1)} GB/s` : '--';
      document.getElementById('warpValue').textContent = `${warps} / ${warpSlots} warps`;

      document.getElementById('resourceHint').textContent = 'Source: GPEmu profiled data';

      const activeModel = data.selectedModel;
      const activeGpu = data.selectedGpu;
      const expected = metrics.total_expected_batches || '?';
      const bs = metrics.batch_size || '32';
      document.getElementById('meta').textContent = `Model ${activeModel.toUpperCase()} | GPU ${activeGpu} | Batch Size ${bs} | Expected Batches ${expected}`;

      updateModelInfo(metrics.model_info, activeModel);
      updateGpuInfo(metrics.gpu_info, activeGpu);
      document.getElementById('algoActive').innerHTML = '<strong>Current mode:</strong> sequential batch execution';
    }

    async function setModelSelection(model) {
      const response = await fetch('/api/set_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });
      const payload = await response.json();
      updateModelInfo(payload.model_info, payload.selected_model);
      document.getElementById('pageTitle').textContent = `GLIDE GPU TASK SCHEDULER SIMULATOR • ${(payload.model_info?.display_name || payload.selected_model || model).toUpperCase()}`;
      return payload;
    }

    async function setGpuSelection(gpu) {
      const response = await fetch('/api/set_gpu', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gpu })
      });
      const payload = await response.json();
      updateGpuInfo(payload.gpu_info, payload.selected_gpu);
      document.getElementById('gpuName').textContent = payload.selected_gpu || gpu;
      return payload;
    }

    async function startNewRun() {
      await fetch('/api/start_new_run', { method: 'POST' });
      resetDashboardState();
      await tick();
    }

    async function tick() {
      try {
        const response = await fetch('/api/metrics', { cache: 'no-store' });
        const metrics = await response.json();
        const batches = Array.isArray(metrics.batches) ? metrics.batches : [];
        console.log('[GLIDE metrics poll]', {
          selectedGpu: metrics.selected_gpu,
          selectedModel: metrics.selected_model,
          gpu_util: metrics.gpu_util,
          compute_util: metrics.compute_util
        });

        populateSelectors(metrics);

        const selector = document.getElementById('modelSelector');
        const selected = metrics.selected_model || 'resnet18';
        if (selector.value !== selected) {
          selector.value = selected;
        }

        const gpuSelector = document.getElementById('gpuSelector');
        const selectedGpu = metrics.selected_gpu || 'Tesla_M40';
        if (gpuSelector.value !== selectedGpu) {
          gpuSelector.value = selectedGpu;
        }

        setStatus(metrics.status, batches.length > 0);
        renderTopStats(metrics);
        renderTaskQueue(metrics);
        renderTimeline(metrics);
        renderNowExecuting(metrics);
        renderResources(metrics);
        renderEventLog(metrics);

        const needsEngine = pageMode === 'overview' || pageMode === 'engine';
        const needsProfiler = pageMode === 'overview' || pageMode === 'profiler';

        let engineData = null;
        if (needsEngine) {
          try {
            const engResp = await fetch('/api/engine/status', { cache: 'no-store' });
            if (engResp.ok) {
              engineData = await engResp.json();
              if (engineData && engineData.ok && engineData.status && pageMode === 'overview') {
                const completed = Number(engineData.status.completed_count) || 0;
                document.getElementById('doneTop').textContent = `${completed}`;
                const qlen = Number(engineData.status.queue_length) || 0;
                const metaEl = document.getElementById('nowExecMeta');
                if (metaEl) {
                  metaEl.textContent = `${metaEl.textContent.split(' | Engine q:')[0]} | Engine q:${qlen}`;
                }
              }
              const resultsResp = await fetch('/api/engine/results', { cache: 'no-store' });
              if (resultsResp.ok) {
                const resultsData = await resultsResp.json();
                if (resultsData && resultsData.ok) {
                  engineData.results = resultsData.results || [];
                }
              }
            }
          } catch (_e) {
            engineData = null;
          }
        }

        let profilerData = null;
        if (needsProfiler) {
          try {
            const profResp = await fetch('/api/profiler/latest', { cache: 'no-store' });
            if (profResp.ok) {
              profilerData = await profResp.json();
              if (profilerData && profilerData.ok && Array.isArray(profilerData.slowest_layers) && pageMode === 'overview') {
                const box = document.getElementById('eventLog');
                if (box) {
                  const slowHtml = profilerData.slowest_layers.slice(0, 5).map((l, idx) => {
                    const ms = Number(l.compute_cost_ms || 0).toFixed(2);
                    return `<div class="log-entry slow">Layer ${idx + 1}: ${l.layer_type} — ${ms} ms</div>`;
                  }).join('');
                  box.innerHTML = slowHtml + box.innerHTML;
                }
              }
            }
          } catch (_e) {
            profilerData = null;
          }
        }

        renderPageSummary(metrics, engineData, profilerData);
      } catch (_err) {
        setStatus('waiting', false);
      }
    }

    document.getElementById('infoToggle').addEventListener('click', () => {
      document.getElementById('gpemuInfo').classList.toggle('open');
    });

    document.getElementById('modelSelector').addEventListener('change', async (evt) => {
      try {
        resetDashboardState();
        const payload = await setModelSelection(evt.target.value);
        applyLookupUtilization(payload);
        console.log('[GLIDE model change response]', payload);
        await tick();
      } catch (_err) {
      }
    });

    document.getElementById('gpuSelector').addEventListener('change', async (evt) => {
      try {
        resetDashboardState();
        const payload = await setGpuSelection(evt.target.value);
        applyLookupUtilization(payload);
        console.log('[GLIDE gpu change response]', payload);
        await tick();
      } catch (_err) {
      }
    });

    document.getElementById('startRunBtn').addEventListener('click', async () => {
      try {
        await startNewRun();
      } catch (_err) {
      }
    });

    document.body.setAttribute('data-page', pageMode || 'overview');
    tick();
    restartTicker();
  </script>
</body>
</html>
"""


def _normalize_model(model_value: Any) -> str:
  if not isinstance(model_value, str):
    return DEFAULT_MODEL
  key = model_value.strip().lower()
  if key in MODEL_CATALOG:
    return key
  profiles = _scan_gpu_profiles()
  if key in profiles['models']:
    return key
  return DEFAULT_MODEL


def _model_info(model_key: str) -> Dict[str, str]:
  model = _normalize_model(model_key)
  if model in MODEL_CATALOG:
    return MODEL_CATALOG[model]
  return {
    'display_name': model,
    'description': 'Profiled model available for GPEmu timing emulation.',
    'layers': 'Unknown',
    'use_case': 'Use for profiled emulation experiments.'
  }


def _load_selected_model() -> str:
  if not os.path.exists(SELECTED_MODEL_PATH):
    return DEFAULT_MODEL

  try:
    with open(SELECTED_MODEL_PATH, 'r', encoding='utf-8') as model_file:
      data = json.load(model_file)
      if isinstance(data, dict):
        return _normalize_model(data.get('model'))
  except (OSError, json.JSONDecodeError):
    return DEFAULT_MODEL

  return DEFAULT_MODEL


def _save_selected_model(model_key: str) -> str:
  model = _normalize_model(model_key)
  _ensure_write_dir()

  with open(SELECTED_MODEL_PATH, 'a+', encoding='utf-8') as model_file:
    fcntl.flock(model_file.fileno(), fcntl.LOCK_EX)
    try:
      model_file.seek(0)
      model_file.truncate()
      json.dump({'model': model, 'updated_at': time.time()}, model_file)
      model_file.flush()
      os.fsync(model_file.fileno())
    finally:
      fcntl.flock(model_file.fileno(), fcntl.LOCK_UN)

  return model


def _clear_metrics_file() -> None:
  _ensure_write_dir()
  with open(METRICS_PATH, 'a+', encoding='utf-8') as metrics_file:
    fcntl.flock(metrics_file.fileno(), fcntl.LOCK_EX)
    try:
      metrics_file.seek(0)
      metrics_file.truncate()
      metrics_file.flush()
      os.fsync(metrics_file.fileno())
    finally:
      fcntl.flock(metrics_file.fileno(), fcntl.LOCK_UN)


def _safe_metrics_payload() -> Dict[str, Any]:
  selected_model = _load_selected_model()
  selected_gpu = _load_selected_gpu()
  profile_stats = _get_profile_stats(selected_gpu, selected_model, 32)
  utilization = get_utilization(selected_gpu, selected_model)
  profiles = _scan_gpu_profiles()
  return {
    'batches': [],
    'status': 'waiting',
    'model': None,
    'selected_model': selected_model,
    'active_model': selected_model,
    'model_info': _model_info(selected_model),
    'selected_gpu': selected_gpu,
    'gpu_info': _gpu_info(selected_gpu),
    'available_gpus': sorted(profiles['gpus'].keys()),
    'available_models': profiles['models'],
    'batch_size': None,
    'start_time': None,
    'total_expected_batches': None,
    'total_batches': 0,
    'avg_compute_time': None,
    'min_compute_time': None,
    'max_compute_time': None,
    'throughput': None,
    'eta_seconds': None,
    'gpu_util': utilization['gpu_util'],
    'compute_util': utilization['compute_util'],
    'profile_stats': profile_stats
  }


def _load_metrics_file() -> Dict[str, Any]:
  # Support two formats:
  # 1) legacy JSON file at METRICS_PATH containing full payload
  # 2) NDJSON per-batch lines at METRICS_PATH with .ndjson extension
  ndjson_path = os.path.splitext(METRICS_PATH)[0] + '.ndjson'

  # If NDJSON exists, read per-line batch records and merge with metadata from JSON if present
  if os.path.exists(ndjson_path):
    batches = []
    try:
      with open(ndjson_path, 'r', encoding='utf-8') as nh:
        for line in nh:
          line = line.strip()
          if not line:
            continue
          try:
            obj = json.loads(line)
            if isinstance(obj, dict):
              batches.append(obj)
          except json.JSONDecodeError:
            continue
    except OSError:
      return _safe_metrics_payload()

    # Base payload: try to read metadata JSON for status/start_time/etc, else safe payload
    payload = _safe_metrics_payload()
    if os.path.exists(METRICS_PATH):
      try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as metrics_file:
          raw = metrics_file.read().strip()
          if raw:
            try:
              data = json.loads(raw)
              if isinstance(data, dict):
                payload.update({k: v for k, v in data.items() if k != 'batches'})
            except json.JSONDecodeError:
              pass
      except OSError:
        pass

    payload['batches'] = batches
    return payload

  # Fallback: legacy JSON file
  if not os.path.exists(METRICS_PATH):
    return _safe_metrics_payload()

  try:
    with open(METRICS_PATH, 'r', encoding='utf-8') as metrics_file:
      raw = metrics_file.read().strip()
      if not raw:
        return _safe_metrics_payload()
      data = json.loads(raw)
      if not isinstance(data, dict):
        return _safe_metrics_payload()
      return data
  except (OSError, json.JSONDecodeError):
    return _safe_metrics_payload()


def get_utilization(gpu: str, model: str, 
                    times: List[float] = None, 
                    elapsed: float = 0) -> Dict[str, int]:
    """
    Get utilization for a GPU + model combo.
    Primary: lookup table (real profiled data)
    Fallback: dynamic calculation from timing
    """
    # Primary: use real lookup table
    gpu_entry = GPU_MODEL_UTILIZATION.get(gpu, {})
    if model in gpu_entry:
        return gpu_entry[model]  # {"gpu_util": X, "compute_util": Y}
    
    # Fallback: dynamic calculation when profile missing
    if times and elapsed > 0:
        total_compute = sum(times)
        avg_compute = total_compute / len(times)
        compute_util = min(100, int((avg_compute / 0.15) * 100))
        gpu_util = min(100, int((total_compute / elapsed) * 100))
        return {"gpu_util": gpu_util, "compute_util": compute_util}
    
    return {"gpu_util": 0, "compute_util": 0}

def _enrich_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
  enriched = _safe_metrics_payload()
  enriched.update(data)

  selected_model = _load_selected_model()
  selected_gpu = _load_selected_gpu()
  active_model = _normalize_model(enriched.get('model') or selected_model)

  enriched['selected_model'] = selected_model
  enriched['active_model'] = active_model
  enriched['model'] = active_model
  enriched['model_info'] = _model_info(active_model)

  enriched['selected_gpu'] = selected_gpu
  enriched['gpu_info'] = _gpu_info(selected_gpu)

  batches: List[Dict[str, Any]] = enriched.get('batches', [])
  if not isinstance(batches, list):
    batches = []
  enriched['batches'] = batches

  times = [
    float(item.get('compute_time'))
    for item in batches
    if isinstance(item, dict) and item.get('compute_time') is not None
  ]
  total_batches = len(times)
  enriched['total_batches'] = total_batches

  if total_batches == 0:
    if enriched.get('status') not in ('running', 'completed'):
      enriched['status'] = 'waiting'
    # Use static lookup when no batches
    utilization = get_utilization(selected_gpu, selected_model)
    enriched['gpu_util'] = utilization['gpu_util']
    enriched['compute_util'] = utilization['compute_util']
    return enriched

  avg_time = sum(times) / total_batches
  min_time = min(times)
  max_time = max(times)

  start_time = enriched.get('start_time')
  elapsed = None
  if start_time is not None:
    try:
      elapsed = max(0.0, time.time() - float(start_time))
    except (TypeError, ValueError):
      elapsed = None

  throughput = None
  if elapsed and elapsed > 0:
    throughput = total_batches / elapsed

  eta_seconds = None
  expected = enriched.get('total_expected_batches')
  try:
    if expected is not None and throughput and throughput > 0:
      remaining = max(0, int(expected) - total_batches)
      eta_seconds = remaining / throughput
  except (TypeError, ValueError):
    eta_seconds = None

  enriched['avg_compute_time'] = avg_time
  enriched['min_compute_time'] = min_time
  enriched['max_compute_time'] = max_time
  enriched['throughput'] = throughput
  enriched['eta_seconds'] = eta_seconds
  
  # Use dynamic utilization based on actual batch data if available.
  if elapsed and elapsed > 0:
    utilization = get_utilization(selected_gpu, selected_model, times=times, elapsed=elapsed)
  else:
    utilization = get_utilization(selected_gpu, selected_model)
  
  enriched['gpu_util'] = utilization['gpu_util']
  enriched['compute_util'] = utilization['compute_util']

  # Derive dynamic VRAM usage and memory bandwidth from profile stats
  # Scale profile values by recent compute time vs profile compute time
  batch_size_for_profile = enriched.get('batch_size') or 32
  profile_stats = _get_profile_stats(selected_gpu, active_model, batch_size_for_profile)
  enriched['profile_stats'] = profile_stats

  try:
    profile_compute = float(profile_stats.get('compute_time_s') or 0)
  except Exception:
    profile_compute = 0

  # Prefer recent per-batch memory/bandwidth telemetry when available
  recent_mem_vals = [float(b.get('memory_gb')) for b in batches if isinstance(b, dict) and b.get('memory_gb') is not None]
  recent_bw_vals = [float(b.get('bandwidth_gbps')) for b in batches if isinstance(b, dict) and b.get('bandwidth_gbps') is not None]
  if recent_mem_vals:
    # use the latest reported memory value
    vram_used = recent_mem_vals[-1]
  else:
    if profile_compute > 0 and total_batches > 0:
      # use average compute time to scale VRAM/BW usage
      scale = avg_time / profile_compute
      try:
        mem_peak = float(profile_stats.get('memory_peak_gb') or 0)
      except Exception:
        mem_peak = 0
      try:
        bw_peak = float(profile_stats.get('estimated_bandwidth_gbps') or 0)
      except Exception:
        bw_peak = 0
      vram_used = max(0.0, min(mem_peak * scale, mem_peak))
      bandwidth = max(0.0, bw_peak * scale)
    else:
      vram_used = float(profile_stats.get('memory_peak_gb') or 0)
      bandwidth = float(profile_stats.get('estimated_bandwidth_gbps') or 0)

  if recent_bw_vals:
    bandwidth = recent_bw_vals[-1]

  enriched['vram_used_gb'] = vram_used
  enriched['bandwidth_gbps'] = bandwidth

  if enriched.get('status') not in ('running', 'completed'):
    enriched['status'] = 'running'

  return enriched


@app.route('/')
def dashboard() -> str:
  return render_template_string(_render_dashboard_page('overview'))


@app.route('/engine')
def engine_dashboard() -> str:
  return render_template_string(_render_dashboard_page('engine'))


@app.route('/profiler')
def profiler_dashboard() -> str:
  return render_template_string(_render_dashboard_page('profiler'))


@app.route('/api/metrics')
def api_metrics():
  data = _load_metrics_file()
  return jsonify(_enrich_metrics(data))


@app.route('/api/set_model', methods=['POST'])
def api_set_model():
  payload = request.get_json(silent=True) or {}
  selected = _save_selected_model(payload.get('model', DEFAULT_MODEL))
  selected_gpu = _load_selected_gpu()
  utilization = get_utilization(selected_gpu, selected)
  print(f"[API set_model] gpu={selected_gpu} model={selected} gpu_util={utilization['gpu_util']} compute_util={utilization['compute_util']}")
  return jsonify({
    'ok': True,
    'selected_model': selected,
    'model_info': _model_info(selected),
    'selected_gpu': selected_gpu,
    'gpu_util': utilization['gpu_util'],
    'compute_util': utilization['compute_util']
  })


@app.route('/api/set_gpu', methods=['POST'])
def api_set_gpu():
  payload = request.get_json(silent=True) or {}
  selected_gpu = _save_selected_gpu(payload.get('gpu', DEFAULT_GPU))
  selected_model = _load_selected_model()
  utilization = get_utilization(selected_gpu, selected_model)
  print(f"[API set_gpu] gpu={selected_gpu} model={selected_model} gpu_util={utilization['gpu_util']} compute_util={utilization['compute_util']}")
  return jsonify({
    'ok': True,
    'selected_gpu': selected_gpu,
    'gpu_info': _gpu_info(selected_gpu),
    'selected_model': selected_model,
    'gpu_util': utilization['gpu_util'],
    'compute_util': utilization['compute_util']
  })


@app.route('/api/start_new_run', methods=['POST'])
def api_start_new_run():
  _clear_metrics_file()
  selected = _load_selected_model()
  selected_gpu = _load_selected_gpu()
  utilization = get_utilization(selected_gpu, selected)
  print(f"[API start_new_run] gpu={selected_gpu} model={selected} gpu_util={utilization['gpu_util']} compute_util={utilization['compute_util']}")
  return jsonify({
    'ok': True,
    'status': 'waiting',
    'selected_model': selected,
    'model_info': _model_info(selected),
    'selected_gpu': selected_gpu,
    'gpu_info': _gpu_info(selected_gpu),
    'gpu_util': utilization['gpu_util'],
    'compute_util': utilization['compute_util']
  })


@app.route('/api/engine/status')
def api_engine_status():
  """Return a compact view of the running engine state."""
  if ENGINE is None:
    return jsonify({'ok': False, 'error': 'engine_unavailable'}), 503
  status = ENGINE.get_queue_status()
  return jsonify({'ok': True, 'selected_gpu': ENGINE.gpu, 'selected_model': ENGINE.model, 'status': status})


@app.route('/api/engine/results')
def api_engine_results():
  """Return completed request results from the engine."""
  if ENGINE is None:
    return jsonify({'ok': False, 'error': 'engine_unavailable'}), 503
  return jsonify({'ok': True, 'results': ENGINE.get_results()})


@app.route('/api/profiler/latest')
def api_profiler_latest():
  """Return the slowest layers from the profiler database (if present)."""
  if _database is None:
    return jsonify({'ok': False, 'error': 'database_unavailable'}), 503
  slowest = _database.get_slowest_layers(limit=10)
  return jsonify({'ok': True, 'slowest_layers': slowest})


if __name__ == '__main__':
  _save_selected_model(_load_selected_model())
  _save_selected_gpu(_load_selected_gpu())
  app.run(host='0.0.0.0', port=5000, debug=False)
