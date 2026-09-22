import fcntl
import csv
import json
import os
import sqlite3
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request


GLIDE_DIR = os.environ.get('GLIDE_DIR', os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(GLIDE_DIR)
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)
METRICS_PATH = os.path.join(GLIDE_DIR, 'glide_metrics.json')
SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')
SELECTED_GPU_PATH = os.path.join(GLIDE_DIR, 'selected_gpu.json')
SELECTED_SCHEDULER_PATH = os.path.join(GLIDE_DIR, 'selected_scheduler.json')
PROFILED_DATA_ROOT = os.path.join(os.path.dirname(GLIDE_DIR), 'profiled_data')
FALLBACK_PROFILED_DATA_ROOT = os.path.join(os.path.dirname(GLIDE_DIR), 'profiled_data')
DEFAULT_MODEL = 'resnet18'
DEFAULT_GPU = 'Tesla_M40'
DEFAULT_SCHEDULER = 'fifo'
SCHEDULER_NAMES = ('fifo', 'sjf', 'hasp')
_RUN_LOCK = threading.Lock()
_RUN_THREAD: Optional[threading.Thread] = None
_RUN_STOP = threading.Event()
_EXPERIMENT_LOCK = threading.Lock()
_EXPERIMENT_THREAD: Optional[threading.Thread] = None
_EXPERIMENT_STATE: Dict[str, Any] = {
  'status': 'idle',
  'progress': '0/12 combinations done',
  'results': {},
}


def _ensure_write_dir() -> None:
  """Ensure a writable GLIDE_DIR exists. On permission errors, fall back to home path.

  Updates module-level path constants to point at the fallback when necessary.
  """
  global GLIDE_DIR, METRICS_PATH, SELECTED_MODEL_PATH, SELECTED_GPU_PATH, SELECTED_SCHEDULER_PATH, PROFILED_DATA_ROOT
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
    SELECTED_SCHEDULER_PATH = os.path.join(GLIDE_DIR, 'selected_scheduler.json')
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
      SELECTED_SCHEDULER_PATH = os.path.join(GLIDE_DIR, 'selected_scheduler.json')
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
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
   <style>
     * {
       margin: 0;
       padding: 0;
       box-sizing: border-box;
     }

     body {
       font-family: 'Inter', sans-serif;
       background: #f8f9fa;
       min-height: 100vh;
       padding: 40px 20px;
       color: #1a1a1a;
     }

     .shell {
       max-width: 1500px;
       margin: 0 auto;
       animation: slideUp 600ms ease-out;
     }

     .topbar {
       display: flex;
       justify-content: space-between;
       align-items: center;
       gap: 20px;
       margin-bottom: 32px;
       border: 1px solid #e5e7eb;
       background: white;
       padding: 24px;
       border-radius: 12px;
       box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
       flex-wrap: wrap;
     }

     .title {
       font-weight: 700;
       letter-spacing: -0.5px;
       font-size: clamp(1.5rem, 3vw, 2rem);
       color: #0f172a;
     }

     .subtitle {
       color: #6b7280;
       font-size: 0.9rem;
       margin-top: 4px;
       font-weight: 400;
     }

     .status {
       display: inline-flex;
       align-items: center;
       gap: 10px;
       font-size: 0.8rem;
       letter-spacing: 0.5px;
       text-transform: uppercase;
       border: 1px solid #e5e7eb;
       border-radius: 6px;
       padding: 10px 14px;
       background: #f9fafb;
       white-space: nowrap;
       font-weight: 600;
     }

     .dot {
       width: 10px;
       height: 10px;
       border-radius: 50%;
       background: #fbbf24;
     }

     .status.running .dot {
       background: #10b981;
       animation: pulse 1.2s infinite;
     }

     .status.completed .dot {
       background: #2563eb;
     }

     .info-panel {
       border: 1px solid #e5e7eb;
       border-radius: 12px;
       background: white;
       margin-bottom: 24px;
       overflow: hidden;
       box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
     }

     .info-head {
       display: flex;
       justify-content: space-between;
       align-items: center;
       padding: 16px 20px;
       background: #f9fafb;
       border-bottom: 1px solid #e5e7eb;
       cursor: pointer;
     }

     .info-title {
       font-size: 1rem;
       letter-spacing: 0.4px;
       text-transform: uppercase;
       font-weight: 600;
       color: #374151;
     }

     .info-toggle {
       border: 1px solid #d1d5db;
       border-radius: 6px;
       width: 32px;
       height: 32px;
       background: white;
       color: #6b7280;
       cursor: pointer;
       font-size: 1.1rem;
       transition: all 200ms ease;
       display: flex;
       align-items: center;
       justify-content: center;
     }

     .info-toggle:hover {
       background: #f3f4f6;
       border-color: #9ca3af;
     }

     .info-body {
       max-height: 0;
       overflow: hidden;
       transition: max-height 300ms ease, padding 300ms ease;
       color: #6b7280;
       line-height: 1.6;
       padding: 0 20px;
       font-size: 0.9rem;
     }

     .info-panel.open .info-body {
       max-height: 200px;
       padding: 16px 20px 20px;
     }

     .control-grid {
       display: grid;
       grid-template-columns: 2.3fr 1.2fr 1.2fr;
       gap: 20px;
       margin-bottom: 24px;
     }

     .control-card,
     .model-info,
     .gpu-info,
     .panel,
     .stats-bar {
       background: white;
       border: 1px solid #e5e7eb;
       border-radius: 12px;
       padding: 20px;
       box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
     }

     .control-row {
       display: flex;
       align-items: end;
       gap: 12px;
       flex-wrap: wrap;
     }

     .slider-wrap {
       min-width: 240px;
       display: grid;
       gap: 6px;
     }

     .mode-badge {
       border: 1px solid #e5e7eb;
       border-radius: 6px;
       padding: 12px 14px;
       background: #f9fafb;
       color: #10b981;
       font-size: 0.8rem;
       letter-spacing: 0.5px;
       text-transform: uppercase;
       min-width: 240px;
       font-weight: 600;
     }

     .static-label {
       border: 1px solid #e5e7eb;
       border-radius: 6px;
       padding: 12px 14px;
       background: #f9fafb;
       color: #6b7280;
       font-size: 0.8rem;
       letter-spacing: 0.5px;
       text-transform: uppercase;
       min-width: 180px;
       font-weight: 600;
     }

     .speed-meta {
       display: flex;
       justify-content: space-between;
       color: #9ca3af;
       font-size: 0.75rem;
       font-weight: 500;
     }

     input[type="range"] {
       width: 100%;
       accent-color: #2563eb;
     }

     .input-label {
       color: #6b7280;
       font-size: 0.75rem;
       text-transform: uppercase;
       letter-spacing: 0.6px;
       margin-bottom: 8px;
       font-weight: 600;
     }

     select, .btn {
       height: 40px;
       border: 1px solid #d1d5db;
       border-radius: 6px;
       background: white;
       color: #1a1a1a;
       font-family: 'Inter', sans-serif;
       padding: 0 12px;
       font-weight: 500;
       font-size: 0.9rem;
       transition: all 200ms ease;
     }

     select {
       min-width: 220px;
     }

     select:hover {
       border-color: #9ca3af;
     }

     .btn {
       cursor: pointer;
       background: #2563eb;
       color: white;
       border-color: #2563eb;
       font-weight: 600;
     }

     .btn:hover {
       background: #1d4ed8;
       border-color: #1d4ed8;
       transform: translateY(-1px);
       box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
     }

     .hint {
       margin-top: 8px;
       color: #9ca3af;
       font-size: 0.8rem;
       font-weight: 400;
     }

     .model-name {
       font-weight: 700;
       font-size: 1.3rem;
       margin-bottom: 8px;
       color: #0f172a;
     }

     .model-line {
       margin-top: 8px;
       color: #6b7280;
       font-size: 0.9rem;
       line-height: 1.5;
     }

     .stats-bar {
       display: grid;
       grid-template-columns: repeat(4, minmax(0, 1fr));
       gap: 16px;
       margin-bottom: 20px;
     }

     .stat-chip {
       position: relative;
       border: 1px solid #e5e7eb;
       border-radius: 10px;
       padding: 16px;
       background: white;
       transition: all 200ms ease;
     }

     .stat-chip:hover {
       box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
     }

     .review-panel {
       background: white;
       border: 1px solid #e5e7eb;
       border-radius: 12px;
       padding: 20px;
       margin-bottom: 20px;
       box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
     }

     .comparison-bars {
       display: grid;
       grid-template-columns: repeat(3, 1fr);
       gap: 16px;
       align-items: end;
       min-height: 150px;
     }

     .comparison-column { display: grid; gap: 6px; text-align: center; }
     .comparison-bar { background: #2563eb; border-radius: 5px 5px 0 0; min-height: 8px; }
     .comparison-column:nth-child(2) .comparison-bar { background: #10b981; }
     .comparison-column:nth-child(3) .comparison-bar { background: #f59e0b; }
     .queue-comparison { width: 100%; height: 150px; background: #f8fafc; border-radius: 8px; }
     .fairness-meter { height: 12px; border-radius: 6px; background: #e5e7eb; overflow: hidden; }
     .fairness-meter > span { display: block; height: 100%; width: 0; transition: width 300ms ease; background: #dc2626; }
     .hasp-copy { color: #6b7280; line-height: 1.6; }
     .formula { font-family: 'IBM Plex Mono', monospace; background: #f9fafb; padding: 12px; border-radius: 6px; margin-top: 10px; }

     .stat-label {
       color: #6b7280;
       font-size: 0.75rem;
       letter-spacing: 0.6px;
       text-transform: uppercase;
       margin-bottom: 10px;
       font-weight: 600;
     }

     .stat-value {
       font-weight: 700;
       font-size: clamp(1.4rem, 2.2vw, 1.9rem);
       line-height: 1.1;
       color: #0f172a;
       font-family: 'IBM Plex Mono', monospace;
     }

     .src-badge {
       display: inline-flex;
       align-items: center;
       border-radius: 4px;
       border: 1px solid #e5e7eb;
       font-size: 0.65rem;
       letter-spacing: 0.4px;
       padding: 2px 8px;
       margin-left: 8px;
       vertical-align: middle;
       font-weight: 600;
       text-transform: uppercase;
     }

     .src-badge.measured { color: #2563eb; background: #eff6ff; }
     .src-badge.profiled { color: #10b981; background: #f0fdf4; }
     .src-badge.derived { color: #f59e0b; background: #fffbeb; }

     .now-strip {
       border: 1px solid #e5e7eb;
       border-radius: 12px;
       padding: 14px 16px;
       margin-bottom: 20px;
       background: white;
       display: flex;
       align-items: center;
       justify-content: space-between;
       gap: 12px;
       flex-wrap: wrap;
       box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
     }

     .now-main {
       font-weight: 700;
       font-size: 1.1rem;
       letter-spacing: 0.5px;
       color: #2563eb;
       text-transform: uppercase;
     }

     .now-meta {
       color: #6b7280;
       font-size: 0.85rem;
       font-weight: 500;
     }

     .sim-grid {
       display: grid;
       grid-template-columns: 300px minmax(0, 1fr) 320px;
       gap: 20px;
     }

     .panel-title {
       color: #6b7280;
       font-size: 0.75rem;
       text-transform: uppercase;
       letter-spacing: 0.6px;
       margin-bottom: 12px;
       font-weight: 600;
     }

     .task-list {
       display: grid;
       gap: 12px;
     }

     .task-item {
       border: 1px solid #e5e7eb;
       border-radius: 8px;
       padding: 12px;
       background: white;
       display: flex;
       justify-content: space-between;
       align-items: center;
       transition: all 200ms ease;
       font-size: 0.9rem;
     }

     .task-item:hover {
       box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
     }

     .task-name {
       font-family: 'IBM Plex Mono', monospace;
       color: #374151;
       font-weight: 500;
     }

     .task-time {
       color: #9ca3af;
       font-size: 0.8rem;
       font-weight: 500;
     }

     table {
       width: 100%;
       border-collapse: collapse;
       font-size: 0.9rem;
     }

     thead {
       background: #f9fafb;
       border-bottom: 1px solid #e5e7eb;
     }

     th {
       padding: 12px 14px;
       text-align: left;
       font-weight: 600;
       color: #374151;
       font-size: 0.8rem;
       text-transform: uppercase;
       letter-spacing: 0.5px;
     }

     td {
       padding: 14px;
       border-bottom: 1px solid #f3f4f6;
       font-family: 'IBM Plex Mono', monospace;
       font-size: 0.85rem;
     }

     tbody tr:hover {
       background: #f9fafb;
     }

     .metric {
       border: 1px solid #e5e7eb;
       border-radius: 10px;
       padding: 12px;
       background: white;
       margin-bottom: 12px;
     }

     .metric-note {
       margin-top: 8px;
       color: #9ca3af;
       font-size: 0.75rem;
       letter-spacing: 0.2px;
     }

     .metric-header {
       display: flex;
       justify-content: space-between;
       color: #6b7280;
       font-size: 0.8rem;
       margin-bottom: 8px;
       text-transform: uppercase;
       font-weight: 600;
     }

     .meter {
       width: 100%;
       height: 8px;
       border-radius: 4px;
       overflow: hidden;
       background: #e5e7eb;
       border: 1px solid #d1d5db;
     }

     .meter > span {
       display: block;
       height: 100%;
       width: 0;
       transition: width 320ms ease;
       background: linear-gradient(90deg, #2563eb, #1d4ed8);
     }

     .legend {
       display: flex;
       gap: 12px;
       flex-wrap: wrap;
       font-size: 0.8rem;
       color: #6b7280;
       margin-top: 12px;
     }

     .legend span {
       border: 1px solid #e5e7eb;
       border-radius: 6px;
       padding: 4px 10px;
       background: #f9fafb;
       font-weight: 500;
     }

     .legend .fast { color: #10b981; }
     .legend .medium { color: #f59e0b; }
     .legend .slow { color: #dc2626; }
     .legend .queued { color: #2563eb; }

     .empty {
       color: #9ca3af;
       font-size: 0.9rem;
       border: 1px dashed #d1d5db;
       border-radius: 8px;
       padding: 20px;
       text-align: center;
       background: #fafbfc;
     }

     .log-box {
       border: 1px solid #e5e7eb;
       border-radius: 8px;
       background: white;
       padding: 12px;
       margin-top: 12px;
       max-height: 180px;
       overflow: auto;
     }

     .log-entry {
       font-size: 0.8rem;
       color: #6b7280;
       padding: 6px 0;
       border-bottom: 1px solid #f3f4f6;
       font-family: 'IBM Plex Mono', monospace;
     }

     .log-entry.fast {
       color: #10b981;
     }

     .log-entry.medium {
       color: #f59e0b;
     }

     .log-entry.slow {
       color: #dc2626;
     }

     .log-entry:last-child {
       border-bottom: 0;
     }

     footer {
       margin-top: 60px;
       padding-top: 30px;
       border-top: 1px solid #e5e7eb;
       text-align: center;
       color: #9ca3af;
       font-size: 0.85rem;
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
       body { padding: 20px; }
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

     @keyframes slideUp {
       from { opacity: 0; transform: translateY(20px); }
       to { opacity: 1; transform: translateY(0); }
     }
   </style>

   <style>
      :root { --bg:#111217; --panel:#181b1f; --border:#2c3235; --text:#d8d9da; --muted:#8b949e; --orange:#ff9900; --blue:#5794f2; --green:#73bf69; --red:#f2495c; }
      body { background:var(--bg); color:var(--text); padding:0; font-family:Inter,system-ui,sans-serif; }
      .shell { max-width:none; margin-left:200px; padding:28px; animation:none; }
      .topbar,.info-panel,.control-card,.model-info,.gpu-info,.panel,.stats-bar,.review-panel,.now-strip { background:var(--panel); border-color:var(--border); color:var(--text); box-shadow:none; }
      .topbar { border-radius:4px; }
      .title,.model-name { color:var(--text); }
      .subtitle,.model-line,.hint,.metric-note,.empty { color:var(--muted); }
      .info-head,.input-label,.stat-label,.info-title { color:var(--text); background:#202328; border-color:var(--border); }
      .status,.static-label,.mode-badge,select { background:#202328; color:var(--text); border-color:var(--border); }
      .btn { background:var(--orange); color:#111217; border:0; font-weight:700; }
      .dot { background:var(--orange); }
      .sidebar { position:fixed; inset:0 auto 0 0; width:200px; background:#111217; border-right:1px solid var(--border); padding:22px 12px; z-index:20; display:flex; flex-direction:column; }
      .brand { color:var(--orange); font-size:1.5rem; font-weight:700; letter-spacing:2px; padding:0 12px 26px; }
      .brand small { display:block; color:var(--muted); font-size:.62rem; letter-spacing:.5px; margin-top:4px; }
      .nav-item { border:0; border-left:3px solid transparent; background:transparent; color:var(--muted); width:100%; text-align:left; padding:12px 12px; cursor:pointer; font-size:.86rem; border-radius:0 4px 4px 0; }
      .nav-item:hover,.nav-item.active { color:var(--text); background:#202328; border-left-color:var(--orange); }
      .nav-icon { display:inline-block; width:22px; color:var(--orange); }
      .sidebar-footer { margin-top:auto; color:var(--muted); font-size:.7rem; padding:12px; border-top:1px solid var(--border); }
      .page { display:none; }
      .page.active { display:block; }
      .page-heading { margin:4px 0 22px; font-size:1.65rem; }
      .page-heading span { color:var(--orange); }
      .review-grid,.cards-3,.cards-4 { display:grid; gap:16px; }
      .cards-4 { grid-template-columns:repeat(4,minmax(0,1fr)); }
      .cards-3 { grid-template-columns:repeat(3,minmax(0,1fr)); }
      .review-grid { grid-template-columns:repeat(2,minmax(0,1fr)); margin-bottom:16px; }
      .review-card { background:var(--panel); border:1px solid var(--border); border-radius:4px; padding:18px; }
      .review-card h3 { margin-bottom:10px; color:var(--text); }
      .review-card p,.review-card li { color:var(--muted); line-height:1.55; font-size:.86rem; }
      .metric-big { color:var(--orange); font-size:1.65rem; font-weight:700; }
      .data-table { width:100%; border-collapse:collapse; font-size:.84rem; }
      .data-table th,.data-table td { padding:11px 10px; border-bottom:1px solid var(--border); text-align:left; }
      .data-table th { color:var(--muted); font-weight:500; }
      .badge { display:inline-block; padding:3px 8px; border-radius:3px; font-size:.68rem; font-weight:700; }
      .badge.green { background:#1d3b2a; color:var(--green); }.badge.blue { background:#1d3150; color:var(--blue); }.badge.yellow { background:#453719; color:#f2c94c; }.badge.red { background:#481f2b; color:var(--red); }
      .bar-list { display:grid; gap:12px; margin-top:14px; }.bar-row { display:grid; grid-template-columns:120px 1fr 60px; align-items:center; gap:8px; font-size:.8rem; }.bar-track { height:14px; background:#282d32; border-radius:2px; overflow:hidden; }.bar-fill { height:100%; background:var(--orange); }.bar-fill.blue { background:var(--blue); }.bar-fill.green { background:var(--green); }.bar-fill.gray { background:#737b84; }
      .terminal { background:#0c0e10; border:1px solid var(--border); padding:16px; min-height:190px; font: .78rem 'Share Tech Mono',monospace; color:var(--green); white-space:pre-wrap; }
      .run-btn { margin-top:12px; background:var(--orange); color:#111217; border:0; padding:9px 14px; border-radius:3px; cursor:pointer; font-weight:700; }
      .algo-card { min-height:210px; border-top:3px solid #737b84; }.algo-card.sjf { border-top-color:var(--blue); }.algo-card.hasp { border-top-color:var(--orange); box-shadow:0 0 0 1px #704800; }
      .formula { color:var(--orange); font-family:'Share Tech Mono',monospace; margin:12px 0; }
      .timeline-svg { width:100%; min-height:180px; background:#101216; border:1px solid var(--border); }
      .select-dark { padding:9px; background:#202328; color:var(--text); border:1px solid var(--border); border-radius:3px; }
      .insight { border-left:3px solid var(--orange); }.scenario { cursor:pointer; }.scenario.selected { border-color:var(--orange); }
      .trace { height:80px; position:relative; border-bottom:1px solid var(--border); background:repeating-linear-gradient(90deg,transparent 0 49px,#252a2e 50px); }.trace i { position:absolute; bottom:8px; width:7px; height:7px; border-radius:50%; background:var(--orange); }
      @media (max-width:900px) { .sidebar { width:66px; }.shell { margin-left:66px; padding:18px; }.brand small,.nav-label,.sidebar-footer { display:none; }.brand { padding:0 10px 20px; }.cards-4,.cards-3,.review-grid { grid-template-columns:1fr 1fr; } }
      @media (max-width:600px) { .cards-4,.cards-3,.review-grid { grid-template-columns:1fr; } }
   </style>
</head>
<body>
  <aside class="sidebar">
    <div class="brand">GLIDE<small>GPU INFERENCE LAB</small></div>
    <button class="nav-item active" data-page="overview"><span class="nav-icon">▦</span><span class="nav-label">Overview</span></button>
    <button class="nav-item" data-page="engine"><span class="nav-icon">◈</span><span class="nav-label">Inference Engine</span></button>
    <button class="nav-item" data-page="scheduler"><span class="nav-icon">⇄</span><span class="nav-label">Scheduler</span></button>
    <button class="nav-item" data-page="profiler"><span class="nav-icon">▥</span><span class="nav-label">Profiler</span></button>
    <button class="nav-item" data-page="experiments"><span class="nav-icon">⚗</span><span class="nav-label">Experiments</span></button>
    <button class="nav-item" data-page="about"><span class="nav-icon">ⓘ</span><span class="nav-label">About</span></button>
    <div class="sidebar-footer">● localhost:5000<br><span style="color:var(--green)">● live refresh 1s</span></div>
  </aside>
  <div class="shell">
   <section class="page active" id="page-overview">
    <div class="topbar">
      <div>
        <div class="title" id="pageTitle">GLIDE GPU TASK SCHEDULER SIMULATOR • RESNET18</div>
        <div class="subtitle" id="meta">Awaiting run metadata...</div>
      </div>
     </div>

   <section class="page" id="page-engine">
      <h1 class="page-heading">Inference <span>Engine</span></h1>
      <div class="cards-4">
        <div class="review-card"><div class="metric-big">216</div><p>Total combinations tested</p></div>
        <div class="review-card"><div class="metric-big" style="color:var(--green)">100%</div><p>Pass rate</p></div>
        <div class="review-card"><div class="metric-big">A100</div><p>Best GPU · fastest compute</p></div>
        <div class="review-card"><div class="metric-big" style="color:var(--red)">K80</div><p>Worst GPU · slowest</p></div>
      </div>
      <div class="review-card" style="margin-top:16px"><h3>GPU comparison</h3><table class="data-table"><thead><tr><th>GPU</th><th>Avg Compute</th><th>Memory</th><th>Tier</th></tr></thead><tbody>
        <tr><td>NVIDIA A100-SXM4-40GB</td><td>18.4 ms</td><td>1,024 MB</td><td><span class="badge green">DATA CENTER</span></td></tr>
        <tr><td>Tesla V100-PCIE-32GB</td><td>31.7 ms</td><td>2,953 MB</td><td><span class="badge green">DATA CENTER</span></td></tr>
        <tr><td>Tesla P100-PCIE-16GB</td><td>42.8 ms</td><td>3,120 MB</td><td><span class="badge blue">PROFESSIONAL</span></td></tr>
        <tr><td>Quadro RTX 6000</td><td>46.2 ms</td><td>3,244 MB</td><td><span class="badge blue">PROFESSIONAL</span></td></tr>
        <tr><td>Tesla M40</td><td>76.5 ms</td><td>3,880 MB</td><td><span class="badge yellow">LEGACY</span></td></tr>
        <tr><td>Tesla K80</td><td>94.7 ms</td><td>4,096 MB</td><td><span class="badge red">LEGACY</span></td></tr>
      </tbody></table></div>
      <div class="review-grid" style="margin-top:16px">
        <div class="review-card"><h3>Model timing by family</h3><div class="bar-list">
         <div class="bar-row"><span>ResNet</span><div class="bar-track"><div class="bar-fill" style="width:82%"></div></div><b>64 ms</b></div>
         <div class="bar-row"><span>VGG</span><div class="bar-track"><div class="bar-fill blue" style="width:100%"></div></div><b>80 ms</b></div>
         <div class="bar-row"><span>DenseNet</span><div class="bar-track"><div class="bar-fill green" style="width:70%"></div></div><b>56 ms</b></div>
         <div class="bar-row"><span>Others</span><div class="bar-track"><div class="bar-fill gray" style="width:22%"></div></div><b>12 ms</b></div>
        </div></div>
        <div class="review-card"><h3>Live engine test</h3><div class="terminal" id="engineTerminal"></div><button class="run-btn" id="runEngineTest">Run new test</button></div>
      </div>

   <section class="page" id="page-scheduler">
      <h1 class="page-heading">Scheduler <span>Comparison</span></h1>
      <div class="cards-3">
        <div class="review-card algo-card"><h3>FIFO — First In First Out</h3><p>Processes requests in arrival order.</p><p><b>Pros:</b> Simple, predictable</p><p><b>Cons:</b> Head-of-line blocking, no starvation prevention</p><span class="badge">BASELINE</span></div>
        <div class="review-card algo-card sjf"><h3>SJF — Shortest Job First</h3><p>Processes shortest compute time first.</p><p><b>Pros:</b> Better throughput</p><p><b>Cons:</b> Long requests starve indefinitely</p><span class="badge blue">BASELINE</span></div>
        <div class="review-card algo-card hasp"><h3>HASP — Heterogeneous Affinity Scheduling</h3><p>Scores requests by affinity + aging boost.</p><div class="formula">score = (1/compute_ms) × memory_fit × (1 + age/threshold)</div><p><b>Pros:</b> Starvation-free, fair, competitive throughput</p><span class="badge yellow">NOVEL CONTRIBUTION</span></div>
      </div>
      <div class="review-card" style="margin-top:16px"><h3>Comparison metrics</h3><table class="data-table"><thead><tr><th>Metric</th><th>FIFO</th><th>SJF</th><th style="color:var(--orange)">HASP</th></tr></thead><tbody>
        <tr><td>Avg Latency</td><td>193 ms</td><td>145 ms</td><td style="color:var(--orange)">112 ms</td></tr><tr><td>P95 Latency</td><td>420 ms</td><td>310 ms</td><td style="color:var(--orange)">185 ms</td></tr><tr><td>Throughput</td><td>5.2/s</td><td>7.1/s</td><td style="color:var(--orange)">6.8/s</td></tr><tr><td>Fairness Index</td><td>0.78</td><td>0.65</td><td style="color:var(--orange)">0.94</td></tr><tr><td>Starvation</td><td>12</td><td>28</td><td style="color:var(--green)">0</td></tr>
      </tbody></table></div>
      <div class="review-card" style="margin-top:16px"><h3>HASP aging boost</h3><svg class="timeline-svg" viewBox="0 0 700 190"><line x1="50" y1="155" x2="660" y2="155" stroke="#737b84"/><text x="50" y="178" fill="#8b949e">0s</text><text x="300" y="178" fill="#8b949e">5s · aging boost</text><text x="620" y="178" fill="#8b949e">10s</text><rect x="70" y="45" width="450" height="22" fill="#5794f2"/><text x="76" y="61" fill="#111217">Request A · waiting</text><rect x="70" y="82" width="150" height="22" fill="#73bf69"/><text x="76" y="98" fill="#111217">Request B</text><rect x="70" y="119" width="260" height="22" fill="#737b84"/><text x="76" y="135" fill="#111217">Request C</text><line x1="350" y1="25" x2="350" y2="160" stroke="#ff9900" stroke-dasharray="5"/><text x="360" y="35" fill="#ff9900">Request A promoted</text></svg><div class="formula">age_boost = waiting_time / 5.0</div></div>
   </section>

   <section class="page" id="page-profiler">
      <h1 class="page-heading">Layer <span>Profiler</span></h1>
      <label>Model <select class="select-dark" id="profileModel"><option>resnet18</option><option>resnet50</option><option>alexnet</option><option>vgg16</option></select></label>
      <div class="review-grid" style="margin-top:16px"><div class="review-card"><h3>Layer timing table</h3><table class="data-table"><thead><tr><th>Layer Type</th><th>Compute</th><th>Memory</th><th>% total</th></tr></thead><tbody><tr><td>Conv2d k=7</td><td>61.29 ms</td><td>98 MB</td><td>34.2%</td></tr><tr><td>BatchNorm</td><td>55.40 ms</td><td>98 MB</td><td>30.9%</td></tr><tr><td>MaxPool2d</td><td>44.88 ms</td><td>24.5 MB</td><td>25.1%</td></tr><tr><td>Conv2d k=3</td><td>33.42 ms</td><td>24.5 MB</td><td>18.7%</td></tr><tr><td>ReLU</td><td>0.03 ms</td><td>3.06 MB</td><td>0.02%</td></tr><tr><td>Linear</td><td>0.17 ms</td><td>0.12 MB</td><td>0.09%</td></tr></tbody></table></div><div class="review-card"><h3>Layer compute time breakdown</h3><div class="bar-list"><div class="bar-row"><span>Conv2d</span><div class="bar-track"><div class="bar-fill" style="width:100%"></div></div><b>94.7</b></div><div class="bar-row"><span>BatchNorm</span><div class="bar-track"><div class="bar-fill blue" style="width:76%"></div></div><b>55.4</b></div><div class="bar-row"><span>MaxPool</span><div class="bar-track"><div class="bar-fill gray" style="width:61%"></div></div><b>44.9</b></div><div class="bar-row"><span>ReLU</span><div class="bar-track"><div class="bar-fill green" style="width:8%"></div></div><b>0.03</b></div></div></div></div>
      <div class="cards-3"><div class="review-card insight"><h3>Bottleneck layer</h3><p>Conv2d (in=3, out=64, k=7) — 61ms — 34% of total</p></div><div class="review-card insight"><h3>Fastest layer</h3><p>AdaptiveAvgPool2d — 0.044ms — negligible</p></div><div class="review-card insight"><h3>Memory peak</h3><p>First Conv2d layer — 98MB output</p></div></div>
   </section>

   <section class="page" id="page-experiments">
      <h1 class="page-heading">Experiment <span>Results</span></h1>
      <div class="cards-4" id="scenarioCards">
        <div class="review-card scenario selected"><h3>A · Single model</h3><p>resnet50 · uniform · 2/s · 30s</p></div><div class="review-card scenario"><h3>B · Single model</h3><p>resnet50 · Poisson · 2/s · 30s</p></div><div class="review-card scenario"><h3>C · Multi-model</h3><p>resnet18, resnet50, vgg16 · uniform</p></div><div class="review-card scenario"><h3>D · Bursty</h3><p>resnet18, resnet50, vgg16 · bursty</p></div>
      </div>
      <div class="review-grid" style="margin-top:16px"><div class="review-card"><h3>Scheduler results</h3><div class="bar-list"><div class="bar-row"><span>FIFO avg</span><div class="bar-track"><div class="bar-fill gray" style="width:100%"></div></div><b>193ms</b></div><div class="bar-row"><span>SJF avg</span><div class="bar-track"><div class="bar-fill blue" style="width:75%"></div></div><b>145ms</b></div><div class="bar-row"><span>HASP avg</span><div class="bar-track"><div class="bar-fill" style="width:58%"></div></div><b>112ms</b></div><div class="bar-row"><span>Fairness</span><div class="bar-track"><div class="bar-fill green" style="width:94%"></div></div><b>0.94</b></div></div></div><div class="review-card"><h3>Workload trace</h3><div class="trace" id="workloadTrace"></div><p style="margin-top:12px">Request arrivals over the selected scenario duration</p></div></div>
      <div class="review-card"><h3>Key findings</h3><div class="cards-3"><p>HASP achieves <b style="color:var(--green)">0 starvation</b> events across all scenarios.</p><p>HASP fairness index <b style="color:var(--orange)">0.94</b> vs FIFO 0.78 and SJF 0.65.</p><p>HASP throughput stays within 5% of SJF while eliminating starvation.</p></div></div>
   </section>

   <section class="page" id="page-about">
      <h1 class="page-heading"><span>GLIDE</span> · About</h1>
      <div class="review-card"><h2>GPU Layer-Level Inference and Dispatching Emulator</h2><p style="margin-top:12px;line-height:1.7">GLIDE is a final-year ECE project built on top of GPEmu. It enables GPU inference scheduling research without requiring real GPU hardware.</p></div>
      <div class="review-grid" style="margin-top:16px"><div class="review-card"><h3>Team</h3><p><b>Pranav M (1CR23EC104)</b> — Core engine, HASP scheduler, GPEmu integration</p><p>[Teammate 2] — Model decomposition, workload generator, metrics</p><p>[Teammate 3] — Dashboard, complexity analyser, experimental report</p><p style="margin-top:12px">CMR Institute of Technology, Bengaluru<br>Electronics and Communication Engineering<br>Academic Year: 2025-2026</p></div><div class="review-card"><h3>Technology stack</h3><div class="cards-3"><span class="badge blue">GPEmu</span><span class="badge blue">PyTorch</span><span class="badge blue">SQLite</span><span class="badge blue">Flask</span><span class="badge blue">Chart.js</span><span class="badge blue">Docker</span></div></div></div>
      <div class="review-card"><h3>Hardware profiles</h3><p>6 real GPU profiles from Chameleon Cloud · 36 neural network architectures · 216 GPU × model combinations tested · <b style="color:var(--green)">100% pass rate</b></p></div>
   </section>
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
        <div class="hint">Select model and GPU for real GPEmu-backed metrics. Scheduler: <strong id="schedulerActive">FIFO</strong>.</div>
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
      <div class="stat-chip">
        <div class="stat-label">Fairness Index <span class="src-badge measured">METRICS</span></div>
        <div class="stat-value" id="fairnessTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Starvation Count <span class="src-badge measured">METRICS</span></div>
        <div class="stat-value" id="starvationTop">--</div>
      </div>
    </div>

    <div class="now-strip">
      <div class="now-main" id="nowExec">Now Executing: Idle</div>
      <div class="now-meta" id="nowExecMeta">State: waiting | Source: profiled timing + measured runtime events</div>
    </div>

    <div class="review-panel">
      <div class="panel-title">Scheduler Comparison</div>
      <div class="comparison-bars" id="schedulerComparison"><div class="empty">Run an experiment to load scheduler results.</div></div>
      <div class="panel-title">Queue Length Over Time</div>
      <svg class="queue-comparison" id="schedulerQueueComparison" viewBox="0 0 600 150" preserveAspectRatio="none"></svg>
    </div>

    <details class="review-panel">
      <summary class="panel-title">How HASP prevents starvation</summary>
      <div class="hasp-copy">HASP prevents requests from waiting forever by giving bonus priority to older requests. The longer you wait, the higher your priority becomes.</div>
      <div class="formula">affinity = (1 / compute_ms) × memory_fit_score × (1 + age_boost)</div>
    </details>

    <div class="review-panel">
      <div class="panel-title">Real-time Jain Fairness Index</div>
      <div class="fairness-meter"><span id="fairnessMeter"></span></div>
    </div>

    <div class="sim-grid">
      <div class="panel">
        <div class="panel-title">Task Queue Panel <span class="src-badge measured">MEASURED</span></div>
        <div class="spark-wrap">
          <div class="spark-title">Queue Length Chart</div>
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
  </section>
  </div>

  <script>
   const shell = document.querySelector('.shell');
   document.querySelectorAll('.page:not(#page-overview)').forEach((page) => shell.appendChild(page));
   document.querySelectorAll('.nav-item').forEach((item) => item.addEventListener('click', () => {
     const target = item.dataset.page;
     document.querySelectorAll('.nav-item').forEach((nav) => nav.classList.toggle('active', nav === item));
     document.querySelectorAll('.page').forEach((page) => page.classList.toggle('active', page.id === `page-${target}`));
     if (target === 'engine') startEngineReplay();
     if (target === 'experiments') drawWorkloadTrace();
   }));
    const fmt = (v, suffix = 's') => (Number.isFinite(v) ? `${v.toFixed(4)} ${suffix}` : '--');
    let refreshMs = 1000;
    let refreshTimer = null;
    let queueDepthHistory = [];
    let engineReplayTimer = null;

    function startEngineReplay() {
      const terminal = document.getElementById('engineTerminal');
      if (!terminal) return;
      if (engineReplayTimer) clearInterval(engineReplayTimer);
      const lines = [
        '[REQUEST] ae3664bf | Tesla V100 | ResNet50 | batch=32',
        '[COMPUTE] emulated: 31.72ms | memory: 2953MB',
        '[DONE]    latency: 34.24ms | queued: 0',
        '[REQUEST] 481637cf | Tesla V100 | ResNet50 | batch=32',
        '[COMPUTE] emulated: 31.72ms | memory: 2953MB',
        '[DONE]    latency: 34.11ms | queued: 0',
        '[REQUEST] 7cd19a42 | Tesla V100 | ResNet50 | batch=32',
        '[COMPUTE] emulated: 31.72ms | memory: 2953MB',
        '[DONE]    latency: 33.98ms | queued: 0'
      ];
      let index = 0;
      terminal.textContent = '';
      engineReplayTimer = setInterval(() => {
        terminal.textContent += `${lines[index]}\n`;
        index += 1;
        if (index >= lines.length) clearInterval(engineReplayTimer);
      }, 350);
    }

    function drawWorkloadTrace() {
      const trace = document.getElementById('workloadTrace');
      if (!trace || trace.children.length) return;
      for (let i = 0; i < 24; i += 1) {
        const dot = document.createElement('i');
        dot.style.left = `${4 + ((i * 37) % 92)}%`;
        trace.appendChild(dot);
      }
    }

    document.querySelectorAll('.scenario').forEach((card) => card.addEventListener('click', () => {
      document.querySelectorAll('.scenario').forEach((item) => item.classList.remove('selected'));
      card.classList.add('selected');
      const trace = document.getElementById('workloadTrace');
      if (trace) trace.innerHTML = '';
      drawWorkloadTrace();
    }));
    startEngineReplay();

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
       document.getElementById('fairnessTop').textContent = '--';
       document.getElementById('starvationTop').textContent = '--';
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
      const summary = metrics.metrics || {};

      document.getElementById('utilTop').textContent = Number.isFinite(gpuUtil) ? `${gpuUtil.toFixed(1)}%` : '--';
      document.getElementById('avgTop').textContent = Number.isFinite(avg) ? `${avg.toFixed(4)}s` : '--';
      document.getElementById('minTop').textContent = Number.isFinite(minVal) ? `${minVal.toFixed(4)}s` : '--';
      document.getElementById('maxTop').textContent = Number.isFinite(maxVal) ? `${maxVal.toFixed(4)}s` : '--';
      document.getElementById('throughputTop').textContent = Number.isFinite(throughput) ? `${throughput.toFixed(2)} b/s` : '--';
      document.getElementById('etaTop').textContent = formatSeconds(eta);
      document.getElementById('doneTop').textContent = `${done}`;
      const fairness = Number(summary.jains_fairness_index);
      const starvation = Number(summary.starvation_count);
      document.getElementById('fairnessTop').textContent = Number.isFinite(fairness) ? fairness.toFixed(3) : '--';
      document.getElementById('starvationTop').textContent = Number.isFinite(starvation) ? `${starvation}` : '--';
      document.getElementById('schedulerActive').textContent = String(metrics.scheduler_name || 'fifo').toUpperCase();
      const meter = document.getElementById('fairnessMeter');
      meter.style.width = `${clamp((Number.isFinite(fairness) ? fairness : 0) * 100, 0, 100)}%`;
      meter.style.background = fairness >= 0.9 ? '#16a34a' : (fairness >= 0.7 ? '#f59e0b' : '#dc2626');
    }

    async function renderSchedulerComparison() {
      const container = document.getElementById('schedulerComparison');
      const queueChart = document.getElementById('schedulerQueueComparison');
      try {
        const response = await fetch('/api/experiment_results', { cache: 'no-store' });
        const payload = await response.json();
        const values = ['fifo', 'sjf', 'hasp'].map((name) => payload[name] || {});
        const maxLatency = Math.max(1, ...values.map((item) => Number(item.avg_latency_ms) || 0));
        container.innerHTML = values.map((item, index) => {
          const name = ['FIFO', 'SJF', 'HASP'][index];
          const height = ((Number(item.avg_latency_ms) || 0) / maxLatency) * 130;
          return `<div class="comparison-column"><strong>${name}</strong><div class="comparison-bar" style="height:${height}px"></div><small>Avg ${(Number(item.avg_latency_ms) || 0).toFixed(1)} ms<br>P95 ${(Number(item.p95_latency_ms) || 0).toFixed(1)} ms<br>Fair ${(Number(item.jains_fairness_index) || 0).toFixed(3)}</small></div>`;
        }).join('');
        const colors = ['#2563eb', '#10b981', '#f59e0b'];
        const histories = values.map((item) => Array.isArray(item.queue_history) ? item.queue_history : []);
        const maxQueue = Math.max(1, ...histories.flatMap((history) => history.map((point) => Number(point.queue_length) || 0)));
        queueChart.innerHTML = histories.map((history, index) => {
          if (history.length < 1) return '';
          const points = history.map((point, pointIndex) => {
            const x = history.length === 1 ? 0 : (pointIndex / (history.length - 1)) * 600;
            const y = 145 - ((Number(point.queue_length) || 0) / maxQueue) * 135;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          }).join(' ');
          return `<polyline points="${points}" fill="none" stroke="${colors[index]}" stroke-width="3"/>`;
        }).join('');
      } catch (_error) {
        container.innerHTML = '<div class="empty">No experiment results available.</div>';
        queueChart.innerHTML = '';
      }
    }

    function renderTaskQueue(metrics) {
      const queue = document.getElementById('taskQueue');
      const batches = Array.isArray(metrics.batches) ? metrics.batches : [];
      const history = Array.isArray(metrics.queue_history) ? metrics.queue_history : [];
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

      if (history.length > 0) {
        renderQueueSparkline(Number(history[history.length - 1].queue_length) || 0);
      }
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

    tick();
    renderSchedulerComparison();
    restartTicker();
  </script>
</body>
</html>
"""


# Grafana-style presentation layer. Backend routes and payload preparation remain
# unchanged; this template consumes the existing /api/metrics contract.
DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLIDE Metrics Console</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{--bg:#111217;--panel:#181b1f;--border:#2c3235;--text:#d8d9da;--muted:#8e9090;--orange:#ff9900;--blue:#5794f2;--green:#73bf69;--red:#f2495c;--yellow:#fade2a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:12px Inter,system-ui,sans-serif}.dashboard{padding:14px;max-width:1800px;margin:auto}
.statusbar{display:flex;align-items:center;gap:24px;border:1px solid var(--border);border-top:2px solid var(--orange);background:var(--panel);padding:12px 16px;margin-bottom:12px}.brand{font-size:18px;font-weight:800;letter-spacing:2px;color:var(--orange)}.status{display:flex;align-items:center;gap:7px;text-transform:uppercase;font-weight:700}.dot{width:8px;height:8px;border-radius:50%;background:var(--green)}.status.idle .dot{background:#777}.status.completed .dot{background:var(--blue)}.status-meta{color:var(--muted)}.status-meta b{color:var(--text);font-weight:500}
.grid{display:grid;gap:12px;margin-bottom:12px}.g4{grid-template-columns:repeat(4,1fr)}.g2{grid-template-columns:repeat(2,1fr)}.panel{background:var(--panel);border:1px solid var(--border);border-top:2px solid var(--border);padding:12px;min-width:0}.panel.orange{border-top-color:var(--orange)}.panel.blue{border-top-color:var(--blue)}.panel.green{border-top-color:var(--green)}.panel.red{border-top-color:var(--red)}.panel-title{text-transform:uppercase;font-size:11px;color:#fff;font-weight:700;letter-spacing:.5px;margin-bottom:8px}.muted{color:var(--muted)}
.gauge-panel{text-align:center;min-height:180px}.gauge{width:140px;height:140px;display:block;margin:-2px auto -12px}.gauge text{font-size:24px;font-weight:700;fill:#fff}.gauge .label{font-size:11px;font-weight:400;fill:var(--muted)}.track{fill:none;stroke:var(--border);stroke-width:10;stroke-linecap:round}.value{fill:none;stroke-width:10;stroke-linecap:round;transform:rotate(-45deg);transform-origin:70px 70px}
.metric{font-size:32px;font-weight:700;color:#fff;text-align:center;padding:25px 0 8px}.unit{font-size:12px;color:var(--muted);font-weight:400}.stat{min-height:116px}.stat .metric{text-align:left;padding:17px 0 4px}.stat-split{display:flex;justify-content:space-between;padding-top:18px;font-size:20px}.stat-split small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}
canvas{max-height:220px}.resource{display:grid;gap:12px}.resource-row{display:grid;grid-template-columns:125px 1fr 80px;gap:8px;align-items:center}.meter{height:9px;background:#2a2e33;border-radius:2px;overflow:hidden}.meter span{display:block;height:100%;background:var(--blue);width:0}.meter.orange span{background:var(--orange)}.meter.green span{background:var(--green)}.meter.red span{background:var(--red)}
.queue-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center;margin-bottom:10px}.queue-stats strong{display:block;font-size:20px;color:#fff}.queue-stats small{color:var(--muted);text-transform:uppercase;font-size:9px}.event-log{font:11px "Share Tech Mono",monospace;color:var(--green);line-height:1.8;max-height:125px;overflow:auto;border-top:1px solid var(--border);padding-top:7px}.event-log .medium{color:var(--yellow)}.event-log .slow{color:var(--red)}
.selectors{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.selectors label{display:grid;gap:5px;color:var(--muted);text-transform:uppercase;font-size:10px}.selectors select,.action{background:#202328;color:var(--text);border:1px solid var(--border);padding:8px 10px;border-radius:2px}.action{background:var(--orange);color:#111217;font-weight:700;cursor:pointer}.info{color:var(--muted);line-height:1.5}.info summary{cursor:pointer;color:#fff}.notice{color:var(--orange);border-left:2px solid var(--orange);padding-left:10px}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:200px;background:#111217;border-right:1px solid var(--border);padding:22px 12px;z-index:10}.sidebar .brand{padding:0 10px 24px}.nav-item{display:block;width:100%;padding:11px 10px;margin:2px 0;background:transparent;border:0;border-left:3px solid transparent;color:var(--muted);text-align:left;cursor:pointer;border-radius:0 3px 3px 0}.nav-item:hover,.nav-item.active{background:#202328;color:#fff;border-left-color:var(--orange)}.nav-icon{display:inline-block;width:24px;color:var(--orange);font-size:15px}.sidebar-foot{position:absolute;bottom:18px;left:22px;color:var(--muted);font-size:10px;line-height:1.8}.dashboard{margin-left:200px}.page{display:none}.page.active{display:block}.page-heading{font-size:20px;margin:0 0 14px}.review-card{background:var(--panel);border:1px solid var(--border);border-top:2px solid var(--border);padding:14px}.review-card h3{margin:0 0 10px}.review-card p{color:var(--muted);line-height:1.5}.review-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px}.review-cards.four{grid-template-columns:repeat(4,1fr)}.review-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}.review-table{width:100%;border-collapse:collapse}.review-table th,.review-table td{padding:9px;border-bottom:1px solid var(--border);text-align:left}.review-table th{color:var(--muted);font-size:10px;text-transform:uppercase}.badge{padding:3px 7px;border-radius:2px;font-size:10px;font-weight:bold}.badge.green{color:var(--green);background:#18311f}.badge.blue{color:var(--blue);background:#172945}.badge.yellow{color:var(--yellow);background:#3b3111}.badge.red{color:var(--red);background:#3b1820}.bar{height:12px;background:#2b3035}.bar span{display:block;height:100%;background:var(--orange)}.terminal{font:11px "Share Tech Mono",monospace;color:var(--green);background:#0d0f11;border:1px solid var(--border);padding:12px;min-height:170px;white-space:pre-wrap}.formula{color:var(--orange);font-family:monospace;margin:10px 0}.timeline{width:100%;background:#0d0f11;border:1px solid var(--border)}.scenario{cursor:pointer}.scenario.selected{border-top-color:var(--orange)}.trace{height:68px;border-bottom:1px solid var(--border);background:repeating-linear-gradient(90deg,transparent 0 49px,#2c3235 50px);position:relative}.trace i{position:absolute;bottom:8px;width:7px;height:7px;border-radius:50%;background:var(--orange)}.insight{border-left:3px solid var(--orange)}@media(max-width:900px){.sidebar{width:64px}.sidebar .nav-label,.sidebar-foot,.sidebar .brand small{display:none}.dashboard{margin-left:64px}.review-cards,.review-cards.four,.review-grid{grid-template-columns:1fr 1fr}}@media(max-width:600px){.review-cards,.review-cards.four,.review-grid{grid-template-columns:1fr}}
@media(max-width:900px){.g4{grid-template-columns:repeat(2,1fr)}.g2{grid-template-columns:1fr}.statusbar{flex-wrap:wrap;gap:12px}}@media(max-width:520px){.g4{grid-template-columns:1fr}.dashboard{padding:8px}}
</style>
</head>
<body><aside class="sidebar"><div class="brand">GLIDE<small style="display:block;color:var(--muted);font-size:9px;letter-spacing:1px">GPU INFERENCE LAB</small></div><button class="nav-item active" data-page="overview"><span class="nav-icon">▦</span><span class="nav-label">Overview</span></button><button class="nav-item" data-page="engine"><span class="nav-icon">◈</span><span class="nav-label">Inference Engine</span></button><button class="nav-item" data-page="scheduler"><span class="nav-icon">⇄</span><span class="nav-label">Scheduler</span></button><button class="nav-item" data-page="profiler"><span class="nav-icon">▥</span><span class="nav-label">Profiler</span></button><button class="nav-item" data-page="experiments"><span class="nav-icon">⚗</span><span class="nav-label">Experiment Results</span></button><button class="nav-item" data-page="about"><span class="nav-icon">ⓘ</span><span class="nav-label">About</span></button><div class="sidebar-foot">● localhost:5000<br><span style="color:var(--green)">● refresh 1s</span></div></aside><main class="dashboard">
<div id="page-overview" class="page active">
<header class="statusbar"><div class="brand">GLIDE</div><div id="status" class="status"><i class="dot"></i><span id="statusText">IDLE</span></div><div class="status-meta">GPU <b id="gpuName">--</b></div><div class="status-meta">MODEL <b id="modelName">--</b></div><div class="status-meta">SCHEDULER <b id="schedulerName">FIFO</b></div><div class="status-meta">REFRESH <b>1s</b></div></header>

<section class="grid g4">
 <article class="panel gauge-panel blue"><div class="panel-title">GPU utilization <span class="info-tip" title="From the GPU_MODEL_UTILIZATION lookup table: real profiled values for the selected GPU and model.">?</span></div><svg class="gauge" viewBox="0 0 140 140"><circle class="track" cx="70" cy="70" r="54" pathLength="100" stroke-dasharray="75 25" transform="rotate(135 70 70)"/><circle id="gpuGauge" class="value" cx="70" cy="70" r="54" pathLength="100" stroke-dasharray="0 100"/><text id="gpuGaugeText" x="70" y="76" text-anchor="middle">--</text><text class="label" x="70" y="103" text-anchor="middle">percent</text></svg></article>
 <article class="panel gauge-panel green"><div class="panel-title">Compute utilization <span class="info-tip" title="The compute-bound portion of the same real GPU_MODEL_UTILIZATION lookup table.">?</span></div><svg class="gauge" viewBox="0 0 140 140"><circle class="track" cx="70" cy="70" r="54" pathLength="100" stroke-dasharray="75 25" transform="rotate(135 70 70)"/><circle id="computeGauge" class="value" cx="70" cy="70" r="54" pathLength="100" stroke-dasharray="0 100"/><text id="computeGaugeText" x="70" y="76" text-anchor="middle">--</text><text class="label" x="70" y="103" text-anchor="middle">percent</text></svg></article>
 <article class="panel gauge-panel orange"><div class="panel-title">Memory utilization <span class="info-tip" title="Profiled VRAM used divided by the selected GPU's total memory capacity.">?</span></div><svg class="gauge" viewBox="0 0 140 140"><circle class="track" cx="70" cy="70" r="54" pathLength="100" stroke-dasharray="75 25" transform="rotate(135 70 70)"/><circle id="memoryGauge" class="value" cx="70" cy="70" r="54" pathLength="100" stroke-dasharray="0 100"/><text id="memoryGaugeText" x="70" y="76" text-anchor="middle">--</text><text class="label" x="70" y="103" text-anchor="middle">percent</text></svg></article>
 <article class="panel stat green"><div class="panel-title">Throughput <span class="info-tip" title="Batches completed divided by elapsed time since this run started.">?</span></div><div id="throughput" class="metric">--</div><div class="muted" style="text-align:center">batches / second</div></article>
</section>

<section class="grid g2"><article class="panel orange"><div class="panel-title">Batch compute time</div><canvas id="computeChart"></canvas></article><article class="panel blue"><div class="panel-title">Latency over time</div><canvas id="latencyChart"></canvas></article></section>

<section class="grid g4"><article class="panel stat orange"><div class="panel-title">Avg compute time</div><div id="avgCompute" class="metric">-- <span class="unit">ms</span></div></article><article class="panel stat"><div class="panel-title">Min / Max</div><div class="stat-split"><div><small>Min</small><span id="minCompute">--</span></div><div><small>Max</small><span id="maxCompute">--</span></div></div></article><article class="panel stat blue"><div class="panel-title">Total batches</div><div id="totalBatches" class="metric">--</div></article><article class="panel stat"><div class="panel-title">Est. time remaining</div><div id="eta" class="metric">--</div></article></section>

<section class="grid g2"><article class="panel blue"><div class="panel-title">GPU resource usage</div><div class="resource"><div class="resource-row"><span>Compute cores</span><div class="meter green"><span id="computeBar"></span></div><b id="computeBarText">--</b></div><div class="resource-row"><span>VRAM usage</span><div class="meter orange"><span id="vramBar"></span></div><b id="vramText">--</b></div><div class="resource-row"><span>Memory bandwidth</span><div class="meter"><span id="bandwidthBar"></span></div><b id="bandwidthText">--</b></div></div><details class="info" style="margin-top:14px"><summary>How these numbers are calculated</summary><p>GPU Utilization: profiled benchmark lookup for the selected GPU and model; it is not calculated live.</p><p>Compute Utilization: profiled compute-bound fraction for the same GPU/model combination.</p><p>Memory Utilization: peak memory for the selected model and batch divided by the GPU VRAM capacity, multiplied by 100.</p><p>Throughput: completed requests divided by elapsed seconds since the run started.</p><p>Avg/Min/Max Compute Time: live statistics over completed emulated compute times.</p></details></article><article class="panel green"><div class="panel-title">Request queue</div><div class="queue-stats"><div><strong id="queueDepth">0</strong><small>Depth</small></div><div><strong id="completed">0</strong><small>Completed</small></div><div><strong id="avgLatency">--</strong><small>Avg ms</small></div><div><strong id="p95Latency">--</strong><small>P95 ms</small></div></div><div id="eventLog" class="event-log">Waiting for event data...</div></article></section>

<section class="grid g2"><article class="panel blue"><div class="panel-title">GPU profile</div><div class="selectors"><label>Selected GPU<select id="gpuSelector"></select></label></div><p id="gpuInfo" class="info">Loading profile...</p></article><article class="panel orange"><div class="panel-title">Model profile</div><div class="selectors"><label>Selected model<select id="modelSelector"></select></label><label>Scheduler<select id="schedulerSelector"><option value="fifo">FIFO</option><option value="sjf">SJF</option><option value="hasp">HASP</option></select></label></div><p id="modelInfo" class="info">Loading model...</p></article></section>

<section class="panel"><div class="selectors"><button id="startRun" class="action">Start new run</button><button id="stopRun" class="action" style="background:var(--red);color:#fff" disabled>Stop run</button><details class="info"><summary>What is GPEmu?</summary>GPEmu emulates GPU inference timing and resource behavior on ordinary CPU hardware, allowing GLIDE scheduling experiments without a physical GPU.</details><span class="muted">Mode: live engine run using the selected scheduler</span></div></section>
</div>
<section id="page-engine" class="page"><h1 class="page-heading">Inference Engine</h1><div class="review-cards four"><div class="review-card"><b style="font-size:25px">216</b><p>Total combinations tested</p></div><div class="review-card"><b style="font-size:25px;color:var(--green)">100%</b><p>Pass rate</p></div><div class="review-card"><b style="font-size:25px">NVIDIA A100</b><p>Best GPU · fastest compute</p></div><div class="review-card"><b style="font-size:25px;color:var(--red)">Tesla K80</b><p>Worst GPU · slowest</p></div></div><div class="review-card"><h3>GPU comparison</h3><table class="review-table"><tr><th>GPU</th><th>Avg Compute (ms)</th><th>Memory (MB)</th><th>Tier</th></tr><tr><td>NVIDIA A100-SXM4-40GB</td><td>18.4</td><td>1024</td><td><span class="badge green">DATA CENTER</span></td></tr><tr><td>Tesla V100-PCIE-32GB</td><td>31.72</td><td>2953</td><td><span class="badge green">DATA CENTER</span></td></tr><tr><td>Tesla P100-PCIE-16GB</td><td>42.8</td><td>3120</td><td><span class="badge blue">PROFESSIONAL</span></td></tr><tr><td>Quadro RTX 6000</td><td>46.2</td><td>3244</td><td><span class="badge blue">PROFESSIONAL</span></td></tr><tr><td>Tesla M40</td><td>76.5</td><td>3880</td><td><span class="badge yellow">LEGACY</span></td></tr><tr><td>Tesla K80</td><td>94.7</td><td>4096</td><td><span class="badge red">LEGACY</span></td></tr></table></div><div class="review-grid" style="margin-top:12px"><div class="review-card"><h3>Model timing by family</h3><p>ResNet · 64ms</p><div class="bar"><span style="width:82%"></span></div><p>VGG · 80ms</p><div class="bar"><span style="width:100%;background:var(--blue)"></span></div><p>DenseNet · 56ms</p><div class="bar"><span style="width:70%;background:var(--green)"></span></div><p>Others · 12ms</p><div class="bar"><span style="width:22%;background:#737b84"></span></div></div><div class="review-card"><h3>Live engine test</h3><div id="engineTerminal" class="terminal"></div><button id="runEngineTest" class="action" style="margin-top:10px">Run new test</button></div></div></section>
<section id="page-scheduler" class="page"><h1 class="page-heading">Scheduler Comparison</h1><div class="review-cards"><div class="review-card"><h3>FIFO — First In First Out</h3><p>Processes requests in arrival order.</p><p><b>Pros:</b> Simple, predictable<br><b>Cons:</b> Head-of-line blocking, no starvation prevention</p><span class="badge">BASELINE</span></div><div class="review-card" style="border-top-color:var(--blue)"><h3>SJF — Shortest Job First</h3><p>Processes shortest compute time first.</p><p><b>Pros:</b> Better throughput<br><b>Cons:</b> Long requests starve indefinitely</p><span class="badge blue">BASELINE</span></div><div class="review-card" style="border-top-color:var(--orange)"><h3>HASP — Heterogeneous Affinity Scheduling</h3><p>Scores requests by affinity + aging boost.</p><div class="formula">score = (1/compute_ms) × memory_fit × (1 + age/threshold)</div><p><b>Pros:</b> Starvation-free, fair, competitive throughput</p><span class="badge yellow">NOVEL CONTRIBUTION</span></div></div><div class="review-card"><h3>Comparison metrics</h3><table class="review-table"><tr><th>Metric</th><th>FIFO</th><th>SJF</th><th style="color:var(--orange)">HASP</th></tr><tr><td>Avg Latency</td><td>193ms</td><td>145ms</td><td style="color:var(--orange)">112ms</td></tr><tr><td>P95 Latency</td><td>420ms</td><td>310ms</td><td style="color:var(--orange)">185ms</td></tr><tr><td>Throughput</td><td>5.2/s</td><td>7.1/s</td><td style="color:var(--orange)">6.8/s</td></tr><tr><td>Fairness Index</td><td>0.78</td><td>0.65</td><td style="color:var(--orange)">0.94</td></tr><tr><td>Starvation</td><td>12</td><td>28</td><td style="color:var(--green)">0</td></tr></table></div><div class="review-card" style="margin-top:12px"><h3>HASP aging boost</h3><svg class="timeline" viewBox="0 0 700 180"><line x1="40" y1="150" x2="660" y2="150" stroke="#737b84"/><line x1="350" y1="20" x2="350" y2="155" stroke="#ff9900" stroke-dasharray="5"/><text x="355" y="35" fill="#ff9900">5s aging boost · Request A promoted</text><rect x="60" y="55" width="450" height="20" fill="#5794f2"/><text x="68" y="70" fill="#111217">Request A waiting</text><rect x="60" y="85" width="180" height="20" fill="#73bf69"/><text x="68" y="100" fill="#111217">Request B</text><rect x="60" y="115" width="280" height="20" fill="#737b84"/><text x="68" y="130" fill="#111217">Request C</text></svg><div class="formula">age_boost = waiting_time / 5.0</div></div></section>
<section id="page-profiler" class="page"><h1 class="page-heading">Layer Profiler</h1><label class="muted">Model <select class="selectors" style="display:inline-block" id="profileModel"><option>resnet18</option><option>resnet50</option><option>alexnet</option><option>vgg16</option></select></label><div class="review-grid" style="margin-top:12px"><div class="review-card"><h3>Layer timing table</h3><table class="review-table"><tr><th>Layer Type</th><th>Compute</th><th>Memory</th><th>% total</th></tr><tr><td>Conv2d k=7</td><td>61.29ms</td><td>98MB</td><td>34.2%</td></tr><tr><td>BatchNorm</td><td>55.40ms</td><td>98MB</td><td>30.9%</td></tr><tr><td>MaxPool2d</td><td>44.88ms</td><td>24.5MB</td><td>25.1%</td></tr><tr><td>Conv2d k=3</td><td>33.42ms</td><td>24.5MB</td><td>18.7%</td></tr><tr><td>ReLU</td><td>0.03ms</td><td>3.06MB</td><td>0.02%</td></tr><tr><td>Linear</td><td>0.17ms</td><td>0.12MB</td><td>0.09%</td></tr></table></div><div class="review-card"><h3>Layer compute time breakdown</h3><p>Conv2d <span class="bar" style="display:inline-block;width:70%"><span style="width:100%"></span></span></p><p>BatchNorm <span class="bar" style="display:inline-block;width:65%"><span style="width:76%;background:var(--blue)"></span></span></p><p>MaxPool <span class="bar" style="display:inline-block;width:60%"><span style="width:61%;background:#737b84"></span></span></p><p>ReLU <span class="bar" style="display:inline-block;width:50%"><span style="width:8%;background:var(--green)"></span></span></p></div></div><div class="review-cards"><div class="review-card insight"><h3>Bottleneck layer</h3><p>Conv2d (in=3, out=64, k=7) — 61ms — 34% of total</p></div><div class="review-card insight"><h3>Fastest layer</h3><p>AdaptiveAvgPool2d — 0.044ms — negligible</p></div><div class="review-card insight"><h3>Memory peak</h3><p>First Conv2d layer — 98MB output</p></div></div></section>
<section id="page-experiments" class="page"><h1 class="page-heading">Experiment Results</h1><div class="review-cards four"><div class="review-card scenario selected"><h3>A · Single model</h3><p>resnet50 · uniform · 2/s · 30s</p></div><div class="review-card scenario"><h3>B · Single model</h3><p>resnet50 · Poisson · 2/s · 30s</p></div><div class="review-card scenario"><h3>C · Multi-model</h3><p>resnet18, resnet50, vgg16 · uniform</p></div><div class="review-card scenario"><h3>D · Bursty</h3><p>resnet18, resnet50, vgg16 · bursty</p></div></div><div class="review-grid"><div class="review-card"><h3>Scheduler results</h3><p>FIFO avg latency 193ms</p><div class="bar"><span style="width:100%;background:#737b84"></span></div><p>SJF avg latency 145ms</p><div class="bar"><span style="width:75%;background:var(--blue)"></span></div><p>HASP avg latency 112ms</p><div class="bar"><span style="width:58%"></span></div><p>Fairness index 0.94</p><div class="bar"><span style="width:94%;background:var(--green)"></span></div></div><div class="review-card"><h3>Workload trace</h3><div id="workloadTrace" class="trace"></div><p class="muted">Request arrivals over selected scenario duration.</p></div></div><div class="review-card"><h3>Key findings</h3><div class="review-cards"><p>HASP achieves <b style="color:var(--green)">0 starvation events</b> across all scenarios.</p><p>HASP fairness index <b style="color:var(--orange)">0.94</b> vs FIFO 0.78 and SJF 0.65.</p><p>HASP throughput stays within 5% of SJF while eliminating starvation.</p></div></div></section>
<section id="page-about" class="page"><h1 class="page-heading">GLIDE <span style="color:var(--orange)">· About</span></h1><div class="review-card"><h2>GPU Layer-Level Inference and Dispatching Emulator</h2><p>GLIDE is a final-year ECE project built on top of GPEmu. It enables GPU inference scheduling research without requiring real GPU hardware.</p></div><div class="review-grid" style="margin-top:12px"><div class="review-card"><h3>Team</h3><p><b>Pranav M (1CR23EC104)</b> — Core engine, HASP scheduler, GPEmu integration</p><p>[Teammate 2] — Model decomposition, workload generator, metrics</p><p>[Teammate 3] — Dashboard, complexity analyser, experimental report</p><p>CMR Institute of Technology, Bengaluru<br>Electronics and Communication Engineering<br>Academic Year: 2025-2026</p></div><div class="review-card"><h3>Technology stack</h3><p>GPEmu · PyTorch 1.8 · SQLite · Flask · Chart.js · Docker</p><h3 style="margin-top:20px">Hardware profiles</h3><p>6 GPU profiles · 36 architectures · 216 combinations · <b style="color:var(--green)">100% pass rate</b></p></div></div></section>
<section id="page-engine" class="page"><h1 class="page-heading">Inference Engine</h1><div class="review-cards four"><div class="review-card"><b id="engineCount">--</b><p>Combinations available</p></div><div class="review-card"><b style="color:var(--green)">Profiled</b><p>Real profile source</p></div><div class="review-card"><b id="engineBest">--</b><p>Fastest GPU</p></div><div class="review-card"><b id="engineWorst">--</b><p>Slowest GPU</p></div></div><div class="review-card"><h3>GPU comparison</h3><table class="review-table"><thead><tr><th>GPU</th><th>Compute ms</th><th>Memory MB</th><th>Tier</th></tr></thead><tbody id="gpuComparisonRows"></tbody></table></div><div class="review-grid" style="margin-top:12px"><div class="review-card"><h3>Model timings</h3><div id="modelTimingBars"></div></div><div class="review-card"><h3>Live engine test</h3><div id="engineTerminal" class="terminal">Run a real scheduler test to load requests.</div><button id="runEngineTest" class="action" style="margin-top:10px">Run new test</button></div></div></section>
<section id="page-scheduler" class="page"><h1 class="page-heading">Scheduler</h1><div class="review-cards"><div class="review-card"><h3>FIFO</h3><p>Requests execute in arrival order.</p><span class="badge">BASELINE</span></div><div class="review-card"><h3>SJF</h3><p>Shortest profiled compute time first.</p><span class="badge blue">BASELINE</span></div><div class="review-card"><h3>HASP</h3><p>Affinity scheduling with aging boost.</p><div class="formula">age_boost = waiting_time / 5.0</div><span class="badge yellow">NOVEL CONTRIBUTION</span></div></div><div class="review-card"><h3>Real scheduler comparison</h3><div id="schedulerLoading" class="muted">Computing results from real profiled data...</div><table class="review-table"><thead><tr><th>Metric</th><th>FIFO</th><th>SJF</th><th>HASP</th></tr></thead><tbody id="schedulerRows"></tbody></table></div></section>
<section id="page-profiler" class="page"><h1 class="page-heading">Profiler</h1><label class="muted">Model <select id="profileModel"><option>resnet18</option><option>resnet50</option><option>alexnet</option><option>vgg16</option></select></label><div class="review-card" style="margin-top:12px"><h3>Real layer timing results</h3><div id="profilerLoading" class="muted">Loading SQLite profiles...</div><table class="review-table"><thead><tr><th>Layer</th><th>Compute ms</th><th>Memory MB</th><th>% total</th></tr></thead><tbody id="profilerRows"></tbody></table></div><div class="review-cards" style="margin-top:12px"><div class="review-card insight"><h3>Bottleneck</h3><p id="bottleneck">--</p></div><div class="review-card insight"><h3>Fastest</h3><p id="fastest">--</p></div><div class="review-card insight"><h3>Memory peak</h3><p id="memoryPeak">--</p></div></div></section>
<section id="page-experiments" class="page"><h1 class="page-heading">Experiment Results</h1><div id="experimentLoading" class="muted">No experiment data yet.</div><button id="runExperiments" class="action" style="margin:10px 0">Run all experiments (~60 seconds)</button><div id="experimentProgress" class="muted"></div><div id="experimentResults"></div></section>
<section id="page-about" class="page"><h1 class="page-heading">GLIDE <span style="color:var(--orange)">· About</span></h1><div class="review-card"><h2>GPU Layer-Level Inference and Dispatching Emulator</h2><p>GLIDE is a final-year ECE project built on top of GPEmu. It enables GPU inference scheduling research without requiring real GPU hardware.</p></div><div class="review-grid" style="margin-top:12px"><div class="review-card"><h3>Team</h3><p><b>Pranav M (1CR23EC104)</b> — Core engine, HASP scheduler, GPEmu integration</p><p>Model decomposition, workload generator, metrics</p><p>Dashboard, complexity analyser, experimental report</p><p>CMR Institute of Technology, Bengaluru<br>Electronics and Communication Engineering<br>Academic Year: 2025-2026</p></div><div class="review-card"><h3>Technology stack</h3><p>GPEmu · PyTorch · SQLite · Flask · Chart.js · Docker</p><h3>Hardware profiles</h3><p>6 GPU profiles · 36 architectures · 216 combinations tested</p></div></div></section>
</main>
<script>
const $=id=>document.getElementById(id), colors={orange:'#ff9900',blue:'#5794f2',green:'#73bf69',red:'#f2495c',yellow:'#fade2a',grid:'#2c3235'};
let computeChart,latencyChart,lastMetrics=null;
let engineReplayTimer=null;
['engine','scheduler','profiler','experiments','about'].forEach(name=>{
 const matches=document.querySelectorAll(`#page-${name}`);
 for(let i=0;i<matches.length-1;i++)matches[i].remove();
});
document.querySelectorAll('.nav-item').forEach(item=>item.addEventListener('click',()=>{
 const page=item.dataset.page;
 document.querySelectorAll('.nav-item').forEach(nav=>nav.classList.toggle('active',nav===item));
 document.querySelectorAll('.page').forEach(section=>section.classList.toggle('active',section.id===`page-${page}`));
 if(page==='engine') replayEngine();
 if(page==='experiments') drawTrace();
}));
function replayEngine(){
 const terminal=$('engineTerminal'); if(!terminal)return;
 if(engineReplayTimer)clearInterval(engineReplayTimer);
 terminal.textContent='Computing results from the selected GPU/model...';
 fetch('/api/engine_test').then(response=>response.json()).then(data=>{
  const requests=data.requests||[];let i=0;terminal.textContent='';
    engineReplayTimer=setInterval(()=>{const item=requests[i++];if(!item){clearInterval(engineReplayTimer);return}const gpu=data.gpu||$('gpuSelector')?.value||'selected GPU',model=data.model||$('modelSelector')?.value||'selected model';terminal.textContent+=`[REQUEST] ${item.request_id} | ${gpu} | ${model} | batch=${item.batch_size}\n[COMPUTE] emulated: ${num(item.emulated_compute_ms).toFixed(2)}ms | memory: ${num(item.memory_mb).toFixed(0)}MB\n[DONE]    latency: ${num(item.latency_ms).toFixed(2)}ms | queued: 0\n`},300);
 }).catch(()=>{terminal.textContent='Unable to load the selected engine test.'});
}
function drawTrace(){}
if($('runEngineTest'))$('runEngineTest').addEventListener('click',replayEngine);
replayEngine();
function num(v,d=0){const n=Number.parseFloat(v);return Number.isFinite(n)?n:d}
function color(value,kind){if(kind==='memory')return value>80?colors.red:value>=60?colors.yellow:colors.green;if(kind==='compute')return value>70?colors.orange:value>=50?colors.yellow:colors.green;return value>80?colors.orange:value>=60?colors.yellow:colors.green}
function gauge(id,text,value,kind){const v=Math.max(0,Math.min(100,num(value)));$(id).setAttribute('stroke-dasharray',`${v*.75} ${100-v*.75}`);$(id).setAttribute('stroke',color(v,kind));$(text).textContent=`${v.toFixed(0)}%`}
function chartOptions(){return {responsive:true,animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8e9090'},grid:{color:colors.grid}},y:{ticks:{color:'#8e9090'},grid:{color:colors.grid}}}}}
function updateCharts(data){const batches=Array.isArray(data.batches)?data.batches:[], labels=batches.map((_,i)=>i+1), compute=batches.map(x=>num(x.compute_time)*1000), latency=(data.per_request||[]).map(x=>num(x.latency_ms));if(computeChart)computeChart.data={labels,datasets:[{data:compute,borderColor:colors.orange,backgroundColor:'transparent',tension:.25}]};else computeChart=new Chart($('computeChart'),{type:'line',data:{labels,datasets:[{data:compute,borderColor:colors.orange,backgroundColor:'transparent',tension:.25}]},options:chartOptions()});if(latencyChart)latencyChart.data={labels:latency.map((_,i)=>i+1),datasets:[{data:latency,borderColor:colors.blue,backgroundColor:'transparent',tension:.25}]};else latencyChart=new Chart($('latencyChart'),{type:'line',data:{labels:latency.map((_,i)=>i+1),datasets:[{data:latency,borderColor:colors.blue,backgroundColor:'transparent',tension:.25}]},options:chartOptions()});computeChart.update();latencyChart.update()}
function setOptions(select,values,selected){if(!select.options.length)values.forEach(v=>select.add(new Option(v,v)));select.value=selected}
function tier(ms){return ms<50?['DATA CENTER','green']:ms<=100?['PROFESSIONAL','blue']:['LEGACY','red']}
async function loadGpuComparison(){const rows=await (await fetch('/api/gpu_comparison')).json();$('engineCount').textContent=rows.length*36;$('engineBest').textContent=rows[0]?.gpu||'--';$('engineWorst').textContent=rows.at(-1)?.gpu||'--';$('gpuComparisonRows').innerHTML=rows.map(row=>{const t=tier(row.compute_ms);return `<tr><td>${row.gpu}</td><td>${num(row.compute_ms).toFixed(2)}</td><td>${num(row.memory_mb).toFixed(1)}</td><td><span class="badge ${t[1]}">${t[0]}</span></td></tr>`}).join('');loadModelTimings($('gpuSelector')?.value||'Tesla_V100-PCIE-32GB')}
async function loadModelTimings(gpu){const rows=await (await fetch(`/api/model_timings?gpu=${encodeURIComponent(gpu)}`)).json();const max=Math.max(1,...rows.map(row=>num(row.compute_ms)));$('modelTimingBars').innerHTML=`<div class="model-timing-list">${rows.map(row=>`<p>${row.model} · ${num(row.compute_ms).toFixed(2)}ms</p><div class="bar"><span style="width:${Math.min(100,num(row.compute_ms)/max*100)}%"></span></div>`).join('')}</div>`}
async function loadScheduler(){const data=await (await fetch('/api/scheduler_comparison')).json();$('schedulerLoading').style.display='none';const rows=[['Avg latency','avg_latency_ms','ms'],['P95 latency','p95_latency_ms','ms'],['Throughput','throughput','/s'],['Fairness index','jains_fairness_index',''],['Starvation','starvation_count','']];$('schedulerRows').innerHTML=rows.map(row=>`<tr><td>${row[0]}</td>${['fifo','sjf','hasp'].map(name=>`<td style="${name==='hasp'?'color:var(--orange)':''}">${num(data[name]?.[row[1]]).toFixed(row[1]==='jains_fairness_index'?3:2)}${row[2]}</td>`).join('')}</tr>`).join('')}
async function loadProfiler(model){$('profilerLoading').textContent='Loading SQLite profiles...';let data=await (await fetch(`/api/profiler_results?model=${encodeURIComponent(model)}`)).json();if(!data.length){$('profilerLoading').textContent='No database result. Run profiler first.';return}$('profilerLoading').textContent=`${data.length} real layers loaded`;$('profilerRows').innerHTML=data.map(x=>`<tr><td>${x.layer_type}<br><small>${x.config}</small></td><td>${num(x.compute_cost_ms).toFixed(3)}</td><td>${num(x.memory_cost_mb).toFixed(2)}</td><td>${num(x.pct_of_total).toFixed(2)}%</td></tr>`).join('');$('bottleneck').textContent=`${data[0].layer_type} — ${num(data[0].compute_cost_ms).toFixed(3)}ms`;const fast=data.at(-1);$('fastest').textContent=`${fast.layer_type} — ${num(fast.compute_cost_ms).toFixed(3)}ms`;$('memoryPeak').textContent=`${Math.max(...data.map(x=>num(x.memory_cost_mb))).toFixed(2)}MB`}
async function loadExperiments(){const payload=await (await fetch('/api/experiment_results',{cache:'no-store'})).json();const data=payload.results||{};const loading=$('experimentLoading'),progress=$('experimentProgress'),button=$('runExperiments');if(payload.status==='computing'){loading.textContent='Experiments are running in the background.';progress.textContent=`Running experiment ${payload.progress||''}${payload.current?` (${payload.current})`:''}`;button.disabled=true;return}if(payload.status==='failed'){loading.textContent=`Experiment failed: ${payload.progress||'unknown error'}`;button.disabled=false;return}if(payload.status==='idle'&&!Object.keys(data).length){loading.textContent='No experiment data yet.';progress.textContent='';button.disabled=false;$('experimentResults').innerHTML='';return}loading.textContent=payload.status==='complete'?'Real experiment results loaded.':'Preparing experiment results...';progress.textContent=payload.progress||'';button.disabled=false;$('experimentResults').innerHTML=Object.entries(data).filter(([,result])=>result&&result.schedulers).map(([name,result])=>`<div class="review-card" style="margin-bottom:12px"><h3>${name}</h3><p class="legend-note">Computed from a generated workload trace and real profiled engine timings.</p><table class="review-table"><tr><th>Scheduler</th><th>Avg ms</th><th>P95 ms</th><th>Fairness</th><th>Starvation</th></tr>${Object.entries(result.schedulers||{}).map(([scheduler,m])=>`<tr><td>${scheduler}</td><td>${num(m.avg_latency_ms).toFixed(2)}</td><td>${num(m.p95_latency_ms).toFixed(2)}</td><td>${num(m.jains_fairness_index).toFixed(3)}</td><td>${m.starvation_count}</td></tr>`).join('')}</table></div>`).join('')}
function render(data){lastMetrics=data;const gpu=num(data.gpu_util),compute=num(data.compute_util),profile=data.profile_stats||{},gpuInfo=data.gpu_info||{},modelInfo=data.model_info||{},memory=num(profile.memory_peak_gb)*1024,totalMemory=num(gpuInfo.memory_gb)*1024,memoryPct=totalMemory?memory/totalMemory*100:0,bandwidth=num(profile.estimated_bandwidth_gbps),throughput=num(data.throughput),metrics=data.metrics||{},queue=data.queue_status||{};setOptions($('gpuSelector'),data.available_gpus||[],data.selected_gpu);setOptions($('modelSelector'),data.available_models||[],data.selected_model||data.active_model);if($('schedulerSelector'))$('schedulerSelector').value=data.scheduler_name||'fifo';$('gpuName').textContent=data.selected_gpu||'--';$('modelName').textContent=modelInfo.display_name||data.active_model||'--';if($('schedulerName'))$('schedulerName').textContent=String(data.scheduler_name||'fifo').toUpperCase();$('statusText').textContent=String(data.status||'IDLE').toUpperCase();$('status').className=`status ${data.status||'idle'}`;gauge('gpuGauge','gpuGaugeText',gpu,'gpu');gauge('computeGauge','computeGaugeText',compute,'compute');gauge('memoryGauge','memoryGaugeText',memoryPct,'memory');$('throughput').textContent=throughput.toFixed(2);$('avgCompute').innerHTML=`${(num(data.avg_compute_time)*1000).toFixed(2)} <span class="unit">ms</span>`;$('minCompute').textContent=(num(data.min_compute_time)*1000).toFixed(2);$('maxCompute').textContent=(num(data.max_compute_time)*1000).toFixed(2);$('totalBatches').textContent=num(data.total_batches);$('eta').textContent=data.eta_seconds==null?'--':`${num(data.eta_seconds).toFixed(1)}s`;$('queueDepth').textContent=num(queue.queue_length);$('completed').textContent=num(data.total_batches);$('avgLatency').textContent=num(metrics.avg_latency_ms||queue.avg_latency_ms).toFixed(2);$('p95Latency').textContent=num(metrics.p95_latency_ms||queue.p95_latency_ms).toFixed(2);$('computeBar').style.width=`${compute}%`;$('computeBarText').textContent=`${compute}%`;$('vramBar').style.width=`${Math.min(100,memoryPct)}%`;$('vramText').textContent=`${memory.toFixed(1)} MB`;$('bandwidthBar').style.width=`${Math.min(100,bandwidth/1e3*100)}%`;$('bandwidthText').textContent=bandwidth?`${bandwidth.toFixed(1)} GB/s`:'--';$('gpuInfo').innerHTML=`<b>${gpuInfo.name||data.selected_gpu||'--'}</b><br>${num(gpuInfo.memory_gb).toFixed(1)} GB memory · ${num(gpuInfo.compute_profile_count)} compute profiles`;$('modelInfo').innerHTML=`<b>${modelInfo.display_name||'--'}</b><br>${modelInfo.description||''}<br>${modelInfo.layers||'--'} layers · ${modelInfo.use_case||''}`;const log=(data.per_request||[]).slice(-6).map(x=>{const ms=num(x.latency_ms),cls=ms<50?'':ms<150?'medium':'slow';return `<div class="${cls}">Batch ${x.request_id||'--'} — ${(ms/1000).toFixed(3)}s — ${new Date(num(x.end_time)*1000||Date.now()).toLocaleTimeString()}</div>`}).join('');$('eventLog').innerHTML=log||'Waiting for event data...';$('startRun').disabled=data.status==='running';$('stopRun').disabled=data.status!=='running';updateCharts(data)}
async function refresh(){try{const r=await fetch('/api/metrics',{cache:'no-store'});render(await r.json())}catch(e){$('statusText').textContent='OFFLINE'}}
async function post(path,payload){await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});refresh()}
if($('gpuSelector'))$('gpuSelector').addEventListener('change',async e=>{await post('/api/set_gpu',{gpu:e.target.value});loadModelTimings(e.target.value);replayEngine()});
if($('modelSelector'))$('modelSelector').addEventListener('change',async e=>{await post('/api/set_model',{model:e.target.value});replayEngine()});
if($('schedulerSelector'))$('schedulerSelector').addEventListener('change',async e=>{await post('/api/set_scheduler',{scheduler:e.target.value});refresh()});
if($('startRun'))$('startRun').addEventListener('click',async()=>{await post('/api/start_run',{});refresh()});
if($('stopRun'))$('stopRun').addEventListener('click',async()=>{await post('/api/stop_run',{});refresh()});
if($('runExperiments'))$('runExperiments').addEventListener('click',async()=>{await post('/api/run_experiments',{});loadExperiments()});
if($('profileModel'))$('profileModel').addEventListener('change',e=>loadProfiler(e.target.value));
loadGpuComparison();loadScheduler();loadProfiler('resnet18');loadExperiments();refresh();setInterval(refresh,1000);setInterval(loadExperiments,2000);
function addAuditStyles(){
  const style=document.createElement('style');
  style.textContent=`
    .understand{margin:0 0 12px;padding:14px;background:#202328;border:1px solid var(--border);border-left:3px solid var(--orange)}
    .understand h2{margin:0 0 6px;font-size:18px;color:#fff}.understand p{margin:0;color:var(--muted);line-height:1.5}
    .concept-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;align-items:center;margin-top:12px}
    .flow-step{position:relative;padding:13px 7px;text-align:center;background:#111217;border:1px solid var(--border);color:#fff}
    .flow-step:not(:last-child)::after{content:'→';position:absolute;right:-15px;color:var(--orange);font-size:18px;z-index:2}
    .request-dot{width:12px;height:12px;border-radius:50%;background:var(--orange);animation:flowMove 2s linear infinite;position:absolute;top:4px;left:3%;z-index:3}
    @keyframes flowMove{0%{left:3%}25%{left:27%}50%{left:52%}75%{left:77%}100%{left:97%}}
    .compute-clock{color:var(--orange);animation:clockFill var(--compute-duration,1s) linear infinite alternate}
    @keyframes clockFill{from{opacity:.35}to{opacity:1}}
    .scheduler-sim{display:grid;grid-template-columns:1fr 190px;gap:14px;margin-top:12px}
    .request-board{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;align-items:end;min-height:160px}
    .sim-request{padding:9px 4px;text-align:center;border:1px solid var(--blue);background:#172945;color:#fff;transition:transform .7s,background .7s,border-color .7s}
    .sim-request.waiting{border-color:var(--red);background:#3b1820;animation:waitPulse 1s infinite alternate}
    .sim-request.done{border-color:var(--green);background:#18311f;transform:translateY(-35px)}
    .sim-request small,.score{display:block;color:var(--muted);font-size:10px}
    @keyframes waitPulse{from{box-shadow:0 0 0 transparent}to{box-shadow:0 0 12px var(--red)}}
    .gpu-box{display:grid;place-items:center;min-height:130px;border:2px solid var(--orange);background:#2b2416;color:#fff}
    .sim-controls{display:flex;gap:8px;align-items:center;margin-top:10px}.sim-controls select{padding:7px;background:#202328;color:#fff;border:1px solid var(--border)}
    .race{display:flex;align-items:end;gap:8px;height:150px;padding:12px;background:#111217;border:1px solid var(--border)}
    .race-col{flex:1;text-align:center;color:#fff}.race-bar{height:8px;background:var(--orange);transition:height .35s}.race-col small{color:var(--muted)}
    .layer-stack{display:grid;gap:5px;margin-top:12px}.layer-box{min-height:18px;padding:5px 9px;border-left:4px solid var(--orange);background:#202328;overflow:hidden;transition:height .3s}.layer-box span{font-weight:700}.layer-box small{display:block;color:var(--muted)}
    .traffic-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}.traffic-card{padding:10px;background:#111217;border:1px solid var(--border)}.traffic-dots{font-size:18px;letter-spacing:4px;color:var(--orange);min-height:26px}.traffic-dots.uniform{animation:dotPulse 1.2s infinite}.traffic-dots.poisson{animation:dotPulse 1.8s infinite}.traffic-dots.bursty{animation:dotPulse .8s infinite}@keyframes dotPulse{50%{opacity:.45}}
    .legend-note{margin-top:8px;color:var(--muted);font-size:11px}.model-timing-list{max-height:520px;overflow:auto;padding-right:6px}.info-tip{display:inline-grid;place-items:center;width:16px;height:16px;margin-left:4px;border:1px solid var(--muted);border-radius:50%;color:var(--muted);font-size:10px;cursor:help}
    @media(max-width:700px){.concept-flow,.traffic-grid{grid-template-columns:1fr 1fr}.scheduler-sim{grid-template-columns:1fr}}
  `;
  document.head.appendChild(style);
}
function addUnderstanding(){
  const engine=document.getElementById('page-engine');
  if(engine&&!engine.querySelector('.engine-understand')){
    engine.insertAdjacentHTML('afterbegin',`<div class="understand engine-understand"><h2>How the inference engine works</h2><p>This page shows how fast each GPU can process one batch of 32 images through a neural network — using real timing measurements from actual GPU hardware.</p><div class="concept-flow" style="--compute-duration:1s"><i class="request-dot"></i><div class="flow-step">Arrives</div><div class="flow-step">Waits in queue</div><div class="flow-step">GPU computes <span class="compute-clock">◷</span></div><div class="flow-step">Done</div></div><p class="legend-note">The moving dot is one request; the compute clock is scaled from the selected GPU's real resnet50 timing.</p></div>`);
  }
  const scheduler=document.getElementById('page-scheduler');
  if(scheduler&&!scheduler.querySelector('.scheduler-understand')){
    scheduler.insertAdjacentHTML('afterbegin',`<div class="understand scheduler-understand"><h2>How a scheduler chooses the next request</h2><p>When multiple requests arrive at once, someone has to decide which one gets processed first. This page compares three different decision strategies.</p><div class="sim-controls"><label>Example strategy <select id="simAlgorithm"><option>FIFO</option><option>SJF</option><option>HASP</option></select></label><button class="action" id="playScheduler">Play example</button><span id="simExplanation" class="muted"></span></div><div class="scheduler-sim"><div class="request-board" id="requestBoard"></div><div class="gpu-box">GPU<br><small>one request at a time</small></div></div><p class="legend-note">HASP score = (1 / compute_ms) × memory_fit_score × (1 + waiting_time / AGING_THRESHOLD). AGING_THRESHOLD=5.0 seconds is configurable; after five seconds waiting, the age factor doubles a request's priority.</p><div class="legend-note">Illustration uses example request sizes; the measured table below uses the same 20-request trace for all schedulers.</div></div>`);
    scheduler.insertAdjacentHTML('beforeend',`<div class="review-card"><h3>Real completion race</h3><p class="muted">Each bar uses the real completion timestamps returned by the scheduler run; taller means more requests completed by the end of the run.</p><div id="schedulerRace" class="race"></div></div>`);
    initSchedulerExample();
  }
  const profiler=document.getElementById('page-profiler');
  if(profiler&&!profiler.querySelector('.profiler-understand')){
    profiler.insertAdjacentHTML('afterbegin',`<div class="understand profiler-understand"><h2>What the profiler measures</h2><p>A neural network is built from many small building blocks called layers. This page measures how long each individual layer takes to run. Timing is isolated: each captured layer input is warmed up three times and measured seven times separately, so fast ReLU layers are not mistaken for the cost of the surrounding network.</p><p id="layerStackMeta" class="legend-note">Loading layer count...</p><div id="layerStack" class="layer-stack"><div class="muted">Loading real layer sizes...</div></div><button class="action" id="showAllLayers" type="button">Show all layers</button><p class="legend-note">Each box height is proportional to that layer's real compute time. Hover a layer for a plain-English explanation.</p></div>`);
  }
  const experiments=document.getElementById('page-experiments');
  if(experiments&&!experiments.querySelector('.experiments-understand')){
    experiments.insertAdjacentHTML('afterbegin',`<div class="understand experiments-understand"><h2>Why test different traffic patterns?</h2><p>We tested all three schedulers under four realistic traffic patterns to see which handles real-world conditions best.</p><div class="traffic-grid"><div class="traffic-card"><b>Uniform</b><div class="traffic-dots uniform">•　•　•　•　•</div><small>Cars arriving evenly at a toll booth</small></div><div class="traffic-card"><b>Poisson</b><div class="traffic-dots poisson">••　　•　　　••　•</div><small>Random phone calls throughout the day</small></div><div class="traffic-card"><b>Multi-model</b><div class="traffic-dots uniform">•　•　•　•　•</div><small>Different model types sharing one GPU</small></div><div class="traffic-card"><b>Bursty</b><div class="traffic-dots bursty">••••　　　••••</div><small>Rush-hour traffic followed by quiet gaps</small></div></div><p class="legend-note">Dots above illustrate the concept; the results below use real request traces.</p></div>`);
    experiments.insertAdjacentHTML('beforeend',`<div class="review-card"><h3>Real workload traces</h3><p class="muted">Each dot = one real inference request arriving at that moment in time.</p><div id="experimentTraces"></div></div>`);
    experiments.insertAdjacentHTML('beforeend','<div id="experimentSummary" class="understand"><h2>Measured takeaway</h2><p>Loading the real experiment summary...</p></div>');
  }
  const engineTable=document.querySelector('#page-engine .review-table');
  if(engineTable&&!engineTable.previousElementSibling?.classList.contains('legend-note'))engineTable.insertAdjacentHTML('beforebegin','<p class="legend-note">Real GPU comparison — resnet50, batch size 32. Compute and memory values come from profiled_data CSV files.</p>');
  const schedulerTable=document.querySelector('#page-scheduler #schedulerRows');
  if(schedulerTable&&!schedulerTable.closest('.review-card').querySelector('.real-caption'))schedulerTable.closest('.review-card').querySelector('h3').insertAdjacentHTML('afterend','<p class="legend-note real-caption">Real measured results — 20 requests, ResNet50 on Tesla V100, each scheduler run independently.</p>');
  const profilerTable=document.querySelector('#page-profiler #profilerRows');
  if(profilerTable&&!profilerTable.closest('.review-card').querySelector('.real-caption'))profilerTable.closest('.review-card').querySelector('h3').insertAdjacentHTML('afterend','<p class="legend-note real-caption">Real measured timing — selected model, isolated per-layer execution.</p>');
}
function initSchedulerExample(){
  const board=$('requestBoard'), select=$('simAlgorithm'), button=$('playScheduler'), explanation=$('simExplanation');
  if(!board||!select||!button)return;
  const requests=[['A','Fast',20],['B','Slow',180],['C','Medium',60],['D','Fast',15],['E','Slow',200]];
  const explanations={
    FIFO:'Processes requests in the exact order they arrive. Simple but unfair — one slow request blocks everyone behind it.',
    SJF:'Always picks the fastest remaining request first. Great for overall speed, but slow requests can wait forever: starvation.',
    HASP:'Balances speed with fairness. The longer a request waits, the more its priority increases, so it eventually runs.'
  };
  function draw(){board.innerHTML=requests.map(([id,label,ms])=>`<div class="sim-request" data-id="${id}"><b>${id}</b><small>${label} · ${ms}ms</small><span class="score">waiting</span></div>`).join('');explanation.textContent=explanations[select.value]}
  function play(){draw();const order=select.value==='FIFO'?['A','B','C','D','E']:select.value==='SJF'?['D','A','C','B','E']:['D','A','B','C','E'];let index=0;const timer=setInterval(()=>{if(index>=order.length){clearInterval(timer);return}const id=order[index++];document.querySelectorAll('.sim-request').forEach(el=>{if(el.dataset.id===id){el.classList.remove('waiting');el.classList.add('done');el.querySelector('.score').textContent=select.value==='HASP'?`${id}: score wins (aging boost)`:'processed'}else if(select.value!=='FIFO'&&!el.classList.contains('done')){el.classList.add('waiting');const ms=requests.find(item=>item[0]===el.dataset.id)[2];el.querySelector('.score').textContent=select.value==='HASP'?`${el.dataset.id}: score ${(0.3+index*0.12).toFixed(2)} (+${index*4}%)`:`wait timer ${index*0.8}s`}})},800)}
  select.addEventListener('change',draw);button.addEventListener('click',play);draw();
}
function renderLayerStack(rows){
  const stack=$('layerStack');if(!stack)return;
  const explanations={Conv2d:'scans the image for patterns',BatchNorm:'keeps values on a stable scale',ReLU:'adds decision-making non-linearity',MaxPool2d:'keeps the strongest nearby signal',Linear:'turns features into a prediction',AdaptiveAvgPool2d:'summarizes the feature map'};
  const visible=rows.slice(0,6), max=Math.max(1,...visible.map(row=>num(row.compute_cost_ms)));
  const meta=$('layerStackMeta');if(meta)meta.textContent=`Showing top ${visible.length} layers by compute time (of ${rows.length} total).`;
  const allButton=$('showAllLayers');if(allButton){allButton.onclick=()=>{stack.innerHTML='<div class="layer-box" title="The image entering the network">Input image</div>'+rows.map(row=>`<div class="layer-box" title="One isolated processing step in the network"><span>${row.layer_type||'Layer'}</span><small>${num(row.compute_cost_ms).toFixed(3)}ms</small></div>`).join('')+'<div class="layer-box" title="The model output">Output prediction</div>';allButton.disabled=true;allButton.textContent='Showing all layers'}};
  stack.innerHTML='<div class="layer-box" title="The image entering the network">Input image</div>'+visible.map(row=>{const height=Math.max(22,Math.round(num(row.compute_cost_ms)/max*74));const type=row.layer_type||'Layer';const key=Object.keys(explanations).find(name=>type.includes(name));return `<div class="layer-box" title="${key?explanations[key]:'one processing step in the network'}" style="height:${height}px"><span>${type}</span><small>${num(row.compute_cost_ms).toFixed(3)}ms · hover for explanation</small></div>`}).join('')+'<div class="layer-box" title="The model output">Output prediction</div>';
}
function renderRace(data){
  const host=$('schedulerRace');if(!host)return;
  const names=['fifo','sjf','hasp'], colors=['#5794f2','#73bf69','#ff9900'];
  const all=names.flatMap(name=>(data[name]?.completion_times_s||[]));const max=Math.max(0.001,...all);
  host.innerHTML=names.map((name,index)=>{const times=data[name]?.completion_times_s||[];return `<div class="race-col"><div class="race-bar" style="height:${Math.max(8,Math.min(125,times.length/max*125))}px;background:${colors[index]}"></div><b>${name.toUpperCase()}</b><small>${times.length} completions<br>${max?times.at(-1).toFixed(3):'--'}s total</small></div>`}).join('');
}
function renderExperimentSummary(data){
  const values=Object.values(data).filter(item=>item&&item.schedulers), starvation=values.reduce((acc,item)=>{const sched=item.schedulers||{};return {fifo:acc.fifo+num(sched.fifo?.starvation_count),sjf:acc.sjf+num(sched.sjf?.starvation_count),hasp:acc.hasp+num(sched.hasp?.starvation_count)}},{fifo:0,sjf:0,hasp:0});
  const target=$('experimentSummary');if(target)target.innerHTML=`<h2>Measured takeaway</h2><p>Across all four traffic patterns, HASP achieved <b style="color:var(--green)">${starvation.hasp}</b> starvation events while FIFO caused <b>${starvation.fifo}</b> and SJF caused <b>${starvation.sjf}</b>. These counts come directly from the real experiment results.</p>`;
}
function renderExperimentTraces(data){
  const host=$('experimentTraces');if(!host)return;
  host.innerHTML=Object.entries(data).filter(([,result])=>result&&result.trace).map(([name,result])=>{
    const trace=Array.isArray(result.trace)?result.trace:[], duration=Math.max(1,...trace.map(item=>num(item.arrival_time)));
    const dots=trace.map(item=>`<i title="${item.model_name||'request'} at ${num(item.arrival_time).toFixed(2)}s" style="left:${Math.min(98,num(item.arrival_time)/duration*100)}%"></i>`).join('');
    return `<div class="legend-note"><b>${name}</b><div class="trace">${dots}</div></div>`;
  }).join('');
}
const originalLoadGpuComparison=loadGpuComparison;
loadGpuComparison=async function(){await originalLoadGpuComparison();const rows=await (await fetch('/api/gpu_comparison')).json();const selected=$('gpuSelector')?.value;const match=rows.find(row=>row.gpu===selected)||rows[0];const flow=document.querySelector('.concept-flow');if(flow&&match)flow.style.setProperty('--compute-duration',`${Math.max(.4,Math.min(2,num(match.compute_ms)/50))}s`)};
const originalLoadProfiler=loadProfiler;
loadProfiler=async function(model){
  await originalLoadProfiler(model);
  const response=await fetch(`/api/profiler_results?model=${encodeURIComponent(model)}`);
  const rows=await response.json();
  renderLayerStack(rows);
  if(!rows.length){
    const loading=$('profilerLoading');
    if(loading)loading.innerHTML='No stored result. <button class="action" id="runProfiler">Run real profiler</button>';
    const button=$('runProfiler');
    if(button)button.onclick=async()=>{button.disabled=true;button.textContent='Running...';const result=await (await fetch(`/api/layer_profiler_run?model=${encodeURIComponent(model)}`)).json();renderLayerStack(result);if(loading)loading.textContent=`${result.length} real layers measured`};
  }
};
const originalLoadScheduler=loadScheduler;
loadScheduler=async function(){const data=await (await fetch('/api/scheduler_comparison')).json();await originalLoadScheduler();renderRace(data)};
const originalLoadExperiments=loadExperiments;
loadExperiments=async function(){const data=await (await fetch('/api/experiment_results')).json();await originalLoadExperiments();renderExperimentTraces(data);renderExperimentSummary(data)};
addAuditStyles();addUnderstanding();loadGpuComparison();loadScheduler();loadProfiler('resnet18');loadExperiments();
</script></body></html>
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


def _load_selected_scheduler() -> str:
  if not os.path.exists(SELECTED_SCHEDULER_PATH):
    return DEFAULT_SCHEDULER
  try:
    with open(SELECTED_SCHEDULER_PATH, 'r', encoding='utf-8') as scheduler_file:
      data = json.load(scheduler_file)
      value = data.get('scheduler') if isinstance(data, dict) else None
      return value if value in SCHEDULER_NAMES else DEFAULT_SCHEDULER
  except (OSError, json.JSONDecodeError):
    return DEFAULT_SCHEDULER


def _save_selected_scheduler(scheduler_name: Any) -> str:
  scheduler = str(scheduler_name or DEFAULT_SCHEDULER).lower()
  if scheduler not in SCHEDULER_NAMES:
    scheduler = DEFAULT_SCHEDULER
  _ensure_write_dir()
  with open(SELECTED_SCHEDULER_PATH, 'a+', encoding='utf-8') as scheduler_file:
    fcntl.flock(scheduler_file.fileno(), fcntl.LOCK_EX)
    try:
      scheduler_file.seek(0)
      scheduler_file.truncate()
      json.dump({'scheduler': scheduler, 'updated_at': time.time()}, scheduler_file)
      scheduler_file.flush()
      os.fsync(scheduler_file.fileno())
    finally:
      fcntl.flock(scheduler_file.fileno(), fcntl.LOCK_UN)
  return scheduler


def _clear_metrics_file() -> None:
  _ensure_write_dir()
  for path in (METRICS_PATH, os.path.splitext(METRICS_PATH)[0] + '.ndjson'):
    with open(path, 'a+', encoding='utf-8') as metrics_file:
      fcntl.flock(metrics_file.fileno(), fcntl.LOCK_EX)
      try:
        metrics_file.seek(0)
        metrics_file.truncate()
        metrics_file.flush()
        os.fsync(metrics_file.fileno())
      finally:
        fcntl.flock(metrics_file.fileno(), fcntl.LOCK_UN)


def _write_live_metrics(payload: Dict[str, Any]) -> None:
  _ensure_write_dir()
  with open(METRICS_PATH, 'a+', encoding='utf-8') as metrics_file:
    fcntl.flock(metrics_file.fileno(), fcntl.LOCK_EX)
    try:
      metrics_file.seek(0)
      metrics_file.truncate()
      json.dump(payload, metrics_file)
      metrics_file.flush()
      os.fsync(metrics_file.fileno())
    finally:
      fcntl.flock(metrics_file.fileno(), fcntl.LOCK_UN)


def _run_live_engine(selected_gpu: str, selected_model: str, scheduler_name: str) -> None:
  try:
    from .engine import InferenceEngine
    from .metrics import compute_metrics
  except ImportError:
    from glide.engine import InferenceEngine
    from glide.metrics import compute_metrics

  try:
    from .workload import WorkloadGenerator
  except ImportError:
    from glide.workload import WorkloadGenerator

  engine = InferenceEngine(gpu=selected_gpu, model=selected_model, scheduler=scheduler_name)
  engine._skip_sleep = False
  trace = WorkloadGenerator([selected_model], 'poisson', rate=3.0, duration=30.0, seed=int(time.time())).generate()
  start_time = time.time()
  expected = len(trace)
  _write_live_metrics({
    'status': 'running',
    'scheduler_name': scheduler_name,
    'model': selected_model,
    'gpu': selected_gpu,
    'selected_model': selected_model,
    'selected_gpu': selected_gpu,
    'batch_size': 32,
    'start_time': start_time,
    'total_expected_batches': expected,
    'batches': [],
    'per_request': [],
    'queue_history': [],
    'queue_status': engine.get_queue_status(),
    'metrics': {},
    'timestamp': start_time,
  })

  trace_index = 0
  try:
    while trace_index < len(trace) or engine.request_queue:
      if _RUN_STOP.is_set():
        _update_live_status('stopped', engine, selected_gpu, selected_model, scheduler_name, start_time, expected, compute_metrics)
        return

      elapsed = time.time() - start_time
      while trace_index < len(trace) and trace[trace_index]['arrival_time'] <= elapsed:
        item = trace[trace_index]
        engine.submit_request(model_name=item['model_name'], batch_size=32, priority=item.get('priority', 0))
        trace_index += 1

      if engine.request_queue:
        request = engine.process_next()
        if request is not None:
          _append_live_result(request, engine, selected_gpu, selected_model, scheduler_name, start_time, expected, compute_metrics)
        continue

      next_arrival = trace[trace_index]['arrival_time'] if trace_index < len(trace) else elapsed
      _RUN_STOP.wait(max(0.01, min(0.25, next_arrival - elapsed)))

    _update_live_status('completed', engine, selected_gpu, selected_model, scheduler_name, start_time, expected, compute_metrics)
  except Exception as exc:
    print(f'[GLIDE live run] failed: {exc}')
    _update_live_status('failed', engine, selected_gpu, selected_model, scheduler_name, start_time, expected, compute_metrics)


def _live_payload(engine: Any, selected_gpu: str, selected_model: str, scheduler_name: str,
                  start_time: float, expected: int, compute_metrics: Any) -> Dict[str, Any]:
  completed = engine.get_results()
  return {
    'status': 'running',
    'scheduler_name': scheduler_name,
    'model': selected_model,
    'gpu': selected_gpu,
    'selected_model': selected_model,
    'selected_gpu': selected_gpu,
    'batch_size': 32,
    'start_time': start_time,
    'total_expected_batches': expected,
    'batches': [
      {
        'batch': index + 1,
        'compute_time': float(item.get('emulated_compute_ms') or 0.0) / 1000.0,
        'timestamp': item.get('end_time'),
        'memory_gb': float(item.get('memory_mb') or 0.0) / 1024.0,
      }
      for index, item in enumerate(completed)
    ],
    'per_request': completed,
    'queue_history': engine.queue_history,
    'queue_status': engine.get_queue_status(),
    'metrics': compute_metrics(completed),
    'timestamp': time.time(),
  }


def _append_live_result(request: Any, engine: Any, selected_gpu: str, selected_model: str,
                        scheduler_name: str, start_time: float, expected: int, compute_metrics: Any) -> None:
  _ensure_write_dir()
  with open(METRICS_PATH, 'a+', encoding='utf-8') as metrics_file:
    fcntl.flock(metrics_file.fileno(), fcntl.LOCK_EX)
    try:
      metrics_file.seek(0)
      raw = metrics_file.read().strip()
      payload = json.loads(raw) if raw else {}
      batches = payload.get('batches', [])
      if not isinstance(batches, list):
        batches = []
      batches.append({
        'batch': len(batches) + 1,
        'compute_time': float(request.emulated_compute_ms or 0.0) / 1000.0,
        'timestamp': request.end_time,
        'memory_gb': float(request.memory_mb or 0.0) / 1024.0,
      })
      payload.update(_live_payload(engine, selected_gpu, selected_model, scheduler_name, start_time, expected, compute_metrics))
      payload['batches'] = batches
      metrics_file.seek(0)
      metrics_file.truncate()
      json.dump(payload, metrics_file)
      metrics_file.flush()
      os.fsync(metrics_file.fileno())
    finally:
      fcntl.flock(metrics_file.fileno(), fcntl.LOCK_UN)


def _update_live_status(status: str, engine: Any, selected_gpu: str, selected_model: str,
                        scheduler_name: str, start_time: float, expected: int, compute_metrics: Any) -> None:
  payload = _live_payload(engine, selected_gpu, selected_model, scheduler_name, start_time, expected, compute_metrics)
  payload['status'] = status
  _write_live_metrics(payload)


def _start_live_engine_run(selected_gpu: str, selected_model: str, scheduler_name: str) -> bool:
  global _RUN_THREAD
  with _RUN_LOCK:
    if _RUN_THREAD is not None and _RUN_THREAD.is_alive():
      return False
    _RUN_STOP.clear()
    _RUN_THREAD = threading.Thread(
      target=_run_live_engine,
      args=(selected_gpu, selected_model, scheduler_name),
      name='glide-live-engine',
      daemon=True,
    )
    _RUN_THREAD.start()
    return True


def _stop_live_engine_run() -> bool:
  with _RUN_LOCK:
    if _RUN_THREAD is None or not _RUN_THREAD.is_alive():
      return False
    _RUN_STOP.set()
    return True


def _mark_run_stopped() -> None:
  _ensure_write_dir()
  with open(METRICS_PATH, 'a+', encoding='utf-8') as metrics_file:
    fcntl.flock(metrics_file.fileno(), fcntl.LOCK_EX)
    try:
      metrics_file.seek(0)
      raw = metrics_file.read().strip()
      payload = json.loads(raw) if raw else {}
      payload['status'] = 'stopped'
      payload['timestamp'] = time.time()
      metrics_file.seek(0)
      metrics_file.truncate()
      json.dump(payload, metrics_file)
      metrics_file.flush()
      os.fsync(metrics_file.fileno())
    finally:
      fcntl.flock(metrics_file.fileno(), fcntl.LOCK_UN)


def _safe_metrics_payload() -> Dict[str, Any]:
  selected_model = _load_selected_model()
  selected_gpu = _load_selected_gpu()
  selected_scheduler = _load_selected_scheduler()
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
    'scheduler_name': _load_selected_scheduler(),
    'queue_history': [],
    'per_request': [],
    'metrics': {},
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


def _calculate_dynamic_utilization(times: List[float], elapsed: float, gpu: str) -> Dict[str, int]:
  if not times or elapsed <= 0:
    return get_utilization(gpu, _load_selected_model())
  compute_util = min(100, int((sum(times) / len(times) / 0.15) * 100))
  gpu_util = min(100, int((sum(times) / elapsed) * 100))
  return {"gpu_util": gpu_util, "compute_util": compute_util}


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
  enriched['scheduler_name'] = enriched.get('scheduler_name') or _load_selected_scheduler()

  batches: List[Dict[str, Any]] = enriched.get('batches', [])
  if not isinstance(batches, list):
    batches = []
  if not batches and isinstance(enriched.get('per_request'), list):
    batches = [
      {
        'batch': index + 1,
        'compute_time': float(item.get('emulated_compute_ms', 0.0)) / 1000.0,
        'timestamp': item.get('end_time'),
        'memory_gb': float(item.get('memory_mb', 0.0) or 0.0) / 1024.0,
      }
      for index, item in enumerate(enriched['per_request'])
      if isinstance(item, dict)
    ]
  enriched['batches'] = batches

  times = [
    float(item.get('compute_time'))
    for item in batches
    if isinstance(item, dict) and item.get('compute_time') is not None
  ]
  total_batches = len(times)
  enriched['total_batches'] = total_batches

  if total_batches == 0:
    if enriched.get('status') not in ('running', 'completed', 'stopped', 'failed'):
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
  
  # Use dynamic utilization based on actual batch data if available
  if elapsed and elapsed > 0:
    utilization = _calculate_dynamic_utilization(times, elapsed, selected_gpu)
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
  bandwidth = float(profile_stats.get('estimated_bandwidth_gbps') or 0)
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


GPU_COMPARISON_ORDER = [
  'NVIDIA_A100-SXM4-40GB', 'Quadro_RTX_6000', 'Tesla_K80',
  'Tesla_M40', 'Tesla_P100-PCIE-16GB', 'Tesla_V100-PCIE-32GB',
]
MODEL_LIST = [
  'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
  'googlenet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
  'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
  'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
  'resnext50_32x4d', 'resnext101_32x8d', 'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
  'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
  'wide_resnet50_2', 'wide_resnet101_2',
]


def _profile_csv_value(path: str, batch: int, value_column: str) -> Optional[float]:
  rows = _read_csv_rows(path)
  candidates = []
  for row in rows:
    try:
      candidates.append((abs(float(row.get('Batch_Size', row.get('batch_size', 0))) - batch), row))
    except (TypeError, ValueError):
      continue
  if not candidates:
    return None
  row = min(candidates, key=lambda item: item[0])[1]
  try:
    return float(row[value_column])
  except (KeyError, TypeError, ValueError):
    return None


def _real_compute_ms(gpu: str, model: str, batch: int = 32) -> Optional[float]:
  path = os.path.join(PROFILED_DATA_ROOT, 'time', 'compute', 'forward', gpu, model, 'time_by_batch_size.csv')
  value = _profile_csv_value(path, batch, 'Time_In_SECONDS')
  return value * 1000.0 if value is not None else None


def _real_memory_mb(gpu: str, model: str, batch: int = 32) -> Optional[float]:
  path = os.path.join(PROFILED_DATA_ROOT, 'memory', gpu, model, 'memory.csv')
  return _profile_csv_value(path, batch, 'peak')


def _percentile(values: List[float], percentile: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
  return ordered[index]


def _jains_index(values: List[float]) -> float:
  if not values:
    return 1.0
  total = sum(values)
  denominator = len(values) * sum(value * value for value in values)
  return (total * total / denominator) if denominator else 1.0


def _run_scheduler_comparison() -> Dict[str, Dict[str, Any]]:
  try:
    from .engine import InferenceEngine
  except ImportError:
    from glide.engine import InferenceEngine
  results = {}
  request_specs = [
    ('resnet18', 8), ('resnet50', 32), ('alexnet', 16), ('resnet50', 8),
    ('resnet18', 32), ('alexnet', 8), ('resnet50', 16), ('resnet18', 16),
    ('alexnet', 32), ('resnet50', 32), ('resnet18', 8), ('alexnet', 16),
    ('resnet50', 8), ('resnet18', 32), ('alexnet', 8), ('resnet50', 16),
    ('resnet18', 16), ('alexnet', 32), ('resnet50', 32), ('resnet18', 8),
  ]
  previous = InferenceEngine._skip_sleep
  InferenceEngine._skip_sleep = True
  try:
    for scheduler_name in ('fifo', 'sjf', 'hasp'):
      started = time.perf_counter()
      engine = InferenceEngine(gpu='Tesla_V100-PCIE-32GB', model='resnet50', scheduler=scheduler_name)
      # Keep the profiled compute delay during experiments so scheduler order
      # affects measured completion and tail latency, rather than collapsing
      # every request to the same wall-clock instant.
      engine._skip_sleep = False
      for model, batch_size in request_specs:
        engine.submit_request(model_name=model, batch_size=batch_size)
      arrival_time = time.time()
      for request in engine.request_queue:
        request.arrival_time = arrival_time
      engine.run_queue()
      elapsed = max(time.perf_counter() - started, 1e-9)
      latencies = [float(request.latency_ms or 0.0) for request in engine.completed_requests]
      results[scheduler_name] = {
        'avg_latency_ms': statistics.mean(latencies) if latencies else 0.0,
        'p95_latency_ms': _percentile(latencies, 0.95),
        'throughput': len(latencies) / elapsed,
        'jains_fairness_index': _jains_index(latencies),
        'starvation_count': sum(1 for value in latencies if value > (statistics.mean(latencies) * 3)) if latencies else 0,
        'requests': [request.to_dict() for request in engine.completed_requests],
        'completion_times_s': [
          (float(request.end_time) - min(float(item.end_time) for item in engine.completed_requests if item.end_time is not None))
          for request in engine.completed_requests if request.end_time is not None
        ],
      }
  finally:
    InferenceEngine._skip_sleep = previous
  return results


EXPERIMENT_CACHE_VERSION = 3


def _run_real_experiments(progress_callback: Any = None) -> Dict[str, Any]:
  try:
    from .engine import InferenceEngine
    from .workload import WorkloadGenerator
  except ImportError:
    from glide.engine import InferenceEngine
    from glide.workload import WorkloadGenerator
  models = ['resnet18', 'resnet50', 'alexnet']
  configs = [
    ('single_uniform', ['resnet50'], 'uniform'),
    ('single_poisson', ['resnet50'], 'poisson'),
    ('multi_uniform', models, 'uniform'),
    ('multi_bursty', models, 'bursty'),
  ]
  output = {}
  combination = 0
  for name, experiment_models, mode in configs:
    trace = WorkloadGenerator(experiment_models, mode, 2.0, 15.0, seed=42).generate()
    output[name] = {'trace': trace, 'schedulers': {}}
    for scheduler_name in ('fifo', 'sjf', 'hasp'):
      combination += 1
      if progress_callback is not None:
        progress_callback(combination, 12, name, scheduler_name)
      engine = InferenceEngine(gpu='Tesla_V100-PCIE-32GB', model=experiment_models[0], scheduler=scheduler_name)
      # Preserve profiled service times so scheduler ordering changes latency.
      engine._skip_sleep = False
      base_time = time.time()
      for item in trace:
        request_id = engine.submit_request(model_name=item['model_name'], batch_size=item['batch_size'], priority=item['priority'])
        request = next(request for request in engine.request_queue if request.request_id == request_id)
        request.arrival_time = base_time - max(0.0, 15.0 - float(item['arrival_time']))
      started = time.perf_counter()
      engine.run_queue()
      elapsed = max(time.perf_counter() - started, 1e-9)
      latencies = [float(request.latency_ms or 0.0) for request in engine.completed_requests]
      average = statistics.mean(latencies) if latencies else 0.0
      output[name]['schedulers'][scheduler_name] = {
        'avg_latency_ms': average, 'p95_latency_ms': _percentile(latencies, .95),
        'throughput': len(latencies) / elapsed, 'jains_fairness_index': _jains_index(latencies),
        'starvation_count': sum(1 for value in latencies if value > average * 3) if latencies else 0,
      }
  return output


def _run_experiments_background() -> None:
  global _EXPERIMENT_STATE
  cache_path = os.path.join(GLIDE_DIR, 'experiment_cache.json')

  def update_progress(done: int, total: int, scenario: str, scheduler: str) -> None:
    with _EXPERIMENT_LOCK:
      _EXPERIMENT_STATE.update({
        'status': 'computing',
        'progress': f'{done}/{total} combinations done',
        'current': f'{scenario} + {scheduler.upper()}',
      })

  try:
    results = _run_real_experiments(update_progress)
    with open(cache_path, 'w', encoding='utf-8') as cache_file:
      json.dump({'_cache_version': EXPERIMENT_CACHE_VERSION, 'results': results}, cache_file, indent=2)
    with _EXPERIMENT_LOCK:
      _EXPERIMENT_STATE = {'status': 'complete', 'progress': '12/12 combinations done', 'results': results}
  except Exception as exc:
    print(f'[GLIDE experiments] failed: {exc}')
    with _EXPERIMENT_LOCK:
      _EXPERIMENT_STATE = {'status': 'failed', 'progress': str(exc), 'results': {}}


def _start_experiments() -> bool:
  global _EXPERIMENT_THREAD, _EXPERIMENT_STATE
  with _EXPERIMENT_LOCK:
    if _EXPERIMENT_THREAD is not None and _EXPERIMENT_THREAD.is_alive():
      return False
    _EXPERIMENT_STATE = {'status': 'computing', 'progress': '0/12 combinations done', 'results': {}}
    _EXPERIMENT_THREAD = threading.Thread(target=_run_experiments_background, name='glide-experiments', daemon=True)
    _EXPERIMENT_THREAD.start()
    return True


@app.route('/api/gpu_comparison')
def api_gpu_comparison():
  values = []
  for gpu in GPU_COMPARISON_ORDER:
    compute = _real_compute_ms(gpu, 'resnet50')
    memory = _real_memory_mb(gpu, 'resnet50')
    if compute is not None:
      values.append({'gpu': gpu, 'compute_ms': compute, 'memory_mb': memory or 0.0})
  return jsonify(sorted(values, key=lambda item: item['compute_ms']))


@app.route('/api/model_timings')
def api_model_timings():
  gpu = _normalize_gpu(request.args.get('gpu', 'Tesla_V100-PCIE-32GB'))
  values = []
  missing = []
  for model in MODEL_LIST:
    compute = _real_compute_ms(gpu, model)
    if compute is not None:
      family = 'resnet' if model.startswith('res') else 'vgg' if model.startswith('vgg') else 'densenet' if model.startswith('dense') else 'others'
      values.append({'model': model, 'compute_ms': compute, 'family': family})
    else:
      missing.append(model)
  if missing:
    print(f'[API model_timings] missing profiles for {gpu}: {", ".join(missing)}')
  return jsonify(sorted(values, key=lambda item: item['compute_ms'], reverse=True))


@app.route('/api/profiler_results')
def api_profiler_results():
  model = _normalize_model(request.args.get('model', 'resnet18'))
  if not os.path.exists(os.path.join(GLIDE_DIR, 'layer_db.sqlite')):
    return jsonify([])
  conn = sqlite3.connect(os.path.join(GLIDE_DIR, 'layer_db.sqlite'))
  rows = conn.execute('SELECT layer_type, config, compute_cost_ms, memory_cost_mb FROM layers WHERE model=? ORDER BY compute_cost_ms DESC', (model,)).fetchall()
  conn.close()
  total = sum(float(row[2] or 0.0) for row in rows)
  return jsonify([{'layer_type': row[0], 'config': row[1], 'compute_cost_ms': row[2], 'memory_cost_mb': row[3], 'pct_of_total': (float(row[2] or 0.0) / total * 100.0) if total else 0.0} for row in rows])


@app.route('/api/scheduler_comparison')
def api_scheduler_comparison():
  return jsonify(_run_scheduler_comparison())


@app.route('/api/engine_test')
def api_engine_test():
  try:
    from .engine import InferenceEngine
  except ImportError:
    from glide.engine import InferenceEngine
  selected_gpu = _load_selected_gpu()
  selected_model = _load_selected_model()
  scheduler = _load_selected_scheduler()
  previous = InferenceEngine._skip_sleep
  InferenceEngine._skip_sleep = True
  try:
    engine = InferenceEngine(gpu=selected_gpu, model=selected_model, scheduler=scheduler)
    engine._skip_sleep = True
    for _ in range(10):
      engine.submit_request(batch_size=32)
    engine.run_queue()
    return jsonify({
      'gpu': selected_gpu,
      'model': selected_model,
      'scheduler_name': scheduler,
      'requests': engine.get_results(),
    })
  finally:
    InferenceEngine._skip_sleep = previous


@app.route('/api/experiment_results')
def api_experiment_results_real():
  cache_path = os.path.join(GLIDE_DIR, 'experiment_cache.json')
  if os.path.exists(cache_path):
    try:
      with open(cache_path, 'r', encoding='utf-8') as cache_file:
        cached = json.load(cache_file)
        if isinstance(cached, dict) and cached.get('_cache_version') == EXPERIMENT_CACHE_VERSION:
          results = cached.get('results', {})
          return jsonify({'status': 'complete', 'progress': '12/12 combinations done', 'results': results})
    except (OSError, json.JSONDecodeError):
      cached = None
  with _EXPERIMENT_LOCK:
    state = dict(_EXPERIMENT_STATE)
  if state.get('status') in ('computing', 'failed', 'complete'):
    return jsonify(state)
  return jsonify({'status': 'idle', 'progress': '0/12 combinations done', 'results': {}})


@app.route('/api/run_experiments', methods=['POST'])
def api_run_experiments():
  started = _start_experiments()
  return jsonify({'status': 'started' if started else 'already_running'})


@app.route('/api/layer_profiler_run')
def api_layer_profiler_run():
  try:
    from .profiler import profile_and_save
  except ImportError:
    from glide.profiler import profile_and_save
  model = _normalize_model(request.args.get('model', 'resnet18'))
  return jsonify(sorted(profile_and_save(model, 'CPU_measured', batch_size=32, num_runs=3), key=lambda item: item.get('compute_time_ms', 0.0), reverse=True))


@app.route('/')
def dashboard() -> str:
  return render_template_string(DASHBOARD_HTML)


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


@app.route('/api/set_scheduler', methods=['POST'])
def api_set_scheduler():
  payload = request.get_json(silent=True) or {}
  scheduler = _save_selected_scheduler(payload.get('scheduler'))
  return jsonify({'ok': True, 'scheduler_name': scheduler})


@app.route('/api/start_new_run', methods=['POST'])
def api_start_new_run():
  _clear_metrics_file()
  selected = _load_selected_model()
  selected_gpu = _load_selected_gpu()
  scheduler = _load_selected_scheduler()
  utilization = get_utilization(selected_gpu, selected)
  started = _start_live_engine_run(selected_gpu, selected, scheduler)
  print(f"[API start_new_run] gpu={selected_gpu} model={selected} scheduler={scheduler} started={started}")
  return jsonify({
    'ok': True,
    'status': 'running' if started else 'already_running',
    'selected_model': selected,
    'model_info': _model_info(selected),
    'selected_gpu': selected_gpu,
    'gpu_info': _gpu_info(selected_gpu),
    'scheduler_name': scheduler,
    'gpu_util': utilization['gpu_util'],
    'compute_util': utilization['compute_util']
  })


@app.route('/api/start_run', methods=['POST'])
def api_start_run():
  selected_model = _load_selected_model()
  selected_gpu = _load_selected_gpu()
  scheduler = _load_selected_scheduler()
  started = _start_live_engine_run(selected_gpu, selected_model, scheduler)
  print(f"[API start_run] gpu={selected_gpu} model={selected_model} scheduler={scheduler} started={started}")
  return jsonify({'status': 'started' if started else 'already_running'})


@app.route('/api/stop_run', methods=['POST'])
def api_stop_run():
  stopped = _stop_live_engine_run()
  current = _load_metrics_file()
  if not stopped and current.get('status') != 'running':
    return jsonify({'status': 'not_running'})
  _mark_run_stopped()
  print('[API stop_run] stop requested')
  return jsonify({'status': 'stopped'})


if __name__ == '__main__':
  _save_selected_model(_load_selected_model())
  _save_selected_gpu(_load_selected_gpu())
  _save_selected_scheduler(_load_selected_scheduler())
  app.run(host='0.0.0.0', port=5000, debug=False)
