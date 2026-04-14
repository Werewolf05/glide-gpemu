import fcntl
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
  os.makedirs(GLIDE_DIR, exist_ok=True)

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
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <div class="title" id="pageTitle">GLIDE GPU TASK SCHEDULER SIMULATOR • RESNET18</div>
        <div class="subtitle" id="meta">Awaiting run metadata...</div>
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
            <div class="input-label">Scheduling Algorithm</div>
            <select id="algoSelector">
              <option value="FIFO">FIFO</option>
              <option value="Round Robin">Round Robin</option>
              <option value="SJF">SJF</option>
              <option value="HASP" disabled>HASP (coming soon)</option>
            </select>
          </div>
          <div class="slider-wrap">
            <div class="input-label">Simulation Speed</div>
            <input id="speedSlider" type="range" min="0" max="2" step="1" value="1" />
            <div class="speed-meta"><span>Slow</span><span id="speedLabel">Normal (1.0s)</span><span>Fast</span></div>
          </div>
          <button class="btn" id="startRunBtn">Start New Run</button>
        </div>
        <div class="hint">Select model, GPU, algorithm, and speed. For Review 1, all algorithms execute as FIFO behavior while showing the selected policy name.</div>
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
        <div class="model-line" id="algoActive"><strong>Scheduler:</strong> FIFO</div>
      </div>
    </div>

    <div class="stats-bar">
      <div class="stat-chip">
        <div class="stat-label">GPU Utilization <span class="src-badge derived">DERIVED</span></div>
        <div class="stat-value" id="utilTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Avg Wait Time <span class="src-badge derived">DERIVED</span></div>
        <div class="stat-value" id="waitTop">--</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Tasks Completed <span class="src-badge measured">MEASURED</span></div>
        <div class="stat-value" id="doneTop">0</div>
      </div>
      <div class="stat-chip">
        <div class="stat-label">Current Time <span class="src-badge measured">MEASURED</span></div>
        <div class="stat-value" id="timeTop">--</div>
      </div>
    </div>

    <div class="now-strip">
      <div class="now-main" id="nowExec">Now Executing: Idle</div>
      <div class="now-meta" id="nowExecMeta">State: waiting | Source: profiled timing + measured runtime events</div>
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
            <span class="fast">Fast</span>
            <span class="medium">Medium</span>
            <span class="slow">Slow</span>
            <span class="queued">Queued / Processing / Done shown in Task Queue</span>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">GPU Resource Usage Panel</div>
        <div class="metric-block">
          <div class="metric-header"><span>Compute Utilization <span class="src-badge derived">DERIVED</span></span><span id="utilValue">--</span></div>
          <div class="meter"><span id="utilBar"></span></div>
        </div>
        <div class="metric-block">
          <div class="metric-header"><span>VRAM Usage <span class="src-badge derived">DERIVED</span></span><span id="vramValue">--</span></div>
          <div class="meter"><span id="vramBar"></span></div>
        </div>
        <div class="metric-block">
          <div class="metric-header"><span>Memory Bandwidth <span class="src-badge derived">DERIVED</span></span><span id="bwValue">--</span></div>
          <div class="meter"><span id="bwBar"></span></div>
        </div>
        <div class="metric-block">
          <div class="metric-header"><span>Active Warps <span class="src-badge derived">DERIVED</span></span><span id="warpValue">--</span></div>
          <div class="meter"><span id="warpBar"></span></div>
        </div>
        <div class="empty" id="resourceHint">Metrics are derived from live batch timing and selected profile for demo visualization.</div>
        <div class="panel-title" style="margin-top:10px;">Scheduler Decision Log <span class="src-badge measured">EVENTS</span></div>
        <div class="log-box" id="schedulerLog"></div>
      </div>
    </div>
  </div>

  <script>
    const fmt = (v, suffix = 's') => (Number.isFinite(v) ? `${v.toFixed(4)} ${suffix}` : '--');
    let refreshMs = 1000;
    let refreshTimer = null;
    let schedulerName = 'FIFO';
    let schedulerLog = [];
    let queueDepthHistory = [];
    let lastLoggedBatch = -1;

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

    function getProfileFactors(model, gpu) {
      let computeScale = 1.0;
      let utilBias = 1.0;
      let vramScale = 1.0;
      let bwScale = 1.0;

      const m = (model || '').toLowerCase();
      const g = (gpu || '').toUpperCase();

      if (m.includes('resnet18') || m.includes('mobilenet') || m.includes('mnasnet') || m.includes('squeezenet')) {
        computeScale *= 0.78;
        utilBias *= 0.85;
        vramScale *= 0.72;
      } else if (m.includes('resnet50') || m.includes('densenet121') || m.includes('shufflenet')) {
        computeScale *= 0.98;
        utilBias *= 1.0;
      } else if (m.includes('vgg') || m.includes('resnet152') || m.includes('wide_resnet') || m.includes('densenet201')) {
        computeScale *= 1.22;
        utilBias *= 1.18;
        vramScale *= 1.28;
        bwScale *= 1.15;
      }

      if (g.includes('A100')) {
        computeScale *= 0.62;
        utilBias *= 1.2;
        bwScale *= 1.38;
        vramScale *= 1.35;
      } else if (g.includes('V100')) {
        computeScale *= 0.74;
        utilBias *= 1.12;
        bwScale *= 1.22;
      } else if (g.includes('P100')) {
        computeScale *= 0.84;
        utilBias *= 1.05;
        bwScale *= 1.1;
      } else if (g.includes('RTX_6000')) {
        computeScale *= 0.92;
        utilBias *= 1.0;
      } else if (g.includes('M40') || g.includes('K80')) {
        computeScale *= 1.12;
        utilBias *= 0.82;
        bwScale *= 0.78;
      }

      return { computeScale, utilBias, vramScale, bwScale };
    }

    function adjustedTimes(metrics) {
      const { selectedModel, selectedGpu } = getCurrentSelection(metrics);
      const factors = getProfileFactors(selectedModel, selectedGpu);
      const batches = Array.isArray(metrics.batches) ? metrics.batches : [];
      return {
        factors,
        selectedModel,
        selectedGpu,
        adjusted: batches.map((item) => ({
          batch: Number(item.batch),
          compute_time: Math.max(0.004, (Number(item.compute_time) || 0) * factors.computeScale)
        }))
      };
    }

    function restartTicker() {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
      refreshTimer = setInterval(tick, refreshMs);
    }

    function pushLog(message) {
      const stamp = new Date().toLocaleTimeString();
      schedulerLog.push(`[${stamp}] ${message}`);
      if (schedulerLog.length > 22) {
        schedulerLog = schedulerLog.slice(-22);
      }
      renderLog();
    }

    function renderLog() {
      const box = document.getElementById('schedulerLog');
      if (!box) return;
      if (schedulerLog.length === 0) {
        box.innerHTML = '<div class="log-entry">No scheduling events yet.</div>';
        return;
      }
      box.innerHTML = schedulerLog.slice(-12).reverse().map((entry) => `<div class="log-entry">${entry}</div>`).join('');
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
        text.textContent = `RUNNING (${schedulerName})`;
      }
    }

    function resetDashboardState() {
      document.getElementById('utilTop').textContent = '--';
      document.getElementById('waitTop').textContent = '--';
      document.getElementById('doneTop').textContent = '0';
      document.getElementById('timeTop').textContent = '--';
      document.getElementById('meta').textContent = 'Awaiting run metadata...';
      document.getElementById('taskQueue').innerHTML = '<div class="empty">No tasks yet. Start a run to enqueue batches.</div>';
      document.getElementById('timelineCoreContainer').innerHTML = '<div class="timeline-core" id="timelineCore"><div class="core-label">Core 1</div></div>';
      document.getElementById('timelineAxis').textContent = '';
      document.getElementById('coreSummary').textContent = 'Core allocation: waiting for GPU selection';
      setMeter('utilBar', 0);
      setMeter('vramBar', 0);
      setMeter('bwBar', 0);
      setMeter('warpBar', 0);
      document.getElementById('utilValue').textContent = '--';
      document.getElementById('vramValue').textContent = '--';
      document.getElementById('bwValue').textContent = '--';
      document.getElementById('warpValue').textContent = '--';
      document.getElementById('nowExec').textContent = 'Now Executing: Idle';
      document.getElementById('nowExecMeta').textContent = 'State: waiting | Source: profiled timing + measured runtime events';
      queueDepthHistory = [];
      schedulerLog = [];
      lastLoggedBatch = -1;
      renderQueueSparkline(0);
      renderLog();
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
      const data = adjustedTimes(metrics);
      const list = data.adjusted;
      const avg = list.length > 0 ? (list.reduce((acc, it) => acc + it.compute_time, 0) / list.length) : 0;
      const minVal = list.length > 0 ? Math.min(...list.map((it) => it.compute_time)) : avg;
      const throughput = avg > 0 ? (1 / avg) * 0.85 : 0;
      const util = clamp(throughput * avg * 100 * data.factors.utilBias, 0, 100);
      const wait = Math.max(0, avg - minVal);
      const done = Number(metrics.total_batches) || 0;

      let elapsed = null;
      if (metrics.start_time) {
        const raw = (Date.now() / 1000) - Number(metrics.start_time);
        elapsed = Number.isFinite(raw) ? Math.max(0, raw) : null;
      }

      document.getElementById('utilTop').textContent = `${util.toFixed(1)}%`;
      document.getElementById('waitTop').textContent = fmt(wait);
      document.getElementById('doneTop').textContent = `${done}`;
      document.getElementById('timeTop').textContent = formatSeconds(elapsed);
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
        const remain = Math.max(0, expected - total);
        queueDepth = remain;
        const base = preview.length > 0 ? preview[preview.length - 1].batch + 1 : total;
        const queuedCount = Math.min(2, remain);
        for (let i = 0; i < queuedCount; i += 1) {
          preview.push({ batch: base + i, compute_time: null, status: 'QUEUED' });
        }
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
      nowMeta.textContent = `Time: ${Number.isFinite(compute) ? compute.toFixed(4) + 's' : '--'} | Scheduler: ${schedulerName} | Profile: ${data.selectedGpu}/${data.selectedModel}`;
    }

    function renderTimeline(metrics) {
      const coreContainer = document.getElementById('timelineCoreContainer');
      const viewport = document.getElementById('timelineViewport');
      const axis = document.getElementById('timelineAxis');
      const data = adjustedTimes(metrics);
      const batches = data.adjusted;
      const coreSummary = document.getElementById('coreSummary');

      function getGpuCoreCount(gpuName) {
        const g = (gpuName || '').toUpperCase();
        if (g.includes('A100')) return 108;
        if (g.includes('V100')) return 80;
        if (g.includes('P100')) return 56;
        if (g.includes('RTX_6000')) return 72;
        if (g.includes('M40')) return 24;
        if (g.includes('K80')) return 13;
        return 16;
      }

      const coreCount = getGpuCoreCount(data.selectedGpu);
      coreSummary.textContent = `Core allocation: ${coreCount} cores active on ${data.selectedGpu} using ${schedulerName}`;

      if (batches.length === 0) {
        coreContainer.innerHTML = '<div class="timeline-core" id="timelineCore"><div class="core-label">Core 1</div></div>';
        axis.textContent = '';
        return;
      }

      let viewBatches = batches.slice(-120).map((item) => ({
        batch: Number(item.batch),
        compute_time: Number(item.compute_time) || 0
      }));

      if (schedulerName === 'SJF') {
        viewBatches = [...viewBatches].sort((a, b) => a.compute_time - b.compute_time);
      }

      const avg = (viewBatches.reduce((acc, item) => acc + item.compute_time, 0) / viewBatches.length) || 0.0001;
      const scale = 85;
      const cores = Array.from({ length: coreCount }, () => ({ load: 0, bars: [] }));

      const emitBar = (coreIndex, batchId, duration, className, label) => {
        const left = cores[coreIndex].load * scale;
        const width = Math.max(7, duration * scale);
        cores[coreIndex].bars.push(`<div class="timeline-bar ${className}" style="left:${left}px;width:${width}px;">${label || ('B' + batchId)}</div>`);
        cores[coreIndex].load += duration;
      };

      const leastLoadedCore = () => {
        let idx = 0;
        for (let i = 1; i < cores.length; i += 1) {
          if (cores[i].load < cores[idx].load) idx = i;
        }
        return idx;
      };

      const classForDuration = (duration) => {
        if (duration > avg * 1.2) return 'slow';
        if (duration > avg * 0.9) return 'medium';
        return 'fast';
      };

      if (schedulerName === 'Round Robin') {
        const quantum = Math.max(0.018, avg * 0.42);
        const rr = viewBatches.map((it) => ({ ...it, remaining: it.compute_time }));
        let guard = 0;
        let nextCore = 0;
        while (rr.some((it) => it.remaining > 0.0005) && guard < 4000) {
          for (const item of rr) {
            if (item.remaining <= 0.0005) continue;
            const slice = Math.min(item.remaining, quantum);
            const cls = classForDuration(item.compute_time);
            emitBar(nextCore, item.batch, slice, cls, `B${item.batch}`);
            item.remaining -= slice;
            nextCore = (nextCore + 1) % cores.length;
            guard += 1;
          }
        }
      } else {
        for (const item of viewBatches) {
          const duration = Math.max(0.004, item.compute_time);
          const cls = classForDuration(duration);
          const coreIdx = leastLoadedCore();
          emitBar(coreIdx, item.batch, duration, cls, `B${item.batch}`);
        }
      }

      const expected = Number(metrics.total_expected_batches);
      const total = Number(metrics.total_batches);
      if (metrics.status === 'running' && Number.isFinite(expected) && Number.isFinite(total)) {
        const pending = Math.max(0, expected - total);
        const queuedCount = Math.min(coreCount * 2, pending);
        const avgDuration = Math.max(0.02, avg);
        const startBatch = viewBatches.length > 0 ? viewBatches[viewBatches.length - 1].batch + 1 : total;
        for (let i = 0; i < queuedCount; i += 1) {
          const coreIdx = leastLoadedCore();
          const duration = avgDuration * 0.9;
          emitBar(coreIdx, startBatch + i, duration, 'queued', `Q${startBatch + i}`);
        }
      }

      const totalSec = Math.max(...cores.map((c) => c.load));
      const minW = Math.max(780, Math.ceil(totalSec * scale) + 90);
      coreContainer.innerHTML = cores.map((coreLane, idx) => (
        `<div class="timeline-core" style="min-width:${minW}px;"><div class="core-label">Core ${idx + 1}</div>${coreLane.bars.join('')}</div>`
      )).join('');

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
      const list = data.adjusted;
      const avg = list.length > 0 ? (list.reduce((acc, it) => acc + it.compute_time, 0) / list.length) : 0;
      const throughput = avg > 0 ? (1 / avg) * 0.8 : 0;
      const utilRaw = throughput * avg * 100;
      const util = clamp(((utilRaw * 1.2) + (Number(metrics.batch_size) || 0) * 0.35) * data.factors.utilBias, 0, 100);
      const batchSize = Number(metrics.batch_size) || 0;

      const memText = (metrics.gpu_info && metrics.gpu_info.memory_gb) || 'Unknown';
      const totalMem = Number.parseFloat(memText) || 24;
      const vramUsed = clamp(((batchSize * 0.08) + util * 0.07) * data.factors.vramScale, 0, totalMem);
      const vramPct = clamp((vramUsed / Math.max(1, totalMem)) * 100, 0, 100);

      const bw = clamp((throughput * batchSize * 0.35) * data.factors.bwScale, 0, 900);
      const bwPct = clamp((bw / 900) * 100, 0, 100);

      const warps = Math.max(1, Math.ceil(batchSize / 32));
      const warpPct = clamp((warps / 64) * 100, 0, 100);

      setMeter('utilBar', util);
      setMeter('vramBar', vramPct);
      setMeter('bwBar', bwPct);
      setMeter('warpBar', warpPct);

      document.getElementById('utilValue').textContent = `${util.toFixed(1)}%`;
      document.getElementById('vramValue').textContent = `${vramUsed.toFixed(1)} / ${totalMem.toFixed(0)} GB`;
      document.getElementById('bwValue').textContent = `${bw.toFixed(1)} GB/s`;
      document.getElementById('warpValue').textContent = `${warps} warps`;

      const headroom = Math.max(0, 100 - util);
      const hintLevel = util < 45 ? 'low' : (util < 75 ? 'moderate' : 'high');
      document.getElementById('resourceHint').textContent = `Current utilization is ${hintLevel} at ${util.toFixed(1)}%. Optimization headroom: ${headroom.toFixed(1)}%.`;

      const activeModel = data.selectedModel;
      const activeGpu = data.selectedGpu;
      const expected = metrics.total_expected_batches || '?';
      const bs = metrics.batch_size || '32';
      document.getElementById('meta').textContent = `Model ${activeModel.toUpperCase()} | GPU ${activeGpu} | Batch Size ${bs} | Expected Batches ${expected} | Scheduler ${schedulerName}`;

      updateModelInfo(metrics.model_info, activeModel);
      updateGpuInfo(metrics.gpu_info, activeGpu);
      document.getElementById('algoActive').innerHTML = `<strong>Scheduler:</strong> ${schedulerName}`;
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
    }

    async function startNewRun() {
      await fetch('/api/start_new_run', { method: 'POST' });
      resetDashboardState();
      pushLog('New run initialized; queue cleared and scheduler reset.');
      await tick();
    }

    async function tick() {
      try {
        const response = await fetch('/api/metrics', { cache: 'no-store' });
        const metrics = await response.json();
        const batches = Array.isArray(metrics.batches) ? metrics.batches : [];

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

        if (batches.length > 0) {
          const lastBatch = Number(batches[batches.length - 1].batch);
          if (Number.isFinite(lastBatch) && lastBatch > lastLoggedBatch) {
            for (let b = lastLoggedBatch + 1; b <= lastBatch; b += 1) {
              pushLog(`${schedulerName} dispatched Batch ${b}`);
            }
            lastLoggedBatch = lastBatch;
          }
        }

        if (metrics.status === 'completed') {
          const total = Number(metrics.total_batches) || 0;
          if (schedulerLog.length === 0 || !schedulerLog[schedulerLog.length - 1].includes('Simulation marked completed')) {
            pushLog(`Simulation marked completed with ${total} tasks finished.`);
          }
        }
      } catch (_err) {
        setStatus('waiting', false);
      }
    }

    document.getElementById('infoToggle').addEventListener('click', () => {
      document.getElementById('gpemuInfo').classList.toggle('open');
    });

    document.getElementById('modelSelector').addEventListener('change', async (evt) => {
      try {
        await setModelSelection(evt.target.value);
        resetDashboardState();
        await tick();
        pushLog(`Model switched to ${evt.target.value}.`);
      } catch (_err) {
      }
    });

    document.getElementById('gpuSelector').addEventListener('change', async (evt) => {
      try {
        await setGpuSelection(evt.target.value);
        resetDashboardState();
        await tick();
        pushLog(`GPU profile switched to ${evt.target.value}.`);
      } catch (_err) {
      }
    });

    document.getElementById('algoSelector').addEventListener('change', (evt) => {
      schedulerName = evt.target.value;
      document.getElementById('algoActive').innerHTML = `<strong>Scheduler:</strong> ${schedulerName}`;
      pushLog(`Scheduling policy set to ${schedulerName} (demo uses FIFO execution).`);
    });

    document.getElementById('speedSlider').addEventListener('input', (evt) => {
      const value = Number(evt.target.value);
      if (value === 0) {
        refreshMs = 2000;
        document.getElementById('speedLabel').textContent = 'Slow (2.0s)';
      } else if (value === 2) {
        refreshMs = 500;
        document.getElementById('speedLabel').textContent = 'Fast (0.5s)';
      } else {
        refreshMs = 1000;
        document.getElementById('speedLabel').textContent = 'Normal (1.0s)';
      }
      restartTicker();
      pushLog(`Simulation refresh interval changed to ${(refreshMs / 1000).toFixed(1)}s.`);
    });

    document.getElementById('startRunBtn').addEventListener('click', async () => {
      try {
        await startNewRun();
      } catch (_err) {
      }
    });

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
  os.makedirs(GLIDE_DIR, exist_ok=True)

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
  os.makedirs(GLIDE_DIR, exist_ok=True)
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
    'eta_seconds': None
  }


def _load_metrics_file() -> Dict[str, Any]:
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


def _enrich_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
  enriched = _safe_metrics_payload()
  enriched.update(data)

  selected_model = _normalize_model(enriched.get('selected_model'))
  run_model_raw = enriched.get('model')
  run_model = _normalize_model(run_model_raw) if run_model_raw else None
  active_model = run_model or selected_model

  enriched['selected_model'] = selected_model
  enriched['active_model'] = active_model
  enriched['model'] = active_model
  enriched['model_info'] = _model_info(active_model)

  selected_gpu = _normalize_gpu(enriched.get('selected_gpu'))
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

  if enriched.get('status') not in ('running', 'completed'):
    enriched['status'] = 'running'

  return enriched


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
  return jsonify({
    'ok': True,
    'selected_model': selected,
    'model_info': _model_info(selected)
  })


@app.route('/api/set_gpu', methods=['POST'])
def api_set_gpu():
  payload = request.get_json(silent=True) or {}
  selected_gpu = _save_selected_gpu(payload.get('gpu', DEFAULT_GPU))
  return jsonify({
    'ok': True,
    'selected_gpu': selected_gpu,
    'gpu_info': _gpu_info(selected_gpu)
  })


@app.route('/api/start_new_run', methods=['POST'])
def api_start_new_run():
  _clear_metrics_file()
  selected = _load_selected_model()
  selected_gpu = _load_selected_gpu()
  return jsonify({
    'ok': True,
    'status': 'waiting',
    'selected_model': selected,
    'model_info': _model_info(selected),
    'selected_gpu': selected_gpu,
    'gpu_info': _gpu_info(selected_gpu)
  })


if __name__ == '__main__':
  _save_selected_model(_load_selected_model())
  _save_selected_gpu(_load_selected_gpu())
  app.run(host='0.0.0.0', port=5000, debug=False)
