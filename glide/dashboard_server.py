import fcntl
import json
import os
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request


GLIDE_DIR = '/workspace/gpemu/glide'
METRICS_PATH = os.path.join(GLIDE_DIR, 'glide_metrics.json')
SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')
DEFAULT_MODEL = 'resnet18'

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


app = Flask(__name__)


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GLIDE Live Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg0: #07090f;
      --bg1: #0d1327;
      --panel: rgba(17, 24, 42, 0.8);
      --line: #22366a;
      --text: #e7f2ff;
      --muted: #95aed0;
      --neon: #2be4ff;
      --hot: #ff4d6d;
      --ok: #33f0a6;
      --warn: #ffd166;
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
        linear-gradient(165deg, var(--bg0), var(--bg1));
      color: var(--text);
      font-family: 'Share Tech Mono', monospace;
      padding: 20px;
      overflow-x: hidden;
    }

    .shell {
      max-width: 1300px;
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
      background: linear-gradient(90deg, rgba(43, 228, 255, 0.08), rgba(255, 77, 109, 0.08));
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

    .controls {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 12px;
      margin-bottom: 14px;
    }

    .control-card, .model-info {
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

    .grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    .card {
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      backdrop-filter: blur(8px);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
      transform: translateY(10px);
      opacity: 0;
      animation: reveal 520ms ease-out forwards;
    }

    .card:nth-child(2) { animation-delay: 70ms; }
    .card:nth-child(3) { animation-delay: 140ms; }
    .card:nth-child(4) { animation-delay: 210ms; }
    .card:nth-child(5) { animation-delay: 280ms; }
    .card:nth-child(6) { animation-delay: 350ms; }

    .label {
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.7px;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .value {
      font-family: 'Rajdhani', sans-serif;
      font-weight: 700;
      font-size: clamp(1.3rem, 2.4vw, 1.95rem);
      line-height: 1;
    }

    .tooltip {
      position: absolute;
      left: 10px;
      right: 10px;
      top: calc(100% + 8px);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(5, 9, 18, 0.96);
      color: var(--text);
      font-size: 0.72rem;
      line-height: 1.35;
      padding: 8px;
      opacity: 0;
      transform: translateY(-4px);
      pointer-events: none;
      transition: opacity 160ms ease, transform 160ms ease;
      z-index: 10;
    }

    .card:hover .tooltip {
      opacity: 1;
      transform: translateY(0);
    }

    .chart-wrap {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(8, 12, 22, 0.92);
      padding: 14px;
      min-height: 390px;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }

    #emptyState {
      min-height: 340px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 10px;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 10px;
      background: rgba(10, 14, 26, 0.7);
      letter-spacing: 0.8px;
      text-transform: uppercase;
      text-align: center;
      padding: 18px;
    }

    #chartCanvas {
      width: 100%;
      height: 340px;
    }

    @media (max-width: 1120px) {
      .controls {
        grid-template-columns: 1fr;
      }
      .grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
    }

    @media (max-width: 680px) {
      body { padding: 14px; }
      .topbar { flex-direction: column; align-items: flex-start; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chart-wrap { min-height: 320px; }
      #chartCanvas { height: 280px; }
      .control-row { align-items: stretch; }
      select, .btn { width: 100%; }
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

    @keyframes reveal {
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <div class="title" id="pageTitle">GLIDE Metrics Console • RESNET18</div>
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
        It tracks compute behavior batch by batch so we can observe performance trends and compare model choices.
        In this review demo, GLIDE streams those metrics live to this dashboard for easy interpretation.
      </div>
    </div>

    <div class="controls">
      <div class="control-card">
        <div class="input-label">Model Selector</div>
        <div class="control-row">
          <div>
            <select id="modelSelector">
              <option value="resnet18">ResNet18</option>
              <option value="resnet50">ResNet50</option>
              <option value="alexnet">AlexNet</option>
              <option value="vgg16">VGG16</option>
              <option value="squeezenet1_0">SqueezeNet</option>
            </select>
          </div>
          <button class="btn" id="startRunBtn">Start New Run</button>
        </div>
        <div class="hint">Select a model, then launch your training command. The selected model is saved and used by main.py at startup.</div>
      </div>

      <div class="model-info">
        <div class="input-label">Selected Model Info</div>
        <div class="model-name" id="modelName">ResNet18</div>
        <div class="model-line" id="modelDesc">A compact image classifier with residual skip connections.</div>
        <div class="model-line" id="modelLayers"><strong>Layers:</strong> 18</div>
        <div class="model-line" id="modelUseCase"><strong>Use case:</strong> Fast baseline image classification and education demos.</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="label">Avg Compute Time</div>
        <div class="value" id="avgCard">--</div>
        <div class="tooltip">Average time to process one batch of 32 images</div>
      </div>
      <div class="card">
        <div class="label">Min Compute Time</div>
        <div class="value" id="minCard">--</div>
        <div class="tooltip">Fastest/slowest batch seen so far</div>
      </div>
      <div class="card">
        <div class="label">Max Compute Time</div>
        <div class="value" id="maxCard">--</div>
        <div class="tooltip">Fastest/slowest batch seen so far</div>
      </div>
      <div class="card">
        <div class="label">Throughput</div>
        <div class="value" id="throughputCard">--</div>
        <div class="tooltip">How many batches the GPU emulator processes per second</div>
      </div>
      <div class="card">
        <div class="label">Total Batches</div>
        <div class="value" id="totalCard">0</div>
        <div class="tooltip">Number of image batches processed so far</div>
      </div>
      <div class="card">
        <div class="label">Est. Time Remaining</div>
        <div class="value" id="etaCard">--</div>
        <div class="tooltip">Estimated time to finish all batches</div>
      </div>
    </div>

    <div class="chart-wrap">
      <div id="emptyState">
        <div style="font-size: 1.2rem; color: var(--neon);">WAITING FOR DATA...</div>
        <div>Run main.py and stream batch compute times to begin.</div>
      </div>
      <canvas id="chartCanvas" style="display:none;"></canvas>
    </div>
  </div>

  <script>
    const fmt = (v, suffix = 's') => (Number.isFinite(v) ? `${v.toFixed(4)} ${suffix}` : '--');

    function formatSeconds(sec) {
      if (!Number.isFinite(sec)) return '--';
      if (sec < 60) return `${sec.toFixed(1)}s`;
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m}m ${s}s`;
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
        text.textContent = 'COMPLETED';
      } else {
        badge.classList.add('running');
        text.textContent = 'RUNNING';
      }
    }

    const chartCtx = document.getElementById('chartCanvas').getContext('2d');
    const chart = new Chart(chartCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Batch Compute Time (s)',
          data: [],
          borderColor: '#2be4ff',
          backgroundColor: 'rgba(43, 228, 255, 0.24)',
          borderWidth: 2,
          tension: 0.28,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 350 },
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: '#c8dcf8' } } },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.08)' },
            ticks: { color: '#9db7d8' }
          },
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255,255,255,0.08)' },
            ticks: {
              color: '#9db7d8',
              callback: (value) => `${value}s`
            }
          }
        }
      }
    });

    function resetDashboardState() {
      document.getElementById('avgCard').textContent = '--';
      document.getElementById('minCard').textContent = '--';
      document.getElementById('maxCard').textContent = '--';
      document.getElementById('throughputCard').textContent = '--';
      document.getElementById('totalCard').textContent = '0';
      document.getElementById('etaCard').textContent = '--';
      document.getElementById('meta').textContent = 'Awaiting run metadata...';
      setStatus('waiting', false);

      chart.data.labels = [];
      chart.data.datasets[0].data = [];
      chart.update();

      document.getElementById('emptyState').style.display = 'flex';
      document.getElementById('chartCanvas').style.display = 'none';
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

    function updateCards(metrics) {
      document.getElementById('avgCard').textContent = fmt(metrics.avg_compute_time);
      document.getElementById('minCard').textContent = fmt(metrics.min_compute_time);
      document.getElementById('maxCard').textContent = fmt(metrics.max_compute_time);
      document.getElementById('throughputCard').textContent = Number.isFinite(metrics.throughput) ? `${metrics.throughput.toFixed(2)} batch/s` : '--';
      document.getElementById('totalCard').textContent = metrics.total_batches ?? 0;
      document.getElementById('etaCard').textContent = formatSeconds(metrics.eta_seconds);

      const activeModel = metrics.active_model || metrics.selected_model || 'resnet18';
      const expected = metrics.total_expected_batches || '?';
      const bs = metrics.batch_size || '32';
      document.getElementById('meta').textContent = `Model ${activeModel.toUpperCase()} | Batch Size ${bs} | Expected Batches ${expected}`;

      updateModelInfo(metrics.model_info, activeModel);
    }

    function updateChart(batches) {
      const hasData = batches.length > 0;
      const empty = document.getElementById('emptyState');
      const canvas = document.getElementById('chartCanvas');
      empty.style.display = hasData ? 'none' : 'flex';
      canvas.style.display = hasData ? 'block' : 'none';

      if (!hasData) return;
      chart.data.labels = batches.map((b) => b.batch);
      chart.data.datasets[0].data = batches.map((b) => b.compute_time);
      chart.update();
    }

    async function setModelSelection(model) {
      const response = await fetch('/api/set_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });
      const payload = await response.json();
      updateModelInfo(payload.model_info, payload.selected_model);
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

        const selector = document.getElementById('modelSelector');
        const selected = metrics.selected_model || 'resnet18';
        if (selector.value !== selected) {
          selector.value = selected;
        }

        setStatus(metrics.status, batches.length > 0);
        updateCards(metrics);
        updateChart(batches);
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
    setInterval(tick, 1000);
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
    return DEFAULT_MODEL


def _model_info(model_key: str) -> Dict[str, str]:
    model = _normalize_model(model_key)
    return MODEL_CATALOG[model]


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
    return {
        'batches': [],
        'status': 'waiting',
        'model': None,
        'selected_model': selected_model,
        'active_model': selected_model,
        'model_info': _model_info(selected_model),
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


@app.route('/api/start_new_run', methods=['POST'])
def api_start_new_run():
    _clear_metrics_file()
    selected = _load_selected_model()
    return jsonify({
        'ok': True,
        'status': 'waiting',
        'selected_model': selected,
        'model_info': _model_info(selected)
    })


if __name__ == '__main__':
    _save_selected_model(_load_selected_model())
    app.run(host='0.0.0.0', port=5000, debug=False)
