"""
GLIDE Profiler Dashboard - Research Project

Standalone dashboard for analyzing layer profiling results.
Displays per-layer compute time, memory usage, and performance bottlenecks.
"""

import json
import os
import time
from flask import Flask, jsonify, render_template_string, Response
from flask import request
from typing import Dict, Any, List

# Import database for profiler results
DB_INIT_ERROR = None

try:
    from glide import database as db
except Exception as exc_primary:
    try:
        import database as db
    except Exception as exc_fallback:
        db = None
        DB_INIT_ERROR = f"primary={type(exc_primary).__name__}: {exc_primary}; fallback={type(exc_fallback).__name__}: {exc_fallback}"

print("DATABASE FILE:", db.__file__ if db else "DB IMPORT FAILED")
get_profiled_compute_time = None
get_profiled_memory = None
try:
  from glide.engine import get_profiled_compute_time, get_profiled_memory
except Exception:
  try:
    from engine import get_profiled_compute_time, get_profiled_memory  # type: ignore
  except Exception:
    pass

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

GLIDE_DIR = os.path.dirname(os.path.abspath(__file__))
GPEMU_ROOT = os.path.dirname(GLIDE_DIR)
PROFILED_DATA_ROOT = os.path.join(GPEMU_ROOT, 'profiled_data', 'time', 'compute', 'forward')
LAYER_DB_PATH = os.path.join(GLIDE_DIR, 'layer_db.sqlite')
SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')
SELECTED_GPU_PATH = os.path.join(GLIDE_DIR, 'selected_gpu.json')


def _load_selected_model() -> str:
  if not os.path.exists(SELECTED_MODEL_PATH):
    return 'resnet18'
  try:
    with open(SELECTED_MODEL_PATH, 'r') as file_handle:
      data = json.load(file_handle)
    if isinstance(data, dict):
      return data.get('model', 'resnet18')
  except (OSError, json.JSONDecodeError):
    pass
  return 'resnet18'


def _load_selected_gpu() -> str:
  if not os.path.exists(SELECTED_GPU_PATH):
    return 'Tesla_M40'
  try:
    with open(SELECTED_GPU_PATH, 'r') as file_handle:
      data = json.load(file_handle)
    if isinstance(data, dict):
      return data.get('gpu', 'Tesla_M40')
  except (OSError, json.JSONDecodeError):
    pass
  return 'Tesla_M40'


def _save_json_field(file_path: str, field: str, value: str) -> None:
  payload: Dict[str, Any] = {}
  if os.path.exists(file_path):
    try:
      with open(file_path, 'r') as file_handle:
        loaded = json.load(file_handle)
      if isinstance(loaded, dict):
        payload = loaded
    except (OSError, json.JSONDecodeError):
      payload = {}

  payload[field] = value
  payload['updated_at'] = time.time()

  if field == 'gpu':
    payload['profile_path'] = os.path.join(PROFILED_DATA_ROOT, value)
    payload.setdefault('database_path', os.path.join(GPEMU_ROOT, 'profiled_data'))

  with open(file_path, 'w') as file_handle:
    json.dump(payload, file_handle)


def _list_models_for_gpu(gpu: str) -> list:

    if db is None:
        return []

    profiles = db.get_all_profiles()

    return sorted([
        item['model']
        for item in profiles
        if item['gpu'] == gpu
    ])


def _list_available_gpus() -> list:

    if db is None:
        return []

    profiles = db.get_all_profiles()

    return sorted(
        list(
            set(
                item['gpu']
                for item in profiles
            )
        )
    )


def _build_gpu_coverage(model: str) -> list:

    if db is None:
        return []

    profiles = db.get_all_profiles()

    rows = []

    for item in profiles:

        if item['model'] != model:
            continue

        gpu = item['gpu']

        layers = db.get_slowest_layers(
            gpu=gpu,
            model=model,
            limit=1000
        )

        if not layers:
            continue

        total_compute = sum(
            layer['compute_cost_ms']
            for layer in layers
        )

        total_memory = sum(
            layer['memory_cost_mb']
            for layer in layers
        )

        rows.append({
            'gpu': gpu,
            'compute_ms': round(total_compute, 3),
            'memory_mb': round(total_memory, 3),
        })

    return rows


PROFILER_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GLIDE Profiler Dashboard | Layer Analysis</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
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

    .container {
      max-width: 1400px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 60px;
      padding-bottom: 40px;
      border-bottom: 1px solid #e5e7eb;
    }

    h1 {
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 8px;
      color: #0f172a;
      letter-spacing: -0.5px;
    }

    .subtitle {
      font-size: 1rem;
      color: #6b7280;
      font-weight: 400;
      margin-bottom: 24px;
    }

    .metadata {
      display: flex;
      justify-content: center;
      gap: 40px;
      flex-wrap: wrap;
      font-size: 0.9rem;
      color: #4b5563;
    }

    .metadata-item {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .metadata-label {
      font-weight: 600;
      color: #1a1a1a;
    }

    .status-badge {
      display: inline-block;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      background: #dcfce7;
      color: #16a34a;
      letter-spacing: 0.3px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 24px;
      margin-bottom: 48px;
    }

    .card {
      background: white;
      border-radius: 10px;
      padding: 28px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
      border: 1px solid #e5e7eb;
      border-left: 3px solid #dc2626;
      transition: all 0.2s ease;
    }

    .card:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
      border-left-color: #b91c1c;
    }

    .card-label {
      font-size: 0.75rem;
      font-weight: 700;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 12px;
    }

    .card-value {
      font-size: 2.5rem;
      font-weight: 700;
      color: #0f172a;
      font-family: 'IBM Plex Mono', monospace;
      line-height: 1.2;
    }

    .card-unit {
      font-size: 0.85rem;
      color: #9ca3af;
      margin-left: 6px;
      font-weight: 500;
    }

    .section {
      background: white;
      border-radius: 10px;
      padding: 36px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
      border: 1px solid #e5e7eb;
      margin-bottom: 36px;
    }

    .section-title {
      font-size: 1.3rem;
      font-weight: 700;
      margin-bottom: 28px;
      color: #0f172a;
      padding-bottom: 16px;
      border-bottom: 2px solid #f3f4f6;
      letter-spacing: -0.3px;
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
      padding: 14px 16px;
      text-align: left;
      font-weight: 600;
      color: #374151;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    td {
      padding: 16px;
      border-bottom: 1px solid #f3f4f6;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.85rem;
    }

    tbody tr:hover {
      background: #f9fafb;
    }

    .empty-state {
      text-align: center;
      padding: 60px 20px;
      color: #9ca3af;
    }

    .empty-state p {
      font-size: 1rem;
      margin-bottom: 8px;
    }

    footer {
      text-align: center;
      margin-top: 80px;
      padding-top: 40px;
      border-top: 1px solid #e5e7eb;
      color: #9ca3af;
      font-size: 0.85rem;
    }

    .stat-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 24px;
      margin-bottom: 36px;
    }

    .stat {
      padding: 20px;
      background: #f9fafb;
      border-radius: 8px;
      border-left: 3px solid #dc2626;
      transition: all 0.2s ease;
    }

    .stat:hover {
      background: #f3f4f6;
    }

    .stat-label {
      font-size: 0.75rem;
      font-weight: 700;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 8px;
    }

    .stat-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: #0f172a;
      font-family: 'IBM Plex Mono', monospace;
    }

    .bar-chart {
      margin: 24px 0;
    }

    .bar {
      display: flex;
      align-items: center;
      margin-bottom: 16px;
      gap: 14px;
    }

    .bar-label {
      min-width: 160px;
      font-size: 0.9rem;
      font-weight: 500;
      color: #374151;
      font-family: 'IBM Plex Mono', monospace;
    }

    .bar-container {
      flex: 1;
      height: 28px;
      background: #e5e7eb;
      border-radius: 6px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #dc2626, #991b1b);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding-right: 12px;
      color: white;
      font-size: 0.8rem;
      font-weight: 600;
      font-family: 'IBM Plex Mono', monospace;
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: center;
      margin-top: 24px;
    }

    .controls label {
      font-size: 0.85rem;
      font-weight: 600;
      color: #1a1a1a;
    }

    .controls select,
    .controls button {
      padding: 10px 14px;
      border-radius: 6px;
      border: 1px solid #d1d5db;
      background: #fff;
      font-family: 'Inter', sans-serif;
      font-size: 0.9rem;
      transition: all 0.2s ease;
    }

    .controls select:hover,
    .controls button:hover {
      border-color: #9ca3af;
    }

    .controls button {
      background: #dc2626;
      color: #fff;
      border-color: #dc2626;
      cursor: pointer;
      font-weight: 600;
    }

    .controls button:hover {
      background: #b91c1c;
      border-color: #b91c1c;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>GLIDE Layer Profiler</h1>
      <p class="subtitle">Per-layer compute time and memory analysis</p>
      <div class="metadata">
        <div class="metadata-item">
          <span class="metadata-label">Database:</span>
          <span>layer_db.sqlite</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">Layers Profiled:</span>
          <span id="layerCount">0</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">Status:</span>
          <span class="status-badge" id="statusBadge">Ready</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">Model:</span>
          <span id="modelName">resnet18</span>
        </div>
      </div>
      <div class="controls">
        <label for="profGpuSelect">GPU</label>
        <select id="profGpuSelect"></select>
        <label for="profModelSelect">Model</label>
        <select id="profModelSelect"></select>
        <button id="profApplyConfigBtn" type="button">Apply</button>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-label">Slowest Layer</div>
        <div class="card-value"><span id="slowestLayer">--</span></div>
      </div>
      <div class="card">
        <div class="card-label">Peak Compute</div>
        <div class="card-value"><span id="peakCompute">--</span><span class="card-unit">ms</span></div>
      </div>
      <div class="card">
        <div class="card-label">Total Layers</div>
        <div class="card-value"><span id="totalLayers">0</span></div>
      </div>
      <div class="card">
        <div class="card-label">Avg Compute</div>
        <div class="card-value"><span id="avgCompute">--</span><span class="card-unit">ms</span></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Compute Time Ranking</div>
      <div class="bar-chart" id="computeChart"></div>
    </div>

    <div class="section">
      <div class="section-title">Profiled Coverage Across GPUs (Batch 32)</div>
      <table id="gpuCoverageTable">
        <thead>
          <tr>
            <th>GPU</th>
            <th>Profiled Compute (ms)</th>
            <th>Profiled Memory (MB)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td colspan="3" class="empty-state">
              <p>No profiled GPU coverage found for selected model.</p>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title">All Profiled Layers</div>
      <table id="resultsTable">
        <thead>
          <tr>
            <th>Layer Type</th>
            <th>Compute (ms)</th>
            <th>Memory (MB)</th>
            <th>Configuration</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td colspan="4" class="empty-state">
              <p>No layers profiled yet. Run the profiler to generate data.</p>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <footer>
      <p>GLIDE Layer Profiler Dashboard | Research Project</p>
      <p style="margin-top: 8px; font-size: 0.85rem;">Auto-refreshes every 3 seconds</p>
    </footer>
  </div>

  <script>
    async function loadProfilerOptions() {
      const resp = await fetch('/api/profiler/options');
      if (!resp.ok) throw new Error('Failed to load profiler options');
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || 'Profiler options unavailable');

      const gpuSelect = document.getElementById('profGpuSelect');
      const modelSelect = document.getElementById('profModelSelect');

      gpuSelect.innerHTML = (data.gpus || []).map(g => `<option value="${g}">${g}</option>`).join('');
      gpuSelect.value = data.selected?.gpu || '';

      const models = data.models_by_gpu?.[gpuSelect.value] || [];
      modelSelect.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
      if (models.includes(data.selected?.model)) {
        modelSelect.value = data.selected.model;
      }
    }

    async function applyProfilerConfig() {
      const gpu = document.getElementById('profGpuSelect').value;
      const model = document.getElementById('profModelSelect').value;
      const resp = await fetch('/api/profiler/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gpu, model })
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || 'Failed to apply profiler config');
      }
      document.getElementById('statusBadge').textContent = 'Ready';
      document.getElementById('statusBadge').title = 'Configuration updated';
      await updateDashboard();
    }

    function bindProfilerControls() {
      document.getElementById('profGpuSelect').addEventListener('change', async () => {
        const gpu = document.getElementById('profGpuSelect').value;
        const resp = await fetch('/api/profiler/options');
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.ok) return;
        const modelSelect = document.getElementById('profModelSelect');
        const models = data.models_by_gpu?.[gpu] || [];
        modelSelect.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
      });

      document.getElementById('profApplyConfigBtn').addEventListener('click', async () => {
        try {
          await applyProfilerConfig();
        } catch (err) {
          document.getElementById('statusBadge').textContent = 'Error';
          document.getElementById('statusBadge').title = (err && err.message) ? err.message : 'Configuration failed';
        }
      });
    }

    async function updateDashboard() {
      try {
        const resp = await fetch('/api/profiler/latest');
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Profiler unavailable');

        if (!data.ok) {
          document.getElementById('layerCount').textContent = '0';
          document.getElementById('totalLayers').textContent = '0';
          document.getElementById('statusBadge').textContent = 'Unavailable';
          if (data.error) {
            document.getElementById('statusBadge').title = data.error;
          }
          return;
        }

        document.getElementById('statusBadge').textContent = 'Ready';
        document.getElementById('statusBadge').title = '';

        const layers = Array.isArray(data.slowest_layers) ? data.slowest_layers : [];
        document.getElementById('totalLayers').textContent = layers.length;
        document.getElementById('layerCount').textContent = layers.length;

        if (layers.length === 0) {
          document.getElementById('slowestLayer').textContent = '--';
          document.getElementById('peakCompute').textContent = '--';
          document.getElementById('avgCompute').textContent = '--';
          renderEmpty();
          return;
        }

        const slowest = layers[0];
        document.getElementById('slowestLayer').textContent = slowest.layer_type || '--';
        document.getElementById('peakCompute').textContent = Number.isFinite(Number(slowest.compute_cost_ms)) ? Number(slowest.compute_cost_ms).toFixed(2) : '--';

        const avgMs = layers.reduce((acc, l) => acc + Number(l.compute_cost_ms || 0), 0) / layers.length;
        document.getElementById('avgCompute').textContent = Number.isFinite(avgMs) ? avgMs.toFixed(2) : '--';

        renderChart(layers.slice(0, 10));
        renderResults(layers);

        const coverageResp = await fetch('/api/profiler/gpu_coverage');
        if (coverageResp.ok) {
          const coverageData = await coverageResp.json();
          if (coverageData.ok) {
            document.getElementById('modelName').textContent = coverageData.model || 'unknown';
            renderGpuCoverage(coverageData.coverage || []);
          }
        }
      } catch (err) {
        console.error('Dashboard update failed:', err);
        document.getElementById('statusBadge').textContent = 'Error';
        document.getElementById('statusBadge').title = (err && err.message) ? err.message : 'Unknown profiler update error';
      }
    }

    function renderChart(layers) {
      const maxMs = Math.max(...layers.map(l => Number(l.compute_cost_ms || 0)));
      const chart = document.getElementById('computeChart');
      chart.innerHTML = layers.map(l => {
        const ms = Number(l.compute_cost_ms || 0);
        const pct = maxMs > 0 ? (ms / maxMs) * 100 : 0;
        return `
          <div class="bar">
            <div class="bar-label">${l.layer_type || 'Unknown'}</div>
            <div class="bar-container">
              <div class="bar-fill" style="width: ${pct}%;">
                ${ms.toFixed(2)}ms
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderResults(layers) {
      const tbody = document.querySelector('#resultsTable tbody');
      tbody.innerHTML = layers.map(l => `
        <tr>
          <td>${l.layer_type || '-'}</td>
          <td>${Number.isFinite(Number(l.compute_cost_ms)) ? Number(l.compute_cost_ms).toFixed(2) : '--'}</td>
          <td>${Number.isFinite(Number(l.memory_cost_mb)) ? Number(l.memory_cost_mb).toFixed(2) : '--'}</td>
          <td style="max-width: 300px; overflow: auto; font-size: 0.8rem;">${JSON.stringify(l.config || {})}</td>
        </tr>
      `).join('');
    }

    function renderEmpty() {
      document.getElementById('computeChart').innerHTML = '<p class="empty-state">No data</p>';
      document.querySelector('#resultsTable tbody').innerHTML = '<tr><td colspan="4" class="empty-state"><p>No layers profiled yet.</p></td></tr>';
    }

    function renderGpuCoverage(rows) {
      const tbody = document.querySelector('#gpuCoverageTable tbody');
      if (!rows || rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" class="empty-state"><p>No profiled GPU coverage found for selected model.</p></td></tr>';
        return;
      }
      tbody.innerHTML = rows.map(r => `
        <tr>
          <td>${r.gpu || '-'}</td>
          <td>${Number.isFinite(Number(r.compute_ms)) ? Number(r.compute_ms).toFixed(2) : '--'}</td>
          <td>${Number.isFinite(Number(r.memory_mb)) ? Number(r.memory_mb).toFixed(2) : '--'}</td>
        </tr>
      `).join('');
    }

    async function initDashboard() {
      bindProfilerControls();
      await loadProfilerOptions();
      await updateDashboard();
    }

    // Initial load
    initDashboard();
    setInterval(updateDashboard, 3000);
  </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    return render_template_string(PROFILER_DASHBOARD_HTML)


@app.route('/favicon.ico')
def favicon() -> Response:
    return Response(status=204)


@app.route('/api/profiler/latest')
def api_profiler_latest():

    if db is None:
        return jsonify({
            'ok': False,
            'error': f'database_unavailable: {DB_INIT_ERROR or "unknown initialization error"}'
        }), 503

    try:
        gpu = _load_selected_gpu()
        model = _load_selected_model()

        slowest = db.get_slowest_layers(
            gpu=gpu,
            model=model,
            limit=50
        )

        return jsonify({
            'ok': True,
            'gpu': gpu,
            'model': model,
            'slowest_layers': slowest
        })

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@app.route('/api/profiler/gpu_coverage')
def api_profiler_gpu_coverage():
    model = _load_selected_model()
    coverage = _build_gpu_coverage(model=model)
    return jsonify({'ok': True, 'model': model, 'coverage': coverage})


@app.route('/api/profiler/options')
def api_profiler_options():
  gpus = _list_available_gpus()
  models_by_gpu = {gpu: _list_models_for_gpu(gpu) for gpu in gpus}
  selected_gpu = _load_selected_gpu()
  selected_model = _load_selected_model()
  if selected_gpu not in models_by_gpu:
    selected_gpu = gpus[0] if gpus else ''
  models_for_gpu = models_by_gpu.get(selected_gpu, [])
  if selected_model not in models_for_gpu and models_for_gpu:
    selected_model = models_for_gpu[0]

  return jsonify(
    {
      'ok': True,
      'gpus': gpus,
      'models_by_gpu': models_by_gpu,
      'selected': {'gpu': selected_gpu, 'model': selected_model},
    }
  )


@app.route('/api/profiler/config', methods=['POST'])
def api_profiler_config():

    payload = request.get_json(silent=True) or {}

    print("\n===== CONFIG REQUEST =====")
    print("RAW PAYLOAD:", payload)

    gpu = str(payload.get('gpu', '')).strip()
    model = str(payload.get('model', '')).strip()

    print("GPU:", gpu)
    print("MODEL:", model)

    gpus = _list_available_gpus()

    print("AVAILABLE GPUS:", gpus)

    if gpu not in gpus:
        return jsonify({
            'ok': False,
            'error': 'invalid_gpu',
            'received_gpu': gpu,
            'available_gpus': gpus
        }), 400

    models = _list_models_for_gpu(gpu)

    print("MODELS FOR GPU:", models)

    if model not in models:
        return jsonify({
            'ok': False,
            'error': 'invalid_model_for_gpu',
            'received_model': model,
            'available_models': models
        }), 400

    _save_json_field(SELECTED_GPU_PATH, 'gpu', gpu)
    _save_json_field(SELECTED_MODEL_PATH, 'model', model)

    print("CONFIG SAVED SUCCESSFULLY")

    return jsonify({
        'ok': True,
        'selected': {
            'gpu': gpu,
            'model': model
        }
    })
if __name__ == '__main__':
    print('Starting GLIDE Profiler Dashboard on port 5002...')
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False,
        use_reloader=False
    )