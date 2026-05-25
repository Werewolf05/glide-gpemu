"""
GLIDE Engine Dashboard - Research Project

Standalone dashboard for monitoring inference engine performance metrics.
Displays queue status, completed requests, and latency analysis.
"""

import json
import os
import time
from flask import Flask, jsonify, render_template_string, Response, request
from typing import Dict, Any, Optional

# Import engine for live status
ENGINE_INIT_ERROR = None
get_profiled_compute_time = None
get_profiled_memory = None
try:
    from glide.engine import InferenceEngine, get_profiled_compute_time, get_profiled_memory
    ENGINE = InferenceEngine()
except Exception as exc_primary:
  # Support running this file either from repository root or from the glide directory.
  try:
    from engine import InferenceEngine, get_profiled_compute_time, get_profiled_memory  # type: ignore
    ENGINE = InferenceEngine()
  except Exception as exc_fallback:
    ENGINE = None
    ENGINE_INIT_ERROR = f"primary={type(exc_primary).__name__}: {exc_primary}; fallback={type(exc_fallback).__name__}: {exc_fallback}"

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

GLIDE_DIR = os.path.dirname(os.path.abspath(__file__))
GPEMU_ROOT = os.path.dirname(GLIDE_DIR)
PROFILED_DATA_ROOT = os.path.join(GPEMU_ROOT, 'profiled_data', 'time', 'compute', 'forward')
SELECTED_GPU_PATH = os.path.join(GLIDE_DIR, 'selected_gpu.json')
SELECTED_MODEL_PATH = os.path.join(GLIDE_DIR, 'selected_model.json')


def _load_selected_gpu() -> str:
    if not os.path.exists(SELECTED_GPU_PATH):
        return 'Tesla_M40'
    try:
        with open(SELECTED_GPU_PATH, 'r') as f:
            data = json.load(f)
            return data.get('gpu', 'Tesla_M40') if isinstance(data, dict) else 'Tesla_M40'
    except (OSError, json.JSONDecodeError):
        return 'Tesla_M40'


def _load_selected_model() -> str:
    if not os.path.exists(SELECTED_MODEL_PATH):
        return 'resnet18'
    try:
        with open(SELECTED_MODEL_PATH, 'r') as f:
            data = json.load(f)
            return data.get('model', 'resnet18') if isinstance(data, dict) else 'resnet18'
    except (OSError, json.JSONDecodeError):
        return 'resnet18'


def _list_available_gpus() -> list:
    if not os.path.isdir(PROFILED_DATA_ROOT):
        return []
    gpus = []
    for name in sorted(os.listdir(PROFILED_DATA_ROOT)):
        full_path = os.path.join(PROFILED_DATA_ROOT, name)
        if os.path.isdir(full_path):
            gpus.append(name)
    return gpus


def _list_models_for_gpu(gpu: str) -> list:
    if not gpu:
        return []
    gpu_path = os.path.join(PROFILED_DATA_ROOT, gpu)
    if not os.path.isdir(gpu_path):
        return []
    result = []
    for name in sorted(os.listdir(gpu_path)):
        model_path = os.path.join(gpu_path, name)
        if os.path.isdir(model_path):
            result.append(name)
    return result


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

    payload[field] = value
    payload['updated_at'] = time.time()

    if field == 'gpu':
      payload['profile_path'] = os.path.join(PROFILED_DATA_ROOT, value)
      payload.setdefault('database_path', os.path.join(GPEMU_ROOT, 'profiled_data'))

    with open(file_path, 'w') as file_handle:
      json.dump(payload, file_handle)


def _build_gpu_coverage(model: str, batch_size: int = 32) -> list:
    if get_profiled_compute_time is None or get_profiled_memory is None:
        return []

    rows = []
    for gpu in _list_available_gpus():
        compute_ms = get_profiled_compute_time(gpu, model, batch_size)
        memory_mb = get_profiled_memory(gpu, model, batch_size)
        if compute_ms is None and memory_mb is None:
            continue
        rows.append(
            {
                'gpu': gpu,
                'compute_ms': round(compute_ms, 3) if compute_ms is not None else None,
                'memory_mb': round(memory_mb, 3) if memory_mb is not None else None,
            }
        )
    rows.sort(key=lambda item: item['compute_ms'] if item['compute_ms'] is not None else float('inf'))
    return rows


ENGINE_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GLIDE Engine Dashboard | Inference Research</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      min-height: 100vh;
      padding: 40px 20px;
      color: #2c3e50;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 50px;
      padding-bottom: 30px;
      border-bottom: 2px solid rgba(44, 62, 80, 0.1);
    }

    h1 {
      font-size: 2.2rem;
      font-weight: 700;
      margin-bottom: 10px;
      color: #1a252f;
    }

    .subtitle {
      font-size: 1.1rem;
      color: #7f8c8d;
      font-weight: 300;
      margin-bottom: 20px;
    }

    .metadata {
      display: flex;
      justify-content: center;
      gap: 30px;
      font-size: 0.95rem;
      color: #34495e;
    }

    .metadata-item {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .metadata-label {
      font-weight: 600;
      color: #2c3e50;
    }

    .status-badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
      background: #e8f5e9;
      color: #2e7d32;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 24px;
      margin-bottom: 40px;
    }

    .card {
      background: white;
      border-radius: 8px;
      padding: 24px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      border-left: 4px solid #3498db;
      transition: box-shadow 0.3s ease;
    }

    .card:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }

    .card-label {
      font-size: 0.85rem;
      font-weight: 600;
      color: #7f8c8d;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 10px;
    }

    .card-value {
      font-size: 2.5rem;
      font-weight: 700;
      color: #2c3e50;
      font-family: 'IBM Plex Mono', monospace;
    }

    .card-unit {
      font-size: 0.9rem;
      color: #95a5a6;
      margin-left: 8px;
    }

    .section {
      background: white;
      border-radius: 8px;
      padding: 30px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      margin-bottom: 30px;
    }

    .section-title {
      font-size: 1.4rem;
      font-weight: 700;
      margin-bottom: 24px;
      color: #1a252f;
      padding-bottom: 12px;
      border-bottom: 2px solid #ecf0f1;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }

    thead {
      background: #f8f9fa;
      border-bottom: 2px solid #ecf0f1;
    }

    th {
      padding: 12px;
      text-align: left;
      font-weight: 600;
      color: #2c3e50;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }

    td {
      padding: 14px 12px;
      border-bottom: 1px solid #ecf0f1;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.9rem;
    }

    tbody tr:hover {
      background: #f8f9fa;
    }

    .empty-state {
      text-align: center;
      padding: 40px 20px;
      color: #95a5a6;
    }

    .empty-state p {
      font-size: 1.1rem;
      margin-bottom: 10px;
    }

    footer {
      text-align: center;
      margin-top: 60px;
      padding-top: 30px;
      border-top: 1px solid rgba(44, 62, 80, 0.1);
      color: #95a5a6;
      font-size: 0.9rem;
    }

    .stat-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 30px;
    }

    .stat {
      padding: 15px;
      background: #f8f9fa;
      border-radius: 6px;
      border-left: 3px solid #3498db;
    }

    .stat-label {
      font-size: 0.8rem;
      font-weight: 600;
      color: #7f8c8d;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .stat-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: #2c3e50;
      font-family: 'IBM Plex Mono', monospace;
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: center;
      margin-top: 18px;
    }

    .controls label {
      font-size: 0.85rem;
      font-weight: 600;
      color: #2c3e50;
    }

    .controls select,
    .controls button {
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid #d5dbe3;
      background: #fff;
      font-family: 'Inter', sans-serif;
      font-size: 0.9rem;
    }

    .controls button {
      background: #2c3e50;
      color: #fff;
      border-color: #2c3e50;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>GLIDE Inference Engine</h1>
      <p class="subtitle">Real-time queue and performance monitoring</p>
      <div class="metadata">
        <div class="metadata-item">
          <span class="metadata-label">Model:</span>
          <span id="modelName">resnet18</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">GPU:</span>
          <span id="gpuName">Tesla_M40</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">Status:</span>
          <span class="status-badge" id="statusBadge">Ready</span>
        </div>
      </div>
      <div class="controls">
        <label for="gpuSelect">GPU</label>
        <select id="gpuSelect"></select>
        <label for="modelSelect">Model</label>
        <select id="modelSelect"></select>
        <button id="applyConfigBtn" type="button">Apply</button>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-label">Queue Length</div>
        <div class="card-value"><span id="queueLength">0</span><span class="card-unit">requests</span></div>
      </div>
      <div class="card">
        <div class="card-label">Completed</div>
        <div class="card-value"><span id="completedCount">0</span><span class="card-unit">requests</span></div>
      </div>
      <div class="card">
        <div class="card-label">Avg Latency</div>
        <div class="card-value"><span id="avgLatency">--</span><span class="card-unit">ms</span></div>
      </div>
      <div class="card">
        <div class="card-label">P95 Latency</div>
        <div class="card-value"><span id="p95Latency">--</span><span class="card-unit">ms</span></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Performance Summary</div>
      <div class="stat-row">
        <div class="stat">
          <div class="stat-label">Peak Memory</div>
          <div class="stat-value" id="peakMemory">-- MB</div>
        </div>
        <div class="stat">
          <div class="stat-label">Throughput</div>
          <div class="stat-value" id="throughput">-- req/s</div>
        </div>
        <div class="stat">
          <div class="stat-label">Running</div>
          <div class="stat-value" id="isRunning">No</div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Profiled Coverage Across GPUs (Batch 32)</div>
      <table id="coverageTable">
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
              <p>No profiled coverage data found for current model.</p>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title">Recent Completed Requests</div>
      <table id="resultsTable">
        <thead>
          <tr>
            <th>Request ID</th>
            <th>Model</th>
            <th>Batch</th>
            <th>Latency (ms)</th>
            <th>Compute (ms)</th>
            <th>Memory (MB)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td colspan="7" class="empty-state">
              <p>No completed requests yet. Start the engine to generate data.</p>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <footer>
      <p>GLIDE Inference Engine Dashboard | Research Project</p>
      <p style="margin-top: 8px; font-size: 0.85rem;">Auto-refreshes every 2 seconds</p>
    </footer>
  </div>

  <script>
    async function loadOptions() {
      const resp = await fetch('/api/engine/options');
      if (!resp.ok) throw new Error('Failed to load engine options');
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || 'Engine options unavailable');

      const gpuSelect = document.getElementById('gpuSelect');
      const modelSelect = document.getElementById('modelSelect');

      gpuSelect.innerHTML = (data.gpus || []).map(g => `<option value="${g}">${g}</option>`).join('');
      gpuSelect.value = data.selected?.gpu || '';

      const models = data.models_by_gpu?.[gpuSelect.value] || [];
      modelSelect.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
      if (models.includes(data.selected?.model)) {
        modelSelect.value = data.selected.model;
      }
    }

    async function applyConfig() {
      const gpu = document.getElementById('gpuSelect').value;
      const model = document.getElementById('modelSelect').value;
      const resp = await fetch('/api/engine/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gpu, model })
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || 'Failed to apply engine config');
      }
      document.getElementById('statusBadge').textContent = 'Ready';
      document.getElementById('statusBadge').title = 'Configuration updated';
      await updateDashboard();
    }

    function bindControls() {
      document.getElementById('gpuSelect').addEventListener('change', async () => {
        const gpu = document.getElementById('gpuSelect').value;
        const resp = await fetch('/api/engine/options');
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.ok) return;
        const modelSelect = document.getElementById('modelSelect');
        const models = data.models_by_gpu?.[gpu] || [];
        modelSelect.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
      });
      document.getElementById('applyConfigBtn').addEventListener('click', async () => {
        try {
          await applyConfig();
        } catch (err) {
          document.getElementById('statusBadge').textContent = 'Error';
          document.getElementById('statusBadge').title = (err && err.message) ? err.message : 'Configuration failed';
        }
      });
    }

    async function updateDashboard() {
      try {
        // Get engine status
        const statusResp = await fetch('/api/engine/status');
        const status = await statusResp.json();
        if (!statusResp.ok) {
          throw new Error(status.error || 'Status unavailable');
        }

        if (!status.ok) {
          document.getElementById('queueLength').textContent = '--';
          document.getElementById('completedCount').textContent = '--';
          document.getElementById('statusBadge').textContent = 'Unavailable';
          if (status.error) {
            document.getElementById('statusBadge').title = status.error;
          }
          return;
        }

        const data = status.status || {};
        const baseline = status.profiled_baseline || {};
        const selected = status.selected || {};
        document.getElementById('modelName').textContent = selected.model || 'unknown';
        document.getElementById('gpuName').textContent = selected.gpu || 'unknown';

        const avgLatencyVal = Number(data.avg_latency_ms || 0);
        const p95LatencyVal = Number(data.p95_latency_ms || 0);
        const memVal = Number(data.current_memory_mb || 0);

        document.getElementById('queueLength').textContent = data.queue_length || 0;
        document.getElementById('completedCount').textContent = data.completed_count || 0;

        const shownAvg = avgLatencyVal > 0 ? avgLatencyVal : Number(baseline.compute_ms || 0);
        const shownP95 = p95LatencyVal > 0 ? p95LatencyVal : Number(baseline.compute_ms || 0);
        const shownMem = memVal > 0 ? memVal : Number(baseline.memory_mb || 0);

        document.getElementById('avgLatency').textContent = shownAvg > 0 ? shownAvg.toFixed(2) : '--';
        document.getElementById('p95Latency').textContent = shownP95 > 0 ? shownP95.toFixed(2) : '--';
        document.getElementById('peakMemory').textContent = shownMem > 0 ? shownMem.toFixed(1) + ' MB' : '--';
        document.getElementById('isRunning').textContent = data.is_running ? 'Yes' : 'No';
        document.getElementById('throughput').textContent = shownAvg > 0 ? (1000.0 / shownAvg).toFixed(2) + ' req/s' : '-- req/s';

        // Get completed results
        const resultsResp = await fetch('/api/engine/results');
        if (resultsResp.ok) {
          const results = await resultsResp.json();
          if (results.ok && Array.isArray(results.results)) {
            renderResults(results.results.slice(-20).reverse());
          }
        }

        const coverageResp = await fetch('/api/engine/coverage');
        if (coverageResp.ok) {
          const coverage = await coverageResp.json();
          if (coverage.ok && Array.isArray(coverage.coverage)) {
            renderCoverage(coverage.coverage);
          }
        }
      } catch (err) {
        console.error('Dashboard update failed:', err);
        document.getElementById('statusBadge').textContent = 'Error';
        document.getElementById('statusBadge').title = (err && err.message) ? err.message : 'Unknown dashboard update error';
      }
    }

    function renderResults(results) {
      const tbody = document.querySelector('#resultsTable tbody');
      if (results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state"><p>No completed requests yet.</p></td></tr>';
        return;
      }

      tbody.innerHTML = results.map(r => `
        <tr>
          <td>${r.request_id || '-'}</td>
          <td>${r.model_name || '-'}</td>
          <td>${r.batch_size || '-'}</td>
          <td>${Number.isFinite(Number(r.latency_ms)) ? Number(r.latency_ms).toFixed(2) : '--'}</td>
          <td>${Number.isFinite(Number(r.emulated_compute_ms)) ? Number(r.emulated_compute_ms).toFixed(2) : '--'}</td>
          <td>${Number.isFinite(Number(r.memory_mb)) ? Number(r.memory_mb).toFixed(1) : '--'}</td>
          <td>${r.status || 'unknown'}</td>
        </tr>
      `).join('');
    }

    function renderCoverage(rows) {
      const tbody = document.querySelector('#coverageTable tbody');
      if (!rows || rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" class="empty-state"><p>No profiled coverage data found for current model.</p></td></tr>';
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
      bindControls();
      await loadOptions();
      await updateDashboard();
    }

    // Initial load
    initDashboard();
    setInterval(updateDashboard, 2000);
  </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    return render_template_string(ENGINE_DASHBOARD_HTML)


@app.route('/favicon.ico')
def favicon() -> Response:
    return Response(status=204)


@app.route('/api/engine/status')
def api_engine_status():
    if ENGINE is None:
        return jsonify({'ok': False, 'error': f'engine_unavailable: {ENGINE_INIT_ERROR or "unknown initialization error"}'}), 503
    status = ENGINE.get_queue_status()
    selected_gpu = getattr(ENGINE, 'gpu', _load_selected_gpu())
    selected_model = getattr(ENGINE, 'model', _load_selected_model())
    baseline = {
        'compute_ms': get_profiled_compute_time(selected_gpu, selected_model, 32) if get_profiled_compute_time else None,
        'memory_mb': get_profiled_memory(selected_gpu, selected_model, 32) if get_profiled_memory else None,
    }
    return jsonify({'ok': True, 'status': status, 'selected': {'gpu': selected_gpu, 'model': selected_model}, 'profiled_baseline': baseline})


@app.route('/api/engine/results')
def api_engine_results():
    if ENGINE is None:
        return jsonify({'ok': False, 'error': f'engine_unavailable: {ENGINE_INIT_ERROR or "unknown initialization error"}'}), 503
    return jsonify({'ok': True, 'results': ENGINE.get_results()})


@app.route('/api/engine/coverage')
def api_engine_coverage():
    model = _load_selected_model()
    coverage = _build_gpu_coverage(model=model, batch_size=32)
    return jsonify({'ok': True, 'model': model, 'coverage': coverage})


@app.route('/api/engine/options')
def api_engine_options():
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


@app.route('/api/engine/config', methods=['POST'])
def api_engine_config():
  payload = request.get_json(silent=True) or {}
  gpu = str(payload.get('gpu', '')).strip()
  model = str(payload.get('model', '')).strip()

  gpus = _list_available_gpus()
  if gpu not in gpus:
    return jsonify({'ok': False, 'error': 'invalid_gpu'}), 400

  models = _list_models_for_gpu(gpu)
  if model not in models:
    return jsonify({'ok': False, 'error': 'invalid_model_for_gpu'}), 400

  _save_json_field(SELECTED_GPU_PATH, 'gpu', gpu)
  _save_json_field(SELECTED_MODEL_PATH, 'model', model)

  if ENGINE is not None:
    ENGINE.gpu = gpu
    ENGINE.model = model

  return jsonify({'ok': True, 'selected': {'gpu': gpu, 'model': model}})


if __name__ == '__main__':
    print('Starting GLIDE Engine Dashboard on port 5001...')
    app.run(host='0.0.0.0', port=5001, debug=False)
