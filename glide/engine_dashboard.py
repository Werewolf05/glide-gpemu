"""
GLIDE Engine Dashboard - Research Project

Standalone dashboard for monitoring inference engine performance metrics.
Displays queue status, completed requests, and latency analysis.
"""

import json
import os
from flask import Flask, jsonify, render_template_string
from typing import Dict, Any, Optional

# Import engine for live status
try:
    from glide.engine import InferenceEngine
    ENGINE = InferenceEngine()
except Exception:
    ENGINE = None

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

GLIDE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    async function updateDashboard() {
      try {
        // Get engine status
        const statusResp = await fetch('/api/engine/status');
        if (!statusResp.ok) throw new Error('Status unavailable');
        const status = await statusResp.json();

        if (!status.ok) {
          document.getElementById('queueLength').textContent = '--';
          document.getElementById('completedCount').textContent = '--';
          return;
        }

        const data = status.status || {};
        document.getElementById('queueLength').textContent = data.queue_length || 0;
        document.getElementById('completedCount').textContent = data.completed_count || 0;
        document.getElementById('avgLatency').textContent = 
          Number.isFinite(Number(data.avg_latency_ms)) ? Number(data.avg_latency_ms).toFixed(2) : '--';
        document.getElementById('p95Latency').textContent = 
          Number.isFinite(Number(data.p95_latency_ms)) ? Number(data.p95_latency_ms).toFixed(2) : '--';
        document.getElementById('peakMemory').textContent = 
          Number.isFinite(Number(data.current_memory_mb)) ? Number(data.current_memory_mb).toFixed(1) + ' MB' : '--';
        document.getElementById('isRunning').textContent = data.is_running ? 'Yes' : 'No';

        // Get completed results
        const resultsResp = await fetch('/api/engine/results');
        if (resultsResp.ok) {
          const results = await resultsResp.json();
          if (results.ok && Array.isArray(results.results)) {
            renderResults(results.results.slice(-20).reverse());
          }
        }
      } catch (err) {
        console.error('Dashboard update failed:', err);
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

    // Initial load
    updateDashboard();
    setInterval(updateDashboard, 2000);
  </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    return render_template_string(ENGINE_DASHBOARD_HTML)


@app.route('/api/engine/status')
def api_engine_status():
    if ENGINE is None:
        return jsonify({'ok': False, 'error': 'engine_unavailable'}), 503
    status = ENGINE.get_queue_status()
    return jsonify({'ok': True, 'status': status})


@app.route('/api/engine/results')
def api_engine_results():
    if ENGINE is None:
        return jsonify({'ok': False, 'error': 'engine_unavailable'}), 503
    return jsonify({'ok': True, 'results': ENGINE.get_results()})


if __name__ == '__main__':
    print('Starting GLIDE Engine Dashboard on port 5001...')
    app.run(host='0.0.0.0', port=5001, debug=False)
