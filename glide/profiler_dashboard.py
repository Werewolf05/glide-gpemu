"""
GLIDE Profiler Dashboard - Research Project

Standalone dashboard for analyzing layer profiling results.
Displays per-layer compute time, memory usage, and performance bottlenecks.
"""

import json
import os
from flask import Flask, jsonify, render_template_string
from typing import Dict, Any, List

# Import database for profiler results
try:
    from glide import database as db
except Exception:
    db = None

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

GLIDE_DIR = os.path.dirname(os.path.abspath(__file__))
LAYER_DB_PATH = os.path.join(GLIDE_DIR, 'layer_db.sqlite')


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
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      min-height: 100vh;
      padding: 40px 20px;
      color: #2c3e50;
    }

    .container {
      max-width: 1400px;
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
      border-left: 4px solid #e74c3c;
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
      border-left: 3px solid #e74c3c;
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

    .bar-chart {
      margin: 20px 0;
    }

    .bar {
      display: flex;
      align-items: center;
      margin-bottom: 12px;
      gap: 12px;
    }

    .bar-label {
      min-width: 140px;
      font-size: 0.9rem;
      font-weight: 500;
    }

    .bar-container {
      flex: 1;
      height: 24px;
      background: #ecf0f1;
      border-radius: 4px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #e74c3c, #c0392b);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding-right: 8px;
      color: white;
      font-size: 0.75rem;
      font-weight: 600;
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
    async function updateDashboard() {
      try {
        const resp = await fetch('/api/profiler/latest');
        if (!resp.ok) throw new Error('Profiler unavailable');
        const data = await resp.json();

        if (!data.ok) {
          document.getElementById('layerCount').textContent = '0';
          document.getElementById('totalLayers').textContent = '0';
          return;
        }

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
      } catch (err) {
        console.error('Dashboard update failed:', err);
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

    // Initial load
    updateDashboard();
    setInterval(updateDashboard, 3000);
  </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    return render_template_string(PROFILER_DASHBOARD_HTML)


@app.route('/api/profiler/latest')
def api_profiler_latest():
    if db is None:
        return jsonify({'ok': False, 'error': 'database_unavailable'}), 503
    try:
        slowest = db.get_slowest_layers(limit=50)
        return jsonify({'ok': True, 'slowest_layers': slowest})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print('Starting GLIDE Profiler Dashboard on port 5002...')
    app.run(host='0.0.0.0', port=5002, debug=False)
