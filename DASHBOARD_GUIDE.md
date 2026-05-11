# GLIDE Dashboard Suite

Three independent, research-styled dashboards for monitoring GPU inference, engine performance, and layer profiling.

## Overview

- **Overview Dashboard** (`dashboard_server.py`, port 5000)  
  Original unified dashboard with task scheduling, GPU utilization, and performance metrics.

- **Engine Dashboard** (`engine_dashboard.py`, port 5001)  
  Real-time inference engine metrics: queue status, latency analysis, and completed requests.

- **Profiler Dashboard** (`profiler_dashboard.py`, port 5002)  
  Layer profiling results: per-layer compute time, memory usage, and bottleneck analysis.

## Running the Dashboards

### Terminal 1: Overview Dashboard
```bash
cd /home/pranav/gpemu
python glide/dashboard_server.py
# Open http://localhost:5000
```

### Terminal 2: Engine Dashboard
```bash
cd /home/pranav/gpemu
python glide/engine_dashboard.py
# Open http://localhost:5001
```

### Terminal 3: Profiler Dashboard
```bash
cd /home/pranav/gpemu
python glide/profiler_dashboard.py
# Open http://localhost:5002
```

## Design Philosophy

All dashboards use a **research-oriented aesthetic** with:
- Clean typography (Inter + IBM Plex Mono)
- Academic-style metrics cards and tables
- Real-time data polling (2-3 second refresh intervals)
- Minimal, professional color scheme (grays and blue/red accents)
- Clear labeling and per-metric units

## Data Sources

### Overview Dashboard
- `/api/metrics` - GPU profile stats, queue metrics, throughput
- `/api/set_model`, `/api/set_gpu` - Configuration endpoints
- `/api/start_new_run` - Run initialization

### Engine Dashboard
- `/api/engine/status` - Queue length, completed count, latency percentiles, memory
- `/api/engine/results` - List of completed inference requests

### Profiler Dashboard
- `/api/profiler/latest` - Slowest layers by compute time, full layer database

## API Endpoints

All dashboards share endpoints defined in `dashboard_server.py`:

```
GET  /api/metrics              → Overview metrics
POST /api/set_model            → Change model
POST /api/set_gpu              → Change GPU
POST /api/start_new_run        → Clear and restart
GET  /api/engine/status        → Engine queue state
GET  /api/engine/results       → Completed requests
GET  /api/profiler/latest      → Slowest layers (top 10)
```

## Customization

Each dashboard is standalone:
- Modify CSS in the `*_DASHBOARD_HTML` template strings
- Adjust refresh intervals: change `setInterval(updateDashboard, Nms)`
- Add/remove metrics by editing API fetch and render functions

## Notes

- Engine dashboard requires `glide.engine.InferenceEngine` to be importable
- Profiler dashboard queries `glide.database` (SQLite layer_db.sqlite)
- All three dashboards gracefully degrade if backends are unavailable
- No external chart libraries used (pure CSS/vanilla JS)
