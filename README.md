# Retail Store Intelligence Platform

> A production-grade, multi-camera computer vision analytics system that transforms
> raw object-tracking data into real-time retail business intelligence.

---

## Overview

This system ingests asynchronous video streams from multiple IP cameras, runs
**YOLOv11 ONNX inference + ByteTRACK multi-target tracking** at the edge, performs
**multi-zone spatial analytics**, and streams structured business metrics to a
decoupled real-time dashboard — all with sub-second latency.

## Business Metrics Produced

| Metric | Description |
|---|---|
| **Footfall** | Unique entries into the Entrance zone |
| **Conversion Rate** | % of entrants who reach the Checkout zone |
| **Avg Browse Time** | Mean dwell time in the Aisle zone |
| **Queue Wait Time** | Mean dwell time in the Checkout zone |
| **Loitering Alerts** | Any track exceeding the zone's alert threshold |
| **Dwell Heatmaps** | Per-zone thermal overlays on the live video feed |
| **Sankey Flow** | D3 diagram showing the Entrance → Aisle → Checkout journey |

## Architecture

```
config.yaml / config_cam2.yaml          (Zone definitions per camera)
        │
        ▼
src/video_streamer.py                   (Threaded ring-buffer ingestion)
src/detector.py           ──────────►  src/spatial_analytics.py (Multi-zone)
                                        src/journey_tracker.py   (Funnel engine)
        │
        ▼
src/telemetry.py  ──────────────────►  backend/server.py
                                        ├── WebSocket /ws      (Live broadcast)
                                        ├── POST /telemetry    (Edge node ingest)
                                        ├── GET  /snapshot     (Cold-start hydration)
                                        ├── GET  /history      (SQLite time-series)
                                        └── GET  /cameras      (Active node list)
                                              │
                                              ▼
                                        frontend/dashboard.html
                                        ├── KPI strip (footfall, conversion, dwell, queue)
                                        ├── Per-camera zone cards with alert badges
                                        ├── D3 Sankey conversion flow diagram
                                        └── Chart.js rolling 60-min zone traffic chart
```

## Quick Start

```bash
# Terminal 1 — WebSocket broker + REST API
python backend/server.py

# Terminal 2 — Camera 1: Entrance + Checkout zones (sample.mp4)
python src/detector.py --config config.yaml

# Terminal 3 — Camera 2: Aisle zone (sample2.mp4)
python src/detector.py --config config_cam2.yaml

# Open the dashboard
# Open frontend/dashboard.html in any modern browser
```

## Configuration

Edit `config.yaml` (or `config_cam2.yaml`) to define zones for each camera.
Each zone needs a name, colour, polygon, and an optional alert threshold:

```yaml
analytics:
  zones:
    - name: "Entrance"
      color: [0, 255, 150]          # RGB
      alert_dwell_seconds: 120      # Loitering alert after 2 minutes
      polygon:
        - [30,  380]
        - [420, 380]
        - [420, 700]
        - [30,  700]
```

Polygon coordinates are pixel values relative to the camera's resolution.
Use a tool like [labelme](https://github.com/labelmeai/labelme) to draw zones
visually, then paste the coordinates into `config.yaml`.

## Data Persistence

Telemetry is persisted to `data/analytics.db` (SQLite). The server retains
the last **2 hours** of data per camera (~7,200 rows). History is queryable via:

```
GET http://localhost:8000/history?camera=Entrance_Cam&minutes=60
```

## Tech Stack

| Layer | Technology |
|---|---|
| Inference | YOLOv11n ONNX + Ultralytics |
| Tracking | ByteTRACK (via Ultralytics) |
| Video I/O | OpenCV + threaded ring buffer |
| Spatial math | NumPy + OpenCV point-in-polygon |
| Broker | FastAPI + WebSocket |
| Persistence | SQLite (stdlib `sqlite3`) |
| Dashboard | Tailwind CSS + Chart.js + D3 Sankey |
