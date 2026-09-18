# Edge-Optimized Multi-Camera Analytics Engine

![Dashboard Sync](docs/dashboard-preview.png) *(Draft: Add a screenshot or GIF here later)*

## Overview
A high-throughput, low-latency Computer Vision pipeline designed for MLOps and Edge AI deployment. This system ingests asynchronous video streams, performs real-time object detection and multi-target tracking, and dispatches spatial analytics to a decoupled web dashboard.

## Key Features
* **Zero-Latency Ingestion:** A multi-threaded ring buffer guarantees real-time video processing without queue bottlenecks.
* **Edge Optimization:** Utilizes an exported `.onnx` graph for YOLOv11 to minimize inference latency on constrained hardware.
* **Robust Spatial Analytics:** Point-in-polygon math with tracker-flicker mitigation ensures accurate dwell time and occupancy metrics.
* **Microservices Architecture:** A decoupled FastAPI message broker bridges the gap between the GPU inference engine and the React/Tailwind frontend via WebSockets.
* **Dwell-Time Heatmaps:** Generates live thermal overlays showing high-traffic zones based on object centroids.

## Architecture
1. `src/video_streamer.py`: Asynchronous frame ingestion.
2. `src/detector.py`: YOLOv11 ONNX inference and ByteTRACK association.
3. `src/spatial_analytics.py`: Point-in-polygon and Heatmap logic.
4. `src/telemetry.py`: Asynchronous TCP session dispatcher.
5. `backend/server.py`: FastAPI WebSocket broadcaster.
6. `frontend/dashboard.html`: Tailwind + Chart.js SPA.

## Quick Start
```bash
# 1. Start the WebSocket Telemetry Broker
python backend/server.py

# 2. Start the Inference Engine
python src/detector.py

# 3. Open the Dashboard
# Open frontend/dashboard.html in any modern browser
```

## Configuration
Edit `config.yaml` to modify the RTSP stream URL, ROI polygon coordinates, and tracking confidence thresholds.
