# Multi-Camera Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![YOLOv11](https://img.shields.io/badge/YOLO-v11-darkgreen?logo=yolo) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi) ![RabbitMQ](https://img.shields.io/badge/RabbitMQ-3.x-FF6600?logo=rabbitmq) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)

A real-time, multi-camera people-tracking and analytics system. Each camera runs an independent edge inference pipeline (YOLOv11 + ByteTrack + polygon zone analytics + MobileNetV2 ReID embeddings) and streams telemetry via a RabbitMQ broker to a central FastAPI server that performs cross-camera re-identification, persists data to SQLite, and broadcasts live updates to a WebSocket-driven dashboard.

---

## Architecture

```mermaid
flowchart LR
    subgraph Edge["Edge Nodes"]
        C1["Entrance_Cam\ndetector.py\nYOLOv11 · ByteTrack\nZoneAnalyzer · ReIDExtractor"]
        C2["Aisle_Cam\ndetector.py\nYOLOv11 · ByteTrack\nZoneAnalyzer · ReIDExtractor"]
    end

    subgraph Broker["Message Broker"]
        Q["RabbitMQ\nQueue: telemetry"]
    end

    subgraph Server["Central Server (FastAPI)"]
        C["aio_pika Consumer"]
        RM["ReIDManager\nCosine Similarity Gallery\nEMA Update · Global Journeys"]
        DB["SQLite WAL\nPersistence"]
        WS["WebSocket /ws\nBroadcast"]
    end

    subgraph Frontend["Dashboard"]
        D["dashboard.html\nChart.js · Zone Cards\nAlert Panel"]
    end

    C1 -- "AMQP telemetry\n(tracks, zones, signatures)" --> Q
    C2 -- "AMQP telemetry\n(tracks, zones, signatures)" --> Q
    Q --> C
    C --> RM
    RM --> DB
    RM --> WS
    WS --> D
```

---

## Core Components

### Edge Inference `src/`

| Module | Role |
|---|---|
| [`detector.py`](src/detector.py) | Top-level pipeline per camera; orchestrates all edge components |
| YOLOv11 `.onnx` | Object detection - people only class 0 |
| ByteTrack | Multi-object tracking; assigns stable per-camera track IDs across frames |
| [`spatial_analytics.py`](src/spatial_analytics.py) - `ZoneAnalyzer` | Polygon zone membership, per-track dwell time, occupancy counts, dwell-threshold alerts |
| [`journey_tracker.py`](src/journey_tracker.py) - `JourneyTracker` | Per-camera zone transition tracking; builds funnel and Sankey diagram data |
| [`reid_extractor.py`](src/reid_extractor.py) - `ReIDExtractor` | MobileNetV2 classifier head replaced with `Identity`; extracts 1280-d L2-normalised embeddings from person bounding-box crops every 15 frames; returns base64-encoded strings |
| [`telemetry.py`](src/telemetry.py) - `TelemetrySender` | Background-thread AMQP publisher to RabbitMQ |
| [`video_streamer.py`](src/video_streamer.py) - `VideoStreamer` | Threaded ring-buffer; decouples frame capture from inference |

Each camera is configured independently via its own YAML file (see [Configuration Schema](#configuration-schema)).

---

### Central Server `backend/`

| Module | Role |
|---|---|
| [`server.py`](backend/server.py) | FastAPI application; consumes telemetry from RabbitMQ, drives ReIDManager, manages WebSocket connections |
| [`reid_manager.py`](backend/reid_manager.py) - `ReIDManager` | Maintains a cross-camera gallery of global_id to embedding; matches incoming embeddings via cosine similarity; assigns new global IDs or merges existing ones; tracks global zone transitions |
| SQLite WAL mode | Persists telemetry snapshots and history for cold-start hydration and time-series queries |

---

### Presentation `frontend/`

| File | Role |
|---|---|
| [`dashboard.html`](frontend/dashboard.html) | Single-page dashboard; WebSocket client; Chart.js rolling occupancy chart; dynamic per-camera zone cards; alert panel |

---

## ReID Pipeline

Cross-camera person re-identification is performed in five stages:

1. **Embedding extraction** - On the edge node, `ReIDExtractor` crops each tracked bounding box and passes it through a headless MobileNetV2 network ImageNet pre-trained, classifier replaced with `nn.Identity`. The resulting 1280-dimensional feature vector is L2-normalised. Extraction runs every **15 frames** per track to balance accuracy and throughput.

2. **Telemetry transmission** - The normalised embedding is serialised as a base64-encoded string and included in the `signatures` field of the telemetry payload published to the RabbitMQ `telemetry` queue.

3. **Gallery matching** - The server's `ReIDManager` computes **cosine similarity** between the incoming embedding and every embedding in the cross-camera gallery.

4. **Identity assignment** - If the best cosine similarity score exceeds the **0.65 threshold**, the track is matched to the existing `global_id`. Otherwise a new `global_id` is minted and added to the gallery.

5. **Gallery update - EMA blending** - On a successful match the gallery embedding is updated with Exponential Moving Average:

   `e_gallery = (0.8 * e_gallery) + (0.2 * e_new)`

   This keeps the reference stable while slowly adapting to appearance changes. Cross-camera matches are logged as ReID events and used to build global zone-transition journeys.

---

## API Specification

| Protocol / Method | Endpoint / Queue | Description |
|---|---|---|
| `WS` | `/ws` | WebSocket - live analytics broadcast to the dashboard |
| `AMQP` | `telemetry` queue | Asynchronous ingest from edge nodes (tracks, zone states, ReID signatures) |
| `GET` | `/snapshot` | Full current state for cold-start dashboard hydration |
| `GET` | `/history?camera=X&minutes=60` | Time-series occupancy data from SQLite |
| `GET` | `/cameras` | List of active cameras seen by the server |

---

## Deployment Guide

### 1 - Install dependencies

```bash
pip install -r requirements.txt
```

### 2 - Start RabbitMQ & Central Server (Docker)

```bash
docker compose up -d
```
*(Starts both RabbitMQ and the FastAPI backend container with automatic health-checking)*

*(Optional: to run the server locally without Docker instead: `uvicorn backend.server:app --host 0.0.0.0 --port 8000`)*

### 3 - Launch edge detectors one per camera

```bash
python src/detector.py --config config.yaml
python src/detector.py --config config_cam2.yaml
```

### 4 - Open the dashboard

Navigate to `frontend/dashboard.html` in a browser (or serve it from any static file server). The page connects to the WebSocket at `ws://localhost:8000/ws` automatically.

---

## Configuration Schema

Each camera is described by a YAML file. Below is the current configuration for **Entrance_Cam** (640 × 480 @ 6 fps):

```yaml
camera:
  name: "Entrance_Cam"
  source: "data/cam1_mall_entrance.mp4"

analytics:
  classes_to_track: [0]          # 0 = person (COCO)
  zones:
    - name: "Entrance"
      color: [0, 255, 150]
      alert_dwell_seconds: 180
      polygon:
        - [60,  10]
        - [540, 10]
        - [540, 190]
        - [60,  190]
    - name: "Stall"
      color: [255, 215, 0]
      alert_dwell_seconds: 480
      polygon:
        - [120, 375]
        - [290, 375]
        - [290, 480]
        - [120, 480]
    - name: "Seating"
      color: [0, 200, 255]
      alert_dwell_seconds: 900
      polygon:
        - [255, 225]
        - [400, 225]
        - [400, 340]
        - [255, 340]
    - name: "Shop_Entry"
      color: [255, 100, 0]
      alert_dwell_seconds: 300
      polygon:
        - [548, 110]
        - [640, 110]
        - [640, 300]
        - [548, 300]

model:
  weights: "models/yolo11n.onnx"
  confidence_threshold: 0.45
  iou_threshold: 0.4
```

**Aisle_Cam** (cam2) follows the same schema with zones mapped to product-category displays: `Pots`, `Cups`, `Plates`, and `Bowls`.
