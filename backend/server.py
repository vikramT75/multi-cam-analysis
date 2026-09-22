"""
server.py
─────────
Retail Store Intelligence — Central Analytics Broker

Receives telemetry from edge inference nodes over HTTP POST, persists it
to SQLite, and broadcasts live updates to dashboard clients over WebSocket.

Endpoints:
    WS  /ws                        Real-time broadcast to dashboard
    POST /telemetry                Ingest from edge nodes
    GET  /snapshot                 Latest state for all cameras (dashboard cold-start)
    GET  /history?camera=X&minutes=60   Time-series from SQLite
    GET  /cameras                  List of known active cameras

Run:
    python backend/server.py
"""

import asyncio
import json
import sqlite3
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from reid_manager import ReIDManager

DB_PATH = Path("data/analytics.db")
reid_manager = ReIDManager()

MAX_ROWS_PER_CAMERA = 7_200

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _db_connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            camera    TEXT    NOT NULL,
            timestamp REAL    NOT NULL,
            payload   TEXT    NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_camera_ts ON telemetry (camera, timestamp)"
    )
    conn.commit()
    conn.close()

_insert_counter: dict[str, int] = {}
PRUNE_EVERY = 100   # run the expensive DELETE query once per N inserts per camera

def db_insert(camera: str, timestamp: float, payload_json: str) -> None:
    """Persist one telemetry record; prune oldest rows every PRUNE_EVERY inserts."""
    conn = _db_connect()
    conn.execute(
        "INSERT INTO telemetry (camera, timestamp, payload) VALUES (?, ?, ?)",
        (camera, timestamp, payload_json),
    )

    _insert_counter[camera] = _insert_counter.get(camera, 0) + 1
    if _insert_counter[camera] % PRUNE_EVERY == 0:
        conn.execute(
            "DELETE FROM telemetry WHERE camera = ? AND timestamp < ?",
            (camera, timestamp - 7200),  # 2 hours
        )

    conn.commit()
    conn.close()

def db_fetch_history(camera: str, minutes: int) -> list:
    """Return time-series rows for a camera covering the last N minutes."""
    since = time.time() - minutes * 60
    conn  = _db_connect()
    rows  = conn.execute(
        """
        SELECT timestamp, payload
        FROM   telemetry
        WHERE  camera = ? AND timestamp > ?
        ORDER  BY timestamp ASC
        """,
        (camera, since),
    ).fetchall()
    conn.close()
    return [{"timestamp": r[0], **json.loads(r[1])} for r in rows]

class ConnectionManager:
    """
    Thread-safe WebSocket connection registry with async broadcast.
    Handles dead-client cleanup automatically on send failure.
    """

    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, data: dict) -> None:
        payload = json.dumps(data)
        async with self._lock:
            dead: set[WebSocket] = set()
            for client in self._clients:
                try:
                    await client.send_text(payload)
                except Exception:
                    dead.add(client)
            self._clients -= dead

    @property
    def count(self) -> int:
        return len(self._clients)

manager = ConnectionManager()

latest_state: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Multi Cam Analysis API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Persistent WebSocket for dashboard clients.
    Immediately hydrates new connections with the latest known snapshot.
    """
    await manager.connect(ws)

    if latest_state:
        try:
            await ws.send_text(json.dumps({
                "type":    "snapshot",
                "cameras": latest_state,
            }))
        except Exception:
            pass

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)

@app.post("/telemetry")
async def receive_telemetry(data: dict):
    """
    Ingests a telemetry payload from an edge node.
    Persists to SQLite and broadcasts to all connected dashboard clients.
    """
    camera    = data.get("camera", "unknown")
    timestamp = data.get("timestamp", time.time())

    signatures = data.get("signatures", {})
    track_zones = data.get("track_zones", {})

    if signatures:
        reid_manager.resolve_identities(camera, signatures)
    if track_zones:
        reid_manager.update_global_journeys(camera, track_zones)

    data["global_funnel"]        = reid_manager.get_global_funnel()
    data["global_transitions"]   = reid_manager.get_global_transitions()
    data["cross_camera_count"]   = reid_manager.get_cross_camera_count()

    # Strip heavy data before DB insertion and WebSocket broadcast
    data.pop("signatures", None)
    data.pop("track_zones", None)

    latest_state[camera] = data

    payload_str = json.dumps(data)
    await asyncio.to_thread(db_insert, camera, timestamp, payload_str)

    await manager.broadcast({"type": "update", "camera": camera, **data})

    return {"status": "ok", "connected_clients": manager.count}

@app.get("/snapshot")
async def get_snapshot():
    """
    Returns the most recent telemetry payload for every known camera.
    Used by the dashboard on first load to instantly populate KPIs and cards.
    """
    return JSONResponse({"cameras": latest_state})

@app.get("/history")
async def get_history(camera: str, minutes: int = 60):
    """
    Returns historical time-series data for one camera.

    Query params:
        camera  — Camera name (must match config.yaml camera.name)
        minutes — Lookback window in minutes (default: 60, max: 120)
    """
    minutes = min(minutes, 120)
    rows    = await asyncio.to_thread(db_fetch_history, camera, minutes)
    return JSONResponse({"camera": camera, "minutes": minutes, "count": len(rows), "rows": rows})

@app.get("/cameras")
async def get_cameras():
    """Returns the list of cameras that have sent at least one telemetry payload."""
    return JSONResponse({"cameras": list(latest_state.keys())})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
