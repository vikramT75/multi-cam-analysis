from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multi-Camera Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for broadcasting real-time analytics to dashboard clients."""
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text() 
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.post("/telemetry")
async def receive_telemetry(data: dict):
    """Ingests telemetry payloads from edge nodes and broadcasts to connected clients."""
    for client in list(connected_clients):
        try:
            await client.send_json(data)
        except Exception:
            connected_clients.remove(client)
    return {"status": "broadcasted", "clients": len(connected_clients)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
