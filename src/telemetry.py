import threading
import queue
import requests
import logging

class TelemetrySender:
    """
    Asynchronous telemetry dispatcher.
    Offloads HTTP network requests to a background thread to prevent blocking the main inference loop.
    """
    def __init__(self, endpoint_url="http://localhost:8000/telemetry"):
        self.url = endpoint_url
        self.q = queue.Queue(maxsize=10) 
        
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        
    def _worker(self):
        session = requests.Session()
        while True:
            data = self.q.get()
            try:
                session.post(self.url, json=data, timeout=0.5)
            except Exception as e:
                logging.warning(f"Telemetry dispatch failed: {e}")
                
    def send(self, data):
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(data)
