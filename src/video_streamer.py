import cv2
import threading
import queue
import time

class VideoStreamer:
    """
    Asynchronous video stream reader that handles both local files and live RTSP feeds.
    Implements a threaded ring buffer to decouple frame extraction from inference latency.
    """
    def __init__(self, source, name="Camera"):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or self.fps != self.fps: 
            self.fps = 30
            
        self.is_live = str(source).startswith("rtsp://") or str(source).startswith("http://")
        
        # Configure queue size based on stream type (small for live to minimize latency)
        self.q = queue.Queue(maxsize=30 if not self.is_live else 5) 
        self.stopped = False

    def start(self):
        """Starts the background frame extraction thread."""
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Worker thread loop for extracting frames."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                self.stop()
                return
                
            if self.is_live:
                # Live feeds: drop oldest frames if queue is full to maintain 0-latency
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
            else:
                # Local files: block and wait to ensure no frames are dropped
                while not self.stopped:
                    try:
                        self.q.put(frame, timeout=0.1)
                        break 
                    except queue.Full:
                        continue 

    def read(self):
        """Returns the next available frame from the buffer."""
        return self.q.get()

    def stop(self):
        """Signals the thread to terminate and releases resources."""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
