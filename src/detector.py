import cv2
import time
from ultralytics import YOLO
from video_streamer import VideoStreamer
from spatial_analytics import ZoneAnalyzer
from telemetry import TelemetrySender

def main():
    """Main execution loop for edge inference and analytics."""
    print("Initializing Analytics Engine...")
    model = YOLO("models/yolo11n.pt") 
    
    video_path = "data/sample2.mp4"
    streamer = VideoStreamer(video_path, "Camera_1").start()
    telemetry = TelemetrySender()
    
    # Define ROI Polygon
    zone_polygon = [(150, 150), (450, 150), (600, 350), (50, 350)]
    #zone_polygon = [(300, 200), (900, 200), (1200, 700), (100, 700)]
    analyzer = ZoneAnalyzer(zone_polygon)
    
    cv2.namedWindow("Analytics Engine", cv2.WINDOW_NORMAL)
    target_frame_time = 1.0 / streamer.fps
    
    print("Inference started. Press 'q' to quit.")
    
    try:
        while True:
            start_time = time.time()
            
            frame = streamer.read()
            if frame is None:
                break
                
            # Execute object detection and ByteTRACK tracking
            # Using NMS (iou=0.4) and confidence thresholding (conf=0.5) to prevent ghost detections
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, 
                                  classes=[0, 1, 2, 3, 5, 7], conf=0.5, iou=0.4)
            
            annotated_frame = results[0].plot()
            annotated_frame = analyzer.process_tracks(annotated_frame, results)
            annotated_frame = analyzer.draw_zone(annotated_frame)
            
            cv2.imshow("Analytics Engine", annotated_frame)
            
            # Dispatch spatial telemetry asynchronously
            telemetry.send({
                "camera": "Camera_1",
                "occupancy": len(analyzer.occupants),
                "total_entered": analyzer.total_entered,
                "timestamp": time.time()
            })
            
            # Dynamic frame delay calculation to ensure consistent playback speed
            elapsed = time.time() - start_time
            sleep_time = target_frame_time - elapsed
            wait_ms = max(1, int(sleep_time * 1000) if sleep_time > 0 else 1)
            
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        streamer.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1) 

if __name__ == "__main__":
    main()
