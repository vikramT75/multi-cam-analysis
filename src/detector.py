import cv2
import time
import yaml
from ultralytics import YOLO
from video_streamer import VideoStreamer
from spatial_analytics import ZoneAnalyzer
from telemetry import TelemetrySender

def main():
    """Main execution loop for edge inference and analytics."""
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Initializing {config['camera']['name']} Engine...")
    model = YOLO(config['model']['weights']) 
    
    streamer = VideoStreamer(config['camera']['source'], config['camera']['name']).start()
    telemetry = TelemetrySender()
    
    # Load analytics polygon from config
    zone_polygon = config['analytics']['zone_polygon']
    classes_to_track = config['analytics']['classes_to_track']
    conf_thresh = config['model']['confidence_threshold']
    iou_thresh = config['model']['iou_threshold']
    
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
                
            # Execute object detection and ByteTRACK tracking using config thresholds
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, 
                                  classes=classes_to_track, conf=conf_thresh, iou=iou_thresh)
            
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
