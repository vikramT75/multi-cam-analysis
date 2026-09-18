import cv2
import time
import yaml
import argparse
import logging
import sys
import os

# Ensure src directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from video_streamer import VideoStreamer
from spatial_analytics import ZoneAnalyzer
from telemetry import TelemetrySender

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    """Main execution loop for edge inference and analytics."""
    parser = argparse.ArgumentParser(description="Run Edge Analytics Node")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to camera configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    cam_name = config['camera']['name']
    logging.info(f"Initializing {cam_name} Engine...")
    
    model = YOLO(config['model']['weights']) 
    
    streamer = VideoStreamer(config['camera']['source'], cam_name).start()
    telemetry = TelemetrySender()
    
    zone_polygon = config['analytics']['zone_polygon']
    classes_to_track = config['analytics']['classes_to_track']
    conf_thresh = config['model']['confidence_threshold']
    iou_thresh = config['model']['iou_threshold']
    
    analyzer = ZoneAnalyzer(zone_polygon)
    
    cv2.namedWindow(f"Analytics - {cam_name}", cv2.WINDOW_NORMAL)
    target_frame_time = 1.0 / streamer.fps
    
    logging.info("Inference started. Press 'q' to quit.")
    
    try:
        while True:
            start_time = time.time()
            
            frame = streamer.read()
            if frame is None:
                logging.info("Video stream ended.")
                break
                
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, 
                                  classes=classes_to_track, conf=conf_thresh, iou=iou_thresh)
            
            annotated_frame = results[0].plot()
            annotated_frame = analyzer.process_tracks(annotated_frame, results)
            annotated_frame = analyzer.draw_zone(annotated_frame)
            
            cv2.imshow(f"Analytics - {cam_name}", annotated_frame)
            
            telemetry.send({
                "camera": cam_name,
                "occupancy": len(analyzer.occupants),
                "total_entered": analyzer.total_entered,
                "max_dwell": round(analyzer.max_dwell, 1),
                "timestamp": time.time()
            })
            
            elapsed = time.time() - start_time
            sleep_time = target_frame_time - elapsed
            wait_ms = max(1, int(sleep_time * 1000) if sleep_time > 0 else 1)
            
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("Shutting down...")
        streamer.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1) 

if __name__ == "__main__":
    main()
