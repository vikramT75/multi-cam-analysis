import cv2
import numpy as np

class ZoneAnalyzer:
    """
    Provides spatial analytics for tracking systems, including point-in-polygon 
    evaluation and robust historical occupancy tracking.
    """
    def __init__(self, polygon_points):
        self.polygon = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
        self.occupants = set()
        self.all_time_entered = set()
        self.total_entered = 0

    def process_tracks(self, frame, tracks):
        """
        Evaluates bounding box tracks against the defined region of interest.
        """
        current_occupants = set()
        
        if tracks and len(tracks) > 0 and tracks[0].boxes and tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu().numpy()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                
                # Calculate the bottom-center centroid (representing the subject's feet)
                bottom_center = (int((x1 + x2) / 2), int(y2))
                
                # Evaluate if the centroid resides within the defined polygon zone
                is_inside = cv2.pointPolygonTest(self.polygon, bottom_center, False) >= 0
                
                if is_inside:
                    current_occupants.add(track_id)
                    cv2.circle(frame, bottom_center, 6, (0, 0, 255), -1) 
                else:
                    cv2.circle(frame, bottom_center, 6, (0, 255, 0), -1) 
        
        # Maintain a historical set of all unique IDs that have entered the zone
        # This provides robustness against ID flickering or temporary occlusions
        for occupant_id in current_occupants:
            self.all_time_entered.add(occupant_id)
            
        self.total_entered = len(self.all_time_entered)
        self.occupants = current_occupants
        
        return frame

    def draw_zone(self, frame):
        """Renders the polygon zone and analytics overlay onto the frame."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], (0, 255, 255)) 
        
        alpha = 0.25 
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.polylines(frame, [self.polygon], isClosed=True, color=(0, 200, 255), thickness=2)
        
        # Render telemetry HUD
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Zone Occupancy: {len(self.occupants)}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Entered: {self.total_entered}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
        return frame
