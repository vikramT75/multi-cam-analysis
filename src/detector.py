"""
detector.py
───────────
Main execution loop for the Retail Store Intelligence edge node.

Run one instance per camera:
    python src/detector.py --config config.yaml         # Camera 1 (Entrance + Checkout)
    python src/detector.py --config config_cam2.yaml    # Camera 2 (Aisle)
"""

import cv2
import time
import yaml
import argparse
import logging
import sys
import os

# Ensure src/ is importable regardless of the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from video_streamer import VideoStreamer
from spatial_analytics import ZoneAnalyzer
from journey_tracker import JourneyTracker
from telemetry import TelemetrySender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retail Intelligence Edge Node")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to camera configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # ── Load configuration ────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cam_name      = config["camera"]["name"]
    zone_configs  = config["analytics"]["zones"]
    classes_track = config["analytics"].get("classes_to_track", [0])
    conf_thresh   = config["model"]["confidence_threshold"]
    iou_thresh    = config["model"]["iou_threshold"]

    logger.info("Initialising Retail Intelligence node: %s", cam_name)
    logger.info("Zones: %s", [z["name"] for z in zone_configs])

    # ── Initialise components ─────────────────────────────────────────────────
    model     = YOLO(config["model"]["weights"])
    streamer  = VideoStreamer(config["camera"]["source"], cam_name).start()
    telemetry = TelemetrySender()
    analyzer  = ZoneAnalyzer(zone_configs)
    journey   = JourneyTracker([z["name"] for z in zone_configs])

    # Window is created lazily by cv2.imshow() on the first frame --
    # calling namedWindow() before any frame exists produces a blank
    # black ghost window, so we avoid it here.
    window_title = f"Retail Intelligence - {cam_name}"
    target_frame_time = 1.0 / streamer.fps
    logger.info("Inference started. Press 'q' to quit, 'i' to toggle HUD.")
    first_frame = True
    show_hud = True

    try:
        while True:
            t0 = time.time()

            frame = streamer.read()
            if frame is None:
                logger.info("Stream ended for %s.", cam_name)
                break

            # ── Inference + tracking ──────────────────────────────────────────
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                classes=classes_track,
                conf=conf_thresh,
                iou=iou_thresh,
            )

            # ── Spatial analytics ─────────────────────────────────────────────
            annotated = results[0].plot()
            track_zone_map = analyzer.process_tracks(annotated, results)
            journey.update(track_zone_map)
            annotated = analyzer.draw_zones(annotated, show_hud=show_hud)

            cv2.imshow(window_title, annotated)

            # On the very first frame, make the window user-resizable.
            # We do this here (not before the loop) to avoid the blank black
            # ghost window that appears when namedWindow() is called with no content.
            if first_frame:
                cv2.setWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
                first_frame = False

            # ── Telemetry dispatch ────────────────────────────────────────────
            telemetry.send({
                "camera":      cam_name,
                "timestamp":   time.time(),
                "zones":       analyzer.get_zone_states(),
                "funnel":      journey.get_funnel(),
                "transitions": journey.get_transitions(),
                "sankey":      journey.get_sankey_data(),
                "alerts":      analyzer.get_alerts(),
            })

            # ── Frame timing ──────────────────────────────────────────────────
            elapsed    = time.time() - t0
            sleep_time = target_frame_time - elapsed
            wait_ms    = max(1, int(sleep_time * 1000) if sleep_time > 0 else 1)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                show_hud = not show_hud

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down %s.", cam_name)
        streamer.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
