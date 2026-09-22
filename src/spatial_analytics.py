"""
spatial_analytics.py
─────────────────────
Multi-zone spatial analytics engine.

Supports simultaneous occupancy tracking, dwell-time measurement, heatmap
accumulation, and alert generation across any number of named polygonal zones
defined in config.yaml.
"""

import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class ZoneAnalyzer:
    """
    Evaluates object tracks against multiple named polygonal zones simultaneously.

    Args:
        zone_configs: List of zone dicts from config.yaml. Each dict must contain:
            - name  (str):  Unique zone identifier, e.g. "Entrance"
            - polygon (list of [x, y]): Polygon vertices in image coordinates
            - color (list of 3 ints, RGB): Display color
            - alert_dwell_seconds (float, optional): Loitering alert threshold
    """

    def __init__(self, zone_configs: list):
        self._zones: dict = {}

        for zc in zone_configs:
            name = zc["name"]
            # Config stores RGB; OpenCV uses BGR
            rgb = zc.get("color", [0, 255, 255])
            bgr = (rgb[2], rgb[1], rgb[0])

            self._zones[name] = {
                "polygon":     np.array(zc["polygon"], np.int32).reshape((-1, 1, 2)),
                "color":       bgr,
                "alert_dwell": zc.get("alert_dwell_seconds", None),
                # Per-frame state
                "occupants":      set(),
                "all_time_entered": set(),
                "entry_times":    {},   # track_id -> entry timestamp (float)
                "dwell_times":    {},   # track_id -> current dwell seconds (float)
                "heatmap":        None, # np.float32 array, lazy-initialised
            }

    # ── Core processing ───────────────────────────────────────────────────────

    def process_tracks(self, frame: np.ndarray, tracks) -> dict:
        """
        Evaluate all visible tracks against every defined zone.

        Updates per-zone occupancy, dwell times, and heatmaps.

        Args:
            frame:  Annotated video frame (modified in-place with foot-point dots).
            tracks: Ultralytics Results object from model.track().

        Returns:
            track_zone_map: {track_id: zone_name_or_None} — the first zone each
                            track occupies (None if outside all zones). Used by
                            JourneyTracker to detect cross-zone transitions.
        """
        now = time.time()
        track_zone_map: dict = {}

        # ── Initialise / decay heatmaps ──────────────────────────────────────
        h, w = frame.shape[:2]
        for zdata in self._zones.values():
            if zdata["heatmap"] is None:
                zdata["heatmap"] = np.zeros((h, w), dtype=np.float32)
            else:
                zdata["heatmap"] *= 0.98   # Exponential decay — prevents saturation

        # ── Reset per-frame occupant sets ────────────────────────────────────
        current_occupants: dict[str, set] = {name: set() for name in self._zones}

        if (
            tracks
            and len(tracks) > 0
            and tracks[0].boxes is not None
            and tracks[0].boxes.id is not None
        ):
            boxes    = tracks[0].boxes.xyxy.cpu().numpy()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                foot = (int((x1 + x2) / 2), int(y2))

                occupied_zone = None   # first zone this track is inside

                for zone_name, zdata in self._zones.items():
                    inside = cv2.pointPolygonTest(zdata["polygon"], foot, False) >= 0

                    if inside:
                        # Accumulate heat only inside zone boundaries
                        tmp = np.zeros_like(zdata["heatmap"])
                        cv2.circle(tmp, foot, 28, 1.0, -1)
                        zdata["heatmap"] += tmp

                        current_occupants[zone_name].add(track_id)
                        if occupied_zone is None:
                            occupied_zone = zone_name

                        # Dwell time bookkeeping
                        if track_id not in zdata["entry_times"]:
                            zdata["entry_times"][track_id] = now
                        zdata["dwell_times"][track_id] = now - zdata["entry_times"][track_id]

                # ── Visual foot-point indicator ───────────────────────────────
                if occupied_zone:
                    dot_color = self._zones[occupied_zone]["color"]
                else:
                    dot_color = (0, 255, 80)   # bright green — outside all zones

                cv2.circle(frame, foot, 8, dot_color, -1)
                cv2.circle(frame, foot, 8, (255, 255, 255), 1)   # white outline

                track_zone_map[track_id] = occupied_zone

        # ── Update persistent zone state ──────────────────────────────────────
        for zone_name, zdata in self._zones.items():
            occ = current_occupants[zone_name]

            # Remove state for tracks that have left this zone
            for tid in list(zdata["entry_times"]):
                if tid not in occ:
                    zdata["entry_times"].pop(tid, None)
                    zdata["dwell_times"].pop(tid, None)

            zdata["all_time_entered"].update(occ)
            zdata["occupants"] = occ

        return track_zone_map

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw_zones(self, frame: np.ndarray, show_hud: bool = True) -> np.ndarray:
        """
        Renders all zones onto the frame:
          1. Dwell-time heatmap per zone (JET colormap, alpha-blended)
          2. Filled polygon overlay with zone label
          3. Global HUD summarising all zones (togglable)
        """
        # 1. Heatmaps
        for zdata in self._zones.values():
            hm = zdata["heatmap"]
            if hm is None:
                continue
            hm_clipped = np.clip(hm, 0, 80) / 80.0
            hm_uint8   = (hm_clipped * 255).astype(np.uint8)
            mask = hm_uint8 > 8
            if mask.any():
                colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                overlay = frame.copy()
                overlay[mask] = colored[mask]
                frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        # 2. Zone polygons + labels
        for zone_name, zdata in self._zones.items():
            poly  = zdata["polygon"]
            color = zdata["color"]

            # Translucent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

            # Border
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

            # Zone name label at polygon top-left
            pts = poly.reshape(-1, 2)
            lx  = int(pts[:, 0].min())
            ly  = int(pts[:, 1].min())
            (tw, th), _ = cv2.getTextSize(
                zone_name.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            cv2.rectangle(frame, (lx, ly - th - 10), (lx + tw + 10, ly), (0, 0, 0), -1)
            cv2.putText(
                frame, zone_name.upper(), (lx + 5, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

        # 3. HUD overlay
        if show_hud:
            row_h   = 26
            hud_h   = 16 + len(self._zones) * row_h
            cv2.rectangle(frame, (10, 10), (400, 12 + hud_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 12 + hud_h), (60, 60, 60), 1)

            for i, (zone_name, zdata) in enumerate(self._zones.items()):
                occ       = len(zdata["occupants"])
                total     = len(zdata["all_time_entered"])
                dwells    = list(zdata["dwell_times"].values())
                avg_dwell = sum(dwells) / occ if occ > 0 else 0.0

                label = f"{zone_name}: {occ:>2} in | {total:>3} total | avg {avg_dwell:.0f}s"
                y = 28 + i * row_h
                cv2.putText(
                    frame, label, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, zdata["color"], 1
                )

        return frame

    # ── Data export ───────────────────────────────────────────────────────────

    def get_zone_states(self) -> dict:
        """
        Returns a JSON-serialisable snapshot of all zone metrics.

        Schema:
            {
              "Entrance": {
                "occupancy": 5,
                "total_entered": 100,
                "avg_dwell": 45.2,
                "max_dwell": 120.5
              },
              ...
            }
        """
        states = {}
        for zone_name, zdata in self._zones.items():
            occ    = len(zdata["occupants"])
            dwells = list(zdata["dwell_times"].values())
            states[zone_name] = {
                "occupancy":     occ,
                "total_entered": len(zdata["all_time_entered"]),
                "avg_dwell":     round(sum(dwells) / occ, 1) if occ > 0 else 0.0,
                "max_dwell":     round(max(dwells), 1)       if dwells else 0.0,
            }
        return states

    def get_alerts(self) -> list:
        """
        Returns active alert events for tracks exceeding zone dwell thresholds.

        Schema:
            [{"zone": "Entrance", "track_id": 7, "dwell_seconds": 145.2, "type": "LOITERING"}, ...]
        """
        alerts = []
        for zone_name, zdata in self._zones.items():
            threshold = zdata["alert_dwell"]
            if threshold is None:
                continue
            for tid, dwell in zdata["dwell_times"].items():
                if dwell > threshold:
                    alert_type = (
                        "LOITERING"      if zone_name == "Entrance"
                        else "QUEUE_OVERFLOW"
                    )
                    alerts.append({
                        "zone":          zone_name,
                        "track_id":      tid,
                        "dwell_seconds": round(dwell, 1),
                        "type":          alert_type,
                    })
        return alerts
