import numpy as np
import logging

logger = logging.getLogger(__name__)

class ReIDManager:
    """
    Manages Global Identities across multiple cameras using Appearance Signatures.
    """
    def __init__(self, match_threshold=0.85):
        # Maps global_id -> moving average embedding (numpy array)
        self.gallery = {}
        # Maps "camName_localId" -> global_id
        self.local_to_global = {}
        self.next_global_id = 1
        self.match_threshold = match_threshold

        # For global journey tracking
        # Maps global_id -> list of zones visited
        self.global_history = {}
        # Maps "ZoneA->ZoneB" -> count
        self.global_transitions = {}

    def resolve_identities(self, cam_name: str, signatures: dict):
        """
        Processes incoming signatures from a camera, resolving them to Global IDs.
        Args:
            cam_name: The name of the camera (e.g. "Mall")
            signatures: dict of {local_tid: embedding_list}
        """
        for local_tid, emb_list in signatures.items():
            key = f"{cam_name}_{local_tid}"
            emb_np = np.array(emb_list, dtype=np.float32)

            if key in self.local_to_global:
                # We already resolved this track, just slowly update its signature
                gid = self.local_to_global[key]
                self.gallery[gid] = self.gallery[gid] * 0.9 + emb_np * 0.1
                self.gallery[gid] /= np.linalg.norm(self.gallery[gid])
                continue

            # This is a new track from this camera, try to match it globally
            best_match = None
            best_score = -1.0

            for gid, g_emb in self.gallery.items():
                score = np.dot(emb_np, g_emb)
                if score > best_score:
                    best_score = score
                    best_match = gid

            if best_match is not None and best_score >= self.match_threshold:
                # It's a match! Person crossed cameras.
                gid = best_match
                self.local_to_global[key] = gid
                logger.info(f"ReID Match! {key} is Global ID {gid} (Score: {best_score:.2f})")
                
                # Update the global appearance slightly
                self.gallery[gid] = self.gallery[gid] * 0.8 + emb_np * 0.2
                self.gallery[gid] /= np.linalg.norm(self.gallery[gid])
            else:
                # Brand new person
                gid = self.next_global_id
                self.next_global_id += 1
                self.local_to_global[key] = gid
                self.gallery[gid] = emb_np
                logger.info(f"New Identity: {key} assigned Global ID {gid}")

    def update_global_journeys(self, cam_name: str, track_zones: dict):
        """
        Updates the global cross-camera transition counts.
        Args:
            cam_name: Camera name
            track_zones: dict of {local_tid: current_zone_name}
        """
        for local_tid, zone_name in track_zones.items():
            key = f"{cam_name}_{local_tid}"
            gid = self.local_to_global.get(key)
            
            if not gid:
                continue # Signature hasn't been extracted/resolved yet

            if gid not in self.global_history:
                self.global_history[gid] = [zone_name]
            else:
                last_zone = self.global_history[gid][-1]
                if last_zone != zone_name:
                    self.global_history[gid].append(zone_name)
                    
                    # Record the transition! This is true cross-camera tracking
                    t_key = f"{last_zone}->{zone_name}"
                    self.global_transitions[t_key] = self.global_transitions.get(t_key, 0) + 1
                    logger.info(f"Global Transition: ID {gid} moved {t_key}")

    def get_global_transitions(self) -> dict:
        """
        Returns the true cross-camera transition counts.
        """
        return self.global_transitions

    def get_global_funnel(self) -> dict:
        """
        Returns the total unique Global IDs that have visited each zone.
        """
        funnel = {}
        for gid, history in self.global_history.items():
            # Only count unique zones visited by this global ID
            unique_zones = set(history)
            for z in unique_zones:
                funnel[z] = funnel.get(z, 0) + 1
        return funnel
