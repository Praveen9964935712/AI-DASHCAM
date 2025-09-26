"""
risk.py
Calculates risk metrics such as Time-To-Collision, headway, and lane deviation.
"""
import numpy as np

import collections
import numpy as np

class RiskScorer:
    def __init__(self, pixels_per_meter=40, fps=30):
        # Store previous positions and speeds for each track_id
        self.prev_positions = collections.defaultdict(list)
        self.prev_speeds = collections.defaultdict(list)
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter  # Calibrate for your camera

    def compute_ttc(self, tracked_objects, ego_speed=10.0):
        """
        Compute Time-To-Collision (TTC) for each tracked object using bbox center movement and real-world calibration.
        Returns: list of (track_id, ttc)
        """
        ttc_list = []
        for obj in tracked_objects:
            x1, y1, x2, y2, conf, class_id, track_id = obj
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self.prev_positions[track_id].append((cx, cy, y2 - y1))
            # Keep only last 2 positions
            if len(self.prev_positions[track_id]) > 2:
                self.prev_positions[track_id] = self.prev_positions[track_id][-2:]
            # Calculate speed (meters/sec)
            if len(self.prev_positions[track_id]) == 2:
                prev = np.array(self.prev_positions[track_id][0][:2])
                curr = np.array(self.prev_positions[track_id][1][:2])
                pixel_speed = np.linalg.norm(curr - prev) * self.fps
                speed_mps = pixel_speed / self.pixels_per_meter
                self.prev_speeds[track_id].append(speed_mps)
                if len(self.prev_speeds[track_id]) > 2:
                    self.prev_speeds[track_id] = self.prev_speeds[track_id][-2:]
                # Estimate distance in meters using bbox height (approximate)
                bbox_height = self.prev_positions[track_id][1][2]
                distance_m = max(0.5, 20.0 / max(bbox_height, 1))  # 20 is an estimated real car height in pixels at 1m
                # TTC = distance / (relative speed)
                rel_speed = max(speed_mps - ego_speed, 0.1)  # Avoid divide by zero
                ttc = distance_m / rel_speed
                ttc_list.append((track_id, round(ttc, 2)))
            else:
                ttc_list.append((track_id, 99.9))  # Not enough data yet
        return ttc_list

    def detect_harsh_braking(self, tracked_objects, decel_threshold=3.0):
        """
        Detect harsh braking for vehicles in lane. Returns list of track_ids with harsh braking.
        decel_threshold: minimum deceleration (m/s^2) to trigger alert.
        """
        harsh_ids = []
        for obj in tracked_objects:
            x1, y1, x2, y2, conf, class_id, track_id = obj
            # Only consider vehicles
            if class_id not in [2, 3, 5, 7]:
                continue
            speeds = self.prev_speeds.get(track_id, [])
            if len(speeds) == 2:
                decel = speeds[0] - speeds[1]
                if decel > decel_threshold:
                    harsh_ids.append(track_id)
        return harsh_ids

    def compute_headway(self, tracked_objects, ego_speed=10.0):
        # Placeholder: Dummy headway calculation
        return [(obj[-1], 1.2) for obj in tracked_objects]

    def lane_deviation(self, lane_info, tracked_objects):
        # Placeholder: Dummy lane deviation
        return 0.1
