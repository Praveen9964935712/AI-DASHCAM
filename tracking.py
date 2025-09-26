from norfair import Detection, Tracker
import numpy as np

class TrackerWrapper:
    def __init__(self):
        self.tracker = Tracker(distance_function=self.euclidean_distance, distance_threshold=30)

    @staticmethod
    def euclidean_distance(detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def update(self, frame, detections):
        # Convert detections to Norfair format: points=center of bbox, scores=confidence
        norfair_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            norfair_detections.append(Detection(points=np.array([[cx, cy]]), scores=np.array([conf])))

        tracked_objects = self.tracker.update(norfair_detections)
        tracked = []
        for tobj in tracked_objects:
            cx, cy = tobj.estimate[0]
            track_id = tobj.id
            # Find the closest detection to get bbox (optional: improve with IoU matching)
            if detections:
                closest = min(detections, key=lambda d: np.linalg.norm(np.array([(d[0]+d[2])/2, (d[1]+d[3])/2]) - np.array([cx, cy])))
                x1, y1, x2, y2, conf, class_id = closest
                tracked.append([int(x1), int(y1), int(x2), int(y2), conf, class_id, track_id])
        return tracked
