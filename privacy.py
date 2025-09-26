import cv2
import os

class PrivacyFilter:
    def __init__(self, cascade_path=None):
        # Use OpenCV's built-in Haar cascade for license plate detection
        if cascade_path is None:
            # Download from https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml
            cascade_path = 'haarcascade_russian_plate_number.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)

    def blur_sensitive(self, frame, detections=None, fast=True):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Speed up: increase minNeighbors, minSize, and optionally skip frames
            plates = self.plate_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=7 if fast else 4,
                minSize=(60, 60) if fast else (30, 30)
            )
            for (x, y, w, h) in plates:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (31, 31), 0)
        except Exception as e:
            print(f"[PrivacyFilter] Error in blur_sensitive: {e}")
        return frame
