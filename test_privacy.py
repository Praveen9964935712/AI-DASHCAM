"""
test_privacy.py
Test script for privacy.py
"""
import cv2
import numpy as np
from privacy import PrivacyFilter

def main():
    pf = PrivacyFilter()
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)  # White frame
    detections = []  # Not used in dummy
    frame_blur = pf.blur_sensitive(frame, detections)
    cv2.imshow('Privacy Blur Test', frame_blur)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
