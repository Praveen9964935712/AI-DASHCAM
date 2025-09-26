"""
test_tracking.py
Test script for tracking.py
"""
import numpy as np
from tracking import Tracker

def main():
    tracker = Tracker()
    detections = [ [100, 100, 200, 200, 0.9, 0], [300, 120, 400, 220, 0.8, 1] ]
    tracked = tracker.update(None, detections)
    print('Tracked objects:', tracked)

if __name__ == '__main__':
    main()
