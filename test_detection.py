"""
test_detection.py
Test script for detection.py
"""
import cv2
import numpy as np
from detection import Detector

def main():
    detector = Detector(model_path='model.onnx')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy black frame
    detections = detector.detect(frame)
    print('Detections:', detections)
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imshow('Detection Test', frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
