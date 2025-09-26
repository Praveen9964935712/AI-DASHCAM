"""
test_all.py: Integration test for all core modules of Edge AI Dashcam
- Runs detection, tracking, privacy, risk, and incident modules on a webcam frame
"""
import cv2
from detection import Detector
from tracking import TrackerWrapper
from privacy import PrivacyFilter
from risk import RiskScorer
from incident import IncidentLogger

# Config
MODEL_PATH = 'yolov8n.pt'
VIDEO_SOURCE = 0  # webcam

detector = Detector(MODEL_PATH)
tracker = TrackerWrapper()
privacy_filter = PrivacyFilter()
risk_scorer = RiskScorer()
incident_logger = IncidentLogger()

cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, frame = cap.read()
if not ret or frame is None:
    print('[ERROR] Could not capture frame from webcam.')
    exit(1)

# Detection
print('Running detection...')
detections = detector.detect(frame)
print('Detections:', detections)

# Tracking
print('Running tracking...')
tracked = tracker.update(frame, detections)
print('Tracked objects:', tracked)

# Privacy
print('Running privacy filter...')
frame_priv = privacy_filter.blur_sensitive(frame.copy(), detections)

# Risk
print('Running risk scoring...')
ttc = risk_scorer.compute_ttc(tracked)
print('TTC:', ttc)

# Incident
print('Logging incident...')
incident_logger.buffer_frame(frame)
incident_logger.log_incident('Test event', 'output/test_incident.mp4')

# Draw detections
for det in detections:
    x1, y1, x2, y2, conf, class_id = det[:6]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f'ID:{class_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow('Original + Detections', frame)
cv2.imshow('Privacy Filtered', frame_priv)
print('Press any key to exit...')
cv2.waitKey(0)
cv2.destroyAllWindows()
