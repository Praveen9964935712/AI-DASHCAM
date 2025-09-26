import pyttsx3
# === TEXT-TO-SPEECH (TTS) ===
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)  # Adjust speech rate if needed
tts_last_alert = ''
tts_last_time = 0
import time
def speak_alert(alert_text, lang='en'):
    global tts_last_alert, tts_last_time
    now = time.time()
    # Avoid repeating the same alert too frequently
    if alert_text != tts_last_alert or now - tts_last_time > 1:
        # For local language, set voice here if needed
        # Example: for Hindi, use pyttsx3.voice selection or gTTS integration
        tts_engine.say(alert_text)
        tts_engine.runAndWait()
        tts_last_alert = alert_text
        tts_last_time = now
import cv2
import time
from detection import Detector
from tracking import TrackerWrapper
from risk import RiskScorer
from privacy import PrivacyFilter
from incident import IncidentLogger


# === CONFIGURATION ===
DETECTION_MODEL_PATH = 'yolov8n.pt'  # Path to YOLOv8 model
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file (e.g., 'test.mp4')
RUN_DURATION = None  # Set to seconds to limit run time, or None for unlimited
MAX_FRAMES = None   # Set to max frames to process, or None for unlimited

# === MODULES ===
detector = Detector(DETECTION_MODEL_PATH)
tracker = TrackerWrapper()
risk_scorer = RiskScorer()
privacy_filter = PrivacyFilter()
incident_logger = IncidentLogger()

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"[FATAL ERROR] Could not open video source: {VIDEO_SOURCE}")
    exit(1)

frame_count = 0
fps_log = []
incident_count = 0
start_time = time.time()  # Session start time

try:
    # Set detection input size
    DETECT_W, DETECT_H = 320, 240

    while cap.isOpened():
        # Check for run duration or max frames
        elapsed = time.time() - start_time
        if (RUN_DURATION is not None and elapsed >= RUN_DURATION) or (MAX_FRAMES is not None and frame_count >= MAX_FRAMES):
            print("[INFO] Stopping: reached run duration or max frames.")
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Frame not received from video source. Exiting loop.")
            break

        frame_count += 1
        frame_start_time = time.time()  # Per-frame timer

        # Downscale frame for detection
        small_frame = cv2.resize(frame, (DETECT_W, DETECT_H))
        scale_x = frame.shape[1] / DETECT_W
        scale_y = frame.shape[0] / DETECT_H

        try:
            # Detection on small frame
            detections_small = detector.detect(small_frame)
            # Map detections back to original frame size
            detections = []
            for det in detections_small:
                x1, y1, x2, y2, conf, class_id = det[:6]
                # Scale coordinates
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                if len(det) > 6:
                    # If tracker ID is present
                    detections.append((x1, y1, x2, y2, conf, class_id, det[6]))
                else:
                    detections.append((x1, y1, x2, y2, conf, class_id))
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            detections = []


        try:
            # Tracking
            tracked_objects = tracker.update(frame, detections)
        except Exception as e:
            print(f"[ERROR] Tracking failed: {e}")
            tracked_objects = []

        try:
            # Risk scoring
            ttc = risk_scorer.compute_ttc(tracked_objects)
            headway = risk_scorer.compute_headway(tracked_objects)
            lane_dev = risk_scorer.lane_deviation(None, tracked_objects)
        except Exception as e:
            print(f"[ERROR] Risk scoring failed: {e}")
            ttc, headway, lane_dev = [], [], 0


        # --- Helmetless Riding Detection (India-specific) ---
        helmetless_ids = set()
        helmetless_explanations = []
        # Only check for helmetless on motorcycles (class_id==3)
        for obj in tracked_objects:
            x1, y1, x2, y2, conf, class_id, track_id = obj
            if class_id == 3:
                # Stub: Assume no helmet detected (replace with real helmet classifier)
                helmet_detected = False  # TODO: Integrate helmet detection model
                if not helmet_detected:
                    helmetless_ids.add(track_id)
                    helmetless_explanations.append(f"No helmet: Motorcycle ID {track_id}")

        # --- Animal Crossing Detection (India-specific) ---
        animal_ids = set()
        animal_explanations = []
        for obj in tracked_objects:
            x1, y1, x2, y2, conf, class_id, track_id = obj
            # COCO: animal=16 (adjust if your model uses different class IDs)
            if class_id == 16:
                animal_ids.add(track_id)
                animal_explanations.append(f"Animal crossing: ID {track_id}")

        # --- Pothole Detection (India-specific) ---
        pothole_ids = set()
        pothole_explanations = []
        # Only check for potholes (class_id==21, adjust if needed)
        for obj in tracked_objects:
            if len(obj) >= 7:
                x1, y1, x2, y2, conf, class_id, track_id = obj
                if class_id == 21:
                    pothole_ids.add(track_id)
                    pothole_explanations.append(f"Pothole detected: ID {track_id}")

        # Explainable AI Alerts: highlight risky objects and overlay explanation
        risky_ids = set()
        explanations = []
        h, w = frame.shape[:2]
        lane_x1, lane_x2 = int(w * 0.33), int(w * 0.66)
        min_height = int(h * 0.15)
        for obj, (track_id, ttc_val) in zip(tracked_objects, ttc):
            x1, y1, x2, y2, conf, class_id, track_id = obj
            cx = (x1 + x2) // 2
            bbox_h = y2 - y1
            # COCO: car=2, motorcycle=3, bus=5, truck=7
            if class_id in [2, 3, 5, 7] and lane_x1 < cx < lane_x2 and bbox_h > min_height and ttc_val < 3:
                risky_ids.add(track_id)
                explanations.append(f"Tailgating: Vehicle ahead, TTC={ttc_val:.1f}s (ID {track_id})")


        for obj in tracked_objects:
            x1, y1, x2, y2, conf, class_id, track_id = obj
            if track_id in risky_ids:
                # Highlight risky object with thick red box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            elif class_id == 3 and track_id in helmetless_ids:
                # Only highlight helmetless for motorcycles
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,140,255), 4)
                cv2.putText(frame, f'No Helmet ID:{track_id}', (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,140,255), 2)
            elif class_id == 16 and track_id in animal_ids:
                # Only highlight animal for animal class
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 4)
                cv2.putText(frame, f'Animal ID:{track_id}', (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            elif class_id == 21 and track_id in pothole_ids:
                # Only highlight pothole for pothole class
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128,0,128), 4)
                cv2.putText(frame, f'Pothole ID:{track_id}', (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,0,128), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Overlay textual explanations and trigger voice alerts for each risk event
        y_offset = 120
        for explanation in explanations:
            cv2.putText(frame, explanation, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            speak_alert(explanation)
            y_offset += 30
        # Overlay helmetless riding explanations
        for explanation in helmetless_explanations:
            cv2.putText(frame, explanation, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 2)
            speak_alert(explanation)
            y_offset += 30
        # Overlay animal crossing explanations
        for explanation in animal_explanations:
            cv2.putText(frame, explanation, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            speak_alert(explanation)
            y_offset += 30
        # Overlay pothole explanations
        for explanation in pothole_explanations:
            cv2.putText(frame, explanation, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128,0,128), 2)
            speak_alert(explanation)
            y_offset += 30

        # Dummy alert: if any TTC < 3s, show warning and log incident
        try:
            # Only consider vehicles (car, bus, truck, motorcycle) in the center-lane region and with minimum bbox size
            risky = False
            h, w = frame.shape[:2]
            lane_x1, lane_x2 = int(w * 0.33), int(w * 0.66)  # Center third of frame
            min_height = int(h * 0.15)  # Only consider objects with bbox height > 15% of frame
            for obj, (track_id, ttc_val) in zip(tracked_objects, ttc):
                x1, y1, x2, y2, conf, class_id, track_id = obj
                cx = (x1 + x2) // 2
                bbox_h = y2 - y1
                # COCO: car=2, motorcycle=3, bus=5, truck=7
                if class_id in [2, 3, 5, 7] and lane_x1 < cx < lane_x2 and bbox_h > min_height and ttc_val < 3:
                    risky = True
                    break
            if risky:
                cv2.putText(frame, 'WARNING: Risky Situation!', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                incident_logger.log_incident('Risky event detected', 'output/incident_{}.mp4'.format(int(time.time())))
                incident_count += 1
        except Exception as e:
            print(f"[ERROR] Incident logging failed: {e}")

        # Buffer frame for incident video
        try:
            incident_logger.buffer_frame(frame)
        except Exception as e:
            print(f"[ERROR] Buffering frame failed: {e}")

        # Privacy filter: run only every 10th frame for more speed
        try:
            if frame_count % 10 == 0:
                frame = privacy_filter.blur_sensitive(frame, detections, fast=True)
        except Exception as e:
            print(f"[ERROR] Privacy filter failed: {e}")

        # FPS calculation and logging
        frame_end_time = time.time()
        fps = 1.0 / (frame_end_time - frame_start_time + 1e-6)
        fps_log.append(fps)
        if frame_count % 30 == 0:
            avg_fps = sum(fps_log[-30:]) / min(len(fps_log), 30)
            print(f"[INFO] Average FPS (last 30 frames): {avg_fps:.2f}")

        # Show live stats overlay
        cv2.putText(frame, f"Incidents: {incident_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # Show frame
        cv2.imshow('Edge AI Dashcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt received. Exiting gracefully...")
except Exception as e:
    print(f"[FATAL ERROR] {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    # Print trip summary
    total_time = time.time() - start_time
    avg_fps = sum(fps_log) / max(len(fps_log), 1)
    print("\n=== Trip Summary ===")
    print(f"Total incidents: {incident_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Session duration: {total_time:.1f} seconds")
    avg_fps = sum(fps_log) / max(len(fps_log), 1)
    print("\n=== Trip Summary ===")
    print(f"Total incidents: {incident_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Session duration: {total_time:.1f} seconds")
