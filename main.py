import cv2
import mediapipe as mp
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
MODEL_PATH = Path("models/trainer.yml")
LABELS_PATH = Path("models/labels.json")
ATTENDANCE_FILE = Path("attendance.csv")

mp_face = mp.solutions.face_detection

# Load model and labels
if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    raise RuntimeError("Model or labels not found — train the model first")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

label_to_name = {int(k): v for k, v in labels.items()}

# Initialize attendance file
if not ATTENDANCE_FILE.exists():
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

already_marked_today = set()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("[INFO] Starting attendance system...")

CONFIDENCE_THRESHOLD = 60  # lower = stricter, change if needed

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w) - 10
                y1 = int(bbox.ymin * h) - 10
                x2 = int((bbox.xmin + bbox.width) * w) + 10
                y2 = int((bbox.ymin + bbox.height) * h) + 10
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (200, 200))

                label, conf = recognizer.predict(face_resized)

                # Apply confidence filter
                if conf < CONFIDENCE_THRESHOLD:
                    name = label_to_name.get(label, "Unknown")
                else:
                    name = "Unknown"

                # Draw results
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{name} ({int(conf)})"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Mark attendance
                if name != "Unknown":
                    today = datetime.now().date()
                    key = (name, str(today))
                    if key not in already_marked_today:
                        df = pd.read_csv(ATTENDANCE_FILE)
                        time_str = datetime.now().strftime("%H:%M:%S")
                        df = df._append({"Name": name,
                                         "Date": str(today),
                                         "Time": time_str},
                                        ignore_index=True)
                        df.to_csv(ATTENDANCE_FILE, index=False)
                        already_marked_today.add(key)
                        print(f"[INFO] Marked attendance for {name} at {time_str}")

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
