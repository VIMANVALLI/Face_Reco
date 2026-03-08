# build_db.py
import cv2
import mediapipe as mp
import os
import argparse
from pathlib import Path

mp_face = mp.solutions.face_detection

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='Person name (use no spaces)')
parser.add_argument('--count', type=int, default=20, help='Number of images to capture')
parser.add_argument('--output', default='dataset', help='Output dataset folder')
parser.add_argument('--headless', action='store_true', help='Run without imshow (for headless OpenCV)')
args = parser.parse_args()

name = args.name
count = args.count
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
person_dir = output / name
person_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Could not open webcam')

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
    taken = 0
    print('Capturing faces... Press SPACE (GUI mode) or will auto-capture in headless mode')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(img_rgb)

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

                fname = person_dir / f"img_{taken+1}.jpg"
                cv2.imwrite(str(fname), face_resized)
                taken += 1
                print(f"[INFO] Saved {fname}")

                if taken >= count:
                    print("[INFO] Finished capturing.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)

        if not args.headless:
            cv2.imshow('Capture - Press SPACE', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE
                continue  # faces are auto-saved above

cap.release()
cv2.destroyAllWindows()
