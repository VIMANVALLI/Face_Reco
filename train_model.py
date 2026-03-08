# train_model.py
import cv2
import os
import numpy as np
import json
from pathlib import Path

dataset_dir = Path('dataset')
model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

images = []
labels = []
label_map = {}
next_label = 0

# Loop through dataset folders
for person_dir in sorted(dataset_dir.iterdir()):
    if not person_dir.is_dir():
        continue
    name = person_dir.name
    if name not in label_map:
        label_map[name] = next_label
        next_label += 1
    for img_path in person_dir.glob('*.jpg'):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, (200, 200))
        images.append(img_resized)
        labels.append(label_map[name])

# Check if dataset is empty
if len(images) == 0:
    raise RuntimeError('No training images found in dataset/')

# Train LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.save(str(model_dir / 'trainer.yml'))

# Save labels mapping
with open(model_dir / 'labels.json', 'w') as f:
    json.dump({v: k for k, v in label_map.items()}, f)

print('[INFO] Training finished, model saved to models/trainer.yml')
