import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    AutoImageProcessor,
)

# find model dir (handles nested snapshot)
base = Path("models") / "vit-base-violence-detection"
model_dir = base
if (base / "snapshots").exists():
    snaps = list((base / "snapshots").glob("*"))
    if snaps:
        model_dir = snaps[0]
else:
    nested = list(base.glob("models--*"))
    if nested:
        first = nested[0]
        if (first / "snapshots").exists():
            snaps = list((first / "snapshots").glob("*"))
            if snaps:
                model_dir = snaps[0]
        else:
            model_dir = first

print("Using model dir:", model_dir)

# load
fe = None
try:
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
except Exception:
    fe = AutoImageProcessor.from_pretrained(model_dir)

model = AutoModelForImageClassification.from_pretrained(model_dir)
model.eval()

# open video and grab a frame
video_path = Path("data") / "v001_converted.avi"
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

# read a middle frame (10th frame or frame 20)
frame_id = 10
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ret, frame = cap.read()
if not ret:
    # fallback to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read frame")

# convert BGR->RGB and to PIL
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(frame_rgb)

# preprocess
inputs = fe(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

print("logits:", logits.squeeze().cpu().numpy())
print("probs:", probs)
print("model.config.id2label:", getattr(model.config, "id2label", {}))

cap.release()
