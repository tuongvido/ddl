# Violence Detection Model Setup Guide

## Problem

The default YOLOv8n model is trained on COCO dataset (80 classes: person, car, dog, etc.) and **cannot detect violence, fights, or weapons**.

## Solutions

### Option 1: Use Roboflow Pre-trained Models (Recommended - Fastest)

1. **Visit Roboflow Universe**

   - Go to: https://universe.roboflow.com/
   - Search for: "violence detection" or "weapon detection"

2. **Popular Violence Detection Datasets:**

   - Violence Detection: https://universe.roboflow.com/vd-hz6jg/violence-detection-o4rdr
   - Weapon Detection: https://universe.roboflow.com/weapons-uw60y/weapons-lidce
   - Fight Detection: https://universe.roboflow.com/computer-vision-mpfyw/fight-detection-uicvm

3. **Download the Model:**

   ```bash
   pip install roboflow
   ```

   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("WORKSPACE").project("PROJECT_NAME")
   dataset = project.version(1).download("yolov8")
   ```

4. **Use the trained weights:**
   - Copy `runs/detect/train/weights/best.pt` to `models/violence_detection.pt`

### Option 2: Train Your Own Model (Best Accuracy)

1. **Prepare Dataset:**

   - Collect violence/fight videos
   - Extract frames and label them
   - Use Roboflow to label: https://roboflow.com/
   - Export as YOLOv8 format

2. **Train in Google Colab:**

   - Open: `notebooks/Colab_Training_Server.ipynb`
   - Upload your dataset
   - Train YOLOv8 model
   - Download `best.pt` weights

3. **Place trained model:**
   ```bash
   # Copy trained weights
   cp best.pt models/violence_detection.pt
   ```

### Option 3: Use Pre-trained Weapon Detection Model

If you can't find violence detection, use weapon detection as a proxy:

1. **Download weapon detection model:**

   - Search "weapon detection yolov8" on GitHub
   - Example: https://github.com/ultralytics/ultralytics (weapons examples)

2. **Weapons can indicate violence:**
   - Classes: gun, knife, rifle, pistol, sword, axe

### Option 4: Alternative Approach - Action Recognition

Instead of object detection, use action recognition:

1. **Use Pose Estimation:**

   - YOLOv8-pose can detect human poses
   - Analyze poses to detect fighting movements

2. **Use Video Classification:**
   - Train a video classifier (3D CNN, LSTM)
   - Classify video clips as violent/non-violent

## Quick Implementation

### Step 1: Download a Model

**Using Roboflow (Example):**

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("vd-hz6jg").project("violence-detection-o4rdr")
dataset = project.version(2).download("yolov8")

# The trained model will be in:
# {dataset.location}/train/weights/best.pt
```

### Step 2: Update config.py

```python
# Option A: Use custom violence model
YOLO_MODEL_PATH = MODELS_DIR / "violence_detection.pt"

# Option B: Use weapon detection as proxy
YOLO_MODEL_PATH = MODELS_DIR / "weapon_detection.pt"

# Update classes to match your model
HARMFUL_CLASSES = [
    "violence",
    "fight",
    "punch",
    "kick",
    "weapon",
    "gun",
    "knife",
]
```

### Step 3: Verify Model Classes

```python
from ultralytics import YOLO

model = YOLO("models/violence_detection.pt")
print("Model classes:", model.names)
# Output: {0: 'violence', 1: 'fight', 2: 'normal'}
```

### Step 4: Test Detection

```bash
# Restart consumer
python src/consumer_video.py

# Run producer
python src/producer.py --video data/v001_converted.avi
```

## Current Temporary Solution

For now, we've enabled demo mode in `config.py`:

```python
USE_ALL_DETECTIONS_AS_HARMFUL = True
```

This treats ALL detections (person, car, etc.) as harmful for testing purposes.

**⚠️ This is NOT production-ready!** You must replace with a proper violence detection model.

## Recommended Next Steps

1. **Create Roboflow account** (free)
2. **Search and fork** a violence detection dataset
3. **Train YOLOv8** on that dataset (use Colab notebook)
4. **Download trained weights** to `models/violence_detection.pt`
5. **Update config.py** with new model path and classes
6. **Test** with v001_converted.avi violence video
7. **Set** `USE_ALL_DETECTIONS_AS_HARMFUL = False`

## Resources

- Roboflow Universe: https://universe.roboflow.com/
- YOLOv8 Training: https://docs.ultralytics.com/modes/train/
- Violence Detection Papers: https://paperswithcode.com/task/violence-detection
- Dataset Sources:
  - RWF-2000 (Real World Fight Detection)
  - UCF-Crime Dataset
  - ViF (Violence in Films) Dataset
