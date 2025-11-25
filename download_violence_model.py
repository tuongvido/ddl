"""
Download pre-trained violence detection model
This script downloads a YOLO model trained on violence/weapon detection dataset
"""

import os
from pathlib import Path
from ultralytics import YOLO
import requests

# Model options:
# 1. Use Roboflow violence detection model
# 2. Use custom trained model from Hugging Face
# 3. Use open dataset and train our own


def download_from_url(url: str, save_path: Path):
    """Download file from URL"""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✓ Downloaded to {save_path}")


def download_violence_model():
    """Download a pre-trained violence detection model"""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("\n=== Violence Detection Model Setup ===\n")
    print("Options:")
    print("1. Download YOLOv8 fine-tuned for violence detection (Recommended)")
    print("2. Use custom Roboflow model (requires API key)")
    print("3. Use weapon detection model")
    print("4. Skip and use demo mode (all detections as harmful)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        # Option 1: Download from a public violence detection model
        print("\n📥 Downloading violence detection model...")
        print("Note: This will download from a public repository")

        # Example models (you may need to find actual URLs):
        model_urls = {
            "violence": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            # TODO: Replace with actual violence detection model URL
            # Example: https://huggingface.co/username/violence-detection/resolve/main/best.pt
        }

        print("\n⚠️  Note: Default YOLO model doesn't detect violence.")
        print("You need to either:")
        print("   - Train your own model on violence dataset")
        print("   - Use Roboflow API to download their violence model")
        print("   - Find a pre-trained model on Hugging Face")

        print("\n💡 Recommended: Use Roboflow Universe models")
        print("   Visit: https://universe.roboflow.com/")
        print("   Search for: 'violence detection' or 'weapon detection'")

    elif choice == "2":
        # Option 2: Roboflow
        print("\n🔑 Roboflow Setup:")
        print("1. Go to: https://universe.roboflow.com/")
        print("2. Search for 'violence detection' datasets")
        print("3. Get your API key and model version")

        workspace = input("Enter Roboflow workspace: ").strip()
        project = input("Enter project name: ").strip()
        version = input("Enter version (e.g., 1): ").strip()
        api_key = input("Enter API key: ").strip()

        if workspace and project and version and api_key:
            from roboflow import Roboflow

            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download("yolov8")
            print(f"✓ Downloaded to {dataset.location}")

    elif choice == "3":
        # Option 3: Weapon detection (guns, knives, etc.)
        print("\n🔫 Weapon Detection Model")
        print("This can detect weapons which are indicators of violence")
        print("\nNote: You'll need to find a weapon detection model")

    else:
        # Option 4: Demo mode
        print("\n✓ Using demo mode with USE_ALL_DETECTIONS_AS_HARMFUL = True")
        print("All detections will be flagged as harmful for testing")

    print("\n" + "=" * 50)
    print("MANUAL SETUP INSTRUCTIONS:")
    print("=" * 50)
    print("\n1. Train your own model:")
    print("   - Use the Colab notebook: notebooks/Colab_Training_Server.ipynb")
    print(
        "   - Upload violence dataset (search 'violence detection dataset' on Roboflow)"
    )
    print("   - Train YOLOv8 model")
    print("   - Download trained weights to models/violence_detection.pt")

    print("\n2. Use Roboflow pre-trained model:")
    print("   - Visit: https://universe.roboflow.com/")
    print("   - Search: 'violence detection' or 'fight detection'")
    print("   - Popular datasets:")
    print(
        "     * Violence Detection: https://universe.roboflow.com/vd-hz6jg/violence-detection-o4rdr"
    )
    print(
        "     * Weapon Detection: https://universe.roboflow.com/weapons-uw60y/weapons-lidce"
    )
    print("   - Export as YOLOv8 format")
    print("   - Place weights in models/ directory")

    print("\n3. Update config.py:")
    print("   - Set YOLO_MODEL_PATH to your trained model")
    print("   - Update HARMFUL_CLASSES to match model classes")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    try:
        download_violence_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease follow manual instructions above")
