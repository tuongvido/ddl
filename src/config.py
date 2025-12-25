"""
Configuration file for the Harmful Content Detection System
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SCAN_INTERVAL = 5  # seconds
SUPPORTED_EXTS = {".mp4", ".avi", ".mkv"}

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_VIDEO = "livestream-video"
KAFKA_TOPIC_AUDIO = "livestream-audio"
KAFKA_TOPIC_ALERTS = "livestream-alerts"

# Video Processing
VIDEO_FPS = 25
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 640
VIDEO_SAMPLE_RATE = 0.04  # 1/FPS seconds

# Audio Processing
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 3  # seconds
AUDIO_FORMAT = "wav"

# Vision Transformer (ViT) classifier for frame-level violence detection
# This model is a classifier (violence / non-violence). It does NOT return bounding boxes.
# Place the Hugging Face repo or extracted files in: models/vit-base-violence-detection/
VIOLENCE_CLASSIFIER_DIR = str(MODELS_DIR / "vit-base-violence-detection")
# Enable the ViT classifier. If True the consumer will run the classifier on frames
USE_VIOLENCE_CLASSIFIER = True
# Softmax probability threshold for the 'violence' class (0..1)
# Lower threshold = more sensitive detection (more false positives)
# Higher threshold = less sensitive detection (may miss some violence)
VIOLENCE_CLASSIFIER_THRESHOLD = 0.2  # Lowered to 0.2 for boxing/fighting detection
# How many frames to skip between classifier inferences (1 = every frame)
VIOLENCE_CLASSIFIER_FRAME_SKIP = 1  # Changed from 3 to process more frames
# Batch size for classifier inference (if using batched evaluation)
VIOLENCE_CLASSIFIER_BATCH_SIZE = 8

# Harmful object classes (customize based on your model)
# For YOLOv8n default model (COCO dataset), use these real classes:
HARMFUL_CLASSES = [
    "knife",
    "scissors",
    "person",  # Can detect person for violence context
    # Note: Default YOLOv8 doesn't detect violence directly
    # You need a custom-trained model for violence detection
]

# Alternative: Use ALL detections for demo purposes
# Set to False for real inference; True will mark every detection as harmful (testing only)
USE_ALL_DETECTIONS_AS_HARMFUL = False  # TEMPORARILY SET TO TRUE FOR TESTING

# Whisper model configuration
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Toxic words/phrases (Vietnamese and English)
TOXIC_KEYWORDS = [
    # Vietnamese toxic words
    "đồ chó",
    "con chó",
    "đồ ngu",
    "ngu ngốc",
    "khốn nạn",
    "đồ khốn",
    "mất dạy",
    "thằng ngu",
    "con ngu",
    "đồ điên",
    "thằng điên",
    "đồ khùng",
    "vô học",
    "ngu dốt",
    "súc vật",
    "đồ súc sinh",
    "đồ phản bội",
    "đồ phá hoại",
    "đồ lừa đảo",
    "đồ khốn kiếp",
    "chết tiệt",
    "đồ chết",
    "đi chết",
    "bố mày",
    "mẹ mày",
    "cút đi",
    "cút xéo",
    "đồ rác",
    "phế vật",
    "thất bại",
    "thất học",
    "vô dụng",
    # English toxic words
    "fuck",
    "shit",
    "damn",
    "bitch",
    "bastard",
    "asshole",
    "idiot",
    "stupid",
    "moron",
    "dumb",
    "loser",
    "jerk",
    "crap",
    "hell",
    "dickhead",
    "piss",
    "scum",
    "trash",
    "garbage",
    "worthless",
]

# MongoDB Configuration
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "admin123")
MONGO_DB = "livestream_detection"
MONGO_COLLECTION_DETECTIONS = "video_detections"
MONGO_COLLECTION_ALERTS = "alerts"
MONGO_COLLECTION_VIDEO_SUMMARY = "video_summary"

# Alert Configuration
ALERT_COOLDOWN = 5  # seconds between alerts for same type
ALERT_TYPES = {
    "HIGH": {"level": 3, "color": "red"},
    "MEDIUM": {"level": 2, "color": "orange"},
    "LOW": {"level": 1, "color": "yellow"},
}

# Google Colab Training Configuration
COLAB_API_URL = os.getenv("COLAB_API_URL", "http://localhost:8000")
COLAB_TRAINING_ENDPOINT = "/train"
COLAB_STATUS_ENDPOINT = "/status"
GDRIVE_DATASET_PATH = "/content/drive/MyDrive/harmful_detection_dataset"

# Airflow Configuration
AIRFLOW_RETRAIN_SCHEDULE = "@daily"  # Can be: @daily, @weekly, cron expression
MIN_NEW_SAMPLES_FOR_RETRAIN = 100

# Dashboard Configuration
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_INTERVAL = 1  # seconds

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
