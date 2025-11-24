"""
Configuration file for the Harmful Content Detection System
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

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

# Model Configuration
YOLO_MODEL_PATH = str(MODELS_DIR / "yolo_best.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.6
YOLO_IOU_THRESHOLD = 0.45

# Harmful object classes (customize based on your model)
HARMFUL_CLASSES = [
    "knife",
    "gun",
    "weapon",
    "blood",
    "violence",
    "fight",
    "pistol",
    "rifle",
    "sword",
    "axe",
    "explosive",
]

# Whisper model configuration
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Toxic words/phrases (Vietnamese and English)
TOXIC_KEYWORDS = [
    # Vietnamese
    "đồ ngu",
    "chết đi",
    "mẹ mày",
    "địt",
    "lồn",
    "cặc",
    "đéo",
    "vãi",
    "ngu người",
    "chửi bới",
    "thù ghét",
    # English
    "kill",
    "die",
    "hate",
    "violence",
    "fuck",
    "shit",
    "weapon",
    "terrorist",
    "bomb",
    "attack",
]

# MongoDB Configuration
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "admin123")
MONGO_DB = "livestream_detection"
MONGO_COLLECTION_DETECTIONS = "detections"
MONGO_COLLECTION_ALERTS = "alerts"

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
