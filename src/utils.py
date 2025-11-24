"""
Utility functions for the Harmful Content Detection System
"""

import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pymongo import MongoClient
from config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_USERNAME,
    MONGO_PASSWORD,
    MONGO_DB,
    MONGO_COLLECTION_DETECTIONS,
    MONGO_COLLECTION_ALERTS,
    LOG_LEVEL,
    LOG_FORMAT,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class MongoDBHandler:
    """Handle MongoDB connections and operations"""

    def __init__(self):
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        """Connect to MongoDB"""
        try:
            connection_string = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
            self.client = MongoClient(connection_string)
            self.db = self.client[MONGO_DB]
            logger.info(f"Connected to MongoDB at {MONGO_HOST}:{MONGO_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def save_detection(self, detection_data: Dict[str, Any]):
        """Save detection result to database"""
        try:
            detection_data["timestamp"] = datetime.now()
            result = self.db[MONGO_COLLECTION_DETECTIONS].insert_one(detection_data)
            logger.debug(f"Saved detection with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save detection: {e}")
            return None

    def save_alert(self, alert_data: Dict[str, Any]):
        """Save alert to database"""
        try:
            alert_data["timestamp"] = datetime.now()
            result = self.db[MONGO_COLLECTION_ALERTS].insert_one(alert_data)
            logger.info(f"Saved alert: {alert_data.get('type', 'UNKNOWN')}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
            return None

    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """Get recent detections"""
        try:
            detections = (
                self.db[MONGO_COLLECTION_DETECTIONS]
                .find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(detections)
        except Exception as e:
            logger.error(f"Failed to get detections: {e}")
            return []

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        try:
            alerts = (
                self.db[MONGO_COLLECTION_ALERTS]
                .find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(alerts)
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def encode_image_to_base64(frame: np.ndarray) -> str:
    """
    Encode OpenCV frame to base64 string

    Args:
        frame: OpenCV image frame (numpy array)

    Returns:
        Base64 encoded string
    """
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        return jpg_as_text
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""


def decode_base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Decode base64 string to OpenCV frame

    Args:
        base64_string: Base64 encoded image string

    Returns:
        OpenCV image frame (numpy array) or None
    """
    try:
        jpg_original = base64.b64decode(base64_string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame

    Args:
        frame: OpenCV image frame
        detections: List of detection dictionaries with bbox, class, confidence

    Returns:
        Frame with drawn detections
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class"]
        conf = det["confidence"]

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Draw label
        label_text = f"{label}: {conf:.2f}"
        cv2.putText(
            frame,
            label_text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    return frame


def check_toxic_content(text: str, toxic_keywords: List[str]) -> Dict[str, Any]:
    """
    Check if text contains toxic keywords

    Args:
        text: Text to check
        toxic_keywords: List of toxic keywords

    Returns:
        Dictionary with is_toxic flag and matched keywords
    """
    text_lower = text.lower()
    matched_keywords = []

    for keyword in toxic_keywords:
        if keyword.lower() in text_lower:
            matched_keywords.append(keyword)

    return {
        "is_toxic": len(matched_keywords) > 0,
        "matched_keywords": matched_keywords,
        "toxic_score": len(matched_keywords),
    }


def calculate_alert_level(
    detection_type: str, confidence: float, toxic_score: int = 0
) -> str:
    """
    Calculate alert level based on detection type and confidence

    Args:
        detection_type: Type of detection (weapon, violence, toxic_speech)
        confidence: Confidence score
        toxic_score: Number of toxic keywords found

    Returns:
        Alert level: HIGH, MEDIUM, or LOW
    """
    if detection_type == "weapon" and confidence > 0.8:
        return "HIGH"
    elif detection_type == "violence" and confidence > 0.7:
        return "HIGH"
    elif toxic_score >= 3:
        return "HIGH"
    elif confidence > 0.6 or toxic_score >= 2:
        return "MEDIUM"
    else:
        return "LOW"


def save_image_for_training(
    frame: np.ndarray, detection_type: str, save_dir: str = "data/training_samples"
) -> str:
    """
    Save detected frame for future model training

    Args:
        frame: OpenCV image frame
        detection_type: Type of detection
        save_dir: Directory to save image

    Returns:
        Path to saved image
    """
    import os
    from pathlib import Path

    # Create directory if not exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{detection_type}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)

    # Save image
    cv2.imwrite(filepath, frame)
    logger.info(f"Saved training sample: {filepath}")

    return filepath


def format_alert_message(alert_data: Dict[str, Any]) -> str:
    """
    Format alert data into readable message

    Args:
        alert_data: Alert dictionary

    Returns:
        Formatted alert message
    """
    level = alert_data.get("level", "UNKNOWN")
    detection_type = alert_data.get("detection_type", "unknown")
    confidence = alert_data.get("confidence", 0)
    timestamp = alert_data.get("timestamp", datetime.now())

    message = f"""
🚨 ALERT [{level}] 🚨
Type: {detection_type}
Confidence: {confidence:.2%}
Time: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}
    """.strip()

    if "details" in alert_data:
        message += f"\nDetails: {alert_data['details']}"

    return message


class AlertThrottler:
    """Prevent alert spam by throttling similar alerts"""

    def __init__(self, cooldown_seconds: int = 5):
        self.cooldown = cooldown_seconds
        self.last_alerts = {}

    def should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        current_time = datetime.now()

        if alert_type in self.last_alerts:
            time_diff = (current_time - self.last_alerts[alert_type]).total_seconds()
            if time_diff < self.cooldown:
                return False

        self.last_alerts[alert_type] = current_time
        return True


def create_kafka_message(message_type: str, data: Dict[str, Any]) -> bytes:
    """
    Create standardized Kafka message

    Args:
        message_type: Type of message (video, audio, alert)
        data: Message data

    Returns:
        JSON encoded bytes
    """
    message = {
        "type": message_type,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }
    return json.dumps(message).encode("utf-8")


def parse_kafka_message(message_bytes: bytes) -> Dict[str, Any]:
    """
    Parse Kafka message

    Args:
        message_bytes: Raw message bytes

    Returns:
        Parsed message dictionary
    """
    try:
        return json.loads(message_bytes.decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to parse Kafka message: {e}")
        return {}
