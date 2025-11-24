"""
Video Consumer: Process video frames and detect harmful content using YOLOv8
"""

import logging
import argparse
import time
from pathlib import Path
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
from typing import Dict, List, Any

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_VIDEO,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    HARMFUL_CLASSES,
    LOG_LEVEL,
)
from utils import (
    decode_base64_to_image,
    calculate_alert_level,
    save_image_for_training,
    MongoDBHandler,
    AlertThrottler,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Try to import YOLOv8
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("YOLOv8 not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


class VideoConsumer:
    """Consumer for processing video frames and detecting harmful content"""

    def __init__(self, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        """
        Initialize video consumer

        Args:
            kafka_servers: Kafka bootstrap servers
        """
        self.kafka_servers = kafka_servers
        self.consumer = None
        self.model = None
        self.db_handler = MongoDBHandler()
        self.alert_throttler = AlertThrottler(cooldown_seconds=5)
        self.frame_count = 0

        logger.info("Initializing VideoConsumer")

        # Load YOLO model
        self.load_model()

    def load_model(self):
        """Load YOLOv8 model"""
        if not YOLO_AVAILABLE:
            logger.error("YOLOv8 not available, cannot load model")
            return

        try:
            model_path = Path(YOLO_MODEL_PATH)

            if model_path.exists():
                logger.info(f"Loading custom model from {YOLO_MODEL_PATH}")
                self.model = YOLO(YOLO_MODEL_PATH)
            else:
                logger.warning(f"Custom model not found at {YOLO_MODEL_PATH}")
                logger.info("Loading pre-trained YOLOv8 model (yolov8n.pt)")
                self.model = YOLO("yolov8n.pt")  # Use nano model as default

            logger.info("YOLO model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def connect_kafka(self):
        """Connect to Kafka broker"""
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_VIDEO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="video-processing-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_poll_records=10,
                session_timeout_ms=30000,
            )
            logger.info(f"Connected to Kafka at {self.kafka_servers}")
            logger.info(f"Subscribed to topic: {KAFKA_TOPIC_VIDEO}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def detect_objects(self, frame) -> List[Dict[str, Any]]:
        """
        Detect objects in frame using YOLO

        Args:
            frame: OpenCV frame

        Returns:
            List of detections
        """
        if not self.model or not YOLO_AVAILABLE:
            return []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=YOLO_CONFIDENCE_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                verbose=False,
            )

            detections = []

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    detection = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class": class_name,
                        "class_id": class_id,
                    }

                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def check_harmful_content(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Check if detected objects are harmful

        Args:
            detections: List of detected objects

        Returns:
            Dictionary with harmful content information
        """
        harmful_detections = []

        for det in detections:
            class_name = det["class"].lower()

            # Check if object class is in harmful list
            for harmful_class in HARMFUL_CLASSES:
                if harmful_class.lower() in class_name:
                    harmful_detections.append(det)
                    break

        is_harmful = len(harmful_detections) > 0

        result = {
            "is_harmful": is_harmful,
            "harmful_detections": harmful_detections,
            "total_detections": len(detections),
            "harmful_count": len(harmful_detections),
        }

        return result

    def process_frame(self, message: Dict[str, Any]):
        """
        Process a video frame message

        Args:
            message: Kafka message containing frame data
        """
        try:
            frame_id = message.get("frame_id", -1)
            timestamp = message.get("timestamp", time.time())
            frame_data = message.get("data", "")

            if not frame_data:
                logger.warning(f"Empty frame data for frame {frame_id}")
                return

            # Decode frame
            frame = decode_base64_to_image(frame_data)

            if frame is None:
                logger.warning(f"Failed to decode frame {frame_id}")
                return

            # Detect objects
            detections = self.detect_objects(frame)

            # Check for harmful content
            harmful_result = self.check_harmful_content(detections)

            # Save detection result to database
            detection_record = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "total_detections": harmful_result["total_detections"],
                "harmful_count": harmful_result["harmful_count"],
                "is_harmful": harmful_result["is_harmful"],
                "detections": detections,
                "harmful_detections": harmful_result["harmful_detections"],
            }

            self.db_handler.save_detection(detection_record)

            # If harmful content detected, generate alert
            if harmful_result["is_harmful"]:
                self.generate_alert(frame_id, harmful_result, frame)

            self.frame_count += 1

            if self.frame_count % 50 == 0:
                logger.info(f"Processed {self.frame_count} frames")

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def generate_alert(self, frame_id: int, harmful_result: Dict, frame):
        """
        Generate alert for harmful content

        Args:
            frame_id: Frame ID
            harmful_result: Harmful content detection result
            frame: OpenCV frame
        """
        try:
            harmful_detections = harmful_result["harmful_detections"]

            for det in harmful_detections:
                detection_type = det["class"]
                confidence = det["confidence"]

                # Calculate alert level
                alert_level = calculate_alert_level(detection_type, confidence)

                # Check if we should send alert (throttling)
                alert_key = f"video_{detection_type}"

                if not self.alert_throttler.should_send_alert(alert_key):
                    logger.debug(f"Alert throttled for {detection_type}")
                    continue

                # Create alert
                alert_data = {
                    "source": "video",
                    "frame_id": frame_id,
                    "detection_type": detection_type,
                    "confidence": confidence,
                    "level": alert_level,
                    "bbox": det["bbox"],
                    "details": f"Detected {detection_type} with confidence {confidence:.2%}",
                }

                # Save alert to database
                self.db_handler.save_alert(alert_data)

                # Save frame for training
                save_image_for_training(frame, detection_type)

                logger.warning(
                    f"⚠️ ALERT [{alert_level}]: Detected {detection_type} "
                    f"(confidence: {confidence:.2%}) in frame {frame_id}"
                )

        except Exception as e:
            logger.error(f"Error generating alert: {e}")

    def run(self):
        """Run the consumer"""
        try:
            self.connect_kafka()

            logger.info("Starting video consumer...")
            logger.info("Waiting for messages...")

            for message in self.consumer:
                self.process_frame(message.value)

        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")

        if self.db_handler:
            self.db_handler.close()
            logger.info("Database connection closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Video Consumer")
    parser.add_argument(
        "--kafka",
        type=str,
        default=KAFKA_BOOTSTRAP_SERVERS,
        help="Kafka bootstrap servers",
    )

    args = parser.parse_args()

    consumer = VideoConsumer(args.kafka)
    consumer.run()


if __name__ == "__main__":
    main()
