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
import cv2

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_VIDEO,
    VIOLENCE_CLASSIFIER_DIR,
    USE_VIOLENCE_CLASSIFIER,
    VIOLENCE_CLASSIFIER_THRESHOLD,
    VIOLENCE_CLASSIFIER_FRAME_SKIP,
    VIOLENCE_CLASSIFIER_BATCH_SIZE,
    USE_ALL_DETECTIONS_AS_HARMFUL,
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

# Try to import classifier dependencies
try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    import torch

    CLASSIFIER_AVAILABLE = True
except Exception:
    logger.warning(
        "Transformers/torch not available. Install with: pip install transformers torch safetensors"
    )
    CLASSIFIER_AVAILABLE = False


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

        # Load ViT classifier model (if enabled)
        self.load_model()

    def load_model(self):
        """Load ViT/frame-level violence classifier"""
        self.classifier = None
        self.feature_extractor = None
        self.id2label = {}

        if not USE_VIOLENCE_CLASSIFIER:
            logger.info("Violence classifier disabled by configuration")
            return

        if not CLASSIFIER_AVAILABLE:
            logger.error("Transformers/torch not available; cannot load classifier")
            return

        try:
            model_dir = Path(VIOLENCE_CLASSIFIER_DIR)

            # Support either a direct model directory or the huggingface_hub snapshot cache layout
            chosen_dir = None
            if model_dir.exists():
                # direct layout
                if (model_dir / "config.json").exists() or (
                    model_dir / "preprocessor_config.json"
                ).exists():
                    chosen_dir = model_dir
                else:
                    # look for nested cache folder like models--<owner>--<repo>/snapshots/<id>/
                    nested = list(model_dir.glob("models-*"))
                    for n in nested:
                        snaps = list(n.glob("snapshots/*"))
                        for s in snaps:
                            if (s / "config.json").exists() or (
                                s / "preprocessor_config.json"
                            ).exists():
                                chosen_dir = s
                                break
                        if chosen_dir:
                            break
                    # fallback: search deeper
                    if chosen_dir is None:
                        snaps = list(model_dir.rglob("snapshots/*"))
                        for s in snaps:
                            if (s / "config.json").exists() or (
                                s / "preprocessor_config.json"
                            ).exists():
                                chosen_dir = s
                                break

            if chosen_dir is None:
                logger.warning(
                    f"Violence classifier directory not found or missing config: {model_dir}"
                )
                logger.warning(
                    "Place the Hugging Face model files in this folder (model.safetensors, config.json, preprocessor_config.json) or run snapshot_download to fetch the repo"
                )
                return

            model_dir = chosen_dir

            logger.info(f"Loading violence classifier from {model_dir}")
            # Some repos use preprocessor_config.json / AutoImageProcessor instead of feature extractor
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    str(model_dir)
                )
            except Exception:
                from transformers import AutoImageProcessor

                self.feature_extractor = AutoImageProcessor.from_pretrained(
                    str(model_dir)
                )

            self.classifier = AutoModelForImageClassification.from_pretrained(
                str(model_dir), local_files_only=True
            )

            # Move model to GPU if available
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.classifier.to(self.device)

            # Setup id2label mapping
            if (
                hasattr(self.classifier.config, "id2label")
                and self.classifier.config.id2label
            ):
                # Some checkpoints store generic labels like 'LABEL_0', 'LABEL_1'.
                # Normalize to meaningful labels when possible.
                raw = self.classifier.config.id2label
                vals = set([str(v).upper() for v in raw.values()])
                if any(v.startswith("LABEL_") for v in vals):
                    # Assume convention: 0 = non-violence, 1 = violence
                    self.id2label = {0: "non-violence", 1: "violence"}
                else:
                    self.id2label = raw
            else:
                # default mapping assumption
                self.id2label = {0: "non-violence", 1: "violence"}

            logger.info("Violence classifier loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load violence classifier: {e}")
            self.classifier = None

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
        Run frame-level violence classification and return detections list.

        Args:
            frame: OpenCV BGR frame (numpy array)

        Returns:
            List of detections in the form [{"class": str, "confidence": float, "class_id": int}]
        """
        # If classifier not in use or not loaded, return empty
        if (
            not USE_VIOLENCE_CLASSIFIER
            or not self.classifier
            or not self.feature_extractor
        ):
            return []

        try:
            # Frame skip logic
            if self.frame_count % VIOLENCE_CLASSIFIER_FRAME_SKIP != 0:
                return []

            # Convert BGR (OpenCV) to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Prepare inputs
            inputs = self.feature_extractor(images=rgb, return_tensors="pt")

            # Move tensors to device
            import torch

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probs = (
                    torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                )

            # Build top-k label details
            top_k = min(3, len(probs))
            top_indices = probs.argsort()[::-1][:top_k]
            label_details = []
            for idx in top_indices:
                lbl = self.id2label.get(int(idx), str(idx))
                prob = float(probs[int(idx)])
                label_details.append({"label": str(lbl), "prob": prob, "id": int(idx)})

            # Determine if any top label indicates violence.
            # Prefer checking class_id (1 == violence) when available to avoid substring issues
            is_violation = False
            violence_best = None
            # First, look for positive-class id (commonly 1) with sufficient probability
            for item in label_details:
                if (
                    int(item.get("id", -1)) == 1
                    and item["prob"] >= VIOLENCE_CLASSIFIER_THRESHOLD
                ):
                    is_violation = True
                    violence_best = item
                    break

            # Fallback: check explicit label equality (avoid substring matches like 'non-violence')
            if not is_violation:
                for item in label_details:
                    lbl = item["label"].lower().strip()
                    # treat exact 'violence' or 'violent' as positive
                    if (
                        lbl in ("violence", "violent")
                        and item["prob"] >= VIOLENCE_CLASSIFIER_THRESHOLD
                    ):
                        is_violation = True
                        violence_best = item
                        break

            detections = []
            if is_violation:
                detections.append(
                    {
                        "bbox": None,
                        "confidence": violence_best["prob"],
                        "class": violence_best["label"],
                        "class_id": violence_best["id"],
                        "label_details": label_details,
                    }
                )

            return detections

        except Exception as e:
            logger.error(f"Classifier detection error: {e}")
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

        # Demo mode: treat ALL detections as harmful for testing
        if USE_ALL_DETECTIONS_AS_HARMFUL:
            harmful_detections = detections
        else:
            # If using the ViT classifier, any returned 'violence' detection is harmful
            if USE_VIOLENCE_CLASSIFIER:
                for det in detections:
                    # classifier returns class like 'violence' when positive
                    class_name = str(det.get("class", "")).lower()
                    if "violence" in class_name:
                        harmful_detections.append(det)
            else:
                # Fallback: if detections include objects matching HARMFUL_CLASSES
                for det in detections:
                    class_name = det.get("class", "").lower()
                    # Check if object class is in harmful list
                    for harmful_class in []:
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
                    "bbox": det.get("bbox"),
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
