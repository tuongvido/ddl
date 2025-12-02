"""
Video Consumer: Process video frames and detect harmful content using CLIP model
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
import numpy as np

# Import configurations
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_VIDEO,
    USE_VIOLENCE_CLASSIFIER,
    VIOLENCE_CLASSIFIER_THRESHOLD,
    VIOLENCE_CLASSIFIER_FRAME_SKIP,
    USE_ALL_DETECTIONS_AS_HARMFUL,
    LOG_LEVEL,
)

# Import utilities
from utils import (
    decode_base64_to_image,
    calculate_alert_level,
    save_image_for_training,
    MongoDBHandler,
    AlertThrottler,
)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Dependency Check ---
try:
    import torch
    import clip
    from PIL import Image
    import yaml

    CLIP_AVAILABLE = True
except Exception as e:
    logger.warning(
        f"CLIP/torch dependencies missing: {e}. AI detection will be disabled."
    )
    CLIP_AVAILABLE = False


class VideoConsumer:
    """Consumer for processing video frames and detecting harmful content"""

    def __init__(self, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.kafka_servers = kafka_servers
        self.consumer = None
        self.db_handler = MongoDBHandler()
        # Cooldown 5s để tránh spam 100 cái alert trong 1 giây nếu video đánh nhau dài
        self.alert_throttler = AlertThrottler(cooldown_seconds=5)
        self.frame_count = 0

        # Biến chứa model
        self.clip_model = None
        self.clip_preprocess = None
        self.device = None
        self.text_features = None
        self.violence_labels = []
        self.violence_indices = []

        logger.info("Initializing VideoConsumer...")
        self.load_model()

    def load_model(self):
        """Load CLIP model and labels from YAML"""
        if not USE_VIOLENCE_CLASSIFIER:
            logger.info("🚫 Violence classifier disabled by config.")
            return

        if not CLIP_AVAILABLE:
            logger.error("❌ CLIP not available. Cannot load model.")
            return

        # 1. Load Labels from YAML
        settings_path = Path(__file__).parent.parent / "violence_settings.yaml"
        non_violence = []
        violence = []

        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = yaml.safe_load(f)

                label_settings = settings.get("label-settings", {})
                non_violence = label_settings.get("non-violence-labels", [])
                violence = label_settings.get("violence-labels", [])

                logger.info(
                    f"Loaded labels from YAML: {len(non_violence)} Safe, {len(violence)} Violence"
                )
            except Exception as e:
                logger.warning(f"Failed to read YAML: {e}. Using defaults.")

        # Fallback labels if YAML fails
        if not violence:
            non_violence = ["peaceful scene", "people walking", "people talking"]
            violence = ["fighting", "violence", "hitting", "weapons"]

        # IMPORTANT: Order matters!
        # Safe labels first, Violence labels second.
        self.violence_labels = non_violence + violence
        # Indices of violence labels start after the non-violence labels
        self.violence_indices = list(
            range(len(non_violence), len(self.violence_labels))
        )

        logger.info(f"Total Labels: {len(self.violence_labels)}")
        logger.info(f"Violence Class IDs: {self.violence_indices}")

        # 2. Load CLIP Model
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using AI Device: {self.device.upper()}")

            # ViT-B/32 is fast and good for real-time
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=self.device
            )

            # 3. Precompute Text Embeddings (Make "Search Keys")
            logger.info("Precomputing text embeddings...")

            # Prompt Engineering: "a photo of X" helps CLIP understand better
            text_descriptions = [
                f"a photo of {label}" for label in self.violence_labels
            ]
            text_tokens = clip.tokenize(text_descriptions).to(self.device)

            with torch.no_grad():
                self.text_features = self.clip_model.encode_text(text_tokens)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

            logger.info("✅ Model loaded and ready.")

        except Exception as e:
            logger.error(f"❌ Critical Error loading CLIP: {e}")
            self.clip_model = None

    def connect_kafka(self):
        """Connect to Kafka"""
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_VIDEO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",  # Chỉ đọc tin nhắn mới nhất (Real-time)
                enable_auto_commit=True,
                group_id="video-processing-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_poll_records=10,  # Không lấy quá nhiều 1 lúc tránh nghẽn
            )
            logger.info(f"✅ Connected to Kafka topic: {KAFKA_TOPIC_VIDEO}")
        except KafkaError as e:
            logger.error(f"❌ Kafka Connection Failed: {e}")
            raise

    def detect_objects(self, frame) -> List[Dict[str, Any]]:
        """
        Core AI Logic: Classify frame using CLIP
        """
        # Checks
        if not USE_VIOLENCE_CLASSIFIER or self.clip_model is None:
            return []

        # Frame Skip (Optimization)
        if self.frame_count % VIOLENCE_CLASSIFIER_FRAME_SKIP != 0:
            return []

        try:
            # 1. Preprocessing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            # 2. Inference (AI Thinking)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Similarity Calculation
                similarity = (image_features @ self.text_features.T).squeeze(0)
                probs = torch.nn.functional.softmax(similarity, dim=0).cpu().numpy()

            # 3. Analysis
            # Get the single highest probability label (Top 1)
            top_idx = int(probs.argmax())
            top_prob = float(probs[top_idx])
            top_label = self.violence_labels[top_idx]

            detections = []

            # --- LOGIC QUYẾT ĐỊNH (FIXED) ---

            # Case A: Phát hiện bạo lực
            if top_idx in self.violence_indices:
                if top_prob >= VIOLENCE_CLASSIFIER_THRESHOLD:
                    logger.warning(
                        f"🚨 VIOLENCE DETECTED: '{top_label}' ({top_prob:.1%})"
                    )
                    detections.append(
                        {
                            "class": top_label,
                            "class_id": top_idx,
                            "confidence": top_prob,
                            "bbox": None,  # CLIP is classification, not detection
                        }
                    )
                else:
                    # Có vẻ là bạo lực nhưng độ tin cậy thấp (dưới ngưỡng)
                    logger.debug(
                        f"Potential violence ignored (Low confidence): '{top_label}' ({top_prob:.1%})"
                    )

            # Case B: An toàn
            else:
                # Log debug để biết model đang nhìn thấy gì (Rất có ích để chỉnh file YAML)
                if self.frame_count % 30 == 0:  # Log mỗi 30 frame cho đỡ rác
                    logger.debug(f"Safe scene: '{top_label}' ({top_prob:.1%})")

            return detections

        except Exception as e:
            logger.error(f"Detection Error: {e}")
            return []

    def check_harmful_content(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Filter harmful detections.
        FIX: Removed string matching logic. Now purely relies on detect_objects result.
        """

        return {
            "is_harmful": len(detections) > 0,
            "harmful_detections": detections,
            "total_detections": len(detections),
            "harmful_count": len(detections),
        }

    def process_frame(self, message: Dict[str, Any]):
        """Process individual frame from Kafka"""
        try:
            frame_id = message.get("frame_id", -1)
            timestamp = message.get("timestamp", time.time())
            frame_data = message.get("data", "")

            if not frame_data:
                return

            frame = decode_base64_to_image(frame_data)
            if frame is None:
                return

            self.frame_count += 1

            # 1. Detect
            detections = self.detect_objects(frame)

            # 2. Check Harmful
            result = self.check_harmful_content(detections)

            # 3. Database & Alert
            if result["is_harmful"]:
                # Save to DB
                self.db_handler.save_detection(
                    {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "detections": result["harmful_detections"],
                        "is_harmful": True,
                    }
                )

                # Generate Alert
                self.generate_alert(frame_id, result, frame)

            # Progress Log
            if self.frame_count % 100 == 0:
                logger.info(f"Processed {self.frame_count} frames...")

        except Exception as e:
            logger.error(f"Frame Processing Error: {e}")

    def generate_alert(self, frame_id: int, harmful_result: Dict, frame):
        """Send alert if throttler allows"""
        try:
            for det in harmful_result["harmful_detections"]:
                det_type = det["class"]
                conf = det["confidence"]

                # Throttling Key: "video_fighting", "video_weapon", etc.
                alert_key = f"video_{det_type}"

                if self.alert_throttler.should_send_alert(alert_key):
                    level = calculate_alert_level(det_type, conf)

                    alert_data = {
                        "source": "video",
                        "frame_id": frame_id,
                        "detection_type": det_type,
                        "confidence": conf,
                        "level": level,
                        "details": f"Detected {det_type} ({conf:.1%})",
                    }

                    # 1. Save Alert DB
                    self.db_handler.save_alert(alert_data)

                    # 2. Save Evidence Image
                    save_image_for_training(frame, det_type)

                    logger.warning(f"⚠️  ALERT SENT: {det_type} - Frame {frame_id}")

        except Exception as e:
            logger.error(f"Alert Gen Error: {e}")

    def run(self):
        """Main Loop"""
        try:
            self.connect_kafka()
            logger.info("waiting for stream...")

            for message in self.consumer:
                self.process_frame(message.value)

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.consumer:
            self.consumer.close()
        if self.db_handler:
            self.db_handler.close()
        logger.info("Cleanup done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", type=str, default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    consumer = VideoConsumer(args.kafka)
    consumer.run()


if __name__ == "__main__":
    main()
