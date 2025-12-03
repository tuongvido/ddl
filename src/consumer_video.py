"""
Video Consumer: Process video frames and detect harmful content using CLIP model
UPDATED VERSION: Better Logging + Improved Detection Logic
"""

import logging
import argparse
import time
import sys
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

# --- 1. CẤU HÌNH LOGGING ĐỂ XEM ĐƯỢC TRÊN AIRFLOW/FILE ---
# Tạo logger
logger = logging.getLogger("VideoConsumer")
logger.setLevel(logging.DEBUG)  # Bắt buộc để mức DEBUG để xem chi tiết

# Format log
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# A. Handler ghi ra màn hình (Console/Airflow Logs)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# B. Handler ghi ra file (Để bạn check file nếu Console bị trôi)
file_handler = logging.FileHandler("video_consumer.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
        self.alert_throttler = AlertThrottler(
            cooldown_seconds=2
        )  # Giảm cooldown để test dễ hơn
        self.frame_count = 0

        self.clip_model = None
        self.clip_preprocess = None
        self.device = None
        self.text_features = None
        self.violence_labels = []
        self.violence_indices = []

        logger.info("Initializing VideoConsumer...")
        self.load_model()

    def load_model(self):
        """Load CLIP model and labels"""
        if not USE_VIOLENCE_CLASSIFIER:
            logger.info("🚫 Violence classifier disabled by config.")
            return

        if not CLIP_AVAILABLE:
            logger.error("❌ CLIP not available. Cannot load model.")
            return

        # 1. Load Labels
        settings_path = Path(__file__).parent.parent / "src/violence_settings.yaml"
        logger.info(f"Loading label settings from: {settings_path}")
        non_violence = []
        violence = []

        # Default labels (Fallback cực mạnh nếu không load được file)
        default_safe = [
            "peaceful scene",
            "people walking",
            "normal street",
            "friends hugging",
        ]
        default_violence = [
            "violent fighting",
            "punching and hitting",
            "kicking",
            "bloody scene",
            "holding weapon",
        ]

        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = yaml.safe_load(f)
                label_settings = settings.get("label-settings", {})
                non_violence = label_settings.get("non-violence-labels", default_safe)
                violence = label_settings.get("violence-labels", default_violence)
                logger.info(
                    f"Loaded YAML: {len(non_violence)} Safe, {len(violence)} Violence"
                )
            except Exception as e:
                logger.warning(f"YAML Error: {e}. Using defaults.")
                non_violence = default_safe
                violence = default_violence
        else:
            logger.warning("YAML not found. Using defaults.")
            non_violence = default_safe
            violence = default_violence

        self.violence_labels = non_violence + violence
        self.violence_indices = list(
            range(len(non_violence), len(self.violence_labels))
        )

        logger.info(f"Monitor Labels: {self.violence_labels}")
        logger.info(f"Violence IDs: {self.violence_indices}")

        # 2. Load CLIP Model
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using AI Device: {self.device.upper()}")

            # Load model
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=self.device
            )

            # Precompute Embeddings
            logger.info("Precomputing text embeddings...")
            text_descriptions = [
                f"a photo of {label}" for label in self.violence_labels
            ]
            text_tokens = clip.tokenize(text_descriptions).to(self.device)

            with torch.no_grad():
                self.text_features = self.clip_model.encode_text(text_tokens)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

            logger.info("✅ Model loaded successfully.")

        except Exception as e:
            logger.error(f"❌ Critical Error loading CLIP: {e}")
            self.clip_model = None

    def connect_kafka(self):
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_VIDEO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="video-processing-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_poll_records=5,
            )
            logger.info(f"✅ Connected to Kafka: {KAFKA_TOPIC_VIDEO}")
        except KafkaError as e:
            logger.error(f"❌ Kafka Connection Failed: {e}")
            raise

    def detect_objects(self, frame) -> List[Dict[str, Any]]:
        """
        Logic AI Cải tiến: Xem xét Top 5 xác suất thay vì chỉ Top 1
        """
        if not USE_VIOLENCE_CLASSIFIER or self.clip_model is None:
            return []

        if self.frame_count % VIOLENCE_CLASSIFIER_FRAME_SKIP != 0:
            return []

        try:
            # Preprocess
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Tính độ tương đồng
                similarity = (100.0 * image_features @ self.text_features.T).softmax(
                    dim=-1
                )

                # Lấy Top 5 kết quả cao nhất
                values, indices = similarity[0].topk(5)

            detections = []
            log_msg = []

            # Duyệt qua 5 kết quả cao nhất
            for value, index in zip(values, indices):
                idx = index.item()
                prob = value.item()  # Ví dụ: 0.85
                label = self.violence_labels[idx]

                log_msg.append(f"{label}({prob:.2f})")

                # LOGIC QUAN TRỌNG:
                # Nếu label thuộc nhóm bạo lực VÀ độ tin cậy > ngưỡng
                if idx in self.violence_indices:
                    if prob >= VIOLENCE_CLASSIFIER_THRESHOLD:
                        logger.warning(f"🚨 FOUND VIOLENCE: {label} - {prob:.2%}")
                        detections.append(
                            {
                                "class": label,
                                "class_id": idx,
                                "confidence": prob,
                                "bbox": None,
                            }
                        )

            # In log debug mỗi 20 frame để bạn biết model đang nhìn thấy gì
            if self.frame_count % 20 == 0:
                logger.info(f"Frame {self.frame_count} analysis: {', '.join(log_msg)}")

            return detections

        except Exception as e:
            logger.error(f"Detection Error: {e}")
            return []

    def check_harmful_content(self, detections: List[Dict]) -> Dict[str, Any]:
        # Vì hàm detect_objects đã lọc threshold rồi,
        # nên nếu list detections không rỗng nghĩa là có độc hại.
        return {
            "is_harmful": len(detections) > 0,
            "harmful_detections": detections,
            "total_detections": len(detections),
            "harmful_count": len(detections),
        }

    def process_frame(self, message: Dict[str, Any]):
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

            # Detect & Check
            detections = self.detect_objects(frame)
            result = self.check_harmful_content(detections)

            if result["is_harmful"]:
                # Save DB
                self.db_handler.save_detection(
                    {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "detections": result["harmful_detections"],
                        "is_harmful": True,
                        "data": frame_data,
                    }
                )
                # Alert
                self.generate_alert(frame_id, result, frame)

            if self.frame_count % 100 == 0:
                logger.info(f"Processed {self.frame_count} frames...")

        except Exception as e:
            logger.error(f"Frame Processing Error: {e}")

    def generate_alert(self, frame_id: int, harmful_result: Dict, frame):
        try:
            for det in harmful_result["harmful_detections"]:
                det_type = det["class"]
                conf = det["confidence"]
                alert_key = f"video_{det_type}"

                if self.alert_throttler.should_send_alert(alert_key):
                    level = calculate_alert_level(det_type, conf)
                    alert_data = {
                        "source": "video",
                        "frame_id": frame_id,
                        "detection_type": det_type,
                        "confidence": conf,
                        "type": level,
                        "details": f"Detected {det_type} ({conf:.1%})",
                    }
                    self.db_handler.save_alert(alert_data)
                    save_image_for_training(frame, det_type)
                    logger.warning(f"⚠️ ALERT SENT: {det_type} - Frame {frame_id}")

        except Exception as e:
            logger.error(f"Alert Gen Error: {e}")

    def run(self):
        try:
            self.connect_kafka()
            logger.info("👀 Waiting for video stream...")
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
    VideoConsumer(args.kafka).run()


if __name__ == "__main__":
    main()
