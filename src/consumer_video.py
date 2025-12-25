"""Video Consumer: Process video frames and detect harmful content using Swin Transformer Tiny
Detects both normal and violent actions in video frames
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
# file_handler = logging.FileHandler("video_consumer.log")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# --- Dependency Check ---
try:
    import torch
    from PIL import Image
    import yaml
    from ultralytics import YOLO
    
    YOLO_AVAILABLE = True
    logger.info("✅ YOLO (Ultralytics) dependencies loaded successfully")
except Exception as e:
    logger.error(
        f"❌ YOLO/torch dependencies missing: {e}. AI detection will be disabled."
    )
    YOLO_AVAILABLE = False

# OCR for text detection in images
try:
    import easyocr
    OCR_AVAILABLE = True
    logger.info("✅ EasyOCR loaded successfully")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("⚠️ EasyOCR not available - text detection disabled")

# PhoBERT for Vietnamese toxic text classification
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    PHOBERT_AVAILABLE = True
    logger.info("✅ PhoBERT dependencies loaded")
except ImportError:
    PHOBERT_AVAILABLE = False
    logger.warning("⚠️ PhoBERT not available - will use keyword matching")


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

        self.yolo_model = None
        self.device = None
        self.violence_labels = []
        # YOLO model class names: all are harmful content
        # {0: 'alcohol', 1: 'blood', 2: 'cigarette', 3: 'fight detection - v1 2024-05-10 8-55pm', 
        #  4: 'gun', 5: 'insulting_gesture', 6: 'knife'}
        self.id2label = {}
        
        # OCR reader
        self.ocr_reader = None
        
        # PhoBERT for toxic text classification
        self.phobert_tokenizer = None
        self.phobert_model = None
        self.phobert_device = None

        logger.info("Initializing VideoConsumer...")
        self.load_model()
        self.load_ocr()
        self.load_phobert()

    def load_model(self):
        """Load YOLO violence detection model from local best.pt file"""
        if not USE_VIOLENCE_CLASSIFIER:
            logger.info("🚫 Violence classifier disabled by config.")
            return

        if not YOLO_AVAILABLE:
            logger.error("❌ YOLO not available. Cannot load model.")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using AI Device: {self.device.upper()}")

            # Load YOLO model from local file
            model_path = Path(__file__).parent.parent / "models" / "best.pt"
            logger.info(f"Loading YOLO model from {model_path}...")
            
            self.yolo_model = YOLO(str(model_path))
            
            # Get class names from model
            # Expected: {0: 'alcohol', 1: 'blood', 2: 'cigarette', 3: 'fight detection - v1 2024-05-10 8-55pm', 
            #            4: 'gun', 5: 'insulting_gesture', 6: 'knife'}
            self.id2label = self.yolo_model.names
            self.violence_labels = list(self.id2label.values())
            
            logger.info(f"Model Labels: {self.violence_labels}")
            logger.info(f"Label Mapping: {self.id2label}")
            logger.info("✅ YOLO violence detection model loaded successfully.")
            logger.info("ℹ️ All detected classes are harmful content (alcohol, blood, cigarette, fight, gun, insulting_gesture, knife)")

        except Exception as e:
            logger.error(f"❌ Critical Error loading YOLO model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.yolo_model = None
    
    def load_ocr(self):
        """Load EasyOCR for text detection in images"""
        if not OCR_AVAILABLE:
            logger.warning("⚠️ OCR disabled - EasyOCR not available")
            return
        
        try:
            logger.info("⏳ Loading EasyOCR (Vietnamese + English)...")
            # Load Vietnamese and English OCR
            self.ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)
            logger.info("✅ EasyOCR loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading EasyOCR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.ocr_reader = None    
    def load_phobert(self):
        """Load PhoBERT for Vietnamese toxic text classification"""
        if not PHOBERT_AVAILABLE:
            logger.warning("⚠️ PhoBERT not available - will use keyword matching for text")
            return
        
        try:
            logger.info("⏳ Loading PhoBERT for toxic text classification...")
            self.phobert_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use local trained PhoBERT model from models/phobert/
            model_path = Path(__file__).parent.parent / "models" / "phobert"
            logger.info(f"Loading PhoBERT from local path: {model_path}")
            
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.phobert_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.phobert_model.to(self.phobert_device)
            self.phobert_model.eval()
            
            logger.info(f"✅ PhoBERT loaded from local model (Device: {self.phobert_device})")
            logger.info(f"   Model architecture: {self.phobert_model.config.architectures}")
            logger.info(f"   Model config: {self.phobert_model.config.architectures}")
        except Exception as e:
            logger.error(f"❌ Failed to load PhoBERT: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phobert_tokenizer = None
            self.phobert_model = None
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
        YOLO-based violence/harmful content detection
        Processes single frames to detect harmful objects and actions
        Classes: alcohol, blood, cigarette, fight, gun, insulting_gesture, knife
        """
        if not USE_VIOLENCE_CLASSIFIER or self.yolo_model is None:
            return []

        if self.frame_count % VIOLENCE_CLASSIFIER_FRAME_SKIP != 0:
            return []

        try:
            # Run YOLO inference on single frame
            results = self.yolo_model(frame, conf=VIOLENCE_CLASSIFIER_THRESHOLD, verbose=False)
            
            detections = []
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box data
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                    
                    # Get class name
                    class_name = self.id2label.get(class_id, f"class_{class_id}")
                    
                    # All classes from this model are harmful content
                    logger.warning(f"🚨 HARMFUL CONTENT DETECTED: {class_name} - {confidence:.2%} at bbox {bbox}")
                    
                    detections.append({
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "action_type": "violent",  # All detections are harmful
                    })
            
            if detections:
                logger.info(f"Frame {self.frame_count}: {len(detections)} harmful object(s) detected")
            elif self.frame_count % 50 == 0:
                logger.debug(f"Frame {self.frame_count}: No harmful content detected")
            
            return detections

        except Exception as e:
            logger.error(f"YOLO Detection Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def detect_text_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect and check toxic text in video frame using OCR + PhoBERT"""
        if not self.ocr_reader:
            return {"has_text": False, "is_toxic": False, "texts": [], "toxic_score": 0.0, "full_text": ""}
        
        try:
            # Run OCR to extract text from frame
            results = self.ocr_reader.readtext(frame)
            
            if not results:
                logger.debug("📝 OCR: No text detected in frame")
                return {"has_text": False, "is_toxic": False, "texts": [], "toxic_score": 0.0, "full_text": ""}
            
            # Log all OCR results (including low confidence)
            logger.info(f"📝 OCR raw results: {len(results)} text regions found")
            # for i, (bbox, text, confidence) in enumerate(results):
            #     logger.info(f"   [{i+1}] Text: '{text}' | Confidence: {confidence:.2%}")
            
            # Extract all detected texts with confidence > 0.3
            detected_texts = []
            all_text_parts = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    detected_texts.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": bbox
                    })
                    all_text_parts.append(text)
            
            if not all_text_parts:
                logger.info("📝 OCR: Text detected but all below confidence threshold (0.3)")
                return {"has_text": False, "is_toxic": False, "texts": [], "toxic_score": 0.0, "full_text": ""}
            
            # Combine all texts
            full_text = " ".join(all_text_parts)
            logger.warning(f"📝 ===== OCR DETECTED TEXT ===== \n{full_text}\n================================")
            
            # Method 1: PhoBERT Classification (preferred for Vietnamese)
            if self.phobert_model and self.phobert_tokenizer:
                try:
                    inputs = self.phobert_tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=True
                    ).to(self.phobert_device)

                    with torch.no_grad():
                        outputs = self.phobert_model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        
                        # Assuming model outputs [negative, neutral, positive]
                        # negative sentiment indicates potential toxic content
                        negative_score = float(probs[0][0])
                        
                        # Threshold for toxic text (adjust as needed)
                        TOXIC_TEXT_THRESHOLD = 0.6
                        is_toxic = negative_score > TOXIC_TEXT_THRESHOLD
                        toxic_score = negative_score
                    
                    if is_toxic:
                        logger.warning(f"🚨 TOXIC TEXT ON SCREEN (PhoBERT): {full_text} (score: {toxic_score:.2%})")
                    
                    return {
                        "has_text": True,
                        "is_toxic": is_toxic,
                        "texts": detected_texts,
                        "toxic_score": toxic_score,
                        "full_text": full_text,
                        "method": "phobert"
                    }
                    
                except Exception as e:
                    logger.error(f"PhoBERT classification error: {e}")
                    # Fallback to keyword matching
            
            # Method 2: Keyword Matching (fallback)
            from config import TOXIC_KEYWORDS
            from utils import check_toxic_content
            
            toxic_result = check_toxic_content(full_text, TOXIC_KEYWORDS)
            
            if toxic_result["is_toxic"]:
                logger.warning(f"🚨 TOXIC TEXT ON SCREEN (Keywords): {toxic_result.get('matched_keywords', [])} in '{full_text}'")
            
            return {
                "has_text": True,
                "is_toxic": toxic_result["is_toxic"],
                "texts": detected_texts,
                "toxic_score": toxic_result.get("toxic_score", 0),
                "full_text": full_text,
                "toxic_words": toxic_result.get("matched_keywords", []),
                "method": "keywords"
            }
            
        except Exception as e:
            logger.error(f"OCR text detection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"has_text": False, "is_toxic": False, "texts": [], "toxic_score": 0.0, "full_text": ""}

    def check_harmful_content(self, detections: List[Dict], text_result: Dict) -> Dict[str, Any]:
        # Check if harmful from video actions OR toxic text
        is_harmful = len(detections) > 0 or text_result.get("is_toxic", False)
        
        return {
            "is_harmful": is_harmful,
            "harmful_detections": detections,
            "total_detections": len(detections),
            "harmful_count": len(detections),
            "text_result": text_result,
        }

    def process_frame(self, message: Dict[str, Any]):
        try:
            frame_id = message.get("frame_id", -1)
            timestamp = message.get("timestamp", time.time())
            frame_data = message.get("data", "")
            session_id = message.get("session_id", "unknown")  # Get session ID

            if not frame_data:
                return

            frame = decode_base64_to_image(frame_data)
            if frame is None:
                return

            self.frame_count += 1

            # Detect & Check
            detections = self.detect_objects(frame)
            text_result = self.detect_text_in_frame(frame)
            result = self.check_harmful_content(detections, text_result)
            # logger.info(f"result: {result}")

            if result["is_harmful"]:
                # Determine detection type
                detection_types = []
                if len(detections) > 0:
                    detection_types.append("violent_video")
                if text_result.get("is_toxic", False):
                    detection_types.append("toxic_text_on_screen")
                # Save DB
                # logger.info(f"detections: {detections}")
                detection_record = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "detections": result["harmful_detections"],
                    "text_detection": text_result,
                    "is_harmful": True,
                    "data": frame_data,
                    "session_id": session_id,
                    "detection_type": ", ".join(detection_types) if len(detection_types) > 1 else detection_types[0] if detection_types else "violent_video",
                }
                # logger.info(f"result: {detection_record}")
                self.db_handler.save_detection(detection_record)
                
                # Update video session summary for each type
                for det_type in detection_types:
                    session_record = detection_record.copy()
                    session_record["detection_type"] = det_type
                    self.db_handler.update_video_session(session_id, session_record)
                
                self.db_handler.finalize_video_session(session_id)
                # Alert
                self.generate_alert(frame_id, result, frame)

            if self.frame_count % 100 == 0:
                logger.info(f"Processed {self.frame_count} frames...")

        except Exception as e:
            logger.error(f"Frame Processing Error: {e}")

    def generate_alert(self, frame_id: int, harmful_result: Dict, frame):
        try:
            # Video action alerts
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
            
            # Toxic text alert
            text_result = harmful_result.get("text_result", {})
            if text_result.get("is_toxic", False):
                alert_key = "video_toxic_text"
                if self.alert_throttler.should_send_alert(alert_key):
                    alert_data = {
                        "source": "video",
                        "frame_id": frame_id,
                        "detection_type": "Toxic Text on Screen",
                        "confidence": text_result.get("toxic_score", 1.0),
                        "type": "HIGH",
                        "details": f"Toxic words: {', '.join(text_result.get('toxic_words', []))} | Text: '{text_result.get('full_text', '')[:100]}'",
                    }
                    self.db_handler.save_alert(alert_data)
                    save_image_for_training(frame, "toxic_text")
                    logger.warning(f"⚠️ ALERT SENT: Toxic Text - Frame {frame_id}")

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
