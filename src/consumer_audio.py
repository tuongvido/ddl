"""
Audio Consumer Optimized: Rolling Buffer + Faster-Whisper + AST
Target: Lightweight, Fast, Accurate for Vietnamese & Sound Events
"""

import logging
import argparse
import json
import base64
import tempfile
import os
import numpy as np
import librosa
from typing import Dict

from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Configs imports
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_AUDIO,
    LOG_LEVEL,
)
from utils import (
    check_toxic_content,
    MongoDBHandler,
    AlertThrottler,
)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- IMPORT MODELS ---

# 1. YAMNet (Sound Event Detection - Lightweight)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np

    YAMNET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"❌ TensorFlow/YAMNet import failed: {e}")
    YAMNET_AVAILABLE = False

# 2. PhoWhisper (Vietnamese Speech Recognition)
try:
    from transformers import pipeline
    PHOWHISPER_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"❌ Transformers not found. Install: pip install transformers. Error: {e}"
    )
    PHOWHISPER_AVAILABLE = False

# 3. PhoBERT (Vietnamese Toxic Content Classification)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    PHOBERT_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"❌ PhoBERT dependencies not found. Error: {e}"
    )
    PHOBERT_AVAILABLE = False


class AudioConsumer:
    """Consumer for processing audio streams with Rolling Buffer"""

    def __init__(self, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.kafka_servers = kafka_servers
        self.db_handler = MongoDBHandler()
        self.alert_throttler = AlertThrottler(
            cooldown_seconds=5
        )  # Giảm cooldown để test nhanh hơn
        self.chunk_count = 0

        # Audio Params
        self.target_sample_rate = 16000

        # --- ROLLING BUFFER CONFIG ---
        self.buffer_duration = 5.0  # Giữ lại 5 giây ngữ cảnh
        self.max_buffer_samples = int(self.buffer_duration * self.target_sample_rate)
        # Buffer khởi tạo rỗng
        self.audio_buffer = np.array([], dtype=np.float32)

        # Initialize model attributes to None
        self.yamnet_model = None
        self.yamnet_classes = []
        self.phowhisper_model = None
        self.phobert_tokenizer = None
        self.phobert_model = None
        self.device = None

        logger.info("Initializing AudioConsumer (Optimized)...")
        self.load_models()

        # Danh sách âm thanh nguy hiểm
        self.harmful_sound_labels = [
            "Screaming",
            "Yelling",
            "Shouting",
            "Crying, sobbing",
            "Gunshot, gunfire",
            "Explosion",
            "Bang",
            "Aggressive",
        ]

    def load_models(self):
        """Load AI Models (YAMNet for Sound Events + Faster-Whisper for Speech)"""

        # YAMNet uses TensorFlow (CPU-friendly)
        logger.info("Loading AI Models...")

        # 1. Load YAMNet (Google's lightweight sound event detection model)
        if YAMNET_AVAILABLE:
            try:
                logger.info("⏳ Loading YAMNet model...")
                # YAMNet from TensorFlow Hub - efficient sound classification
                yamnet_model_url = "https://tfhub.dev/google/yamnet/1"
                self.yamnet_model = hub.load(yamnet_model_url)
                
                # Load YAMNet class names
                yamnet_classes_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audio_set/yamnet/yamnet_class_map.csv"
                try:
                    import urllib.request
                    import csv
                    from io import StringIO
                    
                    with urllib.request.urlopen(yamnet_classes_url) as response:
                        csv_content = response.read().decode('utf-8')
                        self.yamnet_classes = []
                        reader = csv.DictReader(StringIO(csv_content))
                        for row in reader:
                            self.yamnet_classes.append(row['display_name'])
                except Exception as e:
                    logger.warning(f"Could not load YAMNet classes from URL: {e}. Using default classes.")
                    # Fallback to common harmful sound event classes
                    self.yamnet_classes = ["Speech", "Gunshot", "Explosion", "Screaming", "Yelling"]
                
                logger.info(f"✅ YAMNet Loaded (Total classes: {len(self.yamnet_classes)})")
            except Exception as e:
                logger.error(f"❌ Error loading YAMNet: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.yamnet_model = None
                self.yamnet_classes = []
        else:
            logger.warning("⚠️ YAMNet not available - sound event detection disabled")
            self.yamnet_model = None
            self.yamnet_classes = []

        # 2. Load PhoWhisper (Vietnamese Speech Recognition)
        if PHOWHISPER_AVAILABLE:
            try:
                logger.info("⏳ Loading PhoWhisper model...")
                # Use vinai/PhoWhisper for Vietnamese speech recognition
                # Force PyTorch backend to avoid TensorFlow/Keras compatibility issues
                self.phowhisper_model = pipeline(
                    "automatic-speech-recognition",
                    model="vinai/PhoWhisper-small",
                    device=-1,  # CPU
                    framework="pt"  # Force PyTorch
                )
                logger.info("✅ PhoWhisper Loaded")
            except Exception as e:
                logger.error(f"❌ Error loading PhoWhisper: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.phowhisper_model = None
        else:
            logger.warning("⚠️ PhoWhisper not available - speech transcription disabled")
            self.phowhisper_model = None

        # 3. Load PhoBERT (Vietnamese Toxic Classification)
        if PHOBERT_AVAILABLE:
            try:
                logger.info("⏳ Loading PhoBERT toxic classification model...")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Use a Vietnamese toxic comment classification model
                # You can replace with your own fine-tuned model
                model_name = "wonrax/phobert-base-vietnamese-sentiment"
                
                self.phobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.phobert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.phobert_model.to(self.device)
                self.phobert_model.eval()
                
                logger.info(f"✅ PhoBERT Loaded (Device: {self.device})")
            except Exception as e:
                logger.error(f"❌ Error loading PhoBERT: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.phobert_tokenizer = None
                self.phobert_model = None
        else:
            logger.warning("⚠️ PhoBERT not available - toxic classification disabled")
            self.phobert_tokenizer = None
            self.phobert_model = None

    def connect_kafka(self):
        """Connect to Kafka"""
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_AUDIO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="audio-group-optimized",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            logger.info(f"Connected to Kafka topic: {KAFKA_TOPIC_AUDIO}")
        except KafkaError as e:
            logger.error(f"Kafka connection failed: {e}")
            raise

    def decode_audio(self, base64_data: str) -> np.ndarray:
        """Decode base64 to numpy array"""
        try:
            audio_bytes = base64.b64decode(base64_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            # Load audio & Resample về 16kHZ
            audio_array, _ = librosa.load(temp_path, sr=self.target_sample_rate)
            os.remove(temp_path)
            return audio_array
        except Exception as e:
            logger.error(f"Audio decoding error: {e}")
            return None

    def detect_sound_events(self, audio_array: np.ndarray) -> Dict:
        """YAMNet Sound Event Detection"""
        if not self.yamnet_model:
            return {"is_harmful": False, "label": None, "score": 0.0}

        try:
            # YAMNet expects audio at 16kHz as a 1D numpy array
            # Input should be in the range [-1, 1]
            
            # Normalize audio if needed
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_normalized = audio_array / (max_val + 1e-7)
            else:
                audio_normalized = audio_array

            # Run YAMNet inference
            scores, embeddings, spectrogram = self.yamnet_model(
                tf.constant(audio_normalized, dtype=tf.float32)
            )

            # Get top prediction
            top_idx = tf.argmax(scores, axis=-1).numpy()[-1]  # Get last frame's top class
            top_score = tf.reduce_max(scores, axis=-1).numpy()[-1]

            # Get label name
            if top_idx < len(self.yamnet_classes):
                predicted_label = self.yamnet_classes[int(top_idx)]
            else:
                predicted_label = f"Unknown_class_{top_idx}"

            # Define harmful sound events
            harmful_keywords = [
                "gunshot",
                "explosion",
                "screaming",
                "yelling",
                "shouting",
                "crying",
                "gunfire",
                "bang",
                "fighting",
                "violence",
                "aggressive",
                "alarm",
                "siren",
                "breaking",
                "crash",
                "glass breaking",
                "impact",
                "punch",
                "kick",
                "weapon",
            ]

            # Check if detected sound is harmful
            is_harmful = any(
                keyword in predicted_label.lower() for keyword in harmful_keywords
            ) and top_score > 0.35

            # Log YAMNet detection output
            logger.info(f"🎵 YAMNet detected: {predicted_label} (confidence: {top_score:.2%})")

            return {
                "is_harmful": is_harmful,
                "label": predicted_label,
                "score": float(top_score),
            }
        except Exception as e:
            logger.error(f"YAMNet error: {e}")
            return {"is_harmful": False, "label": None, "score": 0.0}

    def transcribe_and_check_toxic(self, audio_buffer: np.ndarray) -> Dict:
        """PhoWhisper Transcription + PhoBERT Toxic Classification"""
        if not self.phowhisper_model:
            return {"is_toxic": False, "text": "", "keywords": [], "score": 0.0}

        try:
            # Step 1: Transcribe audio to Vietnamese text using PhoWhisper
            # Pipeline expects dict with 'raw' and 'sampling_rate' keys
            audio_input = {
                "raw": audio_buffer,
                "sampling_rate": self.target_sample_rate
            }
            result = self.phowhisper_model(audio_input)
            text = result.get("text", "").strip()

            # Log PhoWhisper output
            if text:
                logger.info(f"📝 PhoWhisper transcribed: {text}")
            else:
                logger.info(f"📝 PhoWhisper transcribed: (no speech detected)")

            if not text:
                return {"is_toxic": False, "text": "", "keywords": [], "score": 0.0}

            # Step 2: Classify toxicity using PhoBERT
            if not self.phobert_model or not self.phobert_tokenizer:
                # Fallback to keyword matching if PhoBERT not available
                from config import TOXIC_KEYWORDS
                toxic_result = check_toxic_content(text, TOXIC_KEYWORDS)
                return {
                    "is_toxic": toxic_result["is_toxic"],
                    "text": text,
                    "keywords": toxic_result.get("matched_keywords", []),
                    "score": toxic_result.get("toxic_score", 0),
                }

            # Use PhoBERT for classification
            inputs = self.phobert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Assuming model outputs [negative, neutral, positive]
                # or [non-toxic, toxic] - adjust based on your model
                # For sentiment model: negative sentiment might indicate toxicity
                negative_score = float(probs[0][0])  # negative class
                
                # Consider negative sentiment as potential toxic content
                is_toxic = negative_score > 0.6  # Adjust threshold as needed
                toxic_score = negative_score

            # Log PhoBERT classification result
            logger.info(f"🤖 PhoBERT toxic score: {toxic_score:.2%} (is_toxic: {is_toxic})")

            return {
                "is_toxic": is_toxic,
                "text": text,
                "keywords": [],  # PhoBERT doesn't return keywords
                "score": toxic_score,
            }

        except Exception as e:
            logger.error(f"PhoWhisper/PhoBERT error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"is_toxic": False, "text": "", "keywords": [], "score": 0.0}

    def process_message(self, message: Dict):
        """
        Main processing with ROLLING BUFFER logic
        """
        chunk_id = message.get("chunk_id")
        timestamp = message.get("timestamp")
        b64_data = message.get("data")
        session_id = message.get("session_id", "unknown")  # Get session ID

        if not b64_data:
            return

        # 1. Decode chunk mới (1 giây)
        new_chunk = self.decode_audio(b64_data)
        if new_chunk is None:
            return

        # 2. CẬP NHẬT ROLLING BUFFER
        # Nối chunk mới vào đuôi buffer hiện tại
        self.audio_buffer = np.concatenate((self.audio_buffer, new_chunk))

        # Nếu buffer dài quá 5 giây, cắt bớt phần đầu (cũ nhất)
        if len(self.audio_buffer) > self.max_buffer_samples:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_samples :]

        # Chỉ xử lý khi buffer đã có ít nhất 1-2 giây để model đoán chuẩn hơn
        # (Lúc mới khởi động có thể bỏ qua vài chunk đầu)
        if len(self.audio_buffer) < 16000:
            return

        # --- PHÂN TÍCH ---

        # A. Detect Sound (YAMNet) - Dùng toàn bộ buffer (5s) để detect chính xác hơn
        sound_event = self.detect_sound_events(self.audio_buffer)

        # B. Transcribe (PhoWhisper + PhoBERT) - Dùng toàn bộ buffer (5s) để lấy ngữ cảnh
        speech_result = self.transcribe_and_check_toxic(self.audio_buffer)

        # Log combined results
        logger.info(f"📊 Chunk {chunk_id} Analysis - Sound: {sound_event['label']} ({sound_event['score']:.2%}) | Text: '{speech_result['text'][:50]}{'...' if len(speech_result['text']) > 50 else ''}' | Toxic: {speech_result['is_toxic']}")

        # 3. Alert Logic
        alert_details = ""

        # --- Xử lý Sound Event Alert ---
        if sound_event["is_harmful"]:
            alert_details = (
                f"Detected: {sound_event['label']} ({sound_event['score']:.1%})"
            )
            logger.warning(f"🔊 {alert_details}")

            if self.alert_throttler.should_send_alert("audio_scream"):
                alert_record = {
                    "source": "audio",
                    "frame_id": chunk_id,
                    "detection_type": "violent_audio",
                    "type": "HIGH",
                    "confidence": sound_event["score"],
                    "details": alert_details,
                    "timestamp": timestamp,
                    "session_id": session_id,
                }
                self.db_handler.save_alert(alert_record)
                # Update video session
                self.db_handler.update_video_session(session_id, alert_record)

        # --- Xử lý Toxic Speech Alert ---
        if speech_result["is_toxic"]:
            alert_details = (
                f"Toxic score: {speech_result['score']:.2%} | '{speech_result['text']}'"
            )
            logger.warning(f"🤬 {alert_details}")

            if self.alert_throttler.should_send_alert("audio_toxic"):
                alert_record = {
                    "source": "audio",
                    "frame_id": chunk_id,
                    "detection_type": "toxic_speech",
                    "type": "MEDIUM",
                    "confidence": speech_result["score"],
                    "details": alert_details,
                    "timestamp": timestamp,
                    "session_id": session_id,
                }
                self.db_handler.save_alert(alert_record)
                # Update video session
                self.db_handler.update_video_session(session_id, alert_record)

        # 4. Save Record
        # Lưu text đầy đủ để hiển thị lên dashboard
        self.db_handler.save_detection(
            {
                "chunk_id": chunk_id,
                "timestamp": timestamp,
                "transcribed_text": speech_result["text"],  # Text này sẽ dài (5s)
                "sound_label": sound_event["label"],
                "sound_confidence": sound_event["score"],
                "is_toxic": speech_result["is_toxic"],
                "toxic_score": speech_result["score"],
                "is_screaming": sound_event["is_harmful"],
                "session_id": session_id,
            }
        )

        if self.chunk_count % 5 == 0:
            short_text = (
                speech_result["text"][-50:]
                if len(speech_result["text"]) > 50
                else speech_result["text"]
            )
            logger.info(
                f"Chunk {chunk_id} | Sound: {sound_event['label']} ({sound_event['score']:.2f}) | Text: ...{short_text}"
            )

        self.chunk_count += 1

    def run(self):
        try:
            self.connect_kafka()
            logger.info("🎧 Audio Consumer (Optimized) listening...")
            for msg in self.consumer:
                self.process_message(msg.value)
        except KeyboardInterrupt:
            logger.info("Stopped.")
        finally:
            if hasattr(self, "consumer") and self.consumer:
                self.consumer.close()
            self.db_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", type=str, default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()
    AudioConsumer(args.kafka).run()
