"""
Audio Consumer: Process audio chunks to detect Toxic Speech (Whisper) and Screaming/Violence (AST)
"""

import logging
import argparse
import time
import json
import base64
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, Tuple

from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Configs imports
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_AUDIO,
    LOG_LEVEL,
)
from utils import (
    check_toxic_content,  # Hàm check text từ khóa (bạn đã có)
    MongoDBHandler,
    AlertThrottler,
)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- IMPORT MODELS ---
# 1. Whisper (Speech to Text)
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("❌ Whisper not found. 'Toxic Speech' detection disabled.")
    WHISPER_AVAILABLE = False

# 2. Transformers (Sound Event Detection - Screaming/Yelling)
try:
    from transformers import ASTImageProcessor, ASTForAudioClassification
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("❌ Transformers not found. 'Screaming' detection disabled.")
    TRANSFORMERS_AVAILABLE = False


class AudioConsumer:
    """Consumer for processing audio streams"""

    def __init__(self, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.kafka_servers = kafka_servers
        self.db_handler = MongoDBHandler()
        self.alert_throttler = AlertThrottler(cooldown_seconds=10)
        self.chunk_count = 0

        # Audio Params
        self.target_sample_rate = 16000  # Whisper & AST đều thích 16k

        logger.info("Initializing AudioConsumer...")
        self.load_models()

        # Danh sách các âm thanh nguy hiểm cần bắt (theo nhãn của AudioSet)
        self.harmful_sound_labels = [
            "Screaming",
            "Yelling",
            "Shouting",
            "Crying, sobbing",
            "Gunshot, gunfire",
            "Explosion",
            "Bang",
        ]

    def load_models(self):
        """Load AI Models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using Device: {self.device}")

        # 1. Load Whisper
        if WHISPER_AVAILABLE:
            try:
                # 'tiny' hoặc 'base' là đủ nhanh cho realtime. 'small' chính xác hơn nhưng chậm.
                self.whisper_model = whisper.load_model("base", device=self.device)
                logger.info("✅ Whisper Model Loaded")
            except Exception as e:
                logger.error(f"Error loading Whisper: {e}")
                self.whisper_model = None

        # 2. Load Audio Spectrogram Transformer (AST)
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
                self.ast_processor = ASTImageProcessor.from_pretrained(model_name)
                self.ast_model = ASTForAudioClassification.from_pretrained(
                    model_name
                ).to(self.device)
                logger.info("✅ AST Model (Event Detection) Loaded")
            except Exception as e:
                logger.error(f"Error loading AST: {e}")
                self.ast_model = None

    def connect_kafka(self):
        """Connect to Kafka"""
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_AUDIO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="audio-processing-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_poll_records=5,  # Xử lý ít thôi vì Audio nặng
            )
            logger.info(f"Connected to Kafka topic: {KAFKA_TOPIC_AUDIO}")
        except KafkaError as e:
            logger.error(f"Kafka connection failed: {e}")
            raise

    def decode_audio(self, base64_data: str) -> np.ndarray:
        """
        Decode base64 -> Save temp .wav -> Load via Librosa -> Return Numpy Array
        """
        try:
            audio_bytes = base64.b64decode(base64_data)

            # Tạo file tạm để librosa đọc (librosa cần file path hoặc file-like object)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            # Load audio & Resample về 16kHZ
            audio_array, _ = librosa.load(temp_path, sr=self.target_sample_rate)

            # Xóa file tạm
            os.remove(temp_path)

            return audio_array
        except Exception as e:
            logger.error(f"Audio decoding error: {e}")
            return None

    def detect_sound_events(self, audio_array: np.ndarray) -> Dict:
        """
        Detect non-speech events (Screaming, Explosions...) using AST
        """
        if not self.ast_model:
            return {"is_harmful": False, "label": None, "score": 0.0}

        try:
            # AST model yêu cầu input độ dài cố định, ta padding hoặc cắt
            # Đơn giản hóa: Chỉ lấy 1024 điểm đặc trưng đầu tiên (khoảng 10s)
            inputs = self.ast_processor(
                audio_array, sampling_rate=self.target_sample_rate, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.ast_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                # Lấy Top 1 sự kiện
                score, idx = torch.max(probs, dim=-1)
                predicted_label = self.ast_model.config.id2label[idx.item()]
                score_val = score.item()

            # Check xem có phải âm thanh nguy hiểm không
            is_harmful = False
            # Ngưỡng thấp (0.3) vì model này detect nhiều class, score thường bị chia nhỏ
            if score_val > 0.3 and predicted_label in self.harmful_sound_labels:
                is_harmful = True

            return {
                "is_harmful": is_harmful,
                "label": predicted_label,
                "score": score_val,
            }

        except Exception as e:
            logger.error(f"AST detection error: {e}")
            return {"is_harmful": False, "label": None, "score": 0.0}

    def transcribe_and_check_toxic(self, audio_array: np.ndarray) -> Dict:
        """
        Speech-to-text -> Check keywords
        """
        if not self.whisper_model:
            return {"is_toxic": False, "text": "", "keywords": []}

        try:
            # Whisper yêu cầu float32
            audio_array = audio_array.astype(np.float32)

            # Transcribe
            # Note: Whisper có thể xử lý trực tiếp numpy array
            result = self.whisper_model.transcribe(
                audio_array, fp16=False, language="vi"
            )  # fp16=False để chạy trên CPU ok
            text = result["text"].strip()

            if not text:
                return {"is_toxic": False, "text": "", "keywords": []}

            # Check toxic (Dùng hàm utils có sẵn)
            # Giả định utils trả về: {'is_toxic': bool, 'matched_keywords': list, 'toxic_score': int}
            from config import TOXIC_KEYWORDS  # Import ở đây để đảm bảo có data

            toxic_result = check_toxic_content(text, TOXIC_KEYWORDS)

            return {
                "is_toxic": toxic_result["is_toxic"],
                "text": text,
                "keywords": toxic_result.get("matched_keywords", []),
                "score": toxic_result.get("toxic_score", 0),
            }

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return {"is_toxic": False, "text": "", "keywords": []}

    def process_message(self, message: Dict):
        """Main processing loop for a chunk"""
        chunk_id = message.get("chunk_id")
        timestamp = message.get("timestamp")
        b64_data = message.get("data")

        if not b64_data:
            return

        # 1. Decode Audio
        audio_array = self.decode_audio(b64_data)
        if audio_array is None or len(audio_array) == 0:
            return

        # 2. Parallel Analysis (Tuần tự trong code này cho đơn giản)

        # A. Detect Sound Events (Gào thét, nổ...)
        sound_event = self.detect_sound_events(audio_array)

        # B. Detect Toxic Speech (Chửi bậy...)
        speech_result = self.transcribe_and_check_toxic(audio_array)

        # 3. Logic Tổng hợp & Alert
        is_alert = False
        alert_type = "INFO"
        alert_details = ""

        # Check Âm thanh (Screaming)
        if sound_event["is_harmful"]:
            is_alert = True
            alert_type = "SCREAMING/VIOLENCE"
            alert_details = (
                f"Detected sound: {sound_event['label']} ({sound_event['score']:.1%})"
            )
            logger.warning(f"🔊 ALERT: {alert_details}")

            if self.alert_throttler.should_send_alert("audio_scream"):
                self.db_handler.save_alert(
                    {
                        "source": "audio",
                        "frame_id": chunk_id,  # Dùng chunk_id thay frame_id
                        "detection_type": "Audio Event",
                        "level": "HIGH",
                        "confidence": sound_event["score"],
                        "details": alert_details,
                        "timestamp": timestamp,
                    }
                )

        # Check Lời nói (Toxic)
        if speech_result["is_toxic"]:
            is_alert = True
            alert_type = "TOXIC SPEECH"
            alert_details = f"Toxic words: {speech_result['keywords']} in text: '{speech_result['text']}'"
            logger.warning(f"🤬 ALERT: {alert_details}")

            if self.alert_throttler.should_send_alert("audio_toxic"):
                self.db_handler.save_alert(
                    {
                        "source": "audio",
                        "frame_id": chunk_id,
                        "detection_type": "Toxic Speech",
                        "level": "MEDIUM",
                        "confidence": 1.0,
                        "details": alert_details,
                        "timestamp": timestamp,
                    }
                )

        # 4. Save Detection Record (Log lại mọi thứ)
        self.db_handler.save_detection(
            {
                "chunk_id": chunk_id,
                "timestamp": timestamp,
                "transcribed_text": speech_result["text"],
                "sound_label": sound_event["label"],
                "sound_confidence": sound_event["score"],
                "is_toxic": speech_result["is_toxic"],
                "is_screaming": sound_event["is_harmful"],
            }
        )

        if self.chunk_count % 10 == 0:
            logger.info(
                f"Processed chunk {chunk_id}: {sound_event['label']} | Text: {speech_result['text'][:30]}..."
            )

        self.chunk_count += 1

    def run(self):
        try:
            self.connect_kafka()
            logger.info("🎧 Audio Consumer listening...")

            for msg in self.consumer:
                self.process_message(msg.value)

        except KeyboardInterrupt:
            logger.info("Stopped.")
        finally:
            if self.consumer:
                self.consumer.close()
            self.db_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", type=str, default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    AudioConsumer(args.kafka).run()
