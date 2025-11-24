"""
Audio Consumer: Process audio chunks and detect toxic speech using Whisper + NLP
"""

import logging
import argparse
import time
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
from typing import Dict, Any

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_AUDIO,
    WHISPER_MODEL,
    TOXIC_KEYWORDS,
    LOG_LEVEL,
)
from utils import (
    check_toxic_content,
    calculate_alert_level,
    MongoDBHandler,
    AlertThrottler,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Try to import Whisper
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("Whisper not available. Install with: pip install openai-whisper")
    WHISPER_AVAILABLE = False


class AudioConsumer:
    """Consumer for processing audio and detecting toxic speech"""

    def __init__(self, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        """
        Initialize audio consumer

        Args:
            kafka_servers: Kafka bootstrap servers
        """
        self.kafka_servers = kafka_servers
        self.consumer = None
        self.model = None
        self.db_handler = MongoDBHandler()
        self.alert_throttler = AlertThrottler(cooldown_seconds=5)
        self.chunk_count = 0

        logger.info("Initializing AudioConsumer")

        # Load Whisper model
        self.load_model()

    def load_model(self):
        """Load Whisper model"""
        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available, cannot load model")
            return

        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
            self.model = whisper.load_model(WHISPER_MODEL)
            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def connect_kafka(self):
        """Connect to Kafka broker"""
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_AUDIO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="audio-processing-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_poll_records=5,
                session_timeout_ms=30000,
            )
            logger.info(f"Connected to Kafka at {self.kafka_servers}")
            logger.info(f"Subscribed to topic: {KAFKA_TOPIC_AUDIO}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def transcribe_audio(self, audio_data: str) -> str:
        """
        Transcribe audio to text using Whisper

        Args:
            audio_data: Audio data (base64 or file path)

        Returns:
            Transcribed text
        """
        if not self.model or not WHISPER_AVAILABLE:
            logger.warning("Whisper model not available, skipping transcription")
            return ""

        try:
            # Note: In real implementation, decode audio_data and save to temp file
            # For now, we'll simulate transcription
            # result = self.model.transcribe(audio_file_path)
            # return result["text"]

            # Placeholder: return empty string
            # In production, implement proper audio decoding and transcription
            logger.debug("Audio transcription placeholder - implement audio decoding")
            return ""

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def process_audio(self, message: Dict[str, Any]):
        """
        Process an audio chunk message

        Args:
            message: Kafka message containing audio data
        """
        try:
            chunk_id = message.get("chunk_id", -1)
            timestamp = message.get("timestamp", time.time())
            audio_data = message.get("data", "")

            if not audio_data:
                logger.debug(f"Empty audio data for chunk {chunk_id}")
                return

            # Transcribe audio to text
            transcribed_text = self.transcribe_audio(audio_data)

            if not transcribed_text:
                logger.debug(f"No transcription for chunk {chunk_id}")
                return

            logger.info(f"Transcribed text: {transcribed_text}")

            # Check for toxic content
            toxic_result = check_toxic_content(transcribed_text, TOXIC_KEYWORDS)

            # Save detection result to database
            detection_record = {
                "chunk_id": chunk_id,
                "timestamp": timestamp,
                "transcribed_text": transcribed_text,
                "is_toxic": toxic_result["is_toxic"],
                "toxic_score": toxic_result["toxic_score"],
                "matched_keywords": toxic_result["matched_keywords"],
            }

            self.db_handler.save_detection(detection_record)

            # If toxic content detected, generate alert
            if toxic_result["is_toxic"]:
                self.generate_alert(chunk_id, transcribed_text, toxic_result)

            self.chunk_count += 1

            if self.chunk_count % 20 == 0:
                logger.info(f"Processed {self.chunk_count} audio chunks")

        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    def generate_alert(self, chunk_id: int, text: str, toxic_result: Dict):
        """
        Generate alert for toxic speech

        Args:
            chunk_id: Audio chunk ID
            text: Transcribed text
            toxic_result: Toxic content detection result
        """
        try:
            toxic_score = toxic_result["toxic_score"]
            matched_keywords = toxic_result["matched_keywords"]

            # Calculate alert level
            alert_level = calculate_alert_level("toxic_speech", 0.8, toxic_score)

            # Check if we should send alert (throttling)
            alert_key = "audio_toxic_speech"

            if not self.alert_throttler.should_send_alert(alert_key):
                logger.debug("Alert throttled for toxic speech")
                return

            # Create alert
            alert_data = {
                "source": "audio",
                "chunk_id": chunk_id,
                "detection_type": "toxic_speech",
                "confidence": min(toxic_score / 5.0, 1.0),  # Normalize to 0-1
                "level": alert_level,
                "transcribed_text": text,
                "matched_keywords": matched_keywords,
                "toxic_score": toxic_score,
                "details": f"Detected {len(matched_keywords)} toxic keywords: {', '.join(matched_keywords)}",
            }

            # Save alert to database
            self.db_handler.save_alert(alert_data)

            logger.warning(
                f"⚠️ ALERT [{alert_level}]: Toxic speech detected "
                f"(score: {toxic_score}) in chunk {chunk_id}"
            )
            logger.warning(f"   Matched keywords: {', '.join(matched_keywords)}")
            logger.warning(f"   Text: {text[:100]}...")

        except Exception as e:
            logger.error(f"Error generating alert: {e}")

    def run(self):
        """Run the consumer"""
        try:
            self.connect_kafka()

            logger.info("Starting audio consumer...")
            logger.info("Waiting for messages...")

            for message in self.consumer:
                self.process_audio(message.value)

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
    parser = argparse.ArgumentParser(description="Audio Consumer")
    parser.add_argument(
        "--kafka",
        type=str,
        default=KAFKA_BOOTSTRAP_SERVERS,
        help="Kafka bootstrap servers",
    )

    args = parser.parse_args()

    consumer = AudioConsumer(args.kafka)
    consumer.run()


if __name__ == "__main__":
    main()
