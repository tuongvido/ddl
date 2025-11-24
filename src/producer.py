"""
Producer: Simulates livestream by reading video file and sending frames to Kafka
"""

import cv2
import time
import logging
import argparse
import json
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError
import numpy as np

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_VIDEO,
    KAFKA_TOPIC_AUDIO,
    VIDEO_FPS,
    VIDEO_FRAME_WIDTH,
    VIDEO_FRAME_HEIGHT,
    VIDEO_SAMPLE_RATE,
    AUDIO_CHUNK_DURATION,
    LOG_LEVEL,
)
from utils import encode_image_to_base64

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class LivestreamProducer:
    """Simulate livestream by reading video file and sending to Kafka"""

    def __init__(self, video_path: str, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        """
        Initialize producer

        Args:
            video_path: Path to video file
            kafka_servers: Kafka bootstrap servers
        """
        self.video_path = video_path
        self.kafka_servers = kafka_servers
        self.producer = None
        self.video_capture = None
        self.frame_count = 0
        self.audio_chunk_count = 0

        # Validate video file
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Initializing LivestreamProducer with video: {video_path}")

    def connect_kafka(self):
        """Connect to Kafka broker"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                max_request_size=10485760,  # 10MB for large frames
                compression_type="gzip",
                acks="all",
                retries=3,
            )
            logger.info(f"Connected to Kafka at {self.kafka_servers}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def open_video(self):
        """Open video file"""
        try:
            self.video_capture = cv2.VideoCapture(self.video_path)

            if not self.video_capture.isOpened():
                raise ValueError(f"Failed to open video: {self.video_path}")

            # Get video properties
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Video opened: FPS={fps}, Frames={frame_count}, Size={width}x{height}"
            )

        except Exception as e:
            logger.error(f"Failed to open video: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process video frame

        Args:
            frame: OpenCV frame

        Returns:
            Processed frame data
        """
        # Resize frame to standard size
        frame_resized = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))

        # Encode to base64
        frame_encoded = encode_image_to_base64(frame_resized)

        # Create message
        message = {
            "frame_id": self.frame_count,
            "timestamp": time.time(),
            "data": frame_encoded,
            "width": VIDEO_FRAME_WIDTH,
            "height": VIDEO_FRAME_HEIGHT,
        }

        return message

    def extract_audio_chunk(self) -> dict:
        """
        Simulate audio extraction (placeholder)
        In real implementation, use libraries like moviepy or ffmpeg

        Returns:
            Audio chunk data
        """
        message = {
            "chunk_id": self.audio_chunk_count,
            "timestamp": time.time(),
            "data": "",  # Placeholder for audio data
            "duration": AUDIO_CHUNK_DURATION,
            "sample_rate": 16000,
        }

        self.audio_chunk_count += 1
        return message

    def send_frame(self, frame_data: dict):
        """
        Send frame to Kafka

        Args:
            frame_data: Frame data dictionary
        """
        try:
            self.producer.send(KAFKA_TOPIC_VIDEO, value=frame_data)
            # Wait for confirmation (optional, can be removed for higher throughput)
            # future.get(timeout=10)
            logger.debug(f"Sent frame {frame_data['frame_id']} to {KAFKA_TOPIC_VIDEO}")
        except Exception as e:
            logger.error(f"Failed to send frame: {e}")

    def send_audio(self, audio_data: dict):
        """
        Send audio chunk to Kafka

        Args:
            audio_data: Audio data dictionary
        """
        try:
            self.producer.send(KAFKA_TOPIC_AUDIO, value=audio_data)
            logger.debug(
                f"Sent audio chunk {audio_data['chunk_id']} to {KAFKA_TOPIC_AUDIO}"
            )
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")

    def run(self, loop: bool = False):
        """
        Run the producer

        Args:
            loop: If True, loop video when it ends
        """
        try:
            self.connect_kafka()
            self.open_video()

            logger.info("Starting livestream simulation...")
            logger.info(
                f"Publishing to topics: {KAFKA_TOPIC_VIDEO}, {KAFKA_TOPIC_AUDIO}"
            )

            frames_per_audio = int(VIDEO_FPS * AUDIO_CHUNK_DURATION)

            while True:
                ret, frame = self.video_capture.read()

                if not ret:
                    if loop:
                        logger.info("Video ended, restarting from beginning...")
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_count = 0
                        continue
                    else:
                        logger.info("Video ended, stopping producer")
                        break

                # Process and send frame
                frame_data = self.process_frame(frame)
                self.send_frame(frame_data)

                # Send audio chunk periodically
                if self.frame_count % frames_per_audio == 0:
                    audio_data = self.extract_audio_chunk()
                    self.send_audio(audio_data)

                self.frame_count += 1

                # Display progress
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames")

                # Sleep to simulate real-time streaming
                time.sleep(VIDEO_SAMPLE_RATE)

            logger.info(f"Finished streaming. Total frames: {self.frame_count}")

        except KeyboardInterrupt:
            logger.info("Producer stopped by user")
        except Exception as e:
            logger.error(f"Producer error: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.video_capture:
            self.video_capture.release()
            logger.info("Video capture released")

        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Livestream Producer")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--kafka",
        type=str,
        default=KAFKA_BOOTSTRAP_SERVERS,
        help="Kafka bootstrap servers",
    )
    parser.add_argument("--loop", action="store_true", help="Loop video when it ends")

    args = parser.parse_args()

    producer = LivestreamProducer(args.video, args.kafka)
    producer.run(loop=args.loop)


if __name__ == "__main__":
    main()
