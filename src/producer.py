import cv2
import time
import logging
import argparse
import json
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError
import numpy as np
from moviepy import VideoFileClip
import base64
import tempfile
import os
import uuid

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
    DATA_DIR,
    SCAN_INTERVAL,
    SUPPORTED_EXTS
)
from utils import encode_image_to_base64, MongoDBHandler

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)
db_handler = MongoDBHandler()


class VideoProducer:

    def __init__(self, video_path: str, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):

        self.video_path = video_path
        self.kafka_servers = kafka_servers
        self.producer = None
        self.video_capture = None
        self.frame_count = 0
        self.audio_chunk_count = 0
        
        # Generate unique session ID for this video
        self.session_id = str(uuid.uuid4())

        try:
            self.video_clip = VideoFileClip(video_path)
            self.audio_clip = self.video_clip.audio
            logger.info("Audio track loaded successfully")
        except Exception as e:
            logger.warning(f"No audio track found: {e}")
            self.audio_clip = None

        # Validate video file
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Initializing VideoProducer with video: {video_path}")
        logger.info(f"Session ID: {self.session_id}")

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
        # Encode to base64
        frame_encoded = encode_image_to_base64(frame)

        # Create message
        message = {
            "frame_id": self.frame_count,
            "timestamp": time.time(),
            "data": frame_encoded,
            "width": VIDEO_FRAME_WIDTH,
            "height": VIDEO_FRAME_HEIGHT,
            "session_id": self.session_id,  # Add session ID to each frame
        }

        return message

    def extract_audio_chunk(self) -> dict:
        """
        Extract audio chunk corresponding to current time
        """
        if self.audio_clip is None:
            return self._empty_audio_msg()

        try:
            # Tính thời gian hiện tại dựa trên số frame
            current_time = self.frame_count / VIDEO_FPS
            duration = AUDIO_CHUNK_DURATION  # Ví dụ 1.0 giây

            # Cắt audio (subclip)
            # Lưu ý: Cần xử lý biên (nếu current_time + duration > tổng thời lượng)
            end_time = min(current_time + duration, self.audio_clip.duration)
            if current_time >= end_time:
                return self._empty_audio_msg()

            sub_clip = self.audio_clip.subclipped(current_time, end_time)

            # Ghi ra file tạm wav -> Đọc bytes -> Base64
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_filename = f.name

            # write_audiofile của moviepy có thể in log, ta tắt verbose
            sub_clip.write_audiofile(
                temp_filename,
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                logger=None,
            )

            with open(temp_filename, "rb") as f:
                audio_bytes = f.read()

            os.remove(temp_filename)  # Dọn dẹp

            # Encode Base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "chunk_id": self.audio_chunk_count,
                "timestamp": time.time(),
                "data": audio_b64,  # Dữ liệu thật
                "duration": duration,
                "sample_rate": 16000,
                "session_id": self.session_id,  # Add session ID to audio
            }

        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return self._empty_audio_msg()

    def _empty_audio_msg(self) -> dict:
        """
        Return empty audio message when no audio is available
        """
        return {
            "chunk_id": self.audio_chunk_count,
            "timestamp": time.time(),
            "data": "",  # Empty data
            "duration": 0,
            "sample_rate": 16000,
            "session_id": self.session_id,
        }

    def send_frame(self, frame_data: dict):
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

    def run(self):
        try:
            self.connect_kafka()
            self.open_video()
            
            # Get video info
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Create video session in database
            video_info = {
                "video_path": str(self.video_path),
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": duration,
            }
            db_handler.create_video_session(self.session_id, video_info)

            logger.info("Starting livestream simulation...")
            logger.info(
                f"Publishing to topics: {KAFKA_TOPIC_VIDEO}, {KAFKA_TOPIC_AUDIO}"
            )

            frames_per_audio = int(VIDEO_FPS * AUDIO_CHUNK_DURATION)

            while True:
                ret, frame = self.video_capture.read()

                if not ret:
                    logger.info("Video ended, finalizing session...")
                    # Finalize session and get summary
                    summary = db_handler.finalize_video_session(self.session_id)
                    logger.info(f"📊 Final Summary: {summary.get('toxic_summary', 'No summary')}")
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
            # Finalize session on interrupt
            summary = db_handler.finalize_video_session(self.session_id)
            logger.info(f"📊 Session Summary: {summary.get('toxic_summary', 'No summary')}")
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
    parser = argparse.ArgumentParser(description="“Detection of harmful content")
    parser.add_argument(
        "--kafka",
        type=str,
        default=KAFKA_BOOTSTRAP_SERVERS,
        help="Kafka bootstrap servers",
    )

    args = parser.parse_args()

    processed_videos = set()

    while True:
        logger.info(f"📂 Watching folder: {DATA_DIR.resolve()}")
        try:
            video_files = sorted(
                [
                    f for f in DATA_DIR.iterdir()
                    if f.suffix.lower() in SUPPORTED_EXTS
                ]
            )

            for video_path in video_files:
                video_path_str = str(video_path.resolve())

                if video_path_str in processed_videos:
                    continue

                # 👉 Skip nếu video đã từng xử lý
                if db_handler.video_exists(video_path_str):
                    logger.info(f"⏭ Skip processed video: {video_path.name}")
                    processed_videos.add(video_path_str)
                    continue

                logger.info(f"🎬 New video detected: {video_path.name}")

                try:
                    producer = VideoProducer(
                        video_path=str(video_path),
                        kafka_servers=args.kafka,
                    )
                    producer.run()

                    processed_videos.add(video_path.name)
                    logger.info(f"✅ Finished processing: {video_path.name}")

                except Exception as e:
                    logger.error(f"❌ Failed processing {video_path.name}: {e}")

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            logger.info("🛑 Folder watcher stopped by user")
            break


if __name__ == "__main__":
    main()
