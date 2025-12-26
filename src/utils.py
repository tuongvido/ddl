"""
Utility functions for the Harmful Content Detection System
"""

import cv2
import numpy as np
import base64
import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from pymongo import MongoClient

from config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_USERNAME,
    MONGO_PASSWORD,
    MONGO_DB,
    MONGO_COLLECTION_DETECTIONS,
    MONGO_COLLECTION_ALERTS,
    MONGO_COLLECTION_VIDEO_SUMMARY,
    LOG_LEVEL,
    LOG_FORMAT,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for MongoDB compatibility"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


class MongoDBHandler:
    """Handle MongoDB connections and operations"""

    def __init__(self):
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        """Connect to MongoDB"""
        try:
            # Nếu không có user/pass, url sẽ gọn hơn. Logic này hỗ trợ cả 2.
            if MONGO_USERNAME and MONGO_PASSWORD:
                connection_string = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
            else:
                connection_string = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"

            self.client = MongoClient(connection_string)
            self.db = self.client[MONGO_DB]
            logger.info(f"Connected to MongoDB at {MONGO_HOST}:{MONGO_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def find_all_video_path(self) -> list[str]:
        cursor = self.db[MONGO_COLLECTION_VIDEO_SUMMARY].find(
            {},
            {"_id": 0, "video_info.video_path": 1}
        )

        return [
            doc["video_info"]["video_path"]
            for doc in cursor
            if "video_info" in doc and "video_path" in doc["video_info"]
        ]

    def save_detection(self, detection_data: Dict[str, Any]):
        """Save detection result to database"""
        try:
            # Thống nhất dùng Unix timestamp nếu chưa có
            if "timestamp" not in detection_data:
                detection_data["timestamp"] = time.time()

            # Convert numpy types to Python native types
            detection_data = convert_numpy_types(detection_data)

            result = self.db[MONGO_COLLECTION_DETECTIONS].insert_one(detection_data)
            logger.info(f"Saved detection with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save detection: {detection_data} | {e}")
            return None

    def save_alert(self, alert_data: Dict[str, Any]):
        """Save alert to database"""
        try:
            # Thống nhất dùng Unix timestamp
            if "timestamp" not in alert_data:
                alert_data["timestamp"] = time.time()

            result = self.db[MONGO_COLLECTION_ALERTS].insert_one(alert_data)
            logger.info(f"Saved alert: {alert_data.get('detection_type', 'UNKNOWN')}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
            return None

    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """Get recent detections"""
        try:
            detections = (
                self.db[MONGO_COLLECTION_DETECTIONS]
                .find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(detections)
        except Exception as e:
            logger.error(f"Failed to get detections: {e}")
            return []

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        try:
            alerts = (
                self.db[MONGO_COLLECTION_ALERTS]
                .find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(alerts)
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
        
    def get_detections_by_session(self, session_id) -> List[Dict]:
        return list(self.db[MONGO_COLLECTION_DETECTIONS].find({"session_id": session_id}))

    def get_alerts_by_session(self, session_id) -> List[Dict]:
        return list(self.db[MONGO_COLLECTION_ALERTS].find({"session_id": session_id}))

    def create_video_session(self, session_id: str, video_info: Dict) -> Any:
        """Create a new video session for tracking"""
        try:
            session_data = {
                "session_id": session_id,
                "start_time": time.time(),
                "end_time": None,
                "video_info": video_info,
                "is_toxic": False,
                "toxic_categories": [],
                "detection_counts": {
                    "violent_video": 0,
                    "violent_audio": 0,
                    "toxic_speech": 0,
                    "toxic_text": 0
                },
                "summary": None,
            }
            result = self.db[MONGO_COLLECTION_VIDEO_SUMMARY].insert_one(session_data)
            logger.info(f"Created video session: {session_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to create video session: {e}")
            return None

    def update_video_session(self, session_id: str, detection_data: Dict):
        """Update video session with new detection"""
        try:
            # Convert numpy types to Python native types
            detection_data = convert_numpy_types(detection_data)
            
            detection_type = detection_data.get("detection_type", "unknown")
            
            # Increment appropriate counter
            update_query = {}
            logger.info("detection_type.lower(): " + detection_type.lower())
            
            if "violent_video" in detection_type.lower():
                update_query["$inc"] = {"detection_counts.violent_video": 1}
                update_query["$addToSet"] = {"toxic_categories": "violent_video"}
                update_query["$set"] = {"is_toxic": True}
            elif "toxic_text_on_screen" in detection_type.lower():
                update_query["$inc"] = {"detection_counts.toxic_text": 1}
                update_query["$addToSet"] = {"toxic_categories": "toxic_text"}
                update_query["$set"] = {"is_toxic": True}
            elif "violent_audio" in detection_type.lower():
                update_query["$inc"] = {"detection_counts.violent_audio": 1}
                update_query["$addToSet"] = {"toxic_categories": "violent_audio"}
                update_query["$set"] = {"is_toxic": True}
            elif "toxic_speech" in detection_type.lower():
                update_query["$inc"] = {"detection_counts.toxic_speech": 1}
                update_query["$addToSet"] = {"toxic_categories": "toxic_speech"}
                update_query["$set"] = {"is_toxic": True}
            
            logger.info(f"Updated video session {session_id} with toxic speech detection.")
            if update_query:
                self.db[MONGO_COLLECTION_VIDEO_SUMMARY].update_one(
                    {"session_id": session_id},
                    update_query
                )
        except Exception as e:
            logger.error(f"Failed to update video session: {e}")

    def track_all_detections(self, session_id: str, all_detections: Dict):
        """Track all detections (including non-harmful) for comprehensive reporting"""
        try:
            self.db[MONGO_COLLECTION_VIDEO_SUMMARY].update_one(
                {"session_id": session_id},
                {
                    "$push": {"all_analyzed_data": all_detections},
                    "$inc": {
                        "analysis_stats.frames_processed": all_detections.get("frames_processed", 0),
                        "analysis_stats.audio_chunks_processed": all_detections.get("audio_chunks_processed", 0)
                    }
                }
            )
        except Exception as e:
            logger.error(f"Failed to track all detections: {e}")

    def finalize_video_session(self, session_id: str) -> Dict:
        """Finalize video session and generate detailed report"""
        try:
            session = self.db[MONGO_COLLECTION_VIDEO_SUMMARY].find_one({"session_id": session_id})
            
            if not session:
                logger.warning(f"Session {session_id} not found")
                return {}
            
            detections = session.get("detections", [])
            video_info = session.get("video_info", {})
            all_analyzed = session.get("all_analyzed_data", [])
            analysis_stats = session.get("analysis_stats", {})
            
            # Analyze ALL labels (harmful and non-harmful)
            video_labels = {}  # {label: count}
            audio_labels = {}  # {label: count}
            toxic_texts = []
            clean_texts = []
            
            # Get all video detections from MongoDB
            video_detections = list(self.db[MONGO_COLLECTION_DETECTIONS].find(
                {"session_id": session_id}
            ))
            
            # Analyze video frames
            for det in video_detections:
                if "detections" in det:
                    # This is video detection
                    for d in det.get("detections", []):
                        label = d.get("class", "unknown")
                        video_labels[label] = video_labels.get(label, 0) + 1
                
                elif "sound_label" in det:
                    # This is audio detection
                    sound_label = det.get("sound_label")
                    if sound_label and sound_label != "Speech":
                        audio_labels[sound_label] = audio_labels.get(sound_label, 0) + 1
                    
                    # Get transcribed text
                    text = det.get("transcribed_text", "").strip()
                    if text and len(text) > 5:
                        if det.get("is_toxic", False):
                            toxic_texts.append({
                                "text": text,
                                "timestamp": det.get("timestamp", 0),
                                "confidence": det.get("toxic_score", 0)
                            })
                        else:
                            # Keep some clean text samples too
                            if len(clean_texts) < 10:
                                clean_texts.append({
                                    "text": text,
                                    "timestamp": det.get("timestamp", 0)
                                })
            
            # Sort labels by frequency
            video_labels_sorted = sorted(video_labels.items(), key=lambda x: x[1], reverse=True)
            audio_labels_sorted = sorted(audio_labels.items(), key=lambda x: x[1], reverse=True)
            
            # Count frames and audio chunks
            frames_processed = len([d for d in video_detections if "frame_id" in d])
            audio_chunks_processed = len([d for d in video_detections if "chunk_id" in d])
            
            # Recalculate is_toxic based on actual detections
            actual_toxic_count = sum(session.get("detection_counts", {}).values())
            is_actually_toxic = actual_toxic_count > 0
            
            # Update session if toxic status changed
            if is_actually_toxic != session.get("is_toxic", False):
                self.db[MONGO_COLLECTION_VIDEO_SUMMARY].update_one(
                    {"session_id": session_id},
                    {"$set": {"is_toxic": is_actually_toxic}}
                )
            
            # Generate comprehensive summary
            summary = {
                "session_id": session_id,
                "video_info": {
                    "video_path": video_info.get("video_path", "unknown"),
                    "video_name": video_info.get("video_path", "unknown").split("/")[-1].split("\\")[-1],
                    "duration_seconds": video_info.get("duration_seconds", 0),
                    "total_frames": video_info.get("total_frames", 0),
                    "fps": video_info.get("fps", 0),
                },
                "analysis_time": {
                    "start_time": datetime.fromtimestamp(session.get("start_time", time.time())).strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_duration_seconds": time.time() - session.get("start_time", time.time())
                },
                "analysis_statistics": {
                    "frames_analyzed": frames_processed,
                    "audio_chunks_analyzed": audio_chunks_processed,
                    "total_video_labels_detected": len(video_labels),
                    "total_audio_labels_detected": len(audio_labels),
                },
                "is_toxic": is_actually_toxic,  # Use recalculated value
                "toxic_categories": session.get("toxic_categories", []),
                "detection_statistics": {
                    "total_detections": sum(session.get("detection_counts", {}).values()),
                    "violent_video_count": session.get("detection_counts", {}).get("violent_video", 0),
                    "violent_audio_count": session.get("detection_counts", {}).get("violent_audio", 0),
                    "toxic_speech_count": session.get("detection_counts", {}).get("toxic_speech", 0),
                },
                "detected_labels": {
                    "video_actions": [{"label": label, "count": count} for label, count in video_labels_sorted],
                    "audio_events": [{"label": label, "count": count} for label, count in audio_labels_sorted],
                },
                "toxic_speech_samples": toxic_texts[:10],  # Top 10 toxic speech samples
                "clean_speech_samples": clean_texts[:5],  # Sample clean speech
            }
            
            # Generate detailed report text
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"VIDEO ANALYSIS REPORT - {summary['video_info']['video_name']}")
            report_lines.append("=" * 80)
            report_lines.append(f"Session ID: {session_id}")
            report_lines.append(f"Video Path: {summary['video_info']['video_path']}")
            report_lines.append(f"Duration: {summary['video_info']['duration_seconds']:.2f}s")
            report_lines.append(f"Analysis Time: {summary['analysis_time']['start_time']} → {summary['analysis_time']['end_time']}")
            report_lines.append("")
            
            report_lines.append("ANALYSIS STATISTICS:")
            report_lines.append("-" * 80)
            report_lines.append(f"   Frames Analyzed: {summary['analysis_statistics']['frames_analyzed']}")
            report_lines.append(f"   Audio Chunks Analyzed: {summary['analysis_statistics']['audio_chunks_analyzed']}")
            report_lines.append(f"   Unique Video Actions Detected: {summary['analysis_statistics']['total_video_labels_detected']}")
            report_lines.append(f"   Unique Audio Events Detected: {summary['analysis_statistics']['total_audio_labels_detected']}")
            report_lines.append("")
            
            report_lines.append("DETECTION SUMMARY:")
            report_lines.append("-" * 80)
            if summary["is_toxic"]:
                report_lines.append(f"⚠️  TOXIC CONTENT DETECTED")
                report_lines.append(f"   Categories: {', '.join(summary['toxic_categories'])}")
                report_lines.append(f"   Total Toxic Detections: {summary['detection_statistics']['total_detections']}")
            else:
                report_lines.append(f"✅ NO TOXIC CONTENT DETECTED")
                report_lines.append(f"   Video analyzed successfully with no harmful content found")
            report_lines.append("")
            
            # Always show detected labels (harmful or not)
            if summary["detected_labels"]["video_actions"]:
                if summary["is_toxic"] and summary.get("toxic_categories") and "violent_video" in summary["toxic_categories"]:
                    report_lines.append("VIOLENT VIDEO ACTIONS DETECTED:")
                else:
                    report_lines.append("VIDEO ACTIONS DETECTED:")
                report_lines.append("-" * 80)
                for item in summary["detected_labels"]["video_actions"][:20]:  # Top 20
                    report_lines.append(f"   • {item['label']}: {item['count']} times")
                report_lines.append("")
            else:
                report_lines.append("VIDEO ACTIONS DETECTED:")
                report_lines.append("-" * 80)
                report_lines.append("   No significant video actions detected")
                report_lines.append("")
            
            if summary["detected_labels"]["audio_events"]:
                if summary["is_toxic"] and summary.get("toxic_categories") and "violent_audio" in summary["toxic_categories"]:
                    report_lines.append("VIOLENT AUDIO EVENTS DETECTED:")
                else:
                    report_lines.append("AUDIO EVENTS DETECTED:")
                report_lines.append("-" * 80)
                for item in summary["detected_labels"]["audio_events"][:20]:  # Top 20
                    report_lines.append(f"   • {item['label']}: {item['count']} times")
                report_lines.append("")
            else:
                report_lines.append("AUDIO EVENTS DETECTED:")
                report_lines.append("-" * 80)
                report_lines.append("   No significant audio events detected")
                report_lines.append("")
            
            if summary["toxic_speech_samples"]:
                report_lines.append("TOXIC SPEECH SAMPLES:")
                report_lines.append("-" * 80)
                for i, sample in enumerate(summary["toxic_speech_samples"][:5], 1):
                    timestamp_str = datetime.fromtimestamp(sample['timestamp']).strftime("%H:%M:%S")
                    report_lines.append(f"   {i}. [{timestamp_str}] (confidence: {sample['confidence']:.2%})")
                    report_lines.append(f"      \"{sample['text']}\"")
                report_lines.append("")
            
            if summary["clean_speech_samples"] and not summary["is_toxic"]:
                report_lines.append("SPEECH TRANSCRIPTION SAMPLES:")
                report_lines.append("-" * 80)
                for i, sample in enumerate(summary["clean_speech_samples"][:3], 1):
                    timestamp_str = datetime.fromtimestamp(sample['timestamp']).strftime("%H:%M:%S")
                    report_lines.append(f"   {i}. [{timestamp_str}] \"{sample['text'][:100]}...\"")
                report_lines.append("")
            
            report_lines.append("=" * 80)
            
            summary["report_text"] = "\n".join(report_lines)
            
            # Update session with summary
            self.db[MONGO_COLLECTION_VIDEO_SUMMARY].update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "end_time": time.time(),
                        "summary": summary,
                        "report_generated": True,
                        "is_toxic": is_actually_toxic  # Update with correct toxic status
                    }
                }
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to finalize video session: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def encode_image_to_base64(frame: np.ndarray, quality: int = 75) -> str:
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_param)

        if not success:
            raise ValueError("cv2.imencode failed")

        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""


def decode_base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV frame"""
    try:
        if not base64_string:
            return None
        jpg_original = base64.b64decode(base64_string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on frame (Skip if bbox is None)"""
    for det in detections:
        bbox = det.get("bbox")
        if bbox is None:
            continue  # Skip drawing if no bounding box (CLIP case)

        x1, y1, x2, y2 = bbox
        label = det["class"]
        conf = det["confidence"]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        label_text = f"{label}: {conf:.2f}"
        cv2.putText(
            frame,
            label_text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return frame


def check_toxic_content(text: str, toxic_keywords: List[str]) -> Dict[str, Any]:
    """Check if text contains toxic keywords (Basic substring matching)"""
    text_lower = text.lower()
    matched_keywords = []

    for keyword in toxic_keywords:
        if keyword.lower() in text_lower:
            matched_keywords.append(keyword)

    return {
        "is_toxic": len(matched_keywords) > 0,
        "matched_keywords": matched_keywords,
        "toxic_score": len(matched_keywords),
    }


def calculate_alert_level(detection_type, confidence):
    """Determine alert level based on confidence"""
    if isinstance(confidence, str):
        try:
            confidence = float(confidence.strip("%")) / 100.0
        except:
            confidence = 0.0

    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.60:
        return "MEDIUM"
    elif confidence >= 0.30:
        return "LOW"

    return "LOW"


def save_image_for_training(
    frame: np.ndarray, detection_type: str, save_dir: str = "../data/training_samples"
) -> str:
    """Save detected frame for future model training"""
    try:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{detection_type}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        logger.info(f"Saved training sample: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save training image: {e}")
        return ""


class AlertThrottler:
    """Prevent alert spam by throttling similar alerts"""

    def __init__(self, cooldown_seconds: int = 5):
        self.cooldown = cooldown_seconds
        self.last_alerts = {}

    def should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed using Unix timestamp"""
        current_time = time.time()  # Use Unix timestamp

        if alert_type in self.last_alerts:
            time_diff = current_time - self.last_alerts[alert_type]
            if time_diff < self.cooldown:
                return False

        self.last_alerts[alert_type] = current_time
        return True
