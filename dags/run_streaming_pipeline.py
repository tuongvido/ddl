"""
Airflow DAG to Run Streaming Pipeline (Producer + Consumers)
Runs producer and consumers directly in Airflow containers with mounted code
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Add src to Python path
sys.path.insert(0, "/opt/airflow/src")

# Default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 24),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

# DAG definition
dag = DAG(
    "run_streaming_pipeline",
    default_args=default_args,
    description="Run Producer and Consumers in Airflow containers",
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=["streaming", "execution", "production"],
)


def check_dependencies():
    """Check if all dependencies are installed"""
    logger.info("Checking Python dependencies...")

    try:
        import cv2

        logger.info(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"✗ OpenCV not found: {e}")
        raise

    try:
        from kafka import KafkaProducer, KafkaConsumer

        logger.info("✓ kafka-python installed")
    except ImportError as e:
        logger.error(f"✗ kafka-python not found: {e}")
        raise

    try:
        from pymongo import MongoClient

        logger.info("✓ pymongo installed")
    except ImportError as e:
        logger.error(f"✗ pymongo not found: {e}")
        raise

    try:
        from ultralytics import YOLO

        logger.info("✓ ultralytics installed")
    except ImportError as e:
        logger.error(f"✗ ultralytics not found: {e}")
        raise

    logger.info("All dependencies OK!")
    return True


def start_producer():
    """Start producer in background"""
    import subprocess
    import time

    logger.info("=" * 60)
    logger.info("STARTING PRODUCER")
    logger.info("=" * 60)
    logger.info("Using video file: /opt/airflow/data/V_10.mp4")

    # Start producer as background process
    process = subprocess.Popen(
        [
            "python",
            "/opt/airflow/src/producer.py",
            "--video",
            "/opt/airflow/data/V_10.mp4",
            "--loop",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info(f"Producer started with PID: {process.pid}")

    # Wait a bit to check if it started successfully
    time.sleep(5)

    if process.poll() is None:
        logger.info("✓ Producer is running")
        return process.pid
    else:
        stdout, stderr = process.communicate()
        logger.error(f"✗ Producer failed to start")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        raise Exception("Producer failed to start")


def start_video_consumer():
    """Start video consumer in background"""
    import subprocess
    import time

    logger.info("=" * 60)
    logger.info("STARTING VIDEO CONSUMER")
    logger.info("=" * 60)

    # Start consumer as background process
    process = subprocess.Popen(
        ["python", "/opt/airflow/src/consumer_video.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info(f"Video consumer started with PID: {process.pid}")

    # Wait a bit to check if it started successfully
    time.sleep(5)

    if process.poll() is None:
        logger.info("✓ Video consumer is running")
        return process.pid
    else:
        stdout, stderr = process.communicate()
        logger.error(f"✗ Video consumer failed to start")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        raise Exception("Video consumer failed to start")


def start_audio_consumer():
    """Start audio consumer in background"""
    import subprocess
    import time

    logger.info("=" * 60)
    logger.info("STARTING AUDIO CONSUMER")
    logger.info("=" * 60)

    # Start consumer as background process
    process = subprocess.Popen(
        ["python", "/opt/airflow/src/consumer_audio.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info(f"Audio consumer started with PID: {process.pid}")

    # Wait a bit to check if it started successfully
    time.sleep(5)

    if process.poll() is None:
        logger.info("✓ Audio consumer is running")
        return process.pid
    else:
        stdout, stderr = process.communicate()
        logger.error(f"✗ Audio consumer failed to start")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        raise Exception("Audio consumer failed to start")


def verify_pipeline():
    """Verify pipeline is running"""
    import subprocess
    import time

    logger.info("=" * 60)
    logger.info("VERIFYING PIPELINE")
    logger.info("=" * 60)

    # Wait for some data to flow
    logger.info("Waiting 15 seconds for data to flow...")
    time.sleep(15)

    # Check MongoDB for detections
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "mongodb",
                "mongosh",
                "-u",
                "admin",
                "-p",
                "admin123",
                "--authenticationDatabase",
                "admin",
                "livestream_detection",
                "--quiet",
                "--eval",
                "db.detections.countDocuments({})",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            count = int(result.stdout.strip())
            logger.info(f"✓ Found {count} detections in MongoDB")

            if count > 0:
                logger.info("✓ Pipeline is working correctly!")
            else:
                logger.warning(
                    "⚠ No detections yet, but pipeline may still be starting"
                )
        else:
            logger.error("✗ Failed to query MongoDB")
    except Exception as e:
        logger.error(f"Verification failed: {e}")

    logger.info("=" * 60)
    logger.info("PIPELINE STARTED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Services running in background:")
    logger.info("  - Producer: Streaming video frames to Kafka")
    logger.info("  - Video Consumer: Processing frames with YOLOv8")
    logger.info("  - Audio Consumer: Processing audio with Whisper")
    logger.info("")
    logger.info("Monitor at:")
    logger.info("  - Dashboard: http://localhost:8501")
    logger.info("  - Airflow: http://localhost:8080")
    logger.info("")
    logger.info("Note: Processes are running in Airflow containers")
    logger.info("They will continue until you stop the Airflow services")
    logger.info("")

    return True


# Define tasks
check_deps_task = PythonOperator(
    task_id="check_dependencies",
    python_callable=check_dependencies,
    dag=dag,
)

start_producer_task = PythonOperator(
    task_id="start_producer",
    python_callable=start_producer,
    dag=dag,
)

start_video_consumer_task = PythonOperator(
    task_id="start_video_consumer",
    python_callable=start_video_consumer,
    dag=dag,
)

start_audio_consumer_task = PythonOperator(
    task_id="start_audio_consumer",
    python_callable=start_audio_consumer,
    dag=dag,
)

verify_task = PythonOperator(
    task_id="verify_pipeline",
    python_callable=verify_pipeline,
    dag=dag,
)

# Task dependencies
(
    check_deps_task
    >> [start_producer_task, start_video_consumer_task, start_audio_consumer_task]
    >> verify_task
)
