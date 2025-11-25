"""
Airflow DAG for Streaming Pipeline Setup Verification
Verifies that all required services are ready for streaming
Note: This DAG only checks infrastructure. Run Producer/Consumers manually on host.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import subprocess

logger = logging.getLogger(__name__)

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
    "streaming_pipeline",
    default_args=default_args,
    description="Verify infrastructure for streaming pipeline (Manual trigger)",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=["streaming", "setup", "verification"],
)


def check_kafka():
    """Check if Kafka is running and accessible"""
    logger.info("=" * 60)
    logger.info("CHECKING KAFKA...")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "kafka",
                "kafka-topics",
                "--list",
                "--bootstrap-server",
                "localhost:9092",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            topics = [t for t in result.stdout.strip().split("\n") if t]
            logger.info(f"✓ Kafka is running")
            logger.info(f"✓ Found {len(topics)} topics: {topics}")

            # Check if our topics exist
            required_topics = ["livestream-video", "livestream-audio"]
            existing = [t for t in required_topics if t in topics]
            missing = [t for t in required_topics if t not in topics]

            if existing:
                logger.info(f"✓ Streaming topics ready: {existing}")
            if missing:
                logger.warning(f"⚠ Topics will be auto-created: {missing}")

            return True
        else:
            logger.error("✗ Kafka topics command failed")
            return False
    except Exception as e:
        logger.error(f"✗ Kafka check failed: {e}")
        return False


def check_mongodb():
    """Check if MongoDB is running and accessible"""
    logger.info("=" * 60)
    logger.info("CHECKING MONGODB...")
    logger.info("=" * 60)

    try:
        # Check connection
        result = subprocess.run(
            [
                "docker",
                "exec",
                "mongodb",
                "mongosh",
                "--quiet",
                "--eval",
                "db.version()",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error("✗ MongoDB is not responding")
            return False

        version = result.stdout.strip()
        logger.info(f"✓ MongoDB is running (version: {version})")

        # Check database
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
                "db.getCollectionNames()",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            logger.info("✓ Database 'livestream_detection' is accessible")

            # Check existing data
            result2 = subprocess.run(
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

            if result2.returncode == 0:
                count = int(result2.stdout.strip())
                if count > 0:
                    logger.info(f"✓ Found {count} existing detection records")
                else:
                    logger.info("ℹ Database is empty (ready for new data)")

        return True
    except Exception as e:
        logger.error(f"✗ MongoDB check failed: {e}")
        return False


def check_zookeeper():
    """Check if Zookeeper is running"""
    logger.info("=" * 60)
    logger.info("CHECKING ZOOKEEPER...")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            ["docker", "exec", "zookeeper", "zkServer.sh", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "Mode:" in result.stdout:
            logger.info("✓ Zookeeper is running")
            return True
        else:
            logger.warning("⚠ Zookeeper status unclear")
            return True  # Don't fail on this
    except Exception as e:
        logger.warning(f"⚠ Zookeeper check skipped: {e}")
        return True  # Don't fail on this


def generate_setup_report():
    """Generate final setup report with instructions"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("INFRASTRUCTURE VERIFICATION COMPLETE")
    logger.info("=" * 60)

    kafka_ok = check_kafka()
    mongo_ok = check_mongodb()
    zk_ok = check_zookeeper()

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS:")
    logger.info(f"  Kafka:     {'✓ READY' if kafka_ok else '✗ FAILED'}")
    logger.info(f"  MongoDB:   {'✓ READY' if mongo_ok else '✗ FAILED'}")
    logger.info(f"  Zookeeper: {'✓ READY' if zk_ok else '⚠ CHECK'}")
    logger.info("=" * 60)

    if not kafka_ok or not mongo_ok:
        logger.error("")
        logger.error("CRITICAL: Infrastructure not ready!")
        logger.error("Please fix the issues above before running the pipeline.")
        logger.error("")
        raise Exception("Infrastructure verification failed")

    logger.info("")
    logger.info("✓ ALL SYSTEMS READY!")
    logger.info("")
    logger.info("=" * 60)
    logger.info("TO START THE STREAMING PIPELINE:")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Run these commands on your HOST machine (not in Docker):")
    logger.info("")
    logger.info("  1. Start Producer (Terminal 1):")
    logger.info("     cd D:/Code/doan")
    logger.info("     .venv\\Scripts\\activate")
    logger.info("     python src/producer.py --video ./data/v001_converted.avi --loop")
    logger.info("")
    logger.info("  2. Start Video Consumer (Terminal 2):")
    logger.info("     cd D:/Code/doan")
    logger.info("     .venv\\Scripts\\activate")
    logger.info("     python src/consumer_video.py")
    logger.info("")
    logger.info("  3. Start Audio Consumer (Terminal 3):")
    logger.info("     cd D:/Code/doan")
    logger.info("     .venv\\Scripts\\activate")
    logger.info("     python src/consumer_audio.py")
    logger.info("")
    logger.info("  4. Start Dashboard (Terminal 4):")
    logger.info("     cd D:/Code/doan")
    logger.info("     .venv\\Scripts\\activate")
    logger.info("     python -m streamlit run src/dashboard.py")
    logger.info("")
    logger.info("=" * 60)
    logger.info("Monitor at: http://localhost:8501")
    logger.info("=" * 60)
    logger.info("")

    return True


# Define tasks
check_kafka_task = PythonOperator(
    task_id="check_kafka",
    python_callable=check_kafka,
    dag=dag,
)

check_mongodb_task = PythonOperator(
    task_id="check_mongodb",
    python_callable=check_mongodb,
    dag=dag,
)

check_zookeeper_task = PythonOperator(
    task_id="check_zookeeper",
    python_callable=check_zookeeper,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id="generate_setup_report",
    python_callable=generate_setup_report,
    dag=dag,
)

# Task dependencies - check all in parallel, then generate report
[check_kafka_task, check_mongodb_task, check_zookeeper_task] >> generate_report_task
