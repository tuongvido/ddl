"""
Airflow DAG for Streaming Pipeline Monitoring
Monitors infrastructure health and data flow every 5 minutes
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
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
}

# DAG definition
dag = DAG(
    "streaming_continuous",
    default_args=default_args,
    description="Monitor streaming pipeline health (Kafka, MongoDB, Data Flow)",
    schedule_interval=timedelta(minutes=5),  # Check every 5 minutes
    catchup=False,
    max_active_runs=1,
    tags=["streaming", "monitoring", "health-check"],
)


def check_kafka_health():
    """Check if Kafka is healthy"""
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
            timeout=10,
        )
        if result.returncode == 0:
            topics = result.stdout.strip().split("\n")
            logger.info(f"✓ Kafka is healthy. Topics: {topics}")
            return True
        else:
            logger.warning("✗ Kafka is not responding properly")
            return False
    except Exception as e:
        logger.error(f"Kafka health check failed: {e}")
        return False


def check_mongodb_health():
    """Check if MongoDB is healthy"""
    try:
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
            logger.warning("✗ MongoDB is not responding")
            return False

        logger.info("✓ MongoDB is healthy")

        # Get detection count
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
            logger.info(f"✓ Total detections: {count}")
        return True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return False


def check_data_flow():
    """Check if data is flowing"""
    try:
        # Check for recent detections (last 5 minutes)
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
                "db.detections.countDocuments({timestamp: {$gte: new Date(Date.now() - 5*60*1000)}})",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            recent_count = int(result.stdout.strip())
            if recent_count > 0:
                logger.info(f"✓ Data flowing: {recent_count} detections in last 5 min")
                return True
            else:
                logger.warning("⚠ No new detections in last 5 minutes")
                return True  # Not critical

        return False
    except Exception as e:
        logger.error(f"Data flow check failed: {e}")
        return False


def generate_report():
    """Generate health report"""
    logger.info("=" * 60)
    logger.info("STREAMING PIPELINE HEALTH REPORT")
    logger.info("=" * 60)

    kafka_ok = check_kafka_health()
    mongo_ok = check_mongodb_health()
    data_ok = check_data_flow()

    logger.info("=" * 60)
    logger.info(f"Kafka:     {'✓ OK' if kafka_ok else '✗ FAILED'}")
    logger.info(f"MongoDB:   {'✓ OK' if mongo_ok else '✗ FAILED'}")
    logger.info(f"Data Flow: {'✓ OK' if data_ok else '⚠ IDLE'}")
    logger.info("=" * 60)

    if not kafka_ok or not mongo_ok:
        logger.error("CRITICAL: Infrastructure issues detected!")
        raise Exception("Infrastructure health check failed")

    logger.info("Pipeline monitoring complete ✓")
    return True


# Tasks
check_kafka_task = PythonOperator(
    task_id="check_kafka",
    python_callable=check_kafka_health,
    dag=dag,
)

check_mongodb_task = PythonOperator(
    task_id="check_mongodb",
    python_callable=check_mongodb_health,
    dag=dag,
)

check_data_flow_task = PythonOperator(
    task_id="check_data_flow",
    python_callable=check_data_flow,
    dag=dag,
)

report_task = PythonOperator(
    task_id="generate_report",
    python_callable=generate_report,
    dag=dag,
)

# Dependencies
[check_kafka_task, check_mongodb_task, check_data_flow_task] >> report_task
