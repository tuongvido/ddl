"""
Airflow DAG for automated model retraining pipeline
Triggers Google Colab training through VS Code tunnel
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    "owner": "harmful_detection_system",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    "retrain_harmful_detection_model",
    default_args=default_args,
    description="Automated model retraining pipeline for harmful content detection",
    schedule_interval="@daily",  # Run daily, can be changed to @weekly or cron expression
    catchup=False,
    tags=["machine_learning", "retraining", "harmful_detection"],
)


def check_new_data(**context):
    """
    Check if there are enough new samples to trigger retraining
    """
    from pathlib import Path

    training_samples_dir = Path("/opt/airflow/data/training_samples")

    if not training_samples_dir.exists():
        logger.info("Training samples directory does not exist")
        return False

    # Count new samples
    sample_files = list(training_samples_dir.glob("*.jpg"))
    num_samples = len(sample_files)

    logger.info(f"Found {num_samples} training samples")

    # Minimum threshold for retraining
    MIN_SAMPLES = 100

    if num_samples >= MIN_SAMPLES:
        logger.info(
            f"Sufficient samples ({num_samples} >= {MIN_SAMPLES}), proceeding with retraining"
        )
        return True
    else:
        logger.info(
            f"Insufficient samples ({num_samples} < {MIN_SAMPLES}), skipping retraining"
        )
        return False


def prepare_training_data(**context):
    """
    Prepare and organize training data
    """
    import shutil
    from pathlib import Path

    logger.info("Preparing training data...")

    source_dir = Path("/opt/airflow/data/training_samples")
    dest_dir = Path("/opt/airflow/data/training_dataset")

    # Create destination directory structure
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "images").mkdir(exist_ok=True)
    (dest_dir / "labels").mkdir(exist_ok=True)

    # Copy images to training dataset
    sample_files = list(source_dir.glob("*.jpg"))

    for i, sample_file in enumerate(sample_files):
        dest_file = dest_dir / "images" / f"sample_{i:05d}.jpg"
        shutil.copy(sample_file, dest_file)

    logger.info(f"Prepared {len(sample_files)} samples for training")

    # Create data.yaml for YOLO training
    data_yaml = dest_dir / "data.yaml"
    yaml_content = f"""
# Training configuration
path: {dest_dir}
train: images
val: images

# Classes
nc: 10
names: ['knife', 'gun', 'weapon', 'blood', 'violence', 'fight', 'pistol', 'rifle', 'sword', 'axe']
"""

    with open(data_yaml, "w") as f:
        f.write(yaml_content)

    logger.info("Training data preparation completed")

    return str(dest_dir)


def sync_to_google_drive(**context):
    """
    Sync training data to Google Drive
    (In production, use rclone or Google Drive API)
    """
    logger.info("Syncing data to Google Drive...")
    logger.info("Note: In production, implement actual Google Drive sync")

    # Placeholder for Google Drive sync
    # In real implementation:
    # 1. Use rclone to sync to Google Drive
    # 2. Or use Google Drive API to upload files
    # 3. Or mount Google Drive and copy files

    return True


# Task 1: Check if there are new training samples
check_data_task = PythonOperator(
    task_id="check_new_training_data",
    python_callable=check_new_data,
    provide_context=True,
    dag=dag,
)

# Task 2: Prepare training data
prepare_data_task = PythonOperator(
    task_id="prepare_training_data",
    python_callable=prepare_training_data,
    provide_context=True,
    dag=dag,
)

# Task 3: Sync data to Google Drive
sync_drive_task = PythonOperator(
    task_id="sync_to_google_drive",
    python_callable=sync_to_google_drive,
    provide_context=True,
    dag=dag,
)

# Task 4: Trigger training on Google Colab
# Note: This requires setting up HTTP connection in Airflow UI
# Connection ID: colab_local_conn
# Host: http://localhost:8000 (or your VS Code tunnel endpoint)
trigger_training_task = SimpleHttpOperator(
    task_id="trigger_colab_training",
    http_conn_id="colab_local_conn",
    endpoint="/train",
    method="POST",
    data='{"epochs": 10, "batch_size": 16, "model": "yolov8n.pt"}',
    headers={"Content-Type": "application/json"},
    response_check=lambda response: response.status_code == 200,
    dag=dag,
)

# Task 5: Wait for training to complete and check status
check_training_status_task = SimpleHttpOperator(
    task_id="check_training_status",
    http_conn_id="colab_local_conn",
    endpoint="/status",
    method="GET",
    response_check=lambda response: response.json().get("status") == "completed",
    dag=dag,
)


# Task 6: Download and deploy new model
def download_new_model(**context):
    """
    Download trained model from Google Colab/Drive
    """
    logger.info("Downloading new model...")
    logger.info(
        "Note: In production, implement actual model download from Google Drive"
    )

    # Placeholder for model download
    # In real implementation:
    # 1. Download model from Google Drive
    # 2. Validate model performance
    # 3. Deploy to models/ directory
    # 4. Restart consumers to load new model

    return True


download_model_task = PythonOperator(
    task_id="download_new_model",
    python_callable=download_new_model,
    provide_context=True,
    dag=dag,
)


# Task 7: Archive processed training samples
def archive_training_samples(**context):
    """
    Move processed samples to archive
    """
    import shutil
    from pathlib import Path
    from datetime import datetime

    logger.info("Archiving training samples...")

    source_dir = Path("/opt/airflow/data/training_samples")
    archive_dir = Path(f"/opt/airflow/data/archive/{datetime.now().strftime('%Y%m%d')}")

    if source_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Move all samples to archive
        sample_files = list(source_dir.glob("*.jpg"))
        for sample_file in sample_files:
            shutil.move(str(sample_file), str(archive_dir / sample_file.name))

        logger.info(f"Archived {len(sample_files)} samples to {archive_dir}")

    return True


archive_task = PythonOperator(
    task_id="archive_training_samples",
    python_callable=archive_training_samples,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
check_data_task >> prepare_data_task >> sync_drive_task >> trigger_training_task
(
    trigger_training_task
    >> check_training_status_task
    >> download_model_task
    >> archive_task
)
