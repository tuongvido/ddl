"""
Airflow DAG to Run Streaming Pipeline (Blocking Mode)
Tasks will run until they finish (or loop forever) and stream logs to UI.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

# Default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 24),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,  # Không retry nếu script chạy mãi mãi
}

# DAG definition
dag = DAG(
    "run_streaming_pipeline_blocking",
    default_args=default_args,
    description="Run Producer and Consumers in Blocking Mode (Real-time Logs)",
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=["streaming", "blocking", "debug"],
)

# Đường dẫn chuẩn
SRC_DIR = "/opt/airflow/src"
DATA_DIR = "/opt/airflow/data"

# 1. Task chạy Producer
# Producer thường sẽ kết thúc khi hết video (nếu không loop)
# Log sẽ hiện ra từng dòng "Sent frame X..."
start_producer = BashOperator(
    task_id="start_producer",
    bash_command=f"cd {SRC_DIR} && python -u producer.py --video {DATA_DIR}/v001_converted.avi",
    dag=dag,
)

# 2. Task chạy Video Consumer
# Lưu ý: Consumer chạy vòng lặp vô tận (while True).
# Task này sẽ KHÔNG BAO GIỜ SUCCESS (mãi mãi màu xanh lá nhạt - Running)
# Trừ khi bạn stop bằng tay hoặc code bị lỗi crash.
start_video_consumer = BashOperator(
    task_id="start_video_consumer",
    bash_command=f"cd {SRC_DIR} && python -u consumer_video.py",
    dag=dag,
)

# 3. Task chạy Audio Consumer
start_audio_consumer = BashOperator(
    task_id="start_audio_consumer",
    bash_command=f"cd {SRC_DIR} && python -u consumer_audio.py",
    dag=dag,
)

# --- CẤU HÌNH THỨ TỰ CHẠY ---

# LƯU Ý QUAN TRỌNG:
# Vì các Consumer chạy vô tận (Blocking), nếu bạn nối mũi tên (>>) theo kiểu tuần tự:
# producer >> consumer
# Thì Consumer sẽ chỉ chạy SAU KHI Producer kết thúc.

# Nếu bạn muốn 3 cái chạy song song cùng lúc để test Real-time:
# Bạn KHÔNG được nối dependency giữa chúng. Hãy để chúng độc lập.

# Khi trigger DAG, cả 3 task sẽ được Airflow chạy đồng thời (nếu Worker đủ slots).
