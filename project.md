# Đồ Án Cao Học: Hệ Thống Phát Hiện Nội Dung Độc Hại Thời Gian Thực Trên Livestream

**(Real-time Harmful Livestream Detection System)**

## 1\. Tổng Quan (Overview)

### 1.1. Vấn đề

Sự bùng nổ của livestream đặt ra thách thức lớn trong việc kiểm duyệt nội dung (bạo lực, vũ khí, lời nói thù ghét). Các phương pháp thủ công không đáp ứng kịp tốc độ và khối lượng dữ liệu.

### 1.2. Mục tiêu

Xây dựng hệ thống Big Data xử lý luồng (Streaming Processing) với các khả năng:

1.  **Giả lập Livestream:** Sử dụng file video có sẵn để giả lập luồng dữ liệu thời gian thực.
2.  **Đa phương thức (Multimodal):** Phát hiện đồng thời hình ảnh (bạo lực, vũ khí) và âm thanh (từ ngữ độc hại).
3.  **Tự động hóa (Automation):** Sử dụng **Apache Airflow** để quản lý quy trình.
4.  **Học liên tục (Continuous Learning):** Cơ chế Hybrid kết hợp **Local** (điều phối) và **Google Colab** (Training GPU) thông qua **VS Code Tunnel**, tối ưu chi phí phần cứng.

---

## 2\. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống hoạt động theo mô hình xử lý phân tán, tách biệt giữa môi trường chạy (Runtime) và môi trường huấn luyện (Training).

### Sơ đồ luồng dữ liệu (Data Flow)

```mermaid
graph TD
    subgraph "1. Data Source (Local)"
        A[File Video .mp4] -->|Python Script| B(Simulated Producer)
    end

    subgraph "2. Streaming Layer (Local - Docker)"
        B -->|Frame Ảnh| C1{Kafka Topic: Video}
        B -->|Audio Chunk| C2{Kafka Topic: Audio}
    end

    subgraph "3. Processing Layer (Local Workers)"
        C1 -->|Consumer 1| D[AI Vision Model<br/>(YOLOv8)]
        C2 -->|Consumer 2| E[AI Audio Model<br/>(Whisper + NLP)]
        D --> F[Database / Dashboard]
        E --> F
        D -.->|Lưu dữ liệu mới| G[(Local Dataset)]
    end

    subgraph "4. Orchestration (Local)"
        H[Apache Airflow] -->|Trigger HTTP| I[VS Code Port Forwarding<br/>(localhost:8000)]
        I -->|Tunnel SSH| J[Google Colab API]
    end

    subgraph "5. Training Env (Google Colab)"
        G -.->|Sync Google Drive| K[Drive Storage]
        K --> J
        J -->|Retrain Model| L[New Weights]
        L -.->|Download| D
    end
```

---

## 3\. Công Nghệ Sử Dụng (Tech Stack)

| Thành phần         | Công nghệ                  | Vai trò                                                                                      |
| :----------------- | :------------------------- | :------------------------------------------------------------------------------------------- |
| **Input Source**   | **Python OpenCV**          | Đọc file video, cắt frame và gửi vào hệ thống giả lập livestream.                            |
| **Message Queue**  | **Apache Kafka**           | Hàng đợi thông điệp, chịu tải cao, điều tiết dữ liệu giữa Producer và Consumer.              |
| **Vision AI**      | **YOLOv8 / ResNet**        | Nhận diện vật thể (dao, súng) và hành động bạo lực.                                          |
| **Audio AI**       | **OpenAI Whisper**         | Chuyển đổi giọng nói thành văn bản (Speech-to-Text) để phân tích ngôn ngữ.                   |
| **Orchestrator**   | **Apache Airflow**         | Lên lịch huấn luyện lại model (Retrain) định kỳ.                                             |
| **Training Infra** | **Google Colab + VS Code** | Tận dụng GPU miễn phí của Google. Sử dụng **VS Code Remote Tunnels** để kết nối mạng LAN ảo. |
| **Dashboard**      | **Streamlit**              | Giao diện web hiển thị cảnh báo thời gian thực.                                              |

---

## 4\. Chi Tiết Triển Khai (Implementation Guide)

### Bước 1: Giả lập Livestream (Data Ingestion)

Thay vì dùng Camera, ta viết script Python đọc file video và đẩy vào Kafka liên tục.

_File: `src/producer.py`_

```python
import cv2
import time
import json
import base64
from kafka import KafkaProducer

# Cấu hình Kafka
producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

video_path = 'data/test_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break # Hết video thì dừng hoặc loop lại

    # Resize và mã hóa ảnh sang base64
    frame = cv2.resize(frame, (640, 640))
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Gửi vào Kafka
    msg = {'timestamp': time.time(), 'data': jpg_as_text}
    producer.send('livestream-video', value=msg)

    time.sleep(0.04) # Giả lập 25 FPS
```

### Bước 2: Thiết lập kết nối Hybrid (VS Code \<-\> Colab)

Đây là bước quan trọng để Airflow ở Local gọi được Colab mà không cần Public IP.

1.  **Trên Google Colab:** Cài đặt SSH server.
    ```python
    # Chạy cell này trên Notebook
    !pip install colab_ssh --upgrade
    from colab_ssh import launch_ssh_cloudflared
    launch_ssh_cloudflared(password="matkhau123")
    ```
2.  **Trên VS Code (Local):**
    - Cài Extension: **Remote - SSH**.
    - Kết nối vào Colab theo config mà code trên in ra.
3.  **Setup Port Forwarding:**
    - Trên Colab, chạy một API server (FastAPI) lắng nghe port `8000`.
    - Trên VS Code: Mở tab **PORTS** $\rightarrow$ Add Port `8000`.
    - _Kết quả:_ Truy cập `http://localhost:8000` trên máy Local sẽ trỏ thẳng tới Colab.

### Bước 3: Airflow DAG (Tự động Retrain)

Airflow sẽ gọi API training thông qua `localhost` nhờ Port Forwarding.

_File: `dags/retrain_pipeline.py`_

```python
from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.dates import days_ago

default_args = {'owner': 'student'}

with DAG('retrain_harmful_model', default_args=default_args, schedule_interval='@daily', start_date=days_ago(1)) as dag:

    # Task: Gọi API trên Colab để bắt đầu train
    trigger_training = SimpleHttpOperator(
        task_id='trigger_colab_training',
        http_conn_id='colab_local_conn', # Cấu hình Host: http://localhost:8000
        endpoint='/train',
        method='POST',
        data='{"epochs": 10, "data_path": "/content/drive/MyDrive/new_data"}',
        headers={"Content-Type": "application/json"},
    )
```

### Bước 4: Xử lý & Phát hiện (Consumer)

Worker nhận ảnh từ Kafka, decode và chạy qua Model.

```python
# Pseudo-code cho Consumer
for msg in consumer:
    # 1. Decode ảnh
    img = decode_base64(msg.value['data'])

    # 2. Predict (Dùng YOLO)
    results = model(img)

    # 3. Logic nghiệp vụ
    if results.conf > 0.8 and results.class in ['knife', 'gun', 'blood']:
        send_alert("Phát hiện nguy hiểm!")
        save_image_to_drive(img) # Lưu để retrain sau này
```

---

## 5\. Cấu Trúc Thư Mục Dự Án (Project Structure)

Để đồ án chuyên nghiệp, hãy tổ chức thư mục như sau:

```text
my-thesis-project/
├── dags/                   # Chứa các file định nghĩa luồng Airflow
│   └── retrain_pipeline.py
├── docker/                 # Cấu hình Docker
│   └── docker-compose.yml  # Định nghĩa Kafka, Zookeeper, Airflow
├── src/                    # Mã nguồn chính
│   ├── producer.py         # Giả lập livestream từ file
│   ├── consumer_video.py   # Xử lý hình ảnh
│   ├── consumer_audio.py   # Xử lý âm thanh
│   └── utils.py
├── notebooks/              # File chạy trên Google Colab
│   └── Colab_Training_Server.ipynb
├── data/                   # Chứa video input và dataset
│   └── sample_video.mp4
└── models/                 # Chứa file weights (.pt, .h5)
    └── yolo_best.pt
```

---

## 6\. Điểm Nhấn Của Đồ Án (Key Highlights)

1.  **Tiết kiệm chi phí:** Giải quyết bài toán cần GPU mạnh (để train model) mà không tốn tiền thuê Cloud Server nhờ kết hợp khéo léo Google Colab và VS Code.
2.  **Tính thực tế:** Kiến trúc Message Queue (Kafka) cho phép hệ thống mở rộng (Scale) dễ dàng, giống với các hệ thống production thực tế.
3.  **Quy trình khép kín (End-to-End MLOps):** Không chỉ dừng lại ở việc chạy model, mà còn có quy trình thu thập dữ liệu mới và cập nhật model tự động.

---

## 7\. Các Bước Cần Làm Tiếp Theo

1.  **Cài Docker:** Cài Docker Desktop lên máy Local để chạy Kafka và Airflow dễ dàng.
2.  **Chuẩn bị Dataset:** Tìm một model YOLO đã pre-train nhận diện vũ khí (có nhiều trên Roboflow) để chạy demo ngay.
3.  **Code phần Producer:** Viết script đọc file video gửi Kafka trước tiên để đảm bảo luồng dữ liệu chạy ổn.
