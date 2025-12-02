# Hệ Thống Phát Hiện Nội Dung Độc Hại Thời Gian Thực Trên Livestream

**(Real-time Harmful Livestream Detection System)**

## 📋 Tổng Quan

Đồ án cao học về xây dựng hệ thống Big Data xử lý luồng để phát hiện nội dung độc hại (bạo lực, vũ khí, lời nói thù ghét) trong livestream. Hệ thống sử dụng kiến trúc đa phương thức (multimodal) với khả năng học liên tục.

### ✨ Tính Năng Chính

- 🎥 **Giả lập Livestream**: Sử dụng file video để giả lập luồng dữ liệu thời gian thực
- 🤖 **Phát hiện đa phương thức**:
  - Hình ảnh: YOLOv8 để nhận diện vũ khí, bạo lực
  - Âm thanh: Whisper + NLP để phát hiện ngôn ngữ độc hại
- 📊 **Dashboard thời gian thực**: Streamlit dashboard để giám sát và cảnh báo
- 🔄 **Học liên tục**: Tự động thu thập dữ liệu và huấn luyện lại model
- ☁️ **Hybrid Architecture**: Kết hợp Local (runtime) + Google Colab (training GPU)

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────┐
│  Video Source   │
│   (.mp4 file)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Producer     │  ← Giả lập livestream
│  (OpenCV + CV2) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Apache Kafka   │  ← Message Queue
│  (2 topics)     │
└───┬─────────┬───┘
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Video  │ │ Audio  │  ← Consumers
│Consumer│ │Consumer│
│(YOLO)  │ │(Whisper)
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌──────────────────┐
│    MongoDB       │  ← Storage
│  + Dashboard     │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│  Apache Airflow  │  ← Orchestration
│  (Retrain DAG)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Google Colab    │  ← GPU Training
│  (via VS Code    │
│   Tunnel)        │
└──────────────────┘
```

## 🚀 Cài Đặt và Chạy

### Yêu Cầu Hệ Thống

- **Docker Desktop** (Windows/Mac) hoặc Docker Engine (Linux)
- **Python 3.8+**
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Disk**: 10GB dung lượng trống
- **GPU** (tùy chọn): NVIDIA GPU với CUDA để chạy model nhanh hơn

### Bước 1: Clone Repository và Chuẩn Bị

```bash
# Clone repository (nếu có)
cd d:\Code\doan

# Tạo môi trường ảo Python
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Cấu Hình Môi Trường

```bash
# Copy file cấu hình mẫu
cp .env.example .env

# Chỉnh sửa .env nếu cần (thông thường không cần thay đổi gì)
```

### Bước 3: Khởi Động Docker Services

```bash
# Di chuyển đến thư mục docker
cd docker

# Khởi động tất cả services
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps
```

**Services sẽ chạy trên các port:**

- Kafka: `localhost:9092`
- Zookeeper: `localhost:2181`
- Airflow Web UI: `http://localhost:8080` (admin/admin)
- MongoDB: `localhost:27017`
- PostgreSQL (Airflow): `localhost:5432`

### Bước 4: Chuẩn Bị Dữ Liệu

```bash
# Tạo thư mục cho video test
mkdir -p data

# Copy video test vào thư mục data/
# Ví dụ: data/test_video.mp4
```

**Lưu ý**: Bạn cần chuẩn bị một file video (.mp4) để test hệ thống.

### Bước 5: Tải Model YOLOv8

```bash
# Tạo thư mục models
mkdir -p models

# Download model mặc định (tự động khi chạy lần đầu)
# Hoặc download model đã train sẵn nếu có
```

### Bước 6: Chạy Hệ Thống

#### Phương Án 1: Sử Dụng Airflow DAG (Khuyến Nghị) ⭐

**Chạy Producer + Consumers trong Docker containers:**

1. **Rebuild Airflow với dependencies:**
   ```powershell
   .\rebuild_airflow.ps1
   ```
2. **Đợi 30 giây** để services khởi động

3. **Trigger DAG:**
   - Mở Airflow UI: `http://localhost:8080`
   - Login: `admin` / `admin`
   - Tìm DAG: **`run_streaming_pipeline`**
   - Click **Trigger DAG** (▶️)
4. **DAG sẽ tự động:**

   - ✅ Check dependencies
   - ✅ Start Producer (video V_10.mp4 loop)
   - ✅ Start Video Consumer (YOLOv8)
   - ✅ Start Audio Consumer (Whisper)
   - ✅ Verify pipeline hoạt động

5. **Mở Dashboard** (optional):
   ```powershell
   cd src
   python -m streamlit run dashboard.py
   ```

**Lưu ý**: Processes chạy trong Docker containers, sẽ tiếp tục chạy background sau khi DAG hoàn thành.

#### Phương Án 2: Chạy Thủ Công Trên Host

**Nếu muốn kiểm soát chi tiết hoặc debug:**

```bash
# Terminal 1 - Producer
cd src
python producer.py --video ../data/V_10.mp4 --loop

# Terminal 2 - Video Consumer
cd src
python consumer_video.py

# Terminal 3 - Audio Consumer
cd src
python consumer_audio.py

# Terminal 4 - Dashboard
cd src
streamlit run dashboard.py
```

## 📊 Sử Dụng Dashboard

1. Mở trình duyệt tại `http://localhost:8501`
2. Xem các tab:
   - **Overview**: Tổng quan hệ thống, số liệu thống kê
   - **Alerts**: Cảnh báo thời gian thực
   - **Video Detection**: Kết quả phát hiện từ video
   - **Audio Detection**: Kết quả phát hiện từ audio

## 🔄 Cấu Hình Airflow

### 1. Truy cập Airflow UI

- URL: `http://localhost:8080`
- Username: `admin`
- Password: `admin`

### 2. DAG Có Sẵn

#### A. `streaming_pipeline` - Pipeline Chạy Thủ Công

- **Mục đích**: Chạy toàn bộ pipeline khi cần
- **Cách dùng**: Click trigger DAG trong Airflow UI
- **Tính năng**:
  - ✅ Kiểm tra Kafka và MongoDB trước khi chạy
  - ✅ Start Producer với video hard-code (`V_10.mp4`)
  - ✅ Start Video + Audio Consumers song song
  - ✅ Verify pipeline sau khi khởi động

#### B. `streaming_continuous` - Pipeline Tự Động 24/7

- **Mục đích**: Giữ pipeline chạy liên tục
- **Cách dùng**: Toggle ON trong Airflow UI
- **Tính năng**:
  - ✅ Auto-start Producer và Consumers
  - ✅ Monitor health liên tục
  - ✅ Auto-restart nếu process crash
  - ✅ Retry vô hạn

#### C. `retrain_harmful_detection_model` - Retraining Pipeline

- **Mục đích**: Tự động huấn luyện lại model
- **Cách dùng**: Chạy theo lịch (daily) hoặc manual trigger
- **Yêu cầu**: Cấu hình Google Colab connection (xem bên dưới)

### 3. Cấu hình HTTP Connection cho Retraining

**Chỉ cần thiết nếu muốn dùng auto-retraining với Google Colab:**

1. Vào **Admin > Connections**
2. Click **+** để thêm connection mới
3. Điền thông tin:
   - **Connection Id**: `colab_local_conn`
   - **Connection Type**: `HTTP`
   - **Host**: `http://localhost:8000` (hoặc ngrok URL nếu dùng Google Colab)
   - **Port**: `8000`

### 4. Kích hoạt DAG

1. Tìm DAG muốn chạy
2. Toggle **ON** để kích hoạt (cho continuous mode)
3. Hoặc click **Trigger DAG** để chạy thủ công

## ☁️ Cấu Hình Google Colab Training

### Bước 1: Mở Colab Notebook

1. Upload file `notebooks/Colab_Training_Server.ipynb` lên Google Colab
2. Chọn Runtime > Change runtime type > GPU (T4 hoặc cao hơn)

### Bước 2: Chạy Notebook

1. Chạy tất cả các cell để khởi động FastAPI server
2. Cell cuối cùng sẽ khởi động server trên port 8000

### Bước 3: Expose Server

**Cách 1: Sử dụng ngrok (Đơn giản)**

```python
# Trong notebook
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(8000)
print(public_url)
```

Copy URL này và cập nhật vào Airflow HTTP Connection.

**Cách 2: Sử dụng VS Code Tunnel (Nâng cao)**

1. Install VS Code extension: **Remote - SSH**
2. Trong Colab, setup SSH:

```python
!pip install colab_ssh --upgrade
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_password")
```

3. Kết nối từ VS Code theo hướng dẫn
4. Forward port 8000 trong VS Code

## 📁 Cấu Trúc Thư Mục

```
doan/
├── dags/                      # Airflow DAGs
│   └── retrain_pipeline.py   # DAG tự động retrain
├── data/                      # Dữ liệu
│   ├── test_video.mp4        # Video test
│   ├── training_samples/     # Mẫu để training
│   └── archive/              # Lưu trữ
├── docker/                    # Docker configuration
│   ├── docker-compose.yml    # Docker services
│   └── .env                  # Environment variables
├── models/                    # Model weights
│   └── yolo_best.pt          # YOLO model
├── notebooks/                 # Jupyter notebooks
│   └── Colab_Training_Server.ipynb
├── src/                       # Source code
│   ├── config.py             # Configuration
│   ├── utils.py              # Utility functions
│   ├── producer.py           # Stream producer
│   ├── consumer_video.py     # Video consumer
│   ├── consumer_audio.py     # Audio consumer
│   └── dashboard.py          # Streamlit dashboard
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
├── project.md               # Project description
└── README.md                # This file
```

## 🔧 Troubleshooting

### Kafka không kết nối được

```bash
# Kiểm tra Kafka đang chạy
docker ps | grep kafka

# Xem logs
docker logs kafka

# Restart Kafka
docker-compose restart kafka
```

### MongoDB không kết nối được

```bash
# Kiểm tra MongoDB
docker ps | grep mongodb

# Test connection
docker exec -it mongodb mongosh -u admin -p admin123
```

### Model không load được

```bash
# Kiểm tra file model
ls -la models/

# Download model mặc định
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Consumer không nhận message

1. Kiểm tra Producer đang chạy
2. Kiểm tra Kafka topics:

```bash
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

## 📈 Performance Tuning

### Tối ưu Producer

- Điều chỉnh FPS trong `config.py`: `VIDEO_FPS`
- Giảm kích thước frame: `VIDEO_FRAME_WIDTH`, `VIDEO_FRAME_HEIGHT`

### Tối ưu Consumer

- Tăng số lượng consumer (chạy nhiều instance)
- Điều chỉnh batch size: `max_poll_records`
- Sử dụng GPU nếu có

### Tối ưu Kafka

- Tăng số partition cho topic
- Điều chỉnh retention policy
- Tăng RAM cho Kafka container

## 🎓 Chi Tiết Kỹ Thuật

### Video Processing Pipeline

1. **Frame Extraction**: OpenCV đọc video, resize về 640x640
2. **Encoding**: Frame được encode sang base64
3. **Publishing**: Gửi vào Kafka topic `livestream-video`
4. **Detection**: YOLOv8 detect objects trong frame
5. **Classification**: Kiểm tra object có phải harmful không
6. **Alert Generation**: Tạo alert nếu phát hiện nội dung độc hại
7. **Storage**: Lưu vào MongoDB và lưu frame để retrain

### Audio Processing Pipeline

1. **Audio Extraction**: (Giả lập - cần implement thực tế)
2. **Publishing**: Gửi vào Kafka topic `livestream-audio`
3. **Transcription**: Whisper chuyển speech to text
4. **Toxic Detection**: NLP check toxic keywords
5. **Alert Generation**: Tạo alert nếu phát hiện ngôn ngữ độc hại
6. **Storage**: Lưu vào MongoDB

### Retraining Pipeline

1. **Data Collection**: Consumer tự động lưu harmful frames
2. **Data Preparation**: Airflow DAG chuẩn bị dataset
3. **Sync to Drive**: Upload lên Google Drive
4. **Trigger Training**: Gọi API Colab để bắt đầu training
5. **Monitor**: Check training status
6. **Deploy**: Download model mới và deploy

## 🤝 Contributing

Đây là đồ án cao học, nếu có góp ý hoặc cải tiến:

1. Fork repository
2. Tạo branch mới
3. Commit changes
4. Push và tạo Pull Request

## 📝 License

Đồ án cao học - Academic Project

## 👨‍🎓 Tác Giả

Đồ án Cao học - Hệ Thống Phát Hiện Nội Dung Độc Hại

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics
- **Whisper**: OpenAI
- **Apache Kafka**: Apache Software Foundation
- **Apache Airflow**: Apache Software Foundation
- **Streamlit**: Streamlit Inc.

---

## 📚 Tài Liệu Tham Khảo

1. [YOLOv8 Documentation](https://docs.ultralytics.com/)
2. [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
3. [Apache Airflow Documentation](https://airflow.apache.org/docs/)
4. [Whisper Documentation](https://github.com/openai/whisper)
5. [Streamlit Documentation](https://docs.streamlit.io/)

---

**Happy Coding! 🚀**
