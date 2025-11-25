# 🎓 Đồ Án Cao Học - Tài Liệu Tổng Hợp

## Thông Tin Đồ Án

**Tên đề tài**: Hệ Thống Phát Hiện Nội Dung Độc Hại Thời Gian Thực Trên Livestream

**Mô tả**: Xây dựng hệ thống Big Data xử lý luồng (Streaming Processing) với khả năng phát hiện đa phương thức (multimodal) và học liên tục (continuous learning).

---

## 📚 Cấu Trúc Tài Liệu

| Tài liệu            | Mục đích                          | Đọc khi nào                 |
| ------------------- | --------------------------------- | --------------------------- |
| **INSTALLATION.md** | Hướng dẫn cài đặt chi tiết từ đầu | Lần đầu setup hệ thống      |
| **QUICKSTART.md**   | Hướng dẫn khởi động nhanh         | Đã cài xong, muốn chạy ngay |
| **README.md**       | Tổng quan và hướng dẫn sử dụng    | Hiểu tổng quan hệ thống     |
| **project.md**      | Mô tả đề tài và yêu cầu           | Hiểu yêu cầu đồ án          |
| Tài liệu này        | Tổng hợp và tham khảo             | Khi cần tra cứu nhanh       |

---

## 🗂️ Cấu Trúc Thư Mục Hoàn Chỉnh

```
doan/
├── 📄 README.md                        # Tổng quan hệ thống
├── 📄 INSTALLATION.md                  # Hướng dẫn cài đặt
├── 📄 QUICKSTART.md                    # Hướng dẫn nhanh
├── 📄 project.md                       # Mô tả đề tài
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .env.example                     # Mẫu cấu hình
├── 📄 .gitignore                       # Git ignore rules
├── 🔧 init_project.py                  # Script khởi tạo
├── 🚀 startup.ps1                      # Script khởi động
├── 🛑 shutdown.ps1                     # Script tắt hệ thống
│
├── 📁 dags/                            # Airflow DAGs
│   └── retrain_pipeline.py            # DAG tự động retrain model
│
├── 📁 data/                            # Dữ liệu
│   ├── .gitkeep
│   ├── test_video.mp4                 # Video test (user cung cấp)
│   ├── training_samples/              # Mẫu để training
│   ├── training_dataset/              # Dataset đã chuẩn bị
│   └── archive/                       # Lưu trữ
│
├── 📁 docker/                          # Docker configuration
│   ├── docker-compose.yml             # Định nghĩa services
│   └── .env                           # Environment variables
│
├── 📁 models/                          # Model weights
│   ├── .gitkeep
│   └── yolo_best.pt                   # YOLO model (tự động tải)
│
├── 📁 notebooks/                       # Jupyter notebooks
│   └── Colab_Training_Server.ipynb    # Server training trên Colab
│
├── 📁 src/                             # Source code
│   ├── config.py                      # Cấu hình hệ thống
│   ├── utils.py                       # Utility functions
│   ├── producer.py                    # Giả lập livestream
│   ├── consumer_video.py              # Xử lý video, detect hình ảnh
│   ├── consumer_audio.py              # Xử lý audio, detect speech
│   └── dashboard.py                   # Dashboard Streamlit
│
└── 📁 venv/                            # Python virtual environment (tự tạo)
```

---

## 🔄 Quy Trình Làm Việc

### 1. Lần Đầu Setup

```
1. Đọc INSTALLATION.md
2. Cài Docker Desktop
3. Cài Python 3.8+
4. Clone/download project
5. Chạy: python init_project.py
6. Chuẩn bị video test
7. Chạy: .\startup.ps1
8. Chạy 4 components (Producer, 2 Consumers, Dashboard)
9. Xem kết quả trên Dashboard
```

https://huggingface.co/jaranohaal/vit-base-violence-detection/resolve/main/model.safetensors?download=true

### 2. Lần Sau Sử Dụng

```
1. Đọc QUICKSTART.md
2. Chạy: .\startup.ps1
3. Chạy 4 components
4. Xem kết quả
5. Khi xong: .\shutdown.ps1
```

### 3. Development Workflow

```
Developer → Producer → Kafka → Consumers → MongoDB → Dashboard
                                    ↓
                            Training Samples
                                    ↓
                    Airflow → Google Colab → New Model
                                    ↓
                            Deploy & Restart
```

---

## 🎯 Các Thành Phần Chính

### 1. Producer (src/producer.py)

- **Chức năng**: Giả lập livestream từ file video
- **Input**: File video (.mp4, .avi, etc.)
- **Output**: Frame images → Kafka topic `livestream-video`
- **Công nghệ**: OpenCV, Kafka Producer

### 2. Video Consumer (src/consumer_video.py)

- **Chức năng**: Phát hiện vật thể độc hại trong video
- **Input**: Kafka topic `livestream-video`
- **Processing**: YOLOv8 object detection
- **Output**: Detections → MongoDB, Alerts
- **Công nghệ**: YOLOv8, PyTorch, Kafka Consumer

### 3. Audio Consumer (src/consumer_audio.py)

- **Chức năng**: Phát hiện ngôn ngữ độc hại trong audio
- **Input**: Kafka topic `livestream-audio`
- **Processing**: Whisper STT + NLP
- **Output**: Detections → MongoDB, Alerts
- **Công nghệ**: OpenAI Whisper, Kafka Consumer

### 4. Dashboard (src/dashboard.py)

- **Chức năng**: Hiển thị kết quả real-time
- **Input**: MongoDB (detections, alerts)
- **Output**: Web interface
- **Công nghệ**: Streamlit, Plotly

### 5. Airflow DAG (dags/retrain_pipeline.py)

- **Chức năng**: Tự động hóa quy trình retrain model
- **Schedule**: Daily (có thể thay đổi)
- **Steps**:
  1. Check new data
  2. Prepare dataset
  3. Sync to Google Drive
  4. Trigger Colab training
  5. Download new model
  6. Archive samples

### 6. Colab Training Server (notebooks/Colab_Training_Server.ipynb)

- **Chức năng**: Training server trên Google Colab với GPU miễn phí
- **API Endpoints**:
  - `GET /`: Health check
  - `GET /status`: Training status
  - `POST /train`: Start training
  - `POST /reset`: Reset status
- **Công nghệ**: FastAPI, YOLOv8, Google Colab

---

## 🛠️ Công Nghệ Stack

| Layer                  | Technology     | Purpose             |
| ---------------------- | -------------- | ------------------- |
| **Input**              | OpenCV         | Video processing    |
| **Message Queue**      | Apache Kafka   | Stream processing   |
| **Object Detection**   | YOLOv8         | Vision AI           |
| **Speech Recognition** | OpenAI Whisper | Audio AI            |
| **Storage**            | MongoDB        | NoSQL database      |
| **Orchestration**      | Apache Airflow | Workflow automation |
| **Training**           | Google Colab   | GPU training        |
| **Dashboard**          | Streamlit      | Visualization       |
| **Containerization**   | Docker         | Service management  |

---

## 📊 Metrics và KPIs

### Performance Metrics

- **Throughput**: Số frame/giây được xử lý
- **Latency**: Thời gian từ input → output
- **Accuracy**: Độ chính xác của detection
- **False Positive Rate**: Tỷ lệ cảnh báo nhầm

### System Metrics

- **CPU Usage**: Sử dụng CPU
- **Memory Usage**: Sử dụng RAM
- **Disk I/O**: Đọc/ghi disk
- **Network**: Kafka throughput

### Business Metrics

- **Detection Rate**: Tỷ lệ phát hiện nội dung độc hại
- **Response Time**: Thời gian phản hồi cảnh báo
- **Model Accuracy**: Độ chính xác model theo thời gian

---

## 🔐 Security & Privacy

### Data Security

- MongoDB authentication (admin/admin123)
- Kafka internal communication
- Docker network isolation

### Privacy Considerations

- No personal data storage by default
- Video processing in-memory
- Configurable data retention

---

## 🚀 Scaling & Performance

### Horizontal Scaling

```yaml
# Scale consumers
docker-compose up -d --scale consumer-video=3

# Or run multiple instances manually
python consumer_video.py &
python consumer_video.py &
python consumer_video.py &
```

### Vertical Scaling

- Increase Docker container resources
- Use GPU for faster inference
- Optimize model (quantization, pruning)

### Performance Optimization

1. **Reduce frame rate**: Adjust `VIDEO_FPS` in config
2. **Batch processing**: Process multiple frames together
3. **Model optimization**: Use smaller YOLO model (nano, small)
4. **Kafka tuning**: Increase partitions, adjust retention

---

## 🧪 Testing

### Unit Tests

```powershell
# Test individual components
python -m pytest tests/
```

### Integration Tests

```powershell
# Test end-to-end flow
python tests/integration_test.py
```

### Load Testing

```powershell
# Test with high load
python tests/load_test.py --fps 60 --duration 300
```

---

## 📈 Monitoring & Logging

### Logs Location

- **Producer**: stdout/stderr
- **Consumers**: stdout/stderr
- **Airflow**: `logs/` directory
- **Docker**: `docker logs <container>`

### Monitoring Tools

- **Dashboard**: Real-time metrics
- **Docker Stats**: `docker stats`
- **Kafka Manager**: Optional UI for Kafka

---

## 🎓 Báo Cáo Đồ Án

### Nội Dung Báo Cáo Nên Có

1. **Giới thiệu**

   - Bối cảnh và động lực
   - Mục tiêu và phạm vi
   - Đóng góp của đồ án

2. **Tổng Quan Hệ Thống**

   - Kiến trúc tổng thể
   - Các thành phần chính
   - Luồng dữ liệu

3. **Thiết Kế Chi Tiết**

   - Data ingestion layer
   - Processing layer
   - Storage layer
   - Orchestration layer
   - Training pipeline

4. **Công Nghệ Sử Dụng**

   - Lý do chọn từng công nghệ
   - So sánh với các giải pháp khác
   - Trade-offs

5. **Triển Khai**

   - Chi tiết implementation
   - Code organization
   - Best practices

6. **Kết Quả**

   - Demo hệ thống
   - Performance metrics
   - Screenshots/videos
   - Test results

7. **Đánh Giá**

   - Ưu điểm
   - Hạn chế
   - Hướng phát triển

8. **Kết Luận**
   - Tóm tắt đóng góp
   - Bài học kinh nghiệm
   - Future work

### Demo Checklist

- [ ] Hệ thống chạy ổn định
- [ ] Dashboard hiển thị real-time
- [ ] Video test có nội dung phù hợp
- [ ] Các alert được trigger
- [ ] Metrics được hiển thị
- [ ] Slide trình bày rõ ràng
- [ ] Video demo backup

---

## 🤔 FAQs

**Q: Cần bao nhiêu thời gian để setup?**
A: 30-60 phút cho lần đầu, 5-10 phút cho các lần sau.

**Q: Có thể chạy trên laptop không?**
A: Có, nhưng cần ít nhất 8GB RAM.

**Q: Có cần internet không?**
A: Cần cho lần đầu (download dependencies), sau đó có thể offline.

**Q: Model đã được train sẵn chưa?**
A: Sử dụng pre-trained YOLOv8, có thể fine-tune thêm.

**Q: Làm sao để thay đổi loại object detect?**
A: Sửa `HARMFUL_CLASSES` trong `config.py` hoặc train model mới.

**Q: Có thể deploy production được không?**
A: Có, nhưng cần một số điều chỉnh:

- Security hardening
- Load balancing
- Monitoring system
- Backup strategy

---

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra logs
2. Xem troubleshooting section
3. Đọc lại documentation
4. Check GitHub issues (nếu có)

---

## 🎉 Conclusion

Đồ án này demo một hệ thống hoàn chỉnh với:
✅ Big Data streaming processing
✅ Real-time AI inference
✅ Automated ML pipeline
✅ Hybrid cloud architecture
✅ Production-ready design

**Good luck với đồ án! 🚀**
