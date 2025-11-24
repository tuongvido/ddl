# Quick Start Guide - Hướng Dẫn Khởi Động Nhanh

## Cách 1: Sử Dụng Airflow DAG (Khuyến nghị) ⭐

### Bước 1: Khởi động Docker Services

```powershell
# Khởi động hệ thống
.\startup.ps1
```

### Bước 2: Chạy Pipeline Bằng Airflow

**Option A: Kiểm tra hạ tầng (Khuyến nghị chạy trước)**

1. Mở trình duyệt: http://localhost:8080
2. Login: `admin` / `admin`
3. Tìm DAG: **streaming_pipeline**
4. Click nút **Trigger DAG** (▶️)
5. Xem logs - nếu thành công sẽ có hướng dẫn start services
6. Copy commands từ logs để chạy Producer/Consumers

**Option B: Monitoring tự động**

1. Sau khi đã start Producer/Consumers thủ công
2. Vào Airflow UI: http://localhost:8080
3. Tìm DAG: **streaming_continuous**
4. Toggle **ON** để monitor mỗi 5 phút
5. DAG sẽ check health và alert nếu có vấn đề

### Bước 3: Mở Dashboard

```powershell
# Mở terminal mới
cd src
python -m streamlit run dashboard.py
```

Hoặc truy cập: http://localhost:8501

### Tắt hệ thống

```powershell
# Tắt Airflow DAG trước (nếu dùng continuous mode)
# Vào Airflow UI > Toggle OFF DAG streaming_continuous

# Sau đó tắt Docker
.\shutdown.ps1
```

---

## Cách 2: Chạy Thủ Công (Debug Mode)

### Windows PowerShell

```powershell
# Khởi động hệ thống
.\startup.ps1

# Mở 4 terminal mới và chạy:

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
python -m streamlit run dashboard.py
```

### Tắt hệ thống

```powershell
# Nhấn Ctrl+C trong mỗi terminal để dừng
# Sau đó:
.\shutdown.ps1
```

## Cách 2: Khởi Động Thủ Công

### Bước 1: Khởi động Docker

```powershell
cd docker
docker-compose up -d
```

### Bước 2: Cài đặt Python dependencies

```powershell
# Tạo virtual environment (chỉ lần đầu)
python -m venv venv

# Kích hoạt virtual environment
.\venv\Scripts\activate

# Cài đặt packages
pip install -r requirements.txt
```

### Bước 3: Chạy các components

```powershell
# Producer (Terminal 1)
cd src
python producer.py --video ../data/test_video.mp4 --loop

# Video Consumer (Terminal 2)
cd src
python consumer_video.py

# Audio Consumer (Terminal 3)
cd src
python consumer_audio.py

# Dashboard (Terminal 4)
cd src
streamlit run dashboard.py
```

## Kiểm Tra Hệ Thống

### 1. Kiểm tra Docker containers

```powershell
docker ps
```

Bạn sẽ thấy các containers: kafka, zookeeper, mongodb, airflow-webserver, airflow-scheduler, etc.

### 2. Kiểm tra Kafka topics

```powershell
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

Bạn sẽ thấy topics: `livestream-video`, `livestream-audio`

### 3. Truy cập các services

- **Dashboard**: http://localhost:8501
- **Airflow**: http://localhost:8080 (admin/admin)
- **MongoDB**: localhost:27017 (admin/admin123)

## Checklist Trước Khi Chạy

- [ ] Docker Desktop đang chạy
- [ ] Python 3.8+ đã cài đặt
- [ ] File video test đã có trong thư mục `data/`
- [ ] Đã cài đặt dependencies (`pip install -r requirements.txt`)
- [ ] Port 8501, 8080, 9092, 27017 chưa bị chiếm bởi ứng dụng khác

## Xử Lý Lỗi Thường Gặp

### 1. "Port already in use"

```powershell
# Tìm process đang dùng port
netstat -ano | findstr :8501

# Tắt process (thay PID bằng số thực tế)
taskkill /PID <PID> /F
```

### 2. "Cannot connect to Kafka"

```powershell
# Restart Kafka
docker-compose restart kafka

# Xem logs
docker logs kafka
```

### 3. "Module not found"

```powershell
# Đảm bảo đã activate virtual environment
.\venv\Scripts\activate

# Cài lại dependencies
pip install -r requirements.txt
```

### 4. "YOLO model not found"

```powershell
# Model sẽ tự động download lần đầu chạy
# Hoặc download thủ công:
cd models
# Model sẽ được tải tự động khi chạy consumer_video.py
```

## Tips & Tricks

### Chạy trong chế độ development

```powershell
# Producer với FPS thấp hơn (tiết kiệm tài nguyên)
python producer.py --video ../data/test_video.mp4 --loop

# Xem logs chi tiết
# Sửa LOG_LEVEL=DEBUG trong config.py
```

### Monitor hệ thống

```powershell
# Xem resource usage của Docker
docker stats

# Xem logs của service cụ thể
docker logs -f airflow-webserver
docker logs -f kafka
```

### Backup và Restore

```powershell
# Backup MongoDB data
docker exec mongodb mongodump --out /backup

# Backup training samples
Copy-Item -Path data\training_samples -Destination backup\ -Recurse
```

## Câu Hỏi Thường Gặp (FAQ)

**Q: Cần bao nhiêu RAM?**
A: Tối thiểu 8GB, khuyến nghị 16GB

**Q: Có cần GPU không?**
A: Không bắt buộc nhưng có GPU sẽ nhanh hơn nhiều. Hệ thống sẽ tự động detect và dùng GPU nếu có.

**Q: Video test nên dài bao nhiêu?**
A: 1-5 phút là đủ để demo. Video dài hơn sẽ tốn thời gian và tài nguyên.

**Q: Làm sao biết hệ thống đang hoạt động?**
A: Mở Dashboard (localhost:8501), bạn sẽ thấy metrics và alerts real-time.

**Q: Có thể chạy nhiều video cùng lúc?**
A: Có, chạy nhiều instances của producer với video khác nhau.

---

**Need Help?** Check README.md for detailed documentation.
