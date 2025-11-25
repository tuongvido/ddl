# Hướng Dẫn Chạy Streaming Pipeline với Airflow DAG

## Bước 1: Rebuild Airflow với Dependencies

```powershell
.\rebuild_airflow.ps1
```

Script này sẽ:

- Stop các Airflow containers cũ
- Build custom Airflow image có opencv-python, kafka-python, ultralytics, etc.
- Mount thư mục src/, data/, models/ vào containers
- Restart các Airflow services

## Bước 2: Đợi Services Khởi Động

Đợi khoảng 30 giây để Airflow khởi động hoàn tất.

## Bước 3: Chạy DAG

1. Mở Airflow UI: http://localhost:8080
2. Login: `admin` / `admin`
3. Tìm DAG: **run_streaming_pipeline**
4. Click nút **Trigger DAG** (▶️)

DAG sẽ:

- ✅ Check dependencies (opencv, kafka-python, ultralytics, etc.)
- ✅ Start Producer (streaming video v001_converted.avi loop)
- ✅ Start Video Consumer (YOLOv8 detection)
- ✅ Start Audio Consumer (Whisper transcription)
- ✅ Verify pipeline (check MongoDB có data)

## Bước 4: Monitor

- **Airflow Logs**: Xem logs từng task trong Airflow UI
- **Dashboard**: Mở http://localhost:8501 (cần start riêng)
- **MongoDB**: Check data với lệnh:
  ```powershell
  docker exec mongodb mongosh -u admin -p admin123 --authenticationDatabase admin livestream_detection --quiet --eval "db.detections.countDocuments({})"
  ```

## Lưu Ý Quan Trọng

1. **Processes chạy trong Docker**: Producer và Consumers chạy trong Airflow worker container, không phải trên host
2. **Chạy background**: Các processes sẽ tiếp tục chạy sau khi DAG hoàn thành
3. **Stop**: Để stop, cần restart Airflow containers hoặc kill processes trong container
4. **Video path**: Đã hard-code `/opt/airflow/data/v001_converted.avi` (mount từ host)

## Troubleshooting

### DAG không xuất hiện

```powershell
# Restart scheduler
docker-compose restart airflow-scheduler
```

### Dependencies lỗi

```powershell
# Rebuild image
.\rebuild_airflow.ps1
```

### Kafka connection error

```powershell
# Check Kafka đang chạy
docker ps | findstr kafka

# Xem logs
docker logs kafka
```

### MongoDB connection error

```powershell
# Check MongoDB
docker exec mongodb mongosh -u admin -p admin123 --authenticationDatabase admin
```
