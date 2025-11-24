# 🚀 BẮT ĐẦU NHANH - 5 PHÚT

## Bước 1: Kiểm Tra Yêu Cầu (1 phút)

```powershell
# Kiểm tra Docker
docker --version

# Kiểm tra Python
python --version

# Nếu OK, tiếp tục. Nếu không, xem INSTALLATION.md
```

## Bước 2: Cài Đặt (2 phút)

```powershell
# Tạo virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt

# Khởi tạo project
python init_project.py
```

## Bước 3: Khởi Động (1 phút)

```powershell
# Start Docker services
.\startup.ps1
```

## Bước 4: Chạy Hệ Thống (1 phút)

### Cách A: Dùng Airflow (Tự động - Khuyến nghị) ⭐

1. Mở trình duyệt: **http://localhost:8080**
2. Login: `admin` / `admin`
3. Tìm DAG: **streaming_pipeline**
4. Click **Trigger DAG** (nút ▶️)
5. Chờ 10 giây để DAG khởi động các services

### Cách B: Chạy Thủ Công (4 Terminal)

**Mở 4 terminal mới:**

```powershell
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

## Bước 5: Xem Kết Quả

- 🌐 **Dashboard**: http://localhost:8501 (Monitoring real-time)
- ⚙️ **Airflow UI**: http://localhost:8080 (admin/admin)
- 📊 **MongoDB**: localhost:27017 (admin/admin123)

### Kiểm Tra Nhanh

```powershell
# Xem số lượng detections trong MongoDB
docker exec mongodb mongosh -u admin -p admin123 --authenticationDatabase admin livestream_detection --quiet --eval "db.detections.countDocuments({})"

# Xem Kafka topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **Cần có video test**: Đặt video vào `data/test_video.mp4`
2. **Docker phải chạy**: Mở Docker Desktop trước
3. **Port 8501, 8080, 9092 phải trống**

## 🆘 Gặp Lỗi?

- **"Module not found"**: Chạy `pip install -r requirements.txt`
- **"Cannot connect"**: Đợi 30 giây sau khi chạy startup.ps1
- **"Port in use"**: Tắt ứng dụng đang dùng port đó

## 📖 Muốn Hiểu Sâu Hơn?

- Chi tiết: Đọc **README.md**
- Cài đặt: Đọc **INSTALLATION.md**
- Video: Đọc **VIDEO_GUIDE.md**
- Tất cả: Đọc **DOCUMENTATION.md**

---

**Chúc may mắn! 🎓✨**
