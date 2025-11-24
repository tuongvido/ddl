# Installation Guide - Hướng Dẫn Cài Đặt Chi Tiết

## Yêu Cầu Hệ Thống

### Phần Cứng

- **CPU**: Intel Core i5 hoặc tương đương (khuyến nghị i7+)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **Disk**: 10GB dung lượng trống
- **GPU** (tùy chọn): NVIDIA GPU với CUDA hỗ trợ (tăng tốc độ xử lý)

### Phần Mềm

- **OS**: Windows 10/11, macOS, hoặc Linux
- **Docker Desktop**: Version 4.0+ (bao gồm Docker Compose)
- **Python**: Version 3.8, 3.9, 3.10, hoặc 3.11
- **Git**: Version 2.0+ (tùy chọn)

---

## Bước 1: Cài Đặt Docker Desktop

### Windows

1. Tải Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Chạy file installer
3. Làm theo hướng dẫn cài đặt
4. Khởi động lại máy tính nếu được yêu cầu
5. Mở Docker Desktop và đảm bảo nó đang chạy

**Kiểm tra:**

```powershell
docker --version
docker-compose --version
```

### macOS

```bash
# Sử dụng Homebrew
brew install --cask docker

# Hoặc tải từ website
# https://www.docker.com/products/docker-desktop/
```

### Linux (Ubuntu/Debian)

```bash
# Cài Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Cài Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Thêm user vào docker group
sudo usermod -aG docker $USER
newgrp docker
```

---

## Bước 2: Cài Đặt Python

### Windows

1. Tải Python từ: https://www.python.org/downloads/
2. **Quan trọng**: Check "Add Python to PATH" khi cài đặt
3. Chạy installer và làm theo hướng dẫn

**Kiểm tra:**

```powershell
python --version
pip --version
```

### macOS

```bash
# Sử dụng Homebrew
brew install python@3.10
```

### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Verify
python3 --version
pip3 --version
```

---

## Bước 3: Clone/Download Project

### Nếu dùng Git

```powershell
git clone <repository-url>
cd doan
```

### Nếu download ZIP

1. Download ZIP file
2. Giải nén vào thư mục mong muốn
3. Mở terminal/PowerShell tại thư mục đó

---

## Bước 4: Setup Python Environment

### Windows

```powershell
# Di chuyển đến thư mục project
cd d:\Code\doan

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
.\venv\Scripts\Activate.ps1

# Nếu gặp lỗi "execution policy", chạy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sau đó chạy lại activate
.\venv\Scripts\Activate.ps1

# Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt
source venv/bin/activate

# Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Lưu ý**: Quá trình cài đặt có thể mất 5-10 phút tùy vào tốc độ internet.

---

## Bước 5: Khởi Tạo Project

```powershell
# Chạy script khởi tạo
python init_project.py
```

Script này sẽ:

- Tạo các thư mục cần thiết
- Kiểm tra dependencies
- Kiểm tra Docker

---

## Bước 6: Chuẩn Bị Video Test

```powershell
# Tạo thư mục data nếu chưa có
mkdir data

# Copy video test vào thư mục data/
# Ví dụ: copy C:\Videos\test.mp4 data\test_video.mp4
```

**Yêu cầu video:**

- Format: .mp4, .avi, .mkv, hoặc các format video phổ biến
- Độ dài: 1-5 phút (cho demo)
- Độ phân giải: Tùy ý (sẽ được resize tự động)

---

## Bước 7: Khởi Động Hệ Thống

### Cách 1: Sử dụng startup script (khuyến nghị)

```powershell
.\startup.ps1
```

### Cách 2: Manual

```powershell
# Khởi động Docker services
cd docker
docker-compose up -d

# Đợi 30 giây để services khởi động
Start-Sleep -Seconds 30

# Quay lại thư mục gốc
cd ..
```

---

## Bước 8: Chạy Components

Mở **4 terminal/PowerShell mới**, mỗi terminal chạy một component:

### Terminal 1: Producer

```powershell
cd d:\Code\doan\src
..\venv\Scripts\Activate.ps1
python producer.py --video ../data/test_video.mp4 --loop
```

### Terminal 2: Video Consumer

```powershell
cd d:\Code\doan\src
..\venv\Scripts\Activate.ps1
python consumer_video.py
```

### Terminal 3: Audio Consumer

```powershell
cd d:\Code\doan\src
..\venv\Scripts\Activate.ps1
python consumer_audio.py
```

### Terminal 4: Dashboard

```powershell
cd d:\Code\doan\src
..\venv\Scripts\Activate.ps1
streamlit run dashboard.py
```

---

## Bước 9: Truy Cập Dashboard

1. Mở trình duyệt
2. Truy cập: http://localhost:8501
3. Xem các tab:
   - Overview: Tổng quan
   - Alerts: Cảnh báo
   - Video Detection: Kết quả video
   - Audio Detection: Kết quả audio

---

## Xác Minh Hệ Thống Hoạt Động

### 1. Kiểm tra Docker containers

```powershell
docker ps
```

**Mong đợi thấy:**

- kafka
- zookeeper
- mongodb
- airflow-webserver
- airflow-scheduler
- airflow-worker
- postgres
- redis

### 2. Kiểm tra logs

```powershell
# Producer logs
# Xem terminal 1, sẽ thấy "Processed XXX frames"

# Consumer logs
# Xem terminal 2 & 3, sẽ thấy "Processed XXX frames/chunks"

# Dashboard
# Xem terminal 4, dashboard sẽ tự động mở browser
```

### 3. Kiểm tra Kafka topics

```powershell
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Mong đợi thấy:**

- livestream-video
- livestream-audio

### 4. Kiểm tra MongoDB

```powershell
docker exec -it mongodb mongosh -u admin -p admin123

# Trong mongo shell:
use livestream_detection
show collections
# Sẽ thấy: detections, alerts
db.detections.countDocuments()
```

---

## Troubleshooting

### Lỗi: "Cannot activate virtual environment"

**Windows:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Lỗi: "Port already in use"

```powershell
# Tìm process đang dùng port
netstat -ano | findstr :8501

# Kill process
taskkill /PID <PID> /F
```

### Lỗi: "Docker daemon is not running"

1. Mở Docker Desktop
2. Đợi nó khởi động hoàn toàn
3. Chạy lại các lệnh

### Lỗi: "Module not found" khi chạy Python

```powershell
# Đảm bảo đã activate venv
.\venv\Scripts\Activate.ps1

# Cài lại dependencies
pip install -r requirements.txt
```

### Lỗi: "CUDA not available" hoặc GPU không được nhận

- Đây không phải lỗi nghiêm trọng
- Hệ thống sẽ tự động sử dụng CPU
- Để dùng GPU, cài CUDA toolkit từ NVIDIA

### Consumer không nhận được message

1. Kiểm tra Producer đang chạy
2. Kiểm tra Kafka đang chạy: `docker ps | grep kafka`
3. Restart Kafka: `docker-compose restart kafka`

---

## Tắt Hệ Thống

### Cách 1: Sử dụng shutdown script

```powershell
.\shutdown.ps1
```

### Cách 2: Manual

```powershell
# Dừng Docker services
cd docker
docker-compose down

# Stop Python processes
# Ctrl+C trong mỗi terminal
```

---

## Gỡ Cài Đặt

```powershell
# Stop và xóa containers
cd docker
docker-compose down -v

# Xóa virtual environment
Remove-Item -Recurse -Force venv

# Xóa thư mục project (nếu muốn)
cd ..
Remove-Item -Recurse -Force doan
```

---

## Next Steps

Sau khi cài đặt thành công:

1. ✅ Đọc **QUICKSTART.md** để sử dụng nhanh
2. ✅ Đọc **README.md** để hiểu chi tiết
3. ✅ Xem **project.md** để hiểu kiến trúc
4. ✅ Cấu hình Airflow cho retraining
5. ✅ Setup Google Colab cho training

---

## Support

Nếu gặp vấn đề:

1. Kiểm tra logs của từng component
2. Xem phần Troubleshooting
3. Kiểm tra Docker logs: `docker logs <container-name>`
4. Đảm bảo đủ RAM và disk space

---

**Chúc bạn thành công! 🚀**
