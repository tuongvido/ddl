# Video Test Guidelines - Hướng Dẫn Chuẩn Bị Video Test

## Yêu Cầu Video

### Format

- **Container**: MP4, AVI, MKV, MOV, hoặc WebM
- **Codec**: H.264, H.265 (khuyến nghị H.264)
- **Audio**: AAC, MP3 (tùy chọn)

### Thông Số Kỹ Thuật

- **Độ phân giải**: Tùy ý (sẽ được resize về 640x640)
- **Frame rate**: 24-30 FPS (khuyến nghị)
- **Độ dài**: 1-5 phút (cho demo)
- **Kích thước**: < 500MB (để dễ xử lý)

### Nội Dung Phù Hợp

Video nên chứa một số trong các nội dung sau để test detection:

#### 1. Cho Video Detection (YOLOv8)

- ✅ Vật thể thường (người, xe, động vật) - để test model hoạt động
- ✅ Vật thể có nguy cơ (dao, súng giả, vũ khí đồ chơi) - nếu có
- ⚠️ **Lưu ý**: Chỉ dùng video hợp pháp, không chứa nội dung bạo lực thật

#### 2. Cho Audio Detection (Whisper + NLP)

- ✅ Có tiếng nói rõ ràng
- ✅ Ngôn ngữ: Tiếng Việt hoặc Tiếng Anh
- ⚠️ Có thể chứa một vài từ trong `TOXIC_KEYWORDS` để test (nhưng không quá thô tục)

---

## Nguồn Video Test

### 1. Sử dụng Video Có Sẵn

```powershell
# Copy video vào thư mục data
Copy-Item "C:\Path\To\Your\Video.mp4" "data\test_video.mp4"
```

### 2. Tải Video Test Từ Internet

#### Nguồn miễn phí:

- **Pexels**: https://www.pexels.com/videos/
- **Pixabay**: https://pixabay.com/videos/
- **Videvo**: https://www.videvo.net/
- **Mixkit**: https://mixkit.co/free-stock-video/

**Ví dụ tìm kiếm:**

- "crowd" (đám đông)
- "city street" (đường phố)
- "people talking" (người nói chuyện)
- "action" (hành động)

#### Download bằng Python:

```python
# Ví dụ download từ URL
import requests

url = "https://example.com/video.mp4"
response = requests.get(url, stream=True)

with open("data/test_video.mp4", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

### 3. Tạo Video Test Đơn Giản

#### Sử dụng webcam:

```python
import cv2

cap = cv2.VideoCapture(0)  # Webcam
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/test_video.mp4', fourcc, 20.0, (640, 480))

for i in range(100):  # Record 100 frames
    ret, frame = cap.read()
    if ret:
        out.write(frame)

cap.release()
out.release()
```

---

## Cấu Trúc Thư Mục Data

```
data/
├── test_video.mp4              # Video chính để test
├── test_video_2.mp4            # Video bổ sung (nếu có)
├── sample_videos/              # Thư mục chứa nhiều video test
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── training_samples/           # Tự động tạo bởi consumer
│   ├── knife_20241124_123456.jpg
│   └── gun_20241124_123457.jpg
└── archive/                    # Lưu trữ
    └── 20241124/
        └── old_samples/
```

---

## Test Videos By Scenario

### Scenario 1: Cơ Bản

**Mục đích**: Test xem hệ thống có chạy được không

**Video cần có:**

- Nội dung đơn giản (người đi bộ, giao thông, etc.)
- Không cần có object đặc biệt
- Chỉ cần có audio để test audio consumer

**Kỳ vọng:**

- Producer gửi frames thành công
- Consumers nhận và xử lý
- Dashboard hiển thị metrics

### Scenario 2: Object Detection

**Mục đích**: Test khả năng detect objects

**Video cần có:**

- Nhiều objects (người, xe, vật thể)
- Có thể có vật thể giống dao, súng (đồ chơi, props)

**Kỳ vọng:**

- YOLOv8 detect được objects
- Nếu có object thuộc `HARMFUL_CLASSES`, tạo alert
- Dashboard hiển thị detections

### Scenario 3: Audio Detection

**Mục đích**: Test khả năng detect toxic speech

**Video cần có:**

- Tiếng nói rõ ràng
- Có thể chứa 1-2 từ trong `TOXIC_KEYWORDS`

**Kỳ vọng:**

- Whisper transcribe được text
- NLP detect được toxic keywords
- Tạo alert khi phát hiện

### Scenario 4: Full System

**Mục đích**: Test toàn bộ hệ thống

**Video cần có:**

- Cả visual và audio
- Có objects và speech
- Đủ dài để test streaming

**Kỳ vọng:**

- Tất cả components hoạt động
- Alerts được tạo và hiển thị
- Dashboard update real-time

---

## Tips & Best Practices

### 1. Chọn Video Phù Hợp

- ✅ Video có nội dung rõ ràng
- ✅ Lighting tốt (không quá tối)
- ✅ Audio rõ ràng (nếu test audio)
- ❌ Tránh video quá dài (> 10 phút)
- ❌ Tránh video có DRM protection

### 2. Chuẩn Bị Nhiều Video

```
data/
├── test_video_1.mp4    # Cơ bản
├── test_video_2.mp4    # Object detection
├── test_video_3.mp4    # Audio detection
└── test_video_full.mp4 # Full test
```

### 3. Test Trước Khi Demo

```powershell
# Test ngắn trước
python producer.py --video data/test_video.mp4
# Ctrl+C sau vài giây

# Nếu OK, chạy với --loop
python producer.py --video data/test_video.mp4 --loop
```

### 4. Backup Video

- Copy video test ra ngoài để backup
- Đặt tên rõ ràng: `test_video_scenario1.mp4`
- Lưu metadata (source, date, scenario)

---

## Video Processing Tips

### Convert Video Format

```bash
# Sử dụng ffmpeg
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
```

### Resize Video

```bash
# Resize về 640x640
ffmpeg -i input.mp4 -vf scale=640:640 output.mp4
```

### Trim Video

```bash
# Cắt 1-5 phút
ffmpeg -i input.mp4 -ss 00:01:00 -t 00:04:00 output.mp4
```

### Extract Audio

```bash
# Extract audio từ video
ffmpeg -i input.mp4 -vn -acodec copy output.aac
```

---

## Troubleshooting

### Video không load được

```python
# Test video có đọc được không
import cv2
cap = cv2.VideoCapture('data/test_video.mp4')
print(f"Can open: {cap.isOpened()}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
cap.release()
```

### Video quá lớn

- Resize về độ phân giải thấp hơn
- Compress với ffmpeg
- Trim bớt độ dài

### Video lag khi process

- Giảm FPS trong config: `VIDEO_FPS = 10`
- Giảm FRAME_WIDTH và FRAME_HEIGHT
- Sử dụng GPU

---

## Sample Test Plan

### Week 1: Basic Testing

- [ ] Chuẩn bị 1 video đơn giản
- [ ] Test producer → kafka
- [ ] Test consumers nhận được message
- [ ] Verify MongoDB có data

### Week 2: Detection Testing

- [ ] Test video với nhiều objects
- [ ] Verify YOLOv8 detect được
- [ ] Test alert generation
- [ ] Check dashboard display

### Week 3: Integration Testing

- [ ] Test với multiple videos
- [ ] Test với video có audio
- [ ] Verify full pipeline
- [ ] Performance testing

### Week 4: Demo Preparation

- [ ] Prepare best demo video
- [ ] Test multiple times
- [ ] Record demo video
- [ ] Prepare backup plan

---

## Recommended Test Videos

### For Object Detection:

1. **Traffic/Street**: Nhiều objects (cars, people)
2. **Kitchen**: Có dao, dụng cụ nhà bếp
3. **Workshop**: Có tools, equipment
4. **Action scenes**: Nếu có (props, fake weapons)

### For Audio Detection:

1. **Interview/Talk**: Rõ ràng, có structure
2. **Debate/Discussion**: Có thể có từ harsh
3. **Movie clips**: Có dialogue
4. **Livestream recordings**: Realistic scenario

---

**Note**: Luôn tuân thủ pháp luật và đạo đức khi chọn video test. Không sử dụng nội dung bất hợp pháp hoặc vi phạm bản quyền.
