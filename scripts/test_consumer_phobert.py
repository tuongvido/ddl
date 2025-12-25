import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Trỏ đến thư mục chứa 5 file vừa tải về
MODEL_PATH = Path(__file__).parent.parent / "models" / "phobert"

print("Đang load model...")
try:
    # Load model từ thư mục offline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )

    print("Load thành công! Sẵn sàng kiểm tra.")

    # Hàm dự đoán
    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs).item()

        return "ĐỘC HẠI 🤬" if pred_label == 1 else "BÌNH THƯỜNG 😊"

    # Test thử
    while True:
        text = input("\nNhập câu bình luận: ")
        if text == "exit":
            break
        print(f"Kết quả: {predict(text)}")

except Exception as e:
    print(f"Lỗi: {e}")
    print(
        "Bạn hãy kiểm tra xem đã tải đủ 5 file (đặc biệt là model.safetensors) vào đúng thư mục chưa nhé."
    )
