"""
Test PhoBERT to find which words/phrases are classified as TOXIC (LABEL_1)
"""
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_toxic_words():
    """Test many Vietnamese words to see which ones are classified as toxic"""
    
    print("=" * 80)
    print("PHOBERT TOXIC WORDS DETECTION")
    print("=" * 80)
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "phobert"
    print(f"\n⏳ Loading model from: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded (Device: {device})")
    print(f"   Labels: {model.config.id2label}")
    
    # List of Vietnamese words/phrases to test
    test_words = [
        # Potentially toxic words
        "ngu", "điên", "chó", "khốn", "đồ", "mẹ", "bố", 
        "đụ", "đéo", "đồ ngu", "thằng ngu", "con chó",
        "mày", "tao", "đm", "vcl", "đcm", "cc", "lol",
        "đồ khốn", "khốn nạn", "chết tiệt", "đồ chó",
        "đồ lừa đảo", "ngu ngốc", "ngớ ngẩn", "dốt",
        "con lợn", "con heo", "đồ điên", "điên khùng",
        "cút đi", "biến đi", "câm miệng", "im mồm",
        "đánh nhau", "giết", "chết", "đập chết",
        
        # Non-toxic words
        "xin chào", "cảm ơn", "vui", "đẹp", "tốt",
        "yêu", "thương", "hạnh phúc", "vui vẻ",
        "bạn", "anh", "chị", "em", "tôi",
        "làm việc", "học tập", "ăn cơm", "ngủ nghỉ",
        "thời tiết", "đẹp trời", "mưa", "nắng",
    ]
    
    print(f"\n🔍 Testing {len(test_words)} words/phrases:")
    print("=" * 80)
    
    toxic_words = []
    clean_words = []
    
    for word in test_words:
        try:
            # Tokenize
            inputs = tokenizer(
                word,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_class].item()
            
            # Store results
            if predicted_class == 1:  # LABEL_1 = toxic
                toxic_words.append({
                    'word': word,
                    'confidence': confidence,
                    'toxic_prob': probs[0][1].item()
                })
            else:  # LABEL_0 = clean
                clean_words.append({
                    'word': word,
                    'confidence': confidence
                })
        except Exception as e:
            print(f"Error testing '{word}': {e}")
    
    # Print TOXIC words
    print("\n" + "=" * 80)
    print(f"🚨 TOXIC WORDS (LABEL_1): {len(toxic_words)}")
    print("=" * 80)
    
    # Sort by confidence
    toxic_words.sort(key=lambda x: x['toxic_prob'], reverse=True)
    
    for i, item in enumerate(toxic_words, 1):
        print(f"{i:2d}. '{item['word']:<20}' → Toxic: {item['toxic_prob']:>6.2%}")
    
    # Print CLEAN words
    print("\n" + "=" * 80)
    print(f"✅ CLEAN WORDS (LABEL_0): {len(clean_words)}")
    print("=" * 80)
    
    for i, item in enumerate(clean_words[:20], 1):  # Show first 20
        print(f"{i:2d}. '{item['word']}'")
    
    if len(clean_words) > 20:
        print(f"... and {len(clean_words) - 20} more clean words")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tested: {len(test_words)}")
    print(f"Toxic (LABEL_1): {len(toxic_words)}")
    print(f"Clean (LABEL_0): {len(clean_words)}")
    print("=" * 80)

if __name__ == "__main__":
    test_toxic_words()
