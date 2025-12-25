"""
Test local PhoBERT model for toxic text classification
"""
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_phobert_model():
    """Test PhoBERT model on sample texts"""
    
    print("=" * 80)
    print("PHOBERT TOXIC TEXT CLASSIFICATION TEST")
    print("=" * 80)
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "phobert"
    print(f"\n📂 Loading model from: {model_path}")
    print(f"   Model directory exists: {model_path.exists()}")
    
    if not model_path.exists():
        print(f"❌ ERROR: Model directory not found!")
        return
    
    # List files in model directory
    print(f"\n📄 Files in model directory:")
    for file in model_path.iterdir():
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"   - {file.name} ({size_mb:.2f} MB)")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device.upper()}")
    
    # Load PhoBERT model
    print("\n⏳ Loading PhoBERT model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.to(device)
        model.eval()
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Architecture: {model.config.architectures}")
        print(f"   Number of labels: {model.config.num_labels}")
        if hasattr(model.config, 'id2label'):
            print(f"   Label mapping: {model.config.id2label}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test sentences
    print("\n" + "=" * 80)
    print("TESTING ON SAMPLE TEXTS")
    print("=" * 80)
    
    test_texts = [
        # Potentially toxic examples
        "Mày là thằng ngu",
        "Đồ khốn nạn",
        "Chết tiệt",
        "Đụ má mày",
        
        # Non-toxic examples  
        "Xin chào, bạn khỏe không?",
        "Hôm nay thời tiết đẹp quá",
        "Cảm ơn bạn nhiều",
        "Tôi rất vui được gặp bạn",
        "Chúc bạn một ngày tốt lành",
        
        # Neutral
        "Tôi không thích điều này",
        "Bạn nói sai rồi",
        "Điều này thật tệ",
    ]
    
    print(f"\n🔍 Testing {len(test_texts)} texts:")
    print("=" * 80)
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        try:
            # Tokenize
            inputs = tokenizer(
                text,
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
                
                # Get prediction
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_class].item()
                
                # Get all probabilities
                all_probs = probs[0].cpu().numpy()
            
            # Store result
            result = {
                'text': text,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probs': all_probs.tolist()
            }
            results.append(result)
            
            # Determine label name if available
            label_name = "Unknown"
            if hasattr(model.config, 'id2label') and predicted_class in model.config.id2label:
                label_name = model.config.id2label[predicted_class]
            
            # Print result
            print(f"\n[{i:2d}] Text: '{text}'")
            print(f"     Predicted: Class {predicted_class} ({label_name})")
            print(f"     Confidence: {confidence:.2%}")
            print(f"     Probabilities: {[f'{p:.2%}' for p in all_probs]}")
            
        except Exception as e:
            print(f"\n[{i:2d}] ❌ ERROR processing text: '{text}'")
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"📊 Total texts tested: {len(results)}")
    
    if results:
        # Count predictions by class
        class_counts = {}
        for r in results:
            cls = r['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"\n📈 Predictions by class:")
        for cls, count in sorted(class_counts.items()):
            label = "Unknown"
            if hasattr(model.config, 'id2label') and cls in model.config.id2label:
                label = model.config.id2label[cls]
            print(f"   Class {cls} ({label}): {count}")
    
    print("\n✅ TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_phobert_model()
