import os
from pathlib import Path
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    AutoImageProcessor,
)

base = Path("models") / "vit-base-violence-detection"
print("base exists:", base.exists())
# detect snapshots or nested HF cache layout like models--owner--repo/snapshots/<id>
model_dir = base
if (base / "snapshots").exists():
    snaps = list((base / "snapshots").glob("*"))
    if snaps:
        model_dir = snaps[0]
        print("Using snapshot dir:", model_dir)
else:
    # look for nested models--owner--repo structure
    nested = list(base.glob("models--*"))
    if nested:
        first = nested[0]
        if (first / "snapshots").exists():
            snaps = list((first / "snapshots").glob("*"))
            if snaps:
                model_dir = snaps[0]
                print("Using nested snapshot dir:", model_dir)
        else:
            # fallback to the nested folder itself
            model_dir = first
            print("Using nested model dir:", model_dir)

print("Listing files:")
for p in model_dir.glob("*"):
    print(" -", p.name)

# try load extractor/processor
fe = None
try:
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    print("Loaded AutoFeatureExtractor")
except Exception as e:
    print("AutoFeatureExtractor failed:", e)
    try:
        fe = AutoImageProcessor.from_pretrained(model_dir)
        print("Loaded AutoImageProcessor")
    except Exception as e2:
        print("AutoImageProcessor failed:", e2)

# load model
try:
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    print("Loaded model class:", type(model))
    cfg = getattr(model, "config", None)
    if cfg is not None:
        id2label = getattr(cfg, "id2label", None)
        if id2label:
            print("id2label keys:", list(id2label.items())[:20])
        else:
            print(
                "No id2label in config; config keys:",
                [k for k in dir(cfg) if not k.startswith("_")][:50],
            )
except Exception as e:
    print("Model load failed:", e)

print("Done")
