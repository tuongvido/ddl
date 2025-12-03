"""
Automated setup for violence detection model
Downloads a pre-configured violence/weapon detection model for the system
"""

from pathlib import Path
import subprocess
import sys


def install_roboflow():
    """Install roboflow package if not already installed"""
    try:
        import roboflow

        print("✓ Roboflow already installed")
        return True
    except ImportError:
        try:
            print("📦 Installing roboflow package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
            print("✓ Roboflow installed")
            return True
        except Exception as e:
            print(f"❌ Failed to install roboflow: {e}")
            return False


def download_violence_model():
    """Download violence detection model from Roboflow"""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("VIOLENCE DETECTION MODEL SETUP")
    print("=" * 60)

    print("\n🎯 This script will help you set up a violence detection model")
    print("\n📋 Requirements:")
    print("   1. Roboflow account (free): https://roboflow.com")
    print("   2. API key from: https://app.roboflow.com/settings/api")

    print("\n🔍 Recommended Datasets on Roboflow Universe:")
    print("   • Violence Detection: vd-hz6jg/violence-detection-o4rdr")
    print("   • Weapon Detection: weapons-uw60y/weapons-lidce")
    print("   • Fight Detection: computer-vision-mpfyw/fight-detection-uicvm")

    proceed = input("\n❓ Do you have a Roboflow API key? (y/n): ").strip().lower()

    if proceed != "y":
        print("\n📖 MANUAL SETUP STEPS:")
        print("\n1. Get API Key:")
        print("   → Visit: https://app.roboflow.com/settings/api")
        print("   → Copy your API key")

        print("\n2. Choose a Model:")
        print("   → Visit: https://universe.roboflow.com/")
        print("   → Search: 'violence detection'")
        print("   → Select a model and note: workspace/project/version")

        print("\n3. Download Model:")
        print("   → Run this script again with your API key")
        print("   → Or use the Roboflow Python API manually")

        print("\n4. Alternative - Train Your Own:")
        print("   → Open: notebooks/Colab_Training_Server.ipynb")
        print("   → Follow training instructions")
        print("   → Download best.pt to models/violence_detection.pt")

        return False

    # Install roboflow if needed
    if not install_roboflow():
        return False

    from roboflow import Roboflow

    # Get API key
    print("\n🔑 Enter your Roboflow API key")
    print("   (Get it from: https://app.roboflow.com/settings/api)")
    api_key = input("API Key: ").strip()

    if not api_key:
        print("❌ No API key provided")
        return False

    # Model selection
    print("\n📦 Select a pre-configured violence detection model:")
    print("\n1. Violence Detection (Recommended)")
    print("   - Workspace: vd-hz6jg")
    print("   - Project: violence-detection-o4rdr")
    print("   - Classes: violence, normal")

    print("\n2. Weapon Detection")
    print("   - Workspace: weapons-uw60y")
    print("   - Project: weapons-lidce")
    print("   - Classes: pistol, rifle, knife, etc.")

    print("\n3. Custom (Enter your own)")

    choice = input("\nChoice (1-3): ").strip()

    if choice == "1":
        workspace = "vd-hz6jg"
        project = "violence-detection-o4rdr"
        version = 2
    elif choice == "2":
        workspace = "weapons-uw60y"
        project = "weapons-lidce"
        version = 1
    elif choice == "3":
        workspace = input("Workspace: ").strip()
        project = input("Project: ").strip()
        version = int(input("Version: ").strip())
    else:
        print("❌ Invalid choice")
        return False

    try:
        print(f"\n📥 Downloading model from Roboflow...")
        print(f"   Workspace: {workspace}")
        print(f"   Project: {project}")
        print(f"   Version: {version}")

        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(
            "yolov8", location=str(models_dir / "dataset")
        )

        print(f"\n✓ Dataset downloaded to: {dataset.location}")

        # Find the trained weights
        weights_path = Path(dataset.location) / "train" / "weights" / "best.pt"
        if not weights_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(dataset.location) / "weights" / "best.pt",
                Path(dataset.location) / "best.pt",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    weights_path = alt_path
                    break

        if weights_path.exists():
            target_path = models_dir / "violence_detection.pt"
            import shutil

            shutil.copy(weights_path, target_path)
            print(f"✓ Model weights copied to: {target_path}")
        else:
            print("\n⚠️  Pre-trained weights not found in dataset")
            print("   You need to train the model first using the notebook")

        # Show model info
        print(f"\n📊 Dataset Info:")
        print(f"   Location: {dataset.location}")
        print(f"   Classes: Check data.yaml in the dataset folder")

        print("\n✅ Setup Complete!")
        print("\n📝 Next Steps:")
        print("   1. Check models/violence_detection.pt exists")
        print("   2. Update config.py if needed")
        print("   3. Restart consumer: python src/consumer_video.py")
        print("   4. Test with: python src/producer.py --video data/v001_converted.avi")

        return True

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Try these alternatives:")
        print("   1. Check your API key is correct")
        print("   2. Verify workspace/project/version exist on Roboflow")
        print("   3. Train your own model using the Colab notebook")
        return False


if __name__ == "__main__":
    try:
        success = download_violence_model()
        if not success:
            print("\n📖 For detailed instructions, see: VIOLENCE_MODEL_SETUP.md")
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n📖 For detailed instructions, see: VIOLENCE_MODEL_SETUP.md")
