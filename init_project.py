"""
Initialize project directories and check dependencies
"""

from pathlib import Path


def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/training_samples",
        "data/training_dataset",
        "data/training_dataset/images",
        "data/training_dataset/labels",
        "data/archive",
        "models",
        "logs",
    ]

    base_dir = Path(__file__).parent

    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "cv2",
        "kafka",
        "pymongo",
        "streamlit",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")

    if missing_packages:
        print("\n⚠️  Missing packages detected!")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def check_docker():
    """Check if Docker is running"""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "ps"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("✓ Docker is running")
            return True
        else:
            print("✗ Docker is not running")
            return False
    except Exception as e:
        print(f"✗ Could not check Docker status: {e}")
        return False


def main():
    """Main initialization"""
    print("🚀 Initializing Harmful Content Detection System\n")

    print("=" * 50)
    print("1. Creating directories...")
    print("=" * 50)
    create_directories()

    print("\n" + "=" * 50)
    print("2. Checking Python dependencies...")
    print("=" * 50)
    deps_ok = check_dependencies()

    print("\n" + "=" * 50)
    print("3. Checking Docker...")
    print("=" * 50)
    docker_ok = check_docker()

    print("\n" + "=" * 50)
    print("Initialization Summary")
    print("=" * 50)

    if deps_ok and docker_ok:
        print("✅ System is ready to run!")
        print("\nNext steps:")
        print("1. Place your test video in data/ folder")
        print("2. Run: .\\startup.ps1 (Windows) to start services")
        print("3. Follow instructions in QUICKSTART.md")
    else:
        print("⚠️  System initialization incomplete")
        if not deps_ok:
            print("   - Install Python dependencies: pip install -r requirements.txt")
        if not docker_ok:
            print("   - Start Docker Desktop")

    print("=" * 50)


if __name__ == "__main__":
    main()
