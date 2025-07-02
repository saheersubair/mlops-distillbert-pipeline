#!/usr/bin/env python3
"""
Simple server startup script for MLOps API
This avoids import issues by properly setting up the environment first
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set up environment variables with defaults
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "8000")
os.environ.setdefault("WORKERS", "1")  # Use 1 worker to avoid metric conflicts
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("RELOAD", "false")


def setup_logging():
    """Setup logging configuration"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "logs",
        "config"
    ]

    for directory in directories:
        dir_path = current_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Directory created/verified: {dir_path}")


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "prometheus_client"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")

    if missing_packages:
        print(f"\nError: Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main startup function"""
    print("🚀 Starting MLOps DistillBERT API Server")
    print("=" * 50)

    # Setup
    setup_logging()
    create_directories()
    check_dependencies()

    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    print(f"\n📋 Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    print(f"   Reload: {reload}")
    print(f"   Log Level: {log_level}")
    print(f"   Python Path: {sys.path[0]}")

    try:
        import uvicorn

        print(f"\n🌟 Starting server at http://{host}:{port}")
        print("   API Documentation: http://localhost:8000/docs")
        print("   Health Check: http://localhost:8000/health")
        print("   Metrics: http://localhost:8000/metrics")
        print("\n   Press Ctrl+C to stop the server")
        print("=" * 50)

        # Start the server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,  # Use 1 worker with reload
            reload=reload,
            log_level=log_level,
            access_log=True,
            loop="asyncio"
        )

    except ImportError as e:
        print(f"❌ Error importing uvicorn: {e}")
        print("Please install uvicorn with: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        logging.exception("Server startup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()