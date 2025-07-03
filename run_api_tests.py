#!/usr/bin/env python3
"""
Test runner for real model integration tests (no mocks)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_virtual_environment():
    """Find and return the path to the virtual environment"""
    current_dir = Path(__file__).parent

    # Common virtual environment names and locations
    venv_candidates = [
        current_dir / "venv",
        current_dir / ".venv",
        current_dir / "env",
        current_dir / ".env"
    ]

    for venv_path in venv_candidates:
        if venv_path.exists():
            # Check for Python executable
            if os.name == 'nt':  # Windows
                python_exe = venv_path / "Scripts" / "python.exe"
                pip_exe = venv_path / "Scripts" / "pip.exe"
            else:  # Unix/Linux/macOS
                python_exe = venv_path / "bin" / "python"
                pip_exe = venv_path / "bin" / "pip"

            if python_exe.exists():
                return venv_path, python_exe, pip_exe

    return None, None, None


def get_python_executable():
    """Get the correct Python executable (preferring virtual environment)"""
    venv_path, python_exe, pip_exe = find_virtual_environment()

    if python_exe and python_exe.exists():
        print(f"✓ Found virtual environment: {venv_path}")
        print(f"✓ Using Python: {python_exe}")
        return str(python_exe), str(pip_exe)
    else:
        print("⚠️  No virtual environment found, using system Python")
        print("💡 Consider creating a virtual environment: python -m venv venv")
        return sys.executable, "pip"


def setup_environment():
    """Setup environment for real model testing"""
    # Add src to Python path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"

    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
        os.environ["PYTHONPATH"] = str(src_dir)
        print(f"✓ Added {src_dir} to Python path")

    # Set environment variables for real model testing
    os.environ["TESTING"] = "false"  # Enable real model loading
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["ENABLE_METRICS"] = "true"

    print("✓ Environment configured for real model testing")


def check_and_install_dependencies(python_exe, pip_exe):
    """Check if required packages are installed and install if missing"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "pytest",
        "numpy",
        "pandas"
    ]

    print("🔍 Checking dependencies...")

    missing = []
    for package in required_packages:
        try:
            result = subprocess.run([python_exe, "-c", f"import {package}"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {package} is installed")
            else:
                missing.append(package)
                print(f"✗ {package} is missing")
        except Exception:
            missing.append(package)
            print(f"✗ {package} is missing")

    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")

        # Check if requirements.txt exists
        req_file = Path("requirements.txt")
        if req_file.exists():
            print("📋 Found requirements.txt, installing all dependencies...")
            try:
                result = subprocess.run([pip_exe, "install", "-r", "requirements.txt"],
                                        check=True, capture_output=True, text=True)
                print("✅ Successfully installed dependencies from requirements.txt")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install from requirements.txt: {e}")
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)
        else:
            # Install individual packages
            for package in missing:
                try:
                    print(f"📦 Installing {package}...")
                    subprocess.run([pip_exe, "install", package], check=True)
                    print(f"✅ Installed {package}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {package}: {e}")
                    return False
        return True

    return True


def download_model_if_needed(python_exe):
    """Pre-download the model if it doesn't exist"""
    print("\n🤖 Checking if DistillBERT model is available...")

    try:
        # Use the virtual environment Python
        test_script = '''
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"📥 Downloading/verifying model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("✅ Model is ready!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)
'''

        result = subprocess.run([python_exe, "-c", test_script],
                                capture_output=True, text=True, timeout=300)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏰ Model download timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Failed to check model: {e}")
        return False


def run_test_file(python_exe, test_file, verbose=False):
    """Run a specific test file"""
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    print(f"\n🧪 Running {test_file}...")

    cmd = [python_exe, "-m", "pytest", test_file, "-m", "integration", "--tb=short"]

    if verbose:
        cmd.extend(["-v", "-s"])

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run {test_file}: {e}")
        return False


def run_api_integration_tests(python_exe, verbose=False):
    """Run API integration tests"""
    return run_test_file(python_exe, "tests/test_api.py", verbose)


def run_model_integration_tests(python_exe, verbose=False):
    """Run model component integration tests"""
    return run_test_file(python_exe, "tests/test_model.py", verbose)


def run_performance_tests(python_exe):
    """Run performance tests with real model"""
    print("\n⚡ Running Performance Tests...")

    perf_script = '''
import sys
import os
import asyncio
import time
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

async def performance_test():
    try:
        from api.utils import ModelManager

        manager = ModelManager()
        print("📊 Loading model for performance testing...")

        start_time = time.time()
        await manager.load_model()
        load_time = time.time() - start_time
        print(f"   Model load time: {load_time:.2f}s")

        # Single prediction performance
        test_text = "This is a great product and I love it!"

        # Warm-up prediction
        await manager.predict(test_text)

        # Measure prediction time
        times = []
        for i in range(5):  # Reduced for faster testing
            start = time.time()
            await manager.predict(test_text)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"   Single prediction avg: {avg_time:.3f}s")
        print(f"   Single prediction range: {min_time:.3f}s - {max_time:.3f}s")

        # Batch prediction performance
        batch_texts = [f"Test message {i}" for i in range(5)]

        start_time = time.time()
        await manager.predict_batch(batch_texts)
        batch_time = time.time() - start_time

        print(f"   Batch prediction (5 items): {batch_time:.3f}s")
        print(f"   Batch efficiency: {(avg_time * 5) / batch_time:.1f}x faster")

        print("✅ Performance tests completed!")
        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(performance_test())
    exit(0 if result else 1)
'''

    try:
        result = subprocess.run([python_exe, "-c", perf_script],
                                check=False, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Performance test timed out")
        return False
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def run_live_demo(python_exe):
    """Run a live demo with real predictions"""
    print("\n🎭 Running Live Demo...")

    demo_script = '''
import sys
import os
import asyncio
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

async def demo():
    try:
        from api.utils import ModelManager

        manager = ModelManager()
        print("🤖 Loading DistillBERT model...")
        await manager.load_model()

        demo_texts = [
            "I absolutely love this product! It's the best purchase I've ever made!",
            "This product is terrible. Complete waste of money and time.",
            "The product is okay. Nothing special but it works as expected.",
            "Amazing quality and fast shipping! Highly recommend to everyone!",
        ]

        print("\\n🎯 Live Sentiment Analysis Demo:")
        print("=" * 70)

        for i, text in enumerate(demo_texts, 1):
            result = await manager.predict(text)

            sentiment = result['label']
            confidence = result['score']
            emoji = "😊" if sentiment == "POSITIVE" else "😞"

            print(f"\\n{i}. Text: \\"{text[:50]}{'...' if len(text) > 50 else ''}\\"")
            print(f"   Result: {sentiment} {emoji} (confidence: {confidence:.1%})")

        print("\\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(demo())
    exit(0 if result else 1)
'''

    try:
        result = subprocess.run([python_exe, "-c", demo_script],
                                check=False, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run real model integration tests")
    parser.add_argument("--api", action="store_true", help="Run only API tests")
    parser.add_argument("--model", action="store_true", help="Run only model tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--demo", action="store_true", help="Run live demo")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download check")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")

    args = parser.parse_args()

    print("🚀 MLOps Real Model Test Runner")
    print("=" * 50)
    print("📝 These tests use the ACTUAL DistillBERT model")
    print("⚠️  NO MOCKS - Real model loading and inference")
    print("⏳ First run may take several minutes...")
    print("=" * 50)

    # Get the correct Python executable
    python_exe, pip_exe = get_python_executable()

    # Setup environment
    setup_environment()

    # Check and install dependencies
    if not args.skip_install:
        if not check_and_install_dependencies(python_exe, pip_exe):
            print("❌ Failed to install dependencies")
            return 1

    # Download model if needed
    if not args.skip_download:
        if not download_model_if_needed(python_exe):
            print("❌ Failed to download/verify model")
            return 1

    # Determine what to run
    run_all = args.all or not any([args.api, args.model, args.performance, args.demo])

    results = []

    if args.api or run_all:
        results.append(("API Integration Tests", run_api_integration_tests(python_exe, args.verbose)))

    if args.model or run_all:
        results.append(("Model Integration Tests", run_model_integration_tests(python_exe, args.verbose)))

    if args.performance or run_all:
        results.append(("Performance Tests", run_performance_tests(python_exe)))

    if args.demo or run_all:
        results.append(("Live Demo", run_live_demo(python_exe)))

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your MLOps pipeline is working with the real model!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())