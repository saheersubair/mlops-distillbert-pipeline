#!/usr/bin/env python3
"""
Troubleshooting script for MLOps DistillBERT Pipeline
This script helps diagnose and fix common issues
"""

import os
import sys
import subprocess
import importlib
import shutil
from pathlib import Path
import tempfile


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_status(message, status="INFO"):
    """Print a status message"""
    symbols = {"INFO": "ℹ️", "OK": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")


def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")

    version = sys.version_info
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status("Python 3.8+ is required!", "ERROR")
        return False
    else:
        print_status("Python version is compatible", "OK")
        return True


def check_virtual_environment():
    """Check if running in virtual environment"""
    print_header("Virtual Environment Check")

    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Running in virtual environment", "OK")
        print_status(f"Virtual env path: {sys.prefix}")
        return True
    else:
        print_status("Not running in virtual environment", "WARNING")
        print_status("Recommendation: Create and activate a virtual environment")
        return False


def check_required_packages():
    """Check if required packages are installed"""
    print_header("Required Packages Check")

    required_packages = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'transformers': 'Hugging Face Transformers',
        'torch': 'PyTorch',
        'prometheus_client': 'Prometheus metrics',
        'pydantic': 'Data validation',
        'numpy': 'Numerical computing',
        'requests': 'HTTP library'
    }

    missing_packages = []
    installed_packages = []

    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print_status(f"{package} - {description}", "OK")
            installed_packages.append(package)
        except ImportError:
            print_status(f"{package} - {description} (MISSING)", "ERROR")
            missing_packages.append(package)

    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        print_status("Run: pip install -r requirements.txt", "INFO")
        return False
    else:
        print_status("All required packages are installed", "OK")
        return True


def check_project_structure():
    """Check if project structure is correct"""
    print_header("Project Structure Check")

    current_dir = Path.cwd()
    required_structure = [
        'src',
        'src/api',
        'src/api/main.py',
        'src/api/models.py',
        'src/api/utils.py',
        'tests',
        'requirements.txt'
    ]

    missing_items = []

    for item in required_structure:
        path = current_dir / item
        if path.exists():
            print_status(f"{item}", "OK")
        else:
            print_status(f"{item} (MISSING)", "ERROR")
            missing_items.append(item)

    if missing_items:
        print_status("Some project files/directories are missing", "ERROR")
        return False
    else:
        print_status("Project structure is correct", "OK")
        return True


def check_python_path():
    """Check Python path configuration"""
    print_header("Python Path Check")

    current_dir = Path.cwd()
    src_dir = current_dir / "src"

    print_status(f"Current directory: {current_dir}")
    print_status(f"Python path: {sys.path}")

    if str(src_dir) in sys.path:
        print_status("src directory is in Python path", "OK")
        return True
    else:
        print_status("src directory is NOT in Python path", "WARNING")
        print_status("This may cause import errors", "WARNING")
        return False


def fix_python_path():
    """Add src directory to Python path"""
    print_header("Fixing Python Path")

    current_dir = Path.cwd()
    src_dir = current_dir / "src"

    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
        print_status(f"Added {src_dir} to Python path", "OK")
        return True
    else:
        print_status("src directory does not exist", "ERROR")
        return False


def test_imports():
    """Test importing main modules"""
    print_header("Import Test")

    modules_to_test = [
        ('api.main', 'FastAPI main module'),
        ('api.models', 'Pydantic models'),
        ('api.utils', 'Utility functions'),
    ]

    failed_imports = []

    for module, description in modules_to_test:
        try:
            importlib.import_module(module)
            print_status(f"{module} - {description}", "OK")
        except ImportError as e:
            print_status(f"{module} - {description} (FAILED: {e})", "ERROR")
            failed_imports.append(module)

    if failed_imports:
        print_status("Some imports failed", "ERROR")
        return False
    else:
        print_status("All imports successful", "OK")
        return True


def check_port_availability():
    """Check if default port is available"""
    print_header("Port Availability Check")

    import socket

    ports_to_check = [8000, 5000, 3000, 9090]

    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()

        if result == 0:
            print_status(f"Port {port} is in use", "WARNING")
        else:
            print_status(f"Port {port} is available", "OK")


def create_missing_directories():
    """Create missing directories"""
    print_header("Creating Missing Directories")

    directories = ['models', 'logs', 'config', 'data']

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_status(f"Created directory: {directory}", "OK")
        else:
            print_status(f"Directory exists: {directory}", "OK")


def create_config_files():
    """Create default configuration files if missing"""
    print_header("Creating Default Configuration Files")

    # Create .env file if missing
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# MLOps DistillBERT Environment Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
RELOAD=false
LOG_LEVEL=INFO
MODEL_CACHE_DIR=./models
ENABLE_METRICS=true
"""
        env_file.write_text(env_content)
        print_status("Created .env file", "OK")
    else:
        print_status(".env file exists", "OK")

    # Create model config if missing
    config_dir = Path('config')
    config_file = config_dir / 'model_config.yaml'
    if not config_file.exists():
        config_content = """model:
  name: "distilbert-base-uncased-finetuned-sst-2-english"
  task: "sentiment-analysis"
  cache_dir: "./models"

serving:
  max_length: 512
  batch_size: 32

cache:
  ttl: 3600
"""
        config_file.write_text(config_content)
        print_status("Created model_config.yaml", "OK")
    else:
        print_status("model_config.yaml exists", "OK")


def fix_prometheus_conflicts():
    """Fix Prometheus metrics conflicts"""
    print_header("Fixing Prometheus Conflicts")

    try:
        from prometheus_client import REGISTRY
        # Clear the default registry to avoid conflicts
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass

        print_status("Cleared Prometheus registry", "OK")
        return True
    except Exception as e:
        print_status(f"Failed to clear Prometheus registry: {e}", "ERROR")
        return False


def test_api_startup():
    """Test if API can start successfully"""
    print_header("API Startup Test")

    try:
        # Try to import the FastAPI app
        fix_python_path()
        from api.main import app
        print_status("FastAPI app imported successfully", "OK")

        # Test if we can create the app instance
        if app:
            print_status("FastAPI app instance created", "OK")
            return True
        else:
            print_status("Failed to create FastAPI app instance", "ERROR")
            return False

    except Exception as e:
        print_status(f"API startup test failed: {e}", "ERROR")
        return False


def run_quick_fix():
    """Run quick fixes for common issues"""
    print_header("Running Quick Fixes")

    # Fix Python path
    fix_python_path()

    # Create missing directories
    create_missing_directories()

    # Create missing config files
    create_config_files()

    # Fix Prometheus conflicts
    fix_prometheus_conflicts()

    print_status("Quick fixes applied", "OK")


def generate_startup_script():
    """Generate a working startup script"""
    print_header("Generating Startup Script")

    script_content = '''#!/usr/bin/env python3
"""
Generated startup script for MLOps API
"""

import os
import sys
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set environment variables
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "8000")
os.environ.setdefault("WORKERS", "1")
os.environ.setdefault("LOG_LEVEL", "INFO")

if __name__ == "__main__":
    import uvicorn

    try:
        print("🚀 Starting MLOps API...")
        uvicorn.run(
            "api.main:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            workers=int(os.getenv("WORKERS", 1)),
            reload=False,
            log_level=os.getenv("LOG_LEVEL", "info").lower()
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
'''

    script_path = Path('start_api.py')
    script_path.write_text(script_content)

    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)

    print_status(f"Created startup script: {script_path}", "OK")
    print_status("Run with: python start_api.py", "INFO")


def main():
    """Main troubleshooting function"""
    print_header("MLOps DistillBERT Pipeline Troubleshooter")

    print("This script will help diagnose and fix common issues.")
    print("Running comprehensive checks...\n")

    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Python Path", check_python_path),
    ]

    all_passed = True

    for check_name, check_func in checks:
        if not check_func():
            all_passed = False

    # Additional checks
    check_port_availability()

    # If basic checks failed, run fixes
    if not all_passed:
        print_header("Issues Found - Running Fixes")
        run_quick_fix()

        # Test again after fixes
        print_header("Re-testing After Fixes")
        if not test_imports():
            print_status("Imports still failing, trying API startup test", "WARNING")
            test_api_startup()

    # Final recommendations
    print_header("Recommendations")

    if all_passed:
        print_status("All checks passed! Your setup looks good.", "OK")
        print_status("Try running: python start_api.py", "INFO")
    else:
        print_status("Some issues were found and fixed.", "WARNING")
        print_status("Try the following steps:", "INFO")
        print("  1. Activate your virtual environment")
        print("  2. Run: pip install -r requirements.txt")
        print("  3. Run: python start_api.py")
        print("  4. Test with: curl http://localhost:8000/health")

    print_header("Troubleshooting Complete")


if __name__ == "__main__":
    main()