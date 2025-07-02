import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Root project folder
root = "mlops-distillbert-pipeline"

# Define all folders to create
all_dirs = [
    ".github/workflows",
    "src/api",
    "src/training",
    "src/monitoring",
    "tests",
    "docker",
    "config"
]

# Define files to create in each folder
files_to_create = {
    ".github/workflows": ["ci-cd.yml", "model-training.yml"],
    "src/api": ["__init__.py", "main.py", "models.py", "utils.py"],
    "src/training": ["__init__.py", "train.py", "evaluate.py"],
    "src/monitoring": ["__init__.py", "metrics.py"],
    "tests": ["test_api.py", "test_model.py"],
    "docker": ["Dockerfile", "docker-compose.yml"],
    "config": ["model_config.yaml", "api_config.yaml"],
    ".": ["requirements.txt", "README.md", ".env.example"]
}

# Step 1: Create all directories
for dir_path in all_dirs:
    full_dir_path = dir_path
    os.makedirs(full_dir_path, exist_ok=True)
    logging.info(f"Created directory: {full_dir_path}")

# Step 2: Create files
for rel_folder, filenames in files_to_create.items():
    for filename in filenames:
        file_path = os.path.join(rel_folder, filename)
        if not os.path.exists(file_path):
            # Ensure the folder exists before file creation
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("")
            logging.info(f"Created file: {file_path}")
        else:
            logging.info(f"File already exists: {file_path}")
