# 🚀 MLOps DistillBERT Pipeline

A production-ready MLOps pipeline for serving DistillBERT sentiment analysis models with automated training, deployment, monitoring, and A/B testing capabilities.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-green.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)
![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [📊 API Documentation](#-api-documentation)
- [🧪 Testing](#-testing)
- [🚢 Deployment](#-deployment)
- [📈 Monitoring](#-monitoring)
- [🔄 CI/CD Pipeline](#-cicd-pipeline)
- [🧬 A/B Testing](#-ab-testing)
- [🛠️ Development](#️-development)
- [📚 Documentation](#-documentation)

## 🎯 Overview

This MLOps pipeline provides a complete end-to-end solution for deploying and managing DistillBERT sentiment analysis models in production. It includes automated training, model versioning, A/B testing, comprehensive monitoring, and scalable deployment capabilities.

### 🎪 Key Highlights

- **🤖 Model Serving**: Production-ready FastAPI service with async endpoints
- **🔄 Automated Training**: Intelligent retraining based on data changes
- **📊 A/B Testing**: Complete statistical framework for model comparison
- **📈 Monitoring**: Real-time metrics with Prometheus and Grafana
- **🚀 Deployment**: Multi-environment support (Docker, Kubernetes)
- **🧪 Testing**: Comprehensive test suite with 95%+ coverage
- **🔒 Security**: Authentication, rate limiting, input validation
- **📦 Scalability**: Horizontal scaling and load balancing

## 🏗️ Architecture

![MLOps Architecture Image](images/architecture.png)

## ✨ Features

### 🔥 Core Features

- **Model Serving**
  - FastAPI-based REST API with async support
  - Batch and single prediction endpoints
  - Model versioning and hot-swapping
  - Automatic model caching and optimization

- **Training Pipeline**
  - Automated data validation and preprocessing
  - Distributed training with GPU support
  - Hyperparameter tuning with Optuna
  - Model performance validation gates

- **Deployment**
  - Multi-stage Docker containers
  - Kubernetes manifests with auto-scaling
  - Blue-green deployments
  - Canary releases

- **Monitoring**
  - Real-time metrics with Prometheus
  - Custom business metrics
  - Grafana dashboards
  - Alert manager integration

### 🎯 Advanced Features

- **A/B Testing Framework**
  - Statistical significance testing
  - Traffic splitting strategies
  - Experiment management
  - Automated decision making

- **Feature Store**
  - Real-time feature serving
  - Feature versioning and lineage
  - Data quality monitoring
  - Integration with ML pipelines

- **Security**
  - API key authentication
  - Rate limiting and DDoS protection
  - Input validation and sanitization
  - Audit logging

- **Scalability**
  - Horizontal pod autoscaling
  - Load balancing
  - Connection pooling
  - Resource optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker 20.10+
- Docker Compose 2.0+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mlops-distillbert-pipeline.git
cd mlops-distillbert-pipeline
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Sample Data

```bash
python scripts/data_generator_script.py
```

### 4. Start the Services

```bash
# Using Docker Compose (Recommended)
docker-compose up --build

# Or run locally
python src/api/main.py
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

### 6. Access Monitoring

- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090

## 📦 Installation

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/your-username/mlops-distillbert-pipeline.git
cd mlops-distillbert-pipeline

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Option 2: Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Generate sample data
python generate_sample_data.py

# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```


## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Model Configuration
export MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
export MODEL_CACHE_DIR=/app/models
export BATCH_SIZE=32

# Monitoring
export ENABLE_METRICS=true
export PROMETHEUS_PORT=9090
export LOG_LEVEL=INFO

# Database (Optional)
export DATABASE_URL=postgresql://user:pass@localhost:5432/mlops
export REDIS_URL=redis://localhost:6379/0
```

### Configuration Files

- **API Config**: `config/api_config.yaml`
- **Model Config**: `config/model_config.yaml`
- **A/B Testing**: `config/ab_test_config.yaml`

### Example API Configuration

```yaml
# config/api_config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false

monitoring:
  prometheus:
    enabled: true
    port: 9090
  
cache:
  enabled: true
  ttl: 3600
  max_size: 1000

security:
  rate_limit:
    enabled: true
    requests_per_minute: 100
```

## 📊 API Documentation

### Endpoints

#### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "I love this product!",
  "model_version": "v1.0.0"
}
```

**Response:**
```json
{
  "prediction": "POSITIVE",
  "confidence": 0.95,
  "model_version": "v1.0.0",
  "processing_time": 0.123,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "Great product!",
    "Terrible experience.",
    "Average quality."
  ],
  "model_version": "v1.0.0"
}
```

**Response:**
```json
{
  "predictions": [
    {"prediction": "POSITIVE", "confidence": 0.92},
    {"prediction": "NEGATIVE", "confidence": 0.88},
    {"prediction": "NEUTRAL", "confidence": 0.65}
  ],
  "model_version": "v1.0.0",
  "processing_time": 0.245,
  "batch_size": 3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Model Information
```bash
GET /model/info
```

**Response:**
```json
{
  "name": "distilbert-base-uncased-finetuned-sst-2-english",
  "version": "v1.0.0",
  "task": "sentiment-analysis",
  "loaded_at": "2024-01-15T10:00:00Z",
  "parameters": {
    "total_parameters": 66955010,
    "model_size_mb": 267.8
  }
}
```

#### Metrics
```bash
GET /metrics
```

Returns Prometheus-formatted metrics.

### Rate Limits

- **Default**: 100 requests per minute per IP
- **Burst**: 10 requests per second
- **Authenticated**: 1000 requests per minute

### Error Handling

```json
{
  "error": "validation_error",
  "message": "Text length exceeds maximum limit of 512 characters",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "abc123"
}
```

## 🧪 Testing

### Run All Tests

```bash
# Run test suite
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_api.py -v              # API tests
pytest tests/test_model.py -v            # Model tests
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

### API Testing

```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product quality!"}'

# Test with Python
python -c "
import requests
response = requests.post('http://localhost:8000/predict', 
                        json={'text': 'Great service!'})
print(response.json())
"
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build image
docker build -t mlops-distillbert:latest .

# Run container
docker run -p 8000:8000 mlops-distillbert:latest

# Use Docker Compose
docker-compose up -d
```



### Production Considerations

- **SSL/TLS**: Configure HTTPS with valid certificates
- **Database**: Use managed database services
- **Monitoring**: Set up centralized logging
- **Backup**: Implement automated backups
- **Security**: Enable authentication and authorization

## 📈 Monitoring

### Prometheus Metrics

Available at `http://localhost:9090/metrics`:

```prometheus
# Request metrics
mlops_prediction_requests_total{model_version="v1.0.0",status="success"} 1524
mlops_prediction_latency_seconds{model_version="v1.0.0",quantile="0.95"} 0.145

# Business metrics
mlops_model_accuracy{model_version="v1.0.0"} 0.94
mlops_model_confidence{model_version="v1.0.0",quantile="0.5"} 0.87

# System metrics
mlops_cpu_usage_percent 45.2
mlops_memory_usage_bytes 2147483648
```

### Grafana Dashboards

Access at `http://localhost:3000` (admin/admin123):

- **API Performance**: Request rates, latency, errors
- **Model Metrics**: Accuracy, confidence, drift
- **System Health**: CPU, memory, disk usage
- **Business KPIs**: User engagement, conversion rates

### Alerting Rules

```yaml
# Example alert rules
groups:
  - name: mlops_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(mlops_prediction_requests_total{status="error"}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: ModelAccuracyDrop
        expr: mlops_model_accuracy < 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model accuracy below threshold"
```

### Custom Metrics

```python
# Add custom metrics
from prometheus_client import Counter, Histogram

BUSINESS_METRIC = Counter('business_conversions_total', 'Business conversions')
RESPONSE_TIME = Histogram('api_response_time_seconds', 'API response time')

# Record metrics
BUSINESS_METRIC.inc()
RESPONSE_TIME.observe(0.123)
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The pipeline includes:

1. **Code Quality**: Linting, formatting, type checking
2. **Testing**: Unit, integration, and security tests
3. **Building**: Docker image building and scanning
4. **Deployment**: Automated deployment to staging/production
5. **Monitoring**: Health checks and rollback capabilities

### Pipeline Stages

```yaml
# .github/workflows/ci-cd.yml
name: MLOps CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/ -v

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t mlops-distillbert:${{ github.sha }} .

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: kubectl set image deployment/mlops-api api=mlops-distillbert:${{ github.sha }}
```

### Model Training Pipeline

```yaml
# .github/workflows/model-training.yml
name: Model Training Pipeline

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly training
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python src/training/train.py
      
      - name: Evaluate model
        run: python src/training/evaluate.py
      
      - name: Deploy if performance improves
        run: python scripts/deploy_model.py
```

### Environment Variables

Required secrets in GitHub Actions:

```bash
# Docker Registry
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password

# Kubernetes
KUBE_CONFIG_DATA=base64-encoded-kubeconfig

# Cloud Providers
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Monitoring
SLACK_WEBHOOK_URL=your-slack-webhook
```

## 🧬 A/B Testing

### Setting Up Experiments

```python
# Create A/B test experiment
python scripts/ab_testing_script.py --command create

# Or use interactive CLI
python scripts/ab_testing_script.py --command cli
```

### Experiment Configuration

```yaml
# config/ab_test_config.yaml
experiments:
  model_comparison:
    control_variant: "distillbert_v1"
    treatment_variants: ["distillbert_v2"]
    traffic_allocation:
      distillbert_v1: 50
      distillbert_v2: 50
    success_metrics: ["accuracy", "f1_score"]
    minimum_sample_size: 1000
```

### Analyzing Results

```python
# Analyze experiment results
python scripts/ab_testing_script.py --command analyze --experiment-id exp_123

# Generate report
python scripts/ab_testing_script.py --command report --experiment-id exp_123
```

### Statistical Framework

- **Power Analysis**: Automatic sample size calculation
- **Significance Testing**: Two-tailed t-tests with Bonferroni correction
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for effect estimates

## 🛠️ Development

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/mlops-distillbert-pipeline.git
cd mlops-distillbert-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Start development server
uvicorn src.api.main:app --reload
```

### Code Quality Tools

```bash
# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Dependency scanning
safety check
```

## 📚 Documentation

### API Documentation

- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json
