# Fixed FastAPI Main Application with Working Metrics
# Path: src/api/main.py

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse
)
from src.api.utils import ModelManager, get_model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create custom registry for our metrics
REGISTRY = CollectorRegistry()

# Custom Prometheus metrics
REQUEST_COUNT = Counter(
    'mlops_prediction_requests_total',
    'Total prediction requests',
    ['model_version', 'status', 'endpoint'],
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    'mlops_prediction_latency_seconds',
    'Prediction request latency',
    ['model_version', 'endpoint'],
    registry=REGISTRY
)

MODEL_LOAD_TIME = Histogram(
    'mlops_model_load_time_seconds',
    'Model loading time',
    ['model_version'],
    registry=REGISTRY
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting MLOps DistillBERT API...")

    # Initialize model manager
    model_manager = get_model_manager()
    start_time = time.time()
    await model_manager.load_model()
    load_time = time.time() - start_time

    # Record model load time
    MODEL_LOAD_TIME.labels(model_version="v1.0.0").observe(load_time)

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="MLOps DistillBERT API",
    description="Production-ready API for DistillBERT sentiment analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation (default metrics)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_manager = get_model_manager()
        model_status = "healthy" if model_manager.is_model_loaded() else "loading"

        return HealthResponse(
            status="healthy",
            model_status=model_status,
            timestamp=datetime.utcnow(),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_status="error",
            timestamp=datetime.utcnow(),
            version="1.0.0"
        )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(model_manager: ModelManager = Depends(get_model_manager)):
    """Get model information"""
    try:
        info = model_manager.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Single text prediction endpoint"""
    start_time = time.time()

    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        if len(request.text) > 512:
            raise HTTPException(status_code=400, detail="Text length exceeds maximum limit of 512 characters")

        # Make prediction
        result = await model_manager.predict(request.text, request.model_version)

        # Record metrics
        latency = time.time() - start_time
        model_version = request.model_version or "v1.0.0"

        REQUEST_LATENCY.labels(
            model_version=model_version,
            endpoint="predict"
        ).observe(latency)

        REQUEST_COUNT.labels(
            model_version=model_version,
            status="success",
            endpoint="predict"
        ).inc()

        return PredictionResponse(
            prediction=result["label"],
            confidence=result["score"],
            model_version=model_version,
            processing_time=latency,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        REQUEST_COUNT.labels(
            model_version=request.model_version or "v1.0.0",
            status="error",
            endpoint="predict"
        ).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(
            model_version=request.model_version or "v1.0.0",
            status="error",
            endpoint="predict"
        ).inc()
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Batch prediction endpoint"""
    start_time = time.time()

    try:
        # Validate input
        if not request.texts or len(request.texts) == 0:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")

        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 100")

        # Make batch predictions
        results = await model_manager.predict_batch(request.texts, request.model_version)

        # Record metrics
        latency = time.time() - start_time
        model_version = request.model_version or "v1.0.0"

        REQUEST_LATENCY.labels(
            model_version=model_version,
            endpoint="batch"
        ).observe(latency)

        REQUEST_COUNT.labels(
            model_version=model_version,
            status="success",
            endpoint="batch"
        ).inc()

        return BatchPredictionResponse(
            predictions=[
                {
                    "prediction": result["label"],
                    "confidence": result["score"]
                }
                for result in results
            ],
            model_version=model_version,
            processing_time=latency,
            batch_size=len(request.texts),
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        REQUEST_COUNT.labels(
            model_version=request.model_version or "v1.0.0",
            status="error",
            endpoint="batch"
        ).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(
            model_version=request.model_version or "v1.0.0",
            status="error",
            endpoint="batch"
        ).inc()
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")

@app.get("/metrics")
async def get_metrics():
    """Custom Prometheus metrics endpoint"""
    try:
        # Generate metrics from our custom registry
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Failed to generate metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")

@app.get("/models/versions")
async def list_model_versions(model_manager: ModelManager = Depends(get_model_manager)):
    """List available model versions"""
    try:
        versions = model_manager.list_available_versions()
        return {"versions": versions, "current": model_manager.current_version}
    except Exception as e:
        logger.error(f"Failed to list model versions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model versions")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1))
    )