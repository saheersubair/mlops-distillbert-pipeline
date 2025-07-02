"""
FastAPI application for DistillBERT model serving
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse
)
from .utils import ModelManager, get_model_manager

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

# Initialize metrics only once
try:
    # Check if metrics already exist to avoid duplicates
    REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests', ['model_version', 'status'])
    REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction request latency')
    MODEL_LOAD_TIME = Histogram('model_load_time_seconds', 'Model loading time')
except ValueError as e:
    if "Duplicated timeseries" in str(e):
        # Metrics already exist, get them from registry
        logger.warning("Metrics already registered, reusing existing metrics")
        for collector in REGISTRY._collector_to_names:
            if hasattr(collector, '_name'):
                if collector._name == 'prediction_requests_total':
                    REQUEST_COUNT = collector
                elif collector._name == 'prediction_latency_seconds':
                    REQUEST_LATENCY = collector
                elif collector._name == 'model_load_time_seconds':
                    MODEL_LOAD_TIME = collector
    else:
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting MLOps DistillBERT API...")

    # Initialize model manager
    model_manager = get_model_manager()
    await model_manager.load_model()

    # Setup custom metrics (import here to avoid circular imports)
    try:
        from .monitoring.metrics import setup_custom_metrics
        setup_custom_metrics()
    except ImportError:
        logger.warning("Custom metrics module not available")
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    # Cleanup resources if needed

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
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)
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
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(model_version=request.model_version or "default", status="success").inc()
        
        # Background task for logging
        background_tasks.add_task(
            log_prediction,
            request.text[:100],  # Log first 100 chars
            result,
            latency,
            request.model_version
        )
        
        return PredictionResponse(
            prediction=result["label"],
            confidence=result["score"],
            model_version=request.model_version or model_manager.current_version,
            processing_time=latency,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        REQUEST_COUNT.labels(model_version=request.model_version or "default", status="error").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(model_version=request.model_version or "default", status="error").inc()
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
        
        if len(request.texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 100")
        
        # Validate individual texts
        for i, text in enumerate(request.texts):
            if not text or len(text.strip()) == 0:
                raise HTTPException(status_code=400, detail=f"Text at index {i} cannot be empty")
            if len(text) > 512:
                raise HTTPException(status_code=400, detail=f"Text at index {i} exceeds maximum length")
        
        # Make batch predictions
        results = await model_manager.predict_batch(request.texts, request.model_version)
        
        # Record metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(model_version=request.model_version or "default", status="success").inc()
        
        # Background task for logging
        background_tasks.add_task(
            log_batch_prediction,
            len(request.texts),
            results,
            latency,
            request.model_version
        )
        
        return BatchPredictionResponse(
            predictions=[
                {
                    "prediction": result["label"],
                    "confidence": result["score"]
                }
                for result in results
            ],
            model_version=request.model_version or model_manager.current_version,
            processing_time=latency,
            batch_size=len(request.texts),
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        REQUEST_COUNT.labels(model_version=request.model_version or "default", status="error").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(model_version=request.model_version or "default", status="error").inc()
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    try:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
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

async def log_prediction(text_preview: str, result: Dict[str, Any], latency: float, model_version: Optional[str]):
    """Background task to log prediction details"""
    logger.info({
        "event": "prediction",
        "text_preview": text_preview,
        "prediction": result["label"],
        "confidence": result["score"],
        "latency": latency,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat()
    })

async def log_batch_prediction(batch_size: int, results: List[Dict[str, Any]], latency: float, model_version: Optional[str]):
    """Background task to log batch prediction details"""
    avg_confidence = sum(r["score"] for r in results) / len(results)
    
    logger.info({
        "event": "batch_prediction",
        "batch_size": batch_size,
        "avg_confidence": avg_confidence,
        "latency": latency,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1))
    )