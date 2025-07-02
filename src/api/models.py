"""
Pydantic models for API request/response validation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    text: str = Field(..., description="Text to analyze", max_length=512)
    model_version: Optional[str] = Field(None, description="Specific model version to use")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    prediction: str = Field(..., description="Predicted label")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    texts: List[str] = Field(..., description="List of texts to analyze", max_items=100)
    model_version: Optional[str] = Field(None, description="Specific model version to use")

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 512:
                raise ValueError(f'Text at index {i} exceeds maximum length of 512 characters')
        return [text.strip() for text in v]


class PredictionItem(BaseModel):
    """Individual prediction item for batch response"""
    prediction: str = Field(..., description="Predicted label")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionItem] = Field(..., description="List of predictions")
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    batch_size: int = Field(..., description="Number of texts processed")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information response"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    task: str = Field(..., description="Model task type")
    loaded_at: datetime = Field(..., description="Model load timestamp")
    size_mb: Optional[float] = Field(None, description="Model size in MB")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall service status")
    model_status: str = Field(..., description="Model loading status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")


class ModelVersionInfo(BaseModel):
    """Model version information"""
    version: str = Field(..., description="Version identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    status: str = Field(..., description="Version status (active, deprecated, etc.)")


class ABTestConfig(BaseModel):
    """A/B testing configuration"""
    enabled: bool = Field(False, description="Whether A/B testing is enabled")
    traffic_split: Dict[str, int] = Field(default_factory=dict, description="Traffic split percentages")
    control_version: str = Field(..., description="Control model version")
    treatment_version: str = Field(..., description="Treatment model version")


class MetricsResponse(BaseModel):
    """Metrics response model"""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    error_rate: float = Field(..., description="Error rate percentage")
    avg_latency: float = Field(..., description="Average latency in seconds")
    model_versions: Dict[str, int] = Field(..., description="Request count per model version")
    timestamp: datetime = Field(..., description="Metrics timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ModelTrainingRequest(BaseModel):
    """Model training request"""
    dataset_path: str = Field(..., description="Path to training dataset")
    model_name: str = Field(..., description="Name for the new model")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Training hyperparameters")
    validation_split: float = Field(0.2, description="Validation split ratio", ge=0.0, le=0.5)


class ModelTrainingResponse(BaseModel):
    """Model training response"""
    job_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="Training job status")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    created_at: datetime = Field(..., description="Job creation timestamp")


class ModelDeploymentRequest(BaseModel):
    """Model deployment request"""
    model_version: str = Field(..., description="Model version to deploy")
    deployment_strategy: str = Field("blue_green", description="Deployment strategy")
    rollback_threshold: float = Field(0.05, description="Error rate threshold for rollback")


class ModelDeploymentResponse(BaseModel):
    """Model deployment response"""
    deployment_id: str = Field(..., description="Deployment identifier")
    status: str = Field(..., description="Deployment status")
    current_version: str = Field(..., description="Currently active version")
    previous_version: Optional[str] = Field(None, description="Previous active version")
    deployed_at: datetime = Field(..., description="Deployment timestamp")


class FeatureStoreRequest(BaseModel):
    """Feature store request"""
    feature_names: List[str] = Field(..., description="List of feature names to retrieve")
    entity_ids: List[str] = Field(..., description="List of entity identifiers")
    timestamp: Optional[datetime] = Field(None, description="Point-in-time timestamp")


class FeatureStoreResponse(BaseModel):
    """Feature store response"""
    features: Dict[str, List[Any]] = Field(..., description="Feature values by name")
    entity_ids: List[str] = Field(..., description="Entity identifiers")
    timestamp: datetime = Field(..., description="Feature retrieval timestamp")


# Configuration models
class APIConfig(BaseModel):
    """API configuration"""
    host: str = Field("0.0.0.0", description="Host address")
    port: int = Field(8000, description="Port number")
    workers: int = Field(4, description="Number of worker processes")
    reload: bool = Field(False, description="Enable auto-reload")
    log_level: str = Field("INFO", description="Logging level")


class ModelConfig(BaseModel):
    """Model configuration"""
    name: str = Field(..., description="Model name")
    task: str = Field(..., description="Model task")
    cache_dir: str = Field("./models", description="Model cache directory")
    max_length: int = Field(512, description="Maximum input length")
    batch_size: int = Field(32, description="Batch size for inference")


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_logging: bool = Field(True, description="Enable request logging")
    log_sampling_rate: float = Field(1.0, description="Log sampling rate")
    metrics_port: int = Field(9090, description="Metrics server port")