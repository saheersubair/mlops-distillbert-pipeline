"""
Model Manager for handling DistillBERT model loading, caching, and inference
"""

import os
import time
import pickle
import hashlib
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import lru_cache
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)
from cachetools import TTLCache
import yaml

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages DistillBERT model loading, caching, and inference"""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.model: Optional[Pipeline] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_info: Dict[str, Any] = {}
        self.current_version: str = "v1.0.0"
        self.loaded_at: Optional[datetime] = None

        # Cache for predictions
        self.prediction_cache = TTLCache(
            maxsize=1000,
            ttl=self.config.get('cache', {}).get('ttl', 3600)
        )

        # Model registry for version management
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self._initialize_registry()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model': {
                'name': 'distilbert-base-uncased-finetuned-sst-2-english',
                'task': 'sentiment-analysis',
                'cache_dir': './models'
            },
            'serving': {
                'max_length': 512,
                'batch_size': 32
            },
            'cache': {
                'ttl': 3600
            }
        }

    def _initialize_registry(self):
        """Initialize model registry"""
        self.model_registry = {
            "v1.0.0": {
                "model_name": self.config['model']['name'],
                "task": self.config['model']['task'],
                "created_at": datetime.utcnow(),
                "status": "active",
                "metrics": {
                    "accuracy": 0.91,
                    "f1_score": 0.90
                }
            }
        }

    async def load_model(self, version: Optional[str] = None) -> None:
        """Load the DistillBERT model"""
        start_time = time.time()
        target_version = version or self.current_version

        try:
            logger.info(f"Loading DistillBERT model version {target_version}...")

            model_config = self.model_registry.get(target_version)
            if not model_config:
                raise ValueError(f"Model version {target_version} not found in registry")

            model_name = model_config['model_name']
            cache_dir = self.config['model']['cache_dir']

            # Create cache directory
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            # Load model using pipeline for simplicity
            self.model = pipeline(
                model_config['task'],
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True,
                model_kwargs={'cache_dir': cache_dir}
            )

            # Load tokenizer separately for advanced operations
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            # Update model info
            self.model_info = {
                'name': model_name,
                'version': target_version,
                'task': model_config['task'],
                'loaded_at': datetime.utcnow(),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'parameters': self._get_model_parameters()
            }

            self.current_version = target_version
            self.loaded_at = datetime.utcnow()

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameter information"""
        if not self.model:
            return {}

        try:
            model = self.model.model
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
                'max_length': self.config['serving']['max_length']
            }
        except Exception as e:
            logger.warning(f"Could not get model parameters: {str(e)}")
            return {}

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        return self.model_info

    async def predict(self, text: str, model_version: Optional[str] = None) -> Dict[str, Any]:
        """Make a single prediction"""
        if not self.is_model_loaded():
            await self.load_model(model_version)

        # Check cache first
        cache_key = self._get_cache_key(text, model_version)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        try:
            # Make prediction
            result = self.model(text)[0]  # Get first (and only) result

            # Process result based on task type
            if self.config['model']['task'] == 'sentiment-analysis':
                prediction = self._process_sentiment_result(result)
            else:
                prediction = self._process_classification_result(result)

            # Cache result
            self.prediction_cache[cache_key] = prediction

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    async def predict_batch(self, texts: List[str], model_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        if not self.is_model_loaded():
            await self.load_model(model_version)

        try:
            # Check for cached results
            results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, model_version)
                if cache_key in self.prediction_cache:
                    results.append(self.prediction_cache[cache_key])
                else:
                    results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Process uncached texts
            if uncached_texts:
                batch_results = self.model(uncached_texts)

                for idx, result in zip(uncached_indices, batch_results):
                    if self.config['model']['task'] == 'sentiment-analysis':
                        prediction = self._process_sentiment_result(result[0])
                    else:
                        prediction = self._process_classification_result(result[0])

                    results[idx] = prediction

                    # Cache result
                    cache_key = self._get_cache_key(texts[idx], model_version)
                    self.prediction_cache[cache_key] = prediction

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise

    def _process_sentiment_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment analysis result"""
        # Find the result with highest score
        if isinstance(result, list):
            best_result = max(result, key=lambda x: x['score'])
        else:
            best_result = result

        return {
            'label': best_result['label'],
            'score': float(best_result['score'])
        }

    def _process_classification_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process general classification result"""
        return {
            'label': result['label'],
            'score': float(result['score'])
        }

    def _get_cache_key(self, text: str, model_version: Optional[str]) -> str:
        """Generate cache key for text and model version"""
        version = model_version or self.current_version
        content = f"{text}:{version}"
        return hashlib.md5(content.encode()).hexdigest()

    def list_available_versions(self) -> List[str]:
        """List available model versions"""
        return list(self.model_registry.keys())

    def register_model_version(self, version: str, model_info: Dict[str, Any]) -> None:
        """Register a new model version"""
        self.model_registry[version] = {
            **model_info,
            'created_at': datetime.utcnow(),
            'status': 'active'
        }
        logger.info(f"Registered model version {version}")

    def set_model_status(self, version: str, status: str) -> None:
        """Set status for a model version"""
        if version in self.model_registry:
            self.model_registry[version]['status'] = status
            logger.info(f"Set model version {version} status to {status}")
        else:
            raise ValueError(f"Model version {version} not found")

    def get_model_metrics(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a model version"""
        target_version = version or self.current_version
        if target_version in self.model_registry:
            return self.model_registry[target_version].get('metrics', {})
        else:
            raise ValueError(f"Model version {target_version} not found")

    async def warm_up(self, sample_texts: List[str]) -> None:
        """Warm up the model with sample predictions"""
        if not self.is_model_loaded():
            await self.load_model()

        logger.info("Warming up model...")
        start_time = time.time()

        try:
            await self.predict_batch(sample_texts)
            warmup_time = time.time() - start_time
            logger.info(f"Model warmed up in {warmup_time:.2f} seconds")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")

    def clear_cache(self) -> None:
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.prediction_cache),
            'max_size': self.prediction_cache.maxsize,
            'ttl': self.prediction_cache.ttl,
            'hit_ratio': getattr(self.prediction_cache, 'hit_ratio', 0.0)
        }


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Utility functions for A/B testing
class ABTestManager:
    """Manages A/B testing for model versions"""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.load_ab_config()

    def load_ab_config(self):
        """Load A/B testing configuration"""
        try:
            with open('config/ab_test_config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {
                'enabled': False,
                'traffic_split': {'v1.0.0': 100},
                'control_version': 'v1.0.0'
            }

    def get_model_version_for_request(self, user_id: Optional[str] = None) -> str:
        """Determine which model version to use for a request"""
        if not self.config.get('enabled', False):
            return self.config.get('control_version', 'v1.0.0')

        # Simple hash-based routing for consistent user experience
        if user_id:
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            percentage = hash_value % 100
        else:
            # Random assignment for anonymous users
            percentage = np.random.randint(0, 100)

        cumulative = 0
        for version, split in self.config.get('traffic_split', {}).items():
            cumulative += split
            if percentage < cumulative:
                return version

        return self.config.get('control_version', 'v1.0.0')

    def record_experiment_result(self, user_id: str, version: str, outcome: Dict[str, Any]):
        """Record experiment result for analysis"""
        # In production, this would write to a metrics store
        logger.info(f"A/B Test Result: user={user_id}, version={version}, outcome={outcome}")


# Feature store simulation
class FeatureStore:
    """Simulated feature store for ML features"""

    def __init__(self):
        self.features: Dict[str, Any] = {}
        self._initialize_mock_features()

    def _initialize_mock_features(self):
        """Initialize with mock features"""
        self.features = {
            'user_sentiment_history': {
                'user_123': [0.8, 0.6, 0.9, 0.7],
                'user_456': [0.2, 0.3, 0.1, 0.4]
            },
            'text_length_stats': {
                'avg_length': 150,
                'max_length': 512
            }
        }

    def get_features(self, feature_names: List[str], entity_ids: List[str]) -> Dict[str, List[Any]]:
        """Retrieve features for given entities"""
        result = {}
        for feature_name in feature_names:
            if feature_name in self.features:
                feature_data = self.features[feature_name]
                if isinstance(feature_data, dict):
                    result[feature_name] = [feature_data.get(entity_id, None) for entity_id in entity_ids]
                else:
                    result[feature_name] = [feature_data] * len(entity_ids)
            else:
                result[feature_name] = [None] * len(entity_ids)

        return result

    def store_features(self, feature_name: str, entity_id: str, value: Any):
        """Store feature value"""
        if feature_name not in self.features:
            self.features[feature_name] = {}

        if isinstance(self.features[feature_name], dict):
            self.features[feature_name][entity_id] = value
        else:
            self.features[feature_name] = {entity_id: value}


# Global instances
_ab_test_manager = None
_feature_store = None


def get_ab_test_manager() -> ABTestManager:
    """Get or create global A/B test manager"""
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager


def get_feature_store() -> FeatureStore:
    """Get or create global feature store"""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store