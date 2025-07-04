"""
Custom Metrics Exporter for MLOps Pipeline
Path: src/monitoring/custom_metrics_exporter.py
"""

import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Info, push_to_gateway
import psutil
import torch
import GPUtil
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class MLOpsMetricsExporter:
    """Custom metrics exporter for MLOps pipeline"""

    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        self.setup_metrics()

    def setup_metrics(self):
        """Setup custom metrics collectors"""

        # Model-specific metrics
        self.model_load_time = Histogram(
            'mlops_model_load_time_seconds',
            'Time taken to load model',
            ['model_name', 'model_version'],
            registry=self.registry
        )

        self.model_memory_usage = Gauge(
            'mlops_model_memory_usage_bytes',
            'Memory usage of loaded model',
            ['model_name', 'model_version'],
            registry=self.registry
        )

        self.model_parameter_count = Gauge(
            'mlops_model_parameter_count',
            'Number of parameters in the model',
            ['model_name', 'model_version'],
            registry=self.registry
        )

        # Prediction quality metrics
        self.prediction_confidence_distribution = Histogram(
            'mlops_prediction_confidence_distribution',
            'Distribution of prediction confidence scores',
            ['model_version'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

        self.prediction_accuracy_realtime = Gauge(
            'mlops_prediction_accuracy_realtime',
            'Real-time prediction accuracy',
            ['model_version', 'time_window'],
            registry=self.registry
        )

        # Data quality metrics
        self.input_text_length_distribution = Histogram(
            'mlops_input_text_length_distribution',
            'Distribution of input text lengths',
            ['endpoint'],
            buckets=[0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            registry=self.registry
        )

        self.data_quality_score = Gauge(
            'mlops_data_quality_score',
            'Data quality score',
            ['quality_metric'],
            registry=self.registry
        )

        # Model drift metrics
        self.model_drift_score = Gauge(
            'mlops_model_drift_score',
            'Model drift detection score',
            ['model_version', 'metric_type'],
            registry=self.registry
        )

        # A/B testing metrics
        self.ab_test_conversion_rate = Gauge(
            'mlops_ab_test_conversion_rate',
            'A/B test conversion rate',
            ['experiment_id', 'variant'],
            registry=self.registry
        )

        self.ab_test_sample_size = Gauge(
            'mlops_ab_test_sample_size',
            'A/B test sample size',
            ['experiment_id', 'variant'],
            registry=self.registry
        )

        # Business metrics
        self.business_conversion_rate = Gauge(
            'mlops_business_conversion_rate',
            'Business conversion rate',
            ['model_version', 'time_window'],
            registry=self.registry
        )

        self.user_satisfaction_score = Gauge(
            'mlops_user_satisfaction_score',
            'User satisfaction score',
            ['model_version'],
            registry=self.registry
        )

        # Infrastructure metrics
        self.gpu_utilization = Gauge(
            'mlops_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )

        self.gpu_memory_usage = Gauge(
            'mlops_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )

        # Model serving metrics
        self.model_serving_queue_size = Gauge(
            'mlops_model_serving_queue_size',
            'Model serving queue size',
            ['model_version'],
            registry=self.registry
        )

        self.model_serving_throughput = Gauge(
            'mlops_model_serving_throughput',
            'Model serving throughput (requests/second)',
            ['model_version'],
            registry=self.registry
        )

        # Feature store metrics
        self.feature_freshness = Gauge(
            'mlops_feature_freshness_seconds',
            'Feature freshness in seconds',
            ['feature_name'],
            registry=self.registry
        )

        self.feature_availability = Gauge(
            'mlops_feature_availability',
            'Feature availability (0 or 1)',
            ['feature_name'],
            registry=self.registry
        )

        # Model version info
        self.model_info = Info(
            'mlops_model_info',
            'Information about the current model',
            registry=self.registry
        )

    def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # GPU metrics
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    self.gpu_utilization.labels(
                        gpu_id=gpu.id,
                        gpu_name=gpu.name
                    ).set(gpu.load * 100)

                    self.gpu_memory_usage.labels(
                        gpu_id=gpu.id,
                        gpu_name=gpu.name
                    ).set(gpu.memoryUsed * 1024 * 1024)  # Convert to bytes

        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")

    def collect_model_metrics(self, model_path: str, model_version: str):
        """Collect model-specific metrics"""
        try:
            start_time = time.time()

            # Load model to get metrics
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            load_time = time.time() - start_time

            # Record load time
            self.model_load_time.labels(
                model_name=model_path,
                model_version=model_version
            ).observe(load_time)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            self.model_parameter_count.labels(
                model_name=model_path,
                model_version=model_version
            ).set(total_params)

            # Estimate memory usage
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            total_memory = param_size + buffer_size

            self.model_memory_usage.labels(
                model_name=model_path,
                model_version=model_version
            ).set(total_memory)

            # Update model info
            self.model_info.info({
                'model_name': model_path,
                'model_version': model_version,
                'total_parameters': str(total_params),
                'memory_usage_mb': str(total_memory / 1024 / 1024),
                'last_updated': datetime.now().isoformat()
            })

            logger.info(f"Model metrics collected for {model_path} v{model_version}")

        except Exception as e:
            logger.error(f"Failed to collect model metrics: {e}")

    def record_prediction_metrics(self, predictions: List[Dict[str, Any]],
                                  model_version: str, endpoint: str):
        """Record prediction-specific metrics"""
        try:
            for prediction in predictions:
                # Record confidence distribution
                confidence = prediction.get('confidence', 0.0)
                self.prediction_confidence_distribution.labels(
                    model_version=model_version
                ).observe(confidence)

                # Record input text length
                text_length = len(prediction.get('text', ''))
                self.input_text_length_distribution.labels(
                    endpoint=endpoint
                ).observe(text_length)

        except Exception as e:
            logger.error(f"Failed to record prediction metrics: {e}")

    def update_data_quality_metrics(self, quality_scores: Dict[str, float]):
        """Update data quality metrics"""
        try:
            for metric_name, score in quality_scores.items():
                self.data_quality_score.labels(
                    quality_metric=metric_name
                ).set(score)

        except Exception as e:
            logger.error(f"Failed to update data quality metrics: {e}")

    def update_model_drift_metrics(self, drift_scores: Dict[str, float],
                                   model_version: str):
        """Update model drift metrics"""
        try:
            for metric_type, score in drift_scores.items():
                self.model_drift_score.labels(
                    model_version=model_version,
                    metric_type=metric_type
                ).set(score)

        except Exception as e:
            logger.error(f"Failed to update model drift metrics: {e}")

    def update_ab_test_metrics(self, experiment_results: Dict[str, Any]):
        """Update A/B testing metrics"""
        try:
            experiment_id = experiment_results.get('experiment_id')

            for variant, results in experiment_results.get('results', {}).items():
                # Update conversion rate
                conversion_rate = results.get('conversion_rate', 0.0)
                self.ab_test_conversion_rate.labels(
                    experiment_id=experiment_id,
                    variant=variant
                ).set(conversion_rate)

                # Update sample size
                sample_size = results.get('sample_size', 0)
                self.ab_test_sample_size.labels(
                    experiment_id=experiment_id,
                    variant=variant
                ).set(sample_size)

        except Exception as e:
            logger.error(f"Failed to update A/B test metrics: {e}")

    def update_business_metrics(self, business_data: Dict[str, Any]):
        """Update business metrics"""
        try:
            # Update conversion rate
            conversion_rate = business_data.get('conversion_rate', 0.0)
            model_version = business_data.get('model_version', 'unknown')
            time_window = business_data.get('time_window', '1h')

            self.business_conversion_rate.labels(
                model_version=model_version,
                time_window=time_window
            ).set(conversion_rate)

            # Update user satisfaction
            satisfaction_score = business_data.get('satisfaction_score', 0.0)
            self.user_satisfaction_score.labels(
                model_version=model_version
            ).set(satisfaction_score)

        except Exception as e:
            logger.error(f"Failed to update business metrics: {e}")

    def update_feature_store_metrics(self, feature_data: Dict[str, Any]):
        """Update feature store metrics"""
        try:
            for feature_name, feature_info in feature_data.items():
                # Update feature freshness
                freshness = feature_info.get('freshness_seconds', 0)
                self.feature_freshness.labels(
                    feature_name=feature_name
                ).set(freshness)

                # Update feature availability
                availability = 1 if feature_info.get('available', False) else 0
                self.feature_availability.labels(
                    feature_name=feature_name
                ).set(availability)

        except Exception as e:
            logger.error(f"Failed to update feature store metrics: {e}")

    def push_metrics_to_gateway(self, gateway_url: str, job_name: str):
        """Push metrics to Prometheus Pushgateway"""
        try:
            push_to_gateway(
                gateway_url,
                job=job_name,
                registry=self.registry
            )
            logger.info(f"Metrics pushed to gateway: {gateway_url}")

        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")

    async def start_metrics_collection(self, interval: int = 30):
        """Start periodic metrics collection"""
        logger.info(f"Starting metrics collection with {interval}s interval")

        while True:
            try:
                # Collect system metrics
                self.collect_system_metrics()

                # Sleep for the specified interval
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)  # Short sleep on error


# Global metrics exporter instance
metrics_exporter = MLOpsMetricsExporter()


def get_metrics_exporter() -> MLOpsMetricsExporter:
    """Get global metrics exporter instance"""
    return metrics_exporter


# Example usage functions
def record_model_inference_metrics(predictions: List[Dict[str, Any]],
                                   model_version: str, endpoint: str):
    """Record metrics for model inference"""
    exporter = get_metrics_exporter()
    exporter.record_prediction_metrics(predictions, model_version, endpoint)


def update_model_performance_metrics(accuracy: float, f1_score: float,
                                     model_version: str):
    """Update model performance metrics"""
    exporter = get_metrics_exporter()
    exporter.prediction_accuracy_realtime.labels(
        model_version=model_version,
        time_window='1h'
    ).set(accuracy)


def record_ab_test_result(experiment_id: str, variant: str,
                          outcome: str, user_data: Dict[str, Any]):
    """Record A/B test result"""
    # This would typically be called from your API endpoints
    # when recording experiment events
    pass


if __name__ == "__main__":
    # Example of running the metrics exporter
    exporter = MLOpsMetricsExporter()

    # Collect some sample metrics
    exporter.collect_system_metrics()

    # Start async metrics collection
    asyncio.run(exporter.start_metrics_collection(interval=30))