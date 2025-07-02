"""
Custom monitoring and metrics for MLOps pipeline
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import wraps

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
    REGISTRY
)
import psutil

logger = logging.getLogger(__name__)

# Use a custom registry to avoid conflicts
CUSTOM_REGISTRY = CollectorRegistry()

# Define metrics with the custom registry to avoid duplicates
def create_or_get_metric(metric_class, name, description, labels=None, **kwargs):
    """Create a metric or get existing one to avoid duplicates"""
    try:
        if labels:
            return metric_class(name, description, labels, registry=CUSTOM_REGISTRY, **kwargs)
        else:
            return metric_class(name, description, registry=CUSTOM_REGISTRY, **kwargs)
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Metric already exists, try to find it
            for collector in CUSTOM_REGISTRY._collector_to_names:
                if hasattr(collector, '_name') and collector._name == name:
                    return collector
            # If not found, create with a unique name
            unique_name = f"{name}_{int(time.time())}"
            logger.warning(f"Metric {name} already exists, creating {unique_name}")
            if labels:
                return metric_class(unique_name, description, labels, registry=CUSTOM_REGISTRY, **kwargs)
            else:
                return metric_class(unique_name, description, registry=CUSTOM_REGISTRY, **kwargs)
        else:
            raise e

# Business Metrics
PREDICTION_REQUESTS = create_or_get_metric(
    Counter,
    'mlops_prediction_requests_total',
    'Total number of prediction requests',
    ['model_version', 'endpoint', 'status']
)

PREDICTION_LATENCY = create_or_get_metric(
    Histogram,
    'mlops_prediction_latency_seconds',
    'Time spent on predictions',
    ['model_version', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

MODEL_ACCURACY = create_or_get_metric(
    Gauge,
    'mlops_model_accuracy',
    'Current model accuracy',
    ['model_version']
)

MODEL_CONFIDENCE = create_or_get_metric(
    Summary,
    'mlops_model_confidence',
    'Model prediction confidence scores',
    ['model_version']
)

BATCH_SIZE = create_or_get_metric(
    Histogram,
    'mlops_batch_size',
    'Batch sizes for batch predictions',
    ['model_version'],
    buckets=[1, 5, 10, 25, 50, 100]
)

# System Metrics
CPU_USAGE = create_or_get_metric(
    Gauge,
    'mlops_cpu_usage_percent',
    'CPU usage percentage'
)

MEMORY_USAGE = create_or_get_metric(
    Gauge,
    'mlops_memory_usage_bytes',
    'Memory usage in bytes'
)

DISK_USAGE = create_or_get_metric(
    Gauge,
    'mlops_disk_usage_bytes',
    'Disk usage in bytes',
    ['path']
)

MODEL_LOAD_TIME = create_or_get_metric(
    Histogram,
    'mlops_model_load_time_seconds',
    'Time taken to load model',
    ['model_version']
)

CACHE_HITS = create_or_get_metric(
    Counter,
    'mlops_cache_hits_total',
    'Number of cache hits',
    ['cache_type']
)

CACHE_MISSES = create_or_get_metric(
    Counter,
    'mlops_cache_misses_total',
    'Number of cache misses',
    ['cache_type']
)

# Model Performance Metrics
MODEL_DRIFT = create_or_get_metric(
    Gauge,
    'mlops_model_drift_score',
    'Model drift detection score',
    ['model_version', 'metric_type']
)

DATA_QUALITY = create_or_get_metric(
    Gauge,
    'mlops_data_quality_score',
    'Data quality score',
    ['quality_metric']
)

ERROR_RATE = create_or_get_metric(
    Gauge,
    'mlops_error_rate',
    'Error rate by endpoint',
    ['endpoint', 'error_type']
)

# A/B Testing Metrics
AB_TEST_REQUESTS = create_or_get_metric(
    Counter,
    'mlops_ab_test_requests_total',
    'A/B test requests by variant',
    ['experiment_id', 'variant', 'outcome']
)

# Model Info
MODEL_INFO = create_or_get_metric(
    Info,
    'mlops_model_info',
    'Information about the current model'
)

class MetricsCollector:
    """Collects and manages custom metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.last_update = time.time()

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)

            # Disk usage
            disk = psutil.disk_usage('/')
            DISK_USAGE.labels(path='/').set(disk.used)

            self.last_update = time.time()

        except Exception as e:
            logger.error(f"Failed to update system metrics: {str(e)}")

    def record_prediction(self,
                         model_version: str,
                         endpoint: str,
                         latency: float,
                         confidence: float,
                         status: str = 'success'):
        """Record prediction metrics"""
        PREDICTION_REQUESTS.labels(
            model_version=model_version,
            endpoint=endpoint,
            status=status
        ).inc()

        PREDICTION_LATENCY.labels(
            model_version=model_version,
            endpoint=endpoint
        ).observe(latency)

        MODEL_CONFIDENCE.labels(
            model_version=model_version
        ).observe(confidence)

    def record_batch_prediction(self,
                               model_version: str,
                               batch_size: int,
                               latency: float,
                               avg_confidence: float):
        """Record batch prediction metrics"""
        BATCH_SIZE.labels(model_version=model_version).observe(batch_size)
        self.record_prediction(model_version, 'batch', latency, avg_confidence)

    def record_model_load(self, model_version: str, load_time: float):
        """Record model loading metrics"""
        MODEL_LOAD_TIME.labels(model_version=model_version).observe(load_time)

    def record_cache_event(self, cache_type: str, hit: bool):
        """Record cache hit/miss events"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()

    def update_model_accuracy(self, model_version: str, accuracy: float):
        """Update model accuracy metric"""
        MODEL_ACCURACY.labels(model_version=model_version).set(accuracy)

    def record_model_drift(self, model_version: str, metric_type: str, score: float):
        """Record model drift metrics"""
        MODEL_DRIFT.labels(
            model_version=model_version,
            metric_type=metric_type
        ).set(score)

    def update_data_quality(self, quality_metric: str, score: float):
        """Update data quality metrics"""
        DATA_QUALITY.labels(quality_metric=quality_metric).set(score)

    def update_error_rate(self, endpoint: str, error_type: str, rate: float):
        """Update error rate metrics"""
        ERROR_RATE.labels(endpoint=endpoint, error_type=error_type).set(rate)

    def record_ab_test(self, experiment_id: str, variant: str, outcome: str):
        """Record A/B test metrics"""
        AB_TEST_REQUESTS.labels(
            experiment_id=experiment_id,
            variant=variant,
            outcome=outcome
        ).inc()

    def update_model_info(self, info: Dict[str, str]):
        """Update model information"""
        MODEL_INFO.info(info)

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        self.update_system_metrics()
        return generate_latest(CUSTOM_REGISTRY)

def track_latency(metric_name: str = None):
    """Decorator to track function execution latency"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                latency = time.time() - start_time
                if metric_name:
                    PREDICTION_LATENCY.labels(
                        model_version=kwargs.get('model_version', 'unknown'),
                        endpoint=metric_name
                    ).observe(latency)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                latency = time.time() - start_time
                if metric_name:
                    PREDICTION_LATENCY.labels(
                        model_version=kwargs.get('model_version', 'unknown'),
                        endpoint=metric_name
                    ).observe(latency)

        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator

def setup_custom_metrics():
    """Initialize custom metrics"""
    logger.info("Setting up custom metrics...")

    # Initialize model info
    global metrics_collector
    metrics_collector = MetricsCollector()

    metrics_collector.update_model_info({
        'name': 'distillbert-base-uncased',
        'task': 'sentiment-analysis',
        'framework': 'transformers',
        'version': '1.0.0'
    })

    # Start system metrics collection
    metrics_collector.update_system_metrics()

    logger.info("Custom metrics setup complete")

class PerformanceMonitor:
    """Monitor model and system performance"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions_window = []
        self.latencies_window = []
        self.confidences_window = []

    def add_prediction(self, latency: float, confidence: float, correct: bool = None):
        """Add a prediction to the monitoring window"""
        self.latencies_window.append(latency)
        self.confidences_window.append(confidence)

        if correct is not None:
            self.predictions_window.append(correct)

        # Keep only the last N predictions
        if len(self.latencies_window) > self.window_size:
            self.latencies_window.pop(0)
        if len(self.confidences_window) > self.window_size:
            self.confidences_window.pop(0)
        if len(self.predictions_window) > self.window_size:
            self.predictions_window.pop(0)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.latencies_window:
            return {}

        stats = {
            'avg_latency': sum(self.latencies_window) / len(self.latencies_window),
            'max_latency': max(self.latencies_window),
            'min_latency': min(self.latencies_window),
            'avg_confidence': sum(self.confidences_window) / len(self.confidences_window),
            'min_confidence': min(self.confidences_window),
            'max_confidence': max(self.confidences_window)
        }

        if self.predictions_window:
            stats['accuracy'] = sum(self.predictions_window) / len(self.predictions_window)

        return stats

    def check_performance_degradation(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """Check for performance degradation"""
        stats = self.get_performance_stats()
        alerts = {}

        if 'max_latency' in thresholds and stats.get('max_latency', 0) > thresholds['max_latency']:
            alerts['high_latency'] = True

        if 'min_accuracy' in thresholds and stats.get('accuracy', 1.0) < thresholds['min_accuracy']:
            alerts['low_accuracy'] = True

        if 'min_confidence' in thresholds and stats.get('avg_confidence', 1.0) < thresholds['min_confidence']:
            alerts['low_confidence'] = True

        return alerts

class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self):
        self.alert_history = []
        self.alert_thresholds = {
            'error_rate': 0.05,
            'latency_p99': 2.0,
            'accuracy': 0.85,
            'confidence': 0.7
        }

    def check_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []

        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics:
                if metric == 'error_rate' and metrics[metric] > threshold:
                    alerts.append({
                        'type': 'high_error_rate',
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'timestamp': datetime.utcnow(),
                        'severity': 'critical'
                    })
                elif metric == 'latency_p99' and metrics[metric] > threshold:
                    alerts.append({
                        'type': 'high_latency',
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'timestamp': datetime.utcnow(),
                        'severity': 'warning'
                    })
                elif metric in ['accuracy', 'confidence'] and metrics[metric] < threshold:
                    alerts.append({
                        'type': f'low_{metric}',
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'timestamp': datetime.utcnow(),
                        'severity': 'warning'
                    })

        # Store alerts
        self.alert_history.extend(alerts)

        return alerts

    def send_alert(self, alert: Dict[str, Any]):
        """Send alert notification (placeholder for actual implementation)"""
        logger.warning(f"ALERT: {alert['type']} - {alert['metric']} = {alert['value']} (threshold: {alert['threshold']})")
        # In production, this would integrate with Slack, PagerDuty, etc.

# Global instances
metrics_collector = None
performance_monitor = PerformanceMonitor()
alert_manager = AlertManager()

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    return metrics_collector

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return performance_monitor

def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    return alert_manager

class MetricsCollector:
    """Collects and manages custom metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.last_update = time.time()

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)

            # Disk usage
            disk = psutil.disk_usage('/')
            DISK_USAGE.labels(path='/').set(disk.used)

            self.last_update = time.time()

        except Exception as e:
            logger.error(f"Failed to update system metrics: {str(e)}")

    def record_prediction(self,
                         model_version: str,
                         endpoint: str,
                         latency: float,
                         confidence: float,
                         status: str = 'success'):
        """Record prediction metrics"""
        PREDICTION_REQUESTS.labels(
            model_version=model_version,
            endpoint=endpoint,
            status=status
        ).inc()

        PREDICTION_LATENCY.labels(
            model_version=model_version,
            endpoint=endpoint
        ).observe(latency)

        MODEL_CONFIDENCE.labels(
            model_version=model_version
        ).observe(confidence)

    def record_batch_prediction(self,
                               model_version: str,
                               batch_size: int,
                               latency: float,
                               avg_confidence: float):
        """Record batch prediction metrics"""
        BATCH_SIZE.labels(model_version=model_version).observe(batch_size)
        self.record_prediction(model_version, 'batch', latency, avg_confidence)

    def record_model_load(self, model_version: str, load_time: float):
        """Record model loading metrics"""
        MODEL_LOAD_TIME.labels(model_version=model_version).observe(load_time)

    def record_cache_event(self, cache_type: str, hit: bool):
        """Record cache hit/miss events"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()

    def update_model_accuracy(self, model_version: str, accuracy: float):
        """Update model accuracy metric"""
        MODEL_ACCURACY.labels(model_version=model_version).set(accuracy)

    def record_model_drift(self, model_version: str, metric_type: str, score: float):
        """Record model drift metrics"""
        MODEL_DRIFT.labels(
            model_version=model_version,
            metric_type=metric_type
        ).set(score)

    def update_data_quality(self, quality_metric: str, score: float):
        """Update data quality metrics"""
        DATA_QUALITY.labels(quality_metric=quality_metric).set(score)

    def update_error_rate(self, endpoint: str, error_type: str, rate: float):
        """Update error rate metrics"""
        ERROR_RATE.labels(endpoint=endpoint, error_type=error_type).set(rate)

    def record_ab_test(self, experiment_id: str, variant: str, outcome: str):
        """Record A/B test metrics"""
        AB_TEST_REQUESTS.labels(
            experiment_id=experiment_id,
            variant=variant,
            outcome=outcome
        ).inc()

    def update_model_info(self, info: Dict[str, str]):
        """Update model information"""
        MODEL_INFO.info(info)

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        self.update_system_metrics()
        return generate_latest(REGISTRY)

# Global metrics collector instance
metrics_collector = MetricsCollector()

def track_latency(metric_name: str = None):
    """Decorator to track function execution latency"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                latency = time.time() - start_time
                if metric_name:
                    PREDICTION_LATENCY.labels(
                        model_version=kwargs.get('model_version', 'unknown'),
                        endpoint=metric_name
                    ).observe(latency)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                latency = time.time() - start_time
                if metric_name:
                    PREDICTION_LATENCY.labels(
                        model_version=kwargs.get('model_version', 'unknown'),
                        endpoint=metric_name
                    ).observe(latency)

        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator

def setup_custom_metrics():
    """Initialize custom metrics"""
    logger.info("Setting up custom metrics...")

    # Initialize model info
    metrics_collector.update_model_info({
        'name': 'distillbert-base-uncased',
        'task': 'sentiment-analysis',
        'framework': 'transformers',
        'version': '1.0.0'
    })

    # Start system metrics collection
    metrics_collector.update_system_metrics()

    logger.info("Custom metrics setup complete")

class PerformanceMonitor:
    """Monitor model and system performance"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions_window = []
        self.latencies_window = []
        self.confidences_window = []

    def add_prediction(self, latency: float, confidence: float, correct: bool = None):
        """Add a prediction to the monitoring window"""
        self.latencies_window.append(latency)
        self.confidences_window.append(confidence)

        if correct is not None:
            self.predictions_window.append(correct)

        # Keep only the last N predictions
        if len(self.latencies_window) > self.window_size:
            self.latencies_window.pop(0)
        if len(self.confidences_window) > self.window_size:
            self.confidences_window.pop(0)
        if len(self.predictions_window) > self.window_size:
            self.predictions_window.pop(0)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.latencies_window:
            return {}

        stats = {
            'avg_latency': sum(self.latencies_window) / len(self.latencies_window),
            'max_latency': max(self.latencies_window),
            'min_latency': min(self.latencies_window),
            'avg_confidence': sum(self.confidences_window) / len(self.confidences_window),
            'min_confidence': min(self.confidences_window),
            'max_confidence': max(self.confidences_window)
        }

        if self.predictions_window:
            stats['accuracy'] = sum(self.predictions_window) / len(self.predictions_window)

        return stats

    def check_performance_degradation(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """Check for performance degradation"""
        stats = self.get_performance_stats()
        alerts = {}

        if 'max_latency' in thresholds and stats.get('max_latency', 0) > thresholds['max_latency']:
            alerts['high_latency'] = True

        if 'min_accuracy' in thresholds and stats.get('accuracy', 1.0) < thresholds['min_accuracy']:
            alerts['low_accuracy'] = True

        if 'min_confidence' in thresholds and stats.get('avg_confidence', 1.0) < thresholds['min_confidence']:
            alerts['low_confidence'] = True

        return alerts

class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self):
        self.alert_history = []
        self.alert_thresholds = {
            'error_rate': 0.05,
            'latency_p99': 2.0,
            'accuracy': 0.85,
            'confidence': 0.7
        }

    def check_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []

        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics:
                if metric == 'error_rate' and metrics[metric] > threshold:
                    alerts.append({
                        'type': 'high_error_rate',
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'timestamp': datetime.utcnow(),
                        'severity': 'critical'
                    })
                elif metric == 'latency_p99' and metrics[metric] > threshold:
                    alerts.append({
                        'type': 'high_latency',
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'timestamp': datetime.utcnow(),
                        'severity': 'warning'
                    })
                elif metric in ['accuracy', 'confidence'] and metrics[metric] < threshold:
                    alerts.append({
                        'type': f'low_{metric}',
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'timestamp': datetime.utcnow(),
                        'severity': 'warning'
                    })

        # Store alerts
        self.alert_history.extend(alerts)

        return alerts

    def send_alert(self, alert: Dict[str, Any]):
        """Send alert notification (placeholder for actual implementation)"""
        logger.warning(f"ALERT: {alert['type']} - {alert['metric']} = {alert['value']} (threshold: {alert['threshold']})")
        # In production, this would integrate with Slack, PagerDuty, etc.

# Global instances
performance_monitor = PerformanceMonitor()
alert_manager = AlertManager()

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return metrics_collector

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return performance_monitor

def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    return alert_manager