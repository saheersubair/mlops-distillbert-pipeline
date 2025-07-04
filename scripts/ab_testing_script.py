"""
A/B Testing Framework for MLOps DistillBERT Model Serving
This script provides comprehensive A/B testing capabilities for model comparison
"""

import os
import json
import hashlib
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ab_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficAllocation(Enum):
    """Traffic allocation strategies"""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    WEIGHTED = "weighted"
    SEGMENT_BASED = "segment_based"


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiment"""
    experiment_id: str
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    control_variant: str
    treatment_variants: List[str]
    traffic_allocation: Dict[str, float]
    allocation_strategy: TrafficAllocation
    success_metrics: List[str]
    minimum_sample_size: int
    confidence_level: float
    power: float
    expected_effect_size: float
    status: ExperimentStatus
    metadata: Dict[str, Any] = None


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment"""
    experiment_id: str
    variant: str
    sample_size: int
    conversion_rate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    statistical_significance: bool
    effect_size: float
    metrics: Dict[str, Any]


class ABTestingFramework:
    """Comprehensive A/B testing framework for ML models"""

    def __init__(self, config_path: str = "config/ab_test_config.yaml",
                 database_path: str = "data/ab_testing.db"):
        self.config_path = config_path
        self.database_path = database_path
        self.config = self.load_config()
        self.init_database()

    def load_config(self) -> Dict[str, Any]:
        """Load A/B testing configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'framework': {
                'name': 'MLOps A/B Testing Framework',
                'version': '1.0.0',
                'author': 'MLOps Team'
            },
            'database': {
                'type': 'sqlite',
                'path': 'data/ab_testing.db'
            },
            'default_experiment': {
                'confidence_level': 0.95,
                'power': 0.8,
                'minimum_sample_size': 100,
                'max_duration_days': 30
            },
            'traffic_allocation': {
                'strategy': 'hash_based',
                'hash_field': 'user_id'
            },
            'metrics': {
                'primary': ['accuracy', 'f1_score'],
                'secondary': ['precision', 'recall', 'response_time']
            }
        }

    def init_database(self):
        """Initialize SQLite database for experiment tracking"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)

        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()

            # Create experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    config TEXT,
                    status TEXT,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create experiment events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    user_id TEXT,
                    variant TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            ''')

            # Create experiment results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    variant TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    sample_size INTEGER,
                    confidence_interval_lower REAL,
                    confidence_interval_upper REAL,
                    p_value REAL,
                    effect_size REAL,
                    statistical_significance BOOLEAN,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            ''')

            conn.commit()

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.database_path)
        try:
            yield conn
        finally:
            conn.close()

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B testing experiment"""
        logger.info(f"Creating experiment: {config.name}")

        # Validate configuration
        self.validate_experiment_config(config)

        # Store experiment in database
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO experiments 
                (experiment_id, name, description, config, status, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.experiment_id,
                config.name,
                config.description,
                json.dumps(config.__dict__, default=str),
                config.status.value,
                config.start_date,
                config.end_date
            ))
            conn.commit()

        logger.info(f"Experiment created: {config.experiment_id}")
        return config.experiment_id

    def validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration"""
        # Check traffic allocation sums to 100%
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 100%, got {total_allocation}")

        # Check variants exist in allocation
        all_variants = [config.control_variant] + config.treatment_variants
        for variant in all_variants:
            if variant not in config.traffic_allocation:
                raise ValueError(f"Variant {variant} not found in traffic allocation")

        # Check date validity
        if config.start_date >= config.end_date:
            raise ValueError("Start date must be before end date")

        return True

    def get_variant_assignment(self, experiment_id: str, user_id: str) -> str:
        """Get variant assignment for a user"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        config = ExperimentConfig(**json.loads(experiment['config']))

        # Check if experiment is active
        if config.status != ExperimentStatus.ACTIVE:
            return config.control_variant

        # Check if experiment is within time bounds
        now = datetime.now()
        if now < config.start_date or now > config.end_date:
            return config.control_variant

        # Determine allocation strategy
        if config.allocation_strategy == TrafficAllocation.HASH_BASED:
            return self._hash_based_allocation(user_id, config)
        elif config.allocation_strategy == TrafficAllocation.RANDOM:
            return self._random_allocation(config)
        elif config.allocation_strategy == TrafficAllocation.WEIGHTED:
            return self._weighted_allocation(config)
        else:
            return config.control_variant

    def _hash_based_allocation(self, user_id: str, config: ExperimentConfig) -> str:
        """Hash-based traffic allocation for consistent user experience"""
        # Create hash from user_id and experiment_id for consistency
        hash_input = f"{user_id}:{config.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100.0  # 0-99.99%

        # Determine variant based on cumulative allocation
        cumulative = 0
        for variant, allocation in config.traffic_allocation.items():
            cumulative += allocation
            if percentage < cumulative:
                return variant

        return config.control_variant

    def _random_allocation(self, config: ExperimentConfig) -> str:
        """Random traffic allocation"""
        percentage = random.uniform(0, 100)

        cumulative = 0
        for variant, allocation in config.traffic_allocation.items():
            cumulative += allocation
            if percentage < cumulative:
                return variant

        return config.control_variant

    def _weighted_allocation(self, config: ExperimentConfig) -> str:
        """Weighted random allocation"""
        variants = list(config.traffic_allocation.keys())
        weights = list(config.traffic_allocation.values())

        return random.choices(variants, weights=weights)[0]

    def record_event(self, experiment_id: str, user_id: str, variant: str,
                     event_type: str, event_data: Dict[str, Any]) -> None:
        """Record an experiment event"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO experiment_events 
                (experiment_id, user_id, variant, event_type, event_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                user_id,
                variant,
                event_type,
                json.dumps(event_data)
            ))
            conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM experiments WHERE experiment_id = ?', (experiment_id,))
            row = cursor.fetchone()

            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by status"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute('SELECT * FROM experiments WHERE status = ?', (status.value,))
            else:
                cursor.execute('SELECT * FROM experiments')

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            return [dict(zip(columns, row)) for row in rows]

    def analyze_experiment(self, experiment_id: str) -> Dict[str, ExperimentResult]:
        """Analyze experiment results and perform statistical tests"""
        logger.info(f"Analyzing experiment: {experiment_id}")

        # Get experiment data
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        config = ExperimentConfig(**json.loads(experiment['config']))

        # Get event data
        event_data = self.get_experiment_events(experiment_id)

        if not event_data:
            logger.warning(f"No event data found for experiment: {experiment_id}")
            return {}

        # Group events by variant
        variant_data = {}
        for event in event_data:
            variant = event['variant']
            if variant not in variant_data:
                variant_data[variant] = []
            variant_data[variant].append(event)

        # Analyze each variant
        results = {}
        control_data = variant_data.get(config.control_variant, [])

        for variant, events in variant_data.items():
            # Calculate metrics
            metrics = self.calculate_variant_metrics(events)

            # Perform statistical tests vs control
            if variant != config.control_variant and control_data:
                control_metrics = self.calculate_variant_metrics(control_data)
                stat_results = self.perform_statistical_test(
                    control_metrics, metrics, config.confidence_level
                )
            else:
                stat_results = {
                    'p_value': 1.0,
                    'statistical_significance': False,
                    'effect_size': 0.0,
                    'confidence_interval': (0.0, 0.0)
                }

            # Create result object
            result = ExperimentResult(
                experiment_id=experiment_id,
                variant=variant,
                sample_size=len(events),
                conversion_rate=metrics.get('conversion_rate', 0.0),
                confidence_interval=stat_results['confidence_interval'],
                p_value=stat_results['p_value'],
                statistical_significance=stat_results['statistical_significance'],
                effect_size=stat_results['effect_size'],
                metrics=metrics
            )

            results[variant] = result

        # Store results in database
        self.store_experiment_results(experiment_id, results)

        return results

    def get_experiment_events(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all events for an experiment"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM experiment_events 
                WHERE experiment_id = ? 
                ORDER BY timestamp
            ''', (experiment_id,))

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            events = []
            for row in rows:
                event = dict(zip(columns, row))
                if event['event_data']:
                    event['event_data'] = json.loads(event['event_data'])
                events.append(event)

            return events

    def calculate_variant_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for a variant"""
        if not events:
            return {}

        # Extract predictions and outcomes
        predictions = []
        actual_labels = []
        response_times = []
        confidences = []

        for event in events:
            event_data = event.get('event_data', {})

            if event['event_type'] == 'prediction':
                predictions.append(event_data.get('prediction', 0))
                actual_labels.append(event_data.get('actual_label', 0))
                response_times.append(event_data.get('response_time', 0))
                confidences.append(event_data.get('confidence', 0))

        metrics = {}

        if predictions and actual_labels:
            # Classification metrics
            metrics['accuracy'] = accuracy_score(actual_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual_labels, predictions, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1

            # Conversion rate (assuming positive predictions are "conversions")
            metrics['conversion_rate'] = sum(predictions) / len(predictions)

        if response_times:
            metrics['avg_response_time'] = np.mean(response_times)
            metrics['p95_response_time'] = np.percentile(response_times, 95)

        if confidences:
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['min_confidence'] = np.min(confidences)

        metrics['sample_size'] = len(events)

        return metrics

    def perform_statistical_test(self, control_metrics: Dict[str, float],
                                 treatment_metrics: Dict[str, float],
                                 confidence_level: float) -> Dict[str, Any]:
        """Perform statistical significance test"""
        # Use accuracy as primary metric for comparison
        control_accuracy = control_metrics.get('accuracy', 0.0)
        treatment_accuracy = treatment_metrics.get('accuracy', 0.0)

        control_size = control_metrics.get('sample_size', 0)
        treatment_size = treatment_metrics.get('sample_size', 0)

        if control_size == 0 or treatment_size == 0:
            return {
                'p_value': 1.0,
                'statistical_significance': False,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0)
            }

        # Two-proportion z-test
        control_successes = control_accuracy * control_size
        treatment_successes = treatment_accuracy * treatment_size

        # Calculate pooled proportion
        pooled_p = (control_successes + treatment_successes) / (control_size + treatment_size)

        # Calculate standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / control_size + 1 / treatment_size))

        # Calculate z-score
        if se == 0:
            z_score = 0
        else:
            z_score = (treatment_accuracy - control_accuracy) / se

        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Check statistical significance
        alpha = 1 - confidence_level
        is_significant = p_value < alpha

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(pooled_p * (1 - pooled_p))
        effect_size = (treatment_accuracy - control_accuracy) / pooled_std if pooled_std > 0 else 0

        # Calculate confidence interval for difference
        diff = treatment_accuracy - control_accuracy
        margin_of_error = stats.norm.ppf(1 - alpha / 2) * se
        ci_lower = diff - margin_of_error
        ci_upper = diff + margin_of_error

        return {
            'p_value': p_value,
            'statistical_significance': is_significant,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'z_score': z_score
        }

    def store_experiment_results(self, experiment_id: str,
                                 results: Dict[str, ExperimentResult]) -> None:
        """Store experiment results in database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Clear existing results
            cursor.execute('DELETE FROM experiment_results WHERE experiment_id = ?',
                           (experiment_id,))

            # Insert new results
            for variant, result in results.items():
                for metric_name, metric_value in result.metrics.items():
                    cursor.execute('''
                        INSERT INTO experiment_results 
                        (experiment_id, variant, metric_name, metric_value, sample_size,
                         confidence_interval_lower, confidence_interval_upper, p_value,
                         effect_size, statistical_significance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        experiment_id,
                        variant,
                        metric_name,
                        metric_value,
                        result.sample_size,
                        result.confidence_interval[0],
                        result.confidence_interval[1],
                        result.p_value,
                        result.effect_size,
                        result.statistical_significance
                    ))

            conn.commit()

    def generate_experiment_report(self, experiment_id: str,
                                   output_dir: str = "reports") -> str:
        """Generate comprehensive experiment report"""
        logger.info(f"Generating report for experiment: {experiment_id}")

        # Get experiment data
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        config = ExperimentConfig(**json.loads(experiment['config']))
        results = self.analyze_experiment(experiment_id)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate report
        report_path = os.path.join(output_dir, f"experiment_report_{experiment_id}.html")

        with open(report_path, 'w') as f:
            f.write(self._generate_html_report(experiment, config, results))

        # Generate visualizations
        self._generate_experiment_visualizations(experiment_id, results, output_dir)

        logger.info(f"Report generated: {report_path}")
        return report_path

    def _generate_html_report(self, experiment: Dict[str, Any],
                              config: ExperimentConfig,
                              results: Dict[str, ExperimentResult]) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B Test Report: {config.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .variant {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
                .significant {{ background-color: #d4edda; }}
                .not-significant {{ background-color: #f8d7da; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>A/B Test Report: {config.name}</h1>
                <p><strong>Experiment ID:</strong> {config.experiment_id}</p>
                <p><strong>Description:</strong> {config.description}</p>
                <p><strong>Period:</strong> {config.start_date} - {config.end_date}</p>
                <p><strong>Status:</strong> {config.status.value}</p>
            </div>

            <div class="section">
                <h2>Experiment Configuration</h2>
                <ul>
                    <li><strong>Control Variant:</strong> {config.control_variant}</li>
                    <li><strong>Treatment Variants:</strong> {', '.join(config.treatment_variants)}</li>
                    <li><strong>Confidence Level:</strong> {config.confidence_level}</li>
                    <li><strong>Minimum Sample Size:</strong> {config.minimum_sample_size}</li>
                </ul>

                <h3>Traffic Allocation</h3>
                <table>
                    <tr><th>Variant</th><th>Allocation (%)</th></tr>
        """

        for variant, allocation in config.traffic_allocation.items():
            html += f"<tr><td>{variant}</td><td>{allocation:.1f}%</td></tr>"

        html += """
                </table>
            </div>

            <div class="section">
                <h2>Results Summary</h2>
        """

        for variant, result in results.items():
            significance_class = "significant" if result.statistical_significance else "not-significant"
            html += f"""
                <div class="variant {significance_class}">
                    <h3>{variant}</h3>
                    <p><strong>Sample Size:</strong> {result.sample_size}</p>
                    <p><strong>Conversion Rate:</strong> {result.conversion_rate:.4f}</p>
                    <p><strong>Statistical Significance:</strong> {result.statistical_significance}</p>
                    <p><strong>P-value:</strong> {result.p_value:.4f}</p>
                    <p><strong>Effect Size:</strong> {result.effect_size:.4f}</p>
                    <p><strong>95% CI:</strong> [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]</p>

                    <h4>Detailed Metrics</h4>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """

            for metric, value in result.metrics.items():
                html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"

            html += """
                    </table>
                </div>
            """

        html += """
            </div>

            <div class="section">
                <h2>Recommendations</h2>
        """

        # Add recommendations based on results
        html += self._generate_recommendations(config, results)

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def _generate_recommendations(self, config: ExperimentConfig,
                                  results: Dict[str, ExperimentResult]) -> str:
        """Generate recommendations based on results"""
        recommendations = []

        # Find best performing variant
        best_variant = None
        best_score = -1

        for variant, result in results.items():
            score = result.metrics.get('f1_score', 0)
            if score > best_score:
                best_score = score
                best_variant = variant

        if best_variant:
            if best_variant == config.control_variant:
                recommendations.append("The control variant performed best. Consider keeping the current model.")
            else:
                result = results[best_variant]
                if result.statistical_significance:
                    recommendations.append(
                        f"Variant '{best_variant}' shows statistically significant improvement. Consider deploying this variant.")
                else:
                    recommendations.append(
                        f"Variant '{best_variant}' performs best but lacks statistical significance. Consider extending the experiment.")

        # Check sample sizes
        min_samples = min(result.sample_size for result in results.values())
        if min_samples < config.minimum_sample_size:
            recommendations.append(
                f"Sample sizes are below minimum threshold ({config.minimum_sample_size}). Consider extending the experiment.")

        # Check for inconclusive results
        significant_variants = [v for v, r in results.items() if r.statistical_significance]
        if len(significant_variants) == 0:
            recommendations.append(
                "No variants show statistical significance. Consider larger sample sizes or different variants.")

        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"

        return html

    def _generate_experiment_visualizations(self, experiment_id: str,
                                            results: Dict[str, ExperimentResult],
                                            output_dir: str) -> None:
        """Generate visualization plots"""
        # Conversion rate comparison
        variants = list(results.keys())
        conversion_rates = [results[v].conversion_rate for v in variants]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(variants, conversion_rates)

        # Color bars based on significance
        for i, variant in enumerate(variants):
            if results[variant].statistical_significance:
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')

        plt.title('Conversion Rate by Variant')
        plt.ylabel('Conversion Rate')
        plt.xlabel('Variant')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'conversion_rates_{experiment_id}.png'))
        plt.close()

        # Metrics comparison heatmap
        metrics_data = []
        metric_names = []

        for variant, result in results.items():
            metrics_data.append([result.metrics.get(m, 0) for m in ['accuracy', 'precision', 'recall', 'f1_score']])
            if not metric_names:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics_data,
                    xticklabels=metric_names,
                    yticklabels=variants,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn')
        plt.title('Metrics Comparison Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_heatmap_{experiment_id}.png'))
        plt.close()

    def stop_experiment(self, experiment_id: str) -> None:
        """Stop a running experiment"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE experiments 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE experiment_id = ?
            ''', (ExperimentStatus.COMPLETED.value, experiment_id))
            conn.commit()

        logger.info(f"Experiment stopped: {experiment_id}")

    def calculate_required_sample_size(self, baseline_rate: float,
                                       minimum_detectable_effect: float,
                                       alpha: float = 0.05,
                                       power: float = 0.8) -> int:
        """Calculate required sample size for experiment"""
        # Two-proportion z-test sample size calculation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect

        # Pooled proportion
        p_pooled = (p1 + p2) / 2

        # Sample size calculation
        numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                     z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2

        n = numerator / denominator

        return int(np.ceil(n))

    def monitor_experiment_progress(self, experiment_id: str) -> Dict[str, Any]:
        """Monitor experiment progress and provide recommendations"""
        results = self.analyze_experiment(experiment_id)
        experiment = self.get_experiment(experiment_id)
        config = ExperimentConfig(**json.loads(experiment['config']))

        # Calculate progress metrics
        total_samples = sum(result.sample_size for result in results.values())
        min_samples = min(result.sample_size for result in results.values()) if results else 0

        progress = {
            'experiment_id': experiment_id,
            'total_samples': total_samples,
            'min_variant_samples': min_samples,
            'target_sample_size': config.minimum_sample_size,
            'progress_percentage': (min_samples / config.minimum_sample_size) * 100,
            'days_running': (datetime.now() - config.start_date).days,
            'days_remaining': (config.end_date - datetime.now()).days,
            'early_stopping_recommendation': None,
            'extend_experiment_recommendation': None
        }

        # Check for early stopping conditions
        if results:
            significant_results = [r for r in results.values() if r.statistical_significance]
            if significant_results and min_samples >= config.minimum_sample_size:
                progress['early_stopping_recommendation'] = "Consider early stopping - significant results detected"

        # Check if extension is needed
        if progress['days_remaining'] < 3 and min_samples < config.minimum_sample_size:
            progress['extend_experiment_recommendation'] = "Consider extending experiment - insufficient sample size"

        return progress


class ABTestingCLI:
    """Command-line interface for A/B testing framework"""

    def __init__(self):
        self.framework = ABTestingFramework()

    def create_experiment_interactive(self):
        """Interactive experiment creation"""
        print("\n=== Create New A/B Test Experiment ===")

        # Get experiment details
        experiment_id = input("Experiment ID: ")
        name = input("Experiment Name: ")
        description = input("Description: ")

        # Get variants
        control_variant = input("Control Variant: ")
        treatment_variants = input("Treatment Variants (comma-separated): ").split(',')
        treatment_variants = [v.strip() for v in treatment_variants]

        # Get traffic allocation
        traffic_allocation = {}
        print("\nTraffic Allocation (percentages):")
        for variant in [control_variant] + treatment_variants:
            allocation = float(input(f"{variant}: "))
            traffic_allocation[variant] = allocation

        # Get experiment parameters
        duration_days = int(input("Duration (days): "))
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)

        confidence_level = float(input("Confidence Level (0.95): ") or "0.95")
        min_sample_size = int(input("Minimum Sample Size (1000): ") or "1000")

        # Create experiment config
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            control_variant=control_variant,
            treatment_variants=treatment_variants,
            traffic_allocation=traffic_allocation,
            allocation_strategy=TrafficAllocation.HASH_BASED,
            success_metrics=['accuracy', 'f1_score'],
            minimum_sample_size=min_sample_size,
            confidence_level=confidence_level,
            power=0.8,
            expected_effect_size=0.05,
            status=ExperimentStatus.DRAFT
        )

        # Create experiment
        experiment_id = self.framework.create_experiment(config)
        print(f"\nExperiment created: {experiment_id}")

        # Ask if user wants to start immediately
        start_now = input("Start experiment now? (y/n): ").lower() == 'y'
        if start_now:
            self.start_experiment(experiment_id)

    def start_experiment(self, experiment_id: str):
        """Start an experiment"""
        with self.framework.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE experiments 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE experiment_id = ?
            ''', (ExperimentStatus.ACTIVE.value, experiment_id))
            conn.commit()

        print(f"Experiment started: {experiment_id}")

    def list_experiments_interactive(self):
        """Interactive experiment listing"""
        print("\n=== Active Experiments ===")
        experiments = self.framework.list_experiments(ExperimentStatus.ACTIVE)

        if not experiments:
            print("No active experiments found.")
            return

        for exp in experiments:
            print(f"\nID: {exp['experiment_id']}")
            print(f"Name: {exp['name']}")
            print(f"Status: {exp['status']}")
            print(f"Created: {exp['created_at']}")

    def analyze_experiment_interactive(self):
        """Interactive experiment analysis"""
        experiment_id = input("Enter Experiment ID to analyze: ")

        try:
            results = self.framework.analyze_experiment(experiment_id)

            print(f"\n=== Analysis Results for {experiment_id} ===")

            for variant, result in results.items():
                print(f"\nVariant: {variant}")
                print(f"Sample Size: {result.sample_size}")
                print(f"Conversion Rate: {result.conversion_rate:.4f}")
                print(f"Statistical Significance: {result.statistical_significance}")
                print(f"P-value: {result.p_value:.4f}")
                print(f"Effect Size: {result.effect_size:.4f}")
                print(f"95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")

                print("\nDetailed Metrics:")
                for metric, value in result.metrics.items():
                    print(f"  {metric}: {value:.4f}")

            # Generate report
            generate_report = input("\nGenerate detailed report? (y/n): ").lower() == 'y'
            if generate_report:
                report_path = self.framework.generate_experiment_report(experiment_id)
                print(f"Report generated: {report_path}")

        except Exception as e:
            print(f"Error analyzing experiment: {str(e)}")

    def monitor_experiment_interactive(self):
        """Interactive experiment monitoring"""
        experiment_id = input("Enter Experiment ID to monitor: ")

        try:
            progress = self.framework.monitor_experiment_progress(experiment_id)

            print(f"\n=== Monitoring {experiment_id} ===")
            print(f"Total Samples: {progress['total_samples']}")
            print(f"Min Variant Samples: {progress['min_variant_samples']}")
            print(f"Target Sample Size: {progress['target_sample_size']}")
            print(f"Progress: {progress['progress_percentage']:.1f}%")
            print(f"Days Running: {progress['days_running']}")
            print(f"Days Remaining: {progress['days_remaining']}")

            if progress['early_stopping_recommendation']:
                print(f"⚠️  {progress['early_stopping_recommendation']}")

            if progress['extend_experiment_recommendation']:
                print(f"⚠️  {progress['extend_experiment_recommendation']}")

        except Exception as e:
            print(f"Error monitoring experiment: {str(e)}")

    def simulate_experiment_data(self):
        """Simulate experiment data for testing"""
        experiment_id = input("Enter Experiment ID to simulate data for: ")
        num_events = int(input("Number of events to simulate (1000): ") or "1000")

        # Get experiment
        experiment = self.framework.get_experiment(experiment_id)
        if not experiment:
            print("Experiment not found!")
            return

        config = ExperimentConfig(**json.loads(experiment['config']))

        print(f"Simulating {num_events} events for experiment {experiment_id}...")

        for i in range(num_events):
            user_id = f"user_{i}"
            variant = self.framework.get_variant_assignment(experiment_id, user_id)

            # Simulate prediction based on variant
            if variant == config.control_variant:
                # Control variant performance
                accuracy = 0.85
                response_time = 0.15
            else:
                # Treatment variant(s) - slightly better performance
                accuracy = 0.87
                response_time = 0.12

            # Generate synthetic prediction
            prediction = 1 if random.random() < accuracy else 0
            actual_label = 1 if random.random() < 0.5 else 0  # Random actual labels
            confidence = random.uniform(0.7, 0.99)

            # Simulate response time
            sim_response_time = response_time + random.uniform(-0.05, 0.05)

            # Record event
            event_data = {
                'prediction': prediction,
                'actual_label': actual_label,
                'confidence': confidence,
                'response_time': sim_response_time
            }

            self.framework.record_event(
                experiment_id, user_id, variant, 'prediction', event_data
            )

        print(f"Simulated {num_events} events successfully!")

    def run_cli(self):
        """Run the command-line interface"""
        while True:
            print("\n=== MLOps A/B Testing Framework ===")
            print("1. Create New Experiment")
            print("2. List Experiments")
            print("3. Start Experiment")
            print("4. Analyze Experiment")
            print("5. Monitor Experiment")
            print("6. Simulate Data (Testing)")
            print("7. Calculate Sample Size")
            print("8. Stop Experiment")
            print("9. Exit")

            choice = input("\nSelect option: ")

            if choice == '1':
                self.create_experiment_interactive()
            elif choice == '2':
                self.list_experiments_interactive()
            elif choice == '3':
                experiment_id = input("Enter Experiment ID to start: ")
                self.start_experiment(experiment_id)
            elif choice == '4':
                self.analyze_experiment_interactive()
            elif choice == '5':
                self.monitor_experiment_interactive()
            elif choice == '6':
                self.simulate_experiment_data()
            elif choice == '7':
                self.calculate_sample_size_interactive()
            elif choice == '8':
                experiment_id = input("Enter Experiment ID to stop: ")
                self.framework.stop_experiment(experiment_id)
            elif choice == '9':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    def calculate_sample_size_interactive(self):
        """Interactive sample size calculation"""
        print("\n=== Sample Size Calculator ===")

        baseline_rate = float(input("Baseline conversion rate (0.0-1.0): "))
        effect_size = float(input("Minimum detectable effect (e.g., 0.05 for 5% improvement): "))
        alpha = float(input("Alpha (significance level, default 0.05): ") or "0.05")
        power = float(input("Power (default 0.8): ") or "0.8")

        sample_size = self.framework.calculate_required_sample_size(
            baseline_rate, effect_size, alpha, power
        )

        print(f"\nRequired sample size per variant: {sample_size}")
        print(f"Total sample size (both variants): {sample_size * 2}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='A/B Testing Framework for MLOps')
    parser.add_argument('--config', default='config/ab_test_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--database', default='data/ab_testing.db',
                        help='Database file path')
    parser.add_argument('--command', choices=['cli', 'create', 'analyze', 'monitor', 'list'],
                        default='cli', help='Command to run')
    parser.add_argument('--experiment-id', help='Experiment ID for specific commands')
    parser.add_argument('--report-dir', default='reports', help='Report output directory')

    args = parser.parse_args()

    # Initialize framework
    framework = ABTestingFramework(args.config, args.database)

    if args.command == 'cli':
        # Run interactive CLI
        cli = ABTestingCLI()
        cli.run_cli()

    elif args.command == 'create':
        # Create example experiment
        config = ExperimentConfig(
            experiment_id=f"exp_{int(time.time())}",
            name="Model A vs Model B",
            description="Comparing DistillBERT models",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=14),
            control_variant="model_a",
            treatment_variants=["model_b"],
            traffic_allocation={"model_a": 50.0, "model_b": 50.0},
            allocation_strategy=TrafficAllocation.HASH_BASED,
            success_metrics=['accuracy', 'f1_score'],
            minimum_sample_size=1000,
            confidence_level=0.95,
            power=0.8,
            expected_effect_size=0.05,
            status=ExperimentStatus.ACTIVE
        )

        experiment_id = framework.create_experiment(config)
        print(f"Created experiment: {experiment_id}")

    elif args.command == 'analyze':
        if not args.experiment_id:
            print("--experiment-id required for analyze command")
            return

        results = framework.analyze_experiment(args.experiment_id)
        report_path = framework.generate_experiment_report(args.experiment_id, args.report_dir)

        print(f"Analysis complete. Report generated: {report_path}")

    elif args.command == 'monitor':
        if not args.experiment_id:
            print("--experiment-id required for monitor command")
            return

        progress = framework.monitor_experiment_progress(args.experiment_id)

        print(json.dumps(progress, indent=2, default=str))

    elif args.command == 'list':
        experiments = framework.list_experiments()

        print("\nAll Experiments:")
        for exp in experiments:
            print(f"ID: {exp['experiment_id']}, Name: {exp['name']}, Status: {exp['status']}")


if __name__ == "__main__":
    main()