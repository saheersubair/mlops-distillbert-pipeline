"""
Model evaluation script for DistillBERT sentiment analysis
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)
from datasets import Dataset
import mlflow
import mlflow.pytorch
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class"""

    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        self.model_path = model_path
        self.config = config or {}
        self.model: Optional[Pipeline] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.raw_model: Optional[AutoModelForSequenceClassification] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.evaluation_results = {}
        self.predictions = []
        self.true_labels = []
        self.prediction_probabilities = []

    def load_model(self) -> None:
        """Load model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")

        try:
            # Load pipeline for easy inference
            self.model = pipeline(
                "sentiment-analysis",
                model=self.model_path,
                tokenizer=self.model_path,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )

            # Load raw model for advanced operations
            self.raw_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Move to device
            self.raw_model.to(self.device)
            self.raw_model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def load_test_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load test data from various formats"""
        logger.info(f"Loading test data from {data_path}")

        if os.path.isfile(data_path):
            # Single file
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path, lines=True)
            elif data_path.endswith('.jsonl'):
                df = pd.read_json(data_path, lines=True)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        else:
            # Directory with test files
            test_file = os.path.join(data_path, 'test.csv')
            if os.path.exists(test_file):
                df = pd.read_csv(test_file)
            else:
                raise FileNotFoundError(f"Test file not found in {data_path}")

        # Extract texts and labels
        if 'text' in df.columns and 'label' in df.columns:
            texts = df['text'].tolist()
            labels = df['label'].tolist()
        elif 'text' in df.columns and 'sentiment' in df.columns:
            texts = df['text'].tolist()
            labels = df['sentiment'].tolist()
        elif 'sentence' in df.columns and 'label' in df.columns:
            texts = df['sentence'].tolist()
            labels = df['label'].tolist()
        else:
            raise ValueError("Expected columns: 'text' and 'label' (or 'sentiment')")

        logger.info(f"Loaded {len(texts)} test samples")
        return texts, labels

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> Tuple[List[int], List[float]]:
        """Make predictions in batches"""
        logger.info(f"Making predictions for {len(texts)} samples")

        predictions = []
        probabilities = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Get predictions
            batch_results = self.model(batch_texts)

            for result in batch_results:
                # Find the class with highest score
                if isinstance(result, list):
                    # Multiple scores returned
                    best_result = max(result, key=lambda x: x['score'])
                    pred_label = 1 if best_result['label'] == 'POSITIVE' else 0
                    pred_prob = best_result['score']
                else:
                    # Single score returned
                    pred_label = 1 if result['label'] == 'POSITIVE' else 0
                    pred_prob = result['score']

                predictions.append(pred_label)
                probabilities.append(pred_prob)

        return predictions, probabilities

    def get_prediction_probabilities(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get prediction probabilities for all classes"""
        logger.info("Getting prediction probabilities...")

        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.raw_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def calculate_basic_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        logger.info("Calculating basic metrics...")

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        return {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support.tolist()
        }

    def calculate_advanced_metrics(self, y_true: List[int], y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate advanced metrics using probabilities"""
        logger.info("Calculating advanced metrics...")

        # ROC AUC
        if y_probs.shape[1] == 2:  # Binary classification
            roc_auc = roc_auc_score(y_true, y_probs[:, 1])
            avg_precision = average_precision_score(y_true, y_probs[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
            avg_precision = average_precision_score(y_true, y_probs[:, 1] if y_probs.shape[1] == 2 else y_probs)

        # Confidence statistics
        max_probs = np.max(y_probs, axis=1)
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs),
            'median_confidence': np.median(max_probs)
        }

        # Calibration metrics (Brier score)
        if y_probs.shape[1] == 2:
            y_true_binary = np.array(y_true)
            brier_score = np.mean((y_probs[:, 1] - y_true_binary) ** 2)
        else:
            # For multi-class, use one-hot encoding
            y_true_onehot = np.eye(y_probs.shape[1])[y_true]
            brier_score = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))

        return {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'brier_score': brier_score,
            **confidence_stats
        }

    def analyze_errors(self, texts: List[str], y_true: List[int], y_pred: List[int],
                       y_probs: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Analyze prediction errors"""
        logger.info("Analyzing prediction errors...")

        # Find misclassified samples
        misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]

        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(y_true),
            'error_examples': []
        }

        # Get examples with highest confidence but wrong predictions
        if misclassified_indices:
            error_confidences = [np.max(y_probs[i]) for i in misclassified_indices]
            # Sort by confidence (descending)
            sorted_errors = sorted(zip(misclassified_indices, error_confidences),
                                   key=lambda x: x[1], reverse=True)

            for i, (idx, conf) in enumerate(sorted_errors[:top_k]):
                error_analysis['error_examples'].append({
                    'text': texts[idx][:200] + '...' if len(texts[idx]) > 200 else texts[idx],
                    'true_label': y_true[idx],
                    'predicted_label': y_pred[idx],
                    'confidence': conf,
                    'text_length': len(texts[idx])
                })

        # Error distribution by text length
        error_lengths = [len(texts[i]) for i in misclassified_indices]
        correct_lengths = [len(texts[i]) for i in range(len(texts)) if i not in misclassified_indices]

        if error_lengths and correct_lengths:
            # Statistical test for length difference
            stat, p_value = stats.ttest_ind(error_lengths, correct_lengths)
            error_analysis['length_analysis'] = {
                'error_mean_length': np.mean(error_lengths),
                'correct_mean_length': np.mean(correct_lengths),
                'length_difference_significant': p_value < 0.05,
                'p_value': p_value
            }

        return error_analysis

    def create_visualizations(self, y_true: List[int], y_pred: List[int],
                              y_probs: np.ndarray, output_dir: str) -> None:
        """Create visualization plots"""
        logger.info("Creating visualizations...")

        os.makedirs(output_dir, exist_ok=True)

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ROC Curve (for binary classification)
        if y_probs.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            auc_score = roc_auc_score(y_true, y_probs[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
            avg_precision = average_precision_score(y_true, y_probs[:, 1])
            plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Confidence Distribution
        plt.figure(figsize=(10, 6))
        max_probs = np.max(y_probs, axis=1)
        correct_mask = np.array(y_true) == np.array(y_pred)

        plt.hist(max_probs[correct_mask], bins=50, alpha=0.7, label='Correct', density=True)
        plt.hist(max_probs[~correct_mask], bins=50, alpha=0.7, label='Incorrect', density=True)
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")

    def evaluate_model(self, test_data_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Main evaluation function"""
        logger.info("Starting model evaluation...")

        if output_dir is None:
            output_dir = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        os.makedirs(output_dir, exist_ok=True)

        # Load model and data
        self.load_model()
        texts, true_labels = self.load_test_data(test_data_path)

        # Make predictions
        predictions, _ = self.predict_batch(texts)
        probabilities = self.get_prediction_probabilities(texts)

        # Store results
        self.predictions = predictions
        self.true_labels = true_labels
        self.prediction_probabilities = probabilities

        # Calculate metrics
        basic_metrics = self.calculate_basic_metrics(true_labels, predictions)
        advanced_metrics = self.calculate_advanced_metrics(true_labels, probabilities)
        error_analysis = self.analyze_errors(texts, true_labels, predictions, probabilities)

        # Create visualizations
        self.create_visualizations(true_labels, predictions, probabilities, output_dir)

        # Compile results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'test_data_path': test_data_path,
            'test_size': len(texts),
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'error_analysis': error_analysis,
            'model_info': {
                'device': str(self.device),
                'num_parameters': self.get_model_parameters()
            }
        }

        self.evaluation_results = evaluation_results

        # Save results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'text': texts,
            'true_label': true_labels,
            'predicted_label': predictions,
            'confidence': np.max(probabilities, axis=1),
            'correct': np.array(true_labels) == np.array(predictions)
        })
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

        # Generate report
        self.generate_report(output_dir)

        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        return evaluation_results

    def get_model_parameters(self) -> int:
        """Get number of model parameters"""
        if self.raw_model is None:
            return 0
        return sum(p.numel() for p in self.raw_model.parameters())

    def generate_report(self, output_dir: str) -> None:
        """Generate a comprehensive evaluation report"""
        logger.info("Generating evaluation report...")

        report_path = os.path.join(output_dir, 'evaluation_report.md')

        with open(report_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model:** {self.model_path}\n\n")

            # Basic metrics
            f.write("## Basic Metrics\n\n")
            basic = self.evaluation_results['basic_metrics']
            f.write(f"- **Accuracy:** {basic['accuracy']:.4f}\n")
            f.write(f"- **Precision (Weighted):** {basic['precision_weighted']:.4f}\n")
            f.write(f"- **Recall (Weighted):** {basic['recall_weighted']:.4f}\n")
            f.write(f"- **F1 Score (Weighted):** {basic['f1_weighted']:.4f}\n\n")

            # Advanced metrics
            f.write("## Advanced Metrics\n\n")
            advanced = self.evaluation_results['advanced_metrics']
            f.write(f"- **ROC AUC:** {advanced['roc_auc']:.4f}\n")
            f.write(f"- **Average Precision:** {advanced['average_precision']:.4f}\n")
            f.write(f"- **Brier Score:** {advanced['brier_score']:.4f}\n")
            f.write(f"- **Mean Confidence:** {advanced['mean_confidence']:.4f}\n\n")

            # Error analysis
            f.write("## Error Analysis\n\n")
            error = self.evaluation_results['error_analysis']
            f.write(f"- **Total Errors:** {error['total_errors']}\n")
            f.write(f"- **Error Rate:** {error['error_rate']:.4f}\n\n")

            if error['error_examples']:
                f.write("### Top Error Examples\n\n")
                for i, example in enumerate(error['error_examples'][:5]):
                    f.write(f"**Example {i + 1}:**\n")
                    f.write(f"- Text: {example['text']}\n")
                    f.write(f"- True Label: {example['true_label']}\n")
                    f.write(f"- Predicted Label: {example['predicted_label']}\n")
                    f.write(f"- Confidence: {example['confidence']:.4f}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            self.generate_recommendations(f)

        logger.info(f"Report generated: {report_path}")

    def generate_recommendations(self, file_handle) -> None:
        """Generate recommendations based on evaluation results"""
        basic = self.evaluation_results['basic_metrics']
        advanced = self.evaluation_results['advanced_metrics']
        error = self.evaluation_results['error_analysis']

        file_handle.write("### Performance Assessment\n\n")

        # Accuracy assessment
        if basic['accuracy'] >= 0.95:
            file_handle.write("✅ **Excellent accuracy** - Model is performing very well\n")
        elif basic['accuracy'] >= 0.90:
            file_handle.write("✅ **Good accuracy** - Model performance is satisfactory\n")
        elif basic['accuracy'] >= 0.80:
            file_handle.write("⚠️ **Moderate accuracy** - Consider model improvements\n")
        else:
            file_handle.write("❌ **Poor accuracy** - Model needs significant improvements\n")

        # Confidence assessment
        if advanced['mean_confidence'] >= 0.90:
            file_handle.write("✅ **High confidence** - Model is confident in predictions\n")
        elif advanced['mean_confidence'] >= 0.70:
            file_handle.write("⚠️ **Moderate confidence** - Some uncertainty in predictions\n")
        else:
            file_handle.write("❌ **Low confidence** - Model shows high uncertainty\n")

        file_handle.write("\n### Actionable Recommendations\n\n")

        # Specific recommendations based on metrics
        if basic['accuracy'] < 0.85:
            file_handle.write("- Consider collecting more training data\n")
            file_handle.write("- Experiment with different model architectures\n")
            file_handle.write("- Fine-tune hyperparameters\n")

        if advanced['mean_confidence'] < 0.70:
            file_handle.write("- Implement confidence-based rejection\n")
            file_handle.write("- Consider model calibration techniques\n")

        if error['error_rate'] > 0.15:
            file_handle.write("- Analyze error patterns for systematic issues\n")
            file_handle.write("- Consider data augmentation for underrepresented cases\n")

        if 'length_analysis' in error and error['length_analysis']['length_difference_significant']:
            file_handle.write("- Address length bias in the model\n")
            file_handle.write("- Consider length-aware training strategies\n")

    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current model with baseline results"""
        logger.info("Comparing with baseline...")

        comparison = {
            'accuracy_improvement': self.evaluation_results['basic_metrics']['accuracy'] - baseline_results['accuracy'],
            'f1_improvement': self.evaluation_results['basic_metrics']['f1_weighted'] - baseline_results['f1_weighted'],
            'roc_auc_improvement': self.evaluation_results['advanced_metrics']['roc_auc'] - baseline_results['roc_auc']
        }

        return comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate DistillBERT sentiment analysis model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-path', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--min-accuracy', type=float, default=0.85, help='Minimum required accuracy')
    parser.add_argument('--min-f1', type=float, default=0.80, help='Minimum required F1 score')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI')
    parser.add_argument('--experiment-name', type=str, default='model-evaluation', help='MLflow experiment name')

    args = parser.parse_args()

    # Setup MLflow if URI provided
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        config={
            'batch_size': args.batch_size,
            'create_visualizations': args.visualize
        }
    )

    try:
        with mlflow.start_run():
            # Run evaluation
            results = evaluator.evaluate_model(
                test_data_path=args.test_data,
                output_dir=args.output_path
            )

            # Log results to MLflow
            if args.mlflow_uri:
                mlflow.log_metrics(results['basic_metrics'])
                mlflow.log_metrics(results['advanced_metrics'])
                mlflow.log_artifact(args.output_path)

            # Check performance thresholds
            accuracy = results['basic_metrics']['accuracy']
            f1_score = results['basic_metrics']['f1_weighted']

            logger.info(f"Evaluation Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  F1 Score: {f1_score:.4f}")
            logger.info(f"  ROC AUC: {results['advanced_metrics']['roc_auc']:.4f}")

            # Validate against thresholds
            if accuracy < args.min_accuracy:
                logger.error(f"Model accuracy {accuracy:.4f} below threshold {args.min_accuracy}")
                exit(1)

            if f1_score < args.min_f1:
                logger.error(f"Model F1 score {f1_score:.4f} below threshold {args.min_f1}")
                exit(1)

            logger.info("✅ Model evaluation completed successfully and passed all thresholds!")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()