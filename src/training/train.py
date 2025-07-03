"""
Training script for DistillBERT sentiment analysis model
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
import mlflow
import mlflow.pytorch
import wandb
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training with experiment tracking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model']['name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tracking
        self.setup_experiment_tracking()

        # Model components
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def setup_experiment_tracking(self):
        """Setup MLflow and W&B tracking"""
        # MLflow setup
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(self.config.get('experiment_name', 'distillbert-sentiment'))

        # W&B setup (optional)
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'mlops-distillbert'),
                config=self.config,
                name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

    def load_data(self, data_path: str) -> DatasetDict:
        """Load and prepare training data"""
        logger.info(f"Loading data from {data_path}")

        # Load data (assuming CSV format)
        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_path, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        logger.info(f"Loaded datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        return dataset_dict

    def preprocess_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Tokenize and preprocess the data"""
        logger.info("Preprocessing data...")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config['training']['max_length']
            )

        # Tokenize datasets
        tokenized_datasets = dataset_dict.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # Rename label column if needed
        if 'sentiment' in tokenized_datasets['train'].column_names:
            tokenized_datasets = tokenized_datasets.rename_column('sentiment', 'labels')

        return tokenized_datasets

    def initialize_model(self) -> None:
        """Initialize the model"""
        logger.info(f"Initializing model: {self.model_name}")

        num_labels = self.config['model'].get('num_labels', 2)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )

        # Move to device
        self.model.to(self.device)

        logger.info(f"Model initialized with {self.model.num_parameters()} parameters")

    def setup_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Setup training arguments"""
        training_config = self.config['training']

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config['epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_steps=training_config.get('warmup_steps', 500),
            logging_dir=f"{output_dir}/logs",
            logging_steps=training_config.get('logging_steps', 10),
            evaluation_strategy="steps",
            eval_steps=training_config.get('eval_steps', 500),
            save_strategy="steps",
            save_steps=training_config.get('save_steps', 500),
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=["mlflow"] + (["wandb"] if self.config.get('use_wandb', False) else []),
            run_name=f"distillbert-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            seed=42,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            push_to_hub=False
        )

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self, data_path: str, output_dir: str) -> Dict[str, Any]:
        """Main training function"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)

            # Load and preprocess data
            dataset_dict = self.load_data(data_path)
            tokenized_datasets = self.preprocess_data(dataset_dict)

            # Initialize model
            self.initialize_model()

            # Setup training arguments
            training_args = self.setup_training_arguments(output_dir)

            # Data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True
            )

            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Train the model
            logger.info("Starting training...")
            train_result = self.trainer.train()

            # Log training metrics
            mlflow.log_metrics({
                'train_loss': train_result.training_loss,
                'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
                'train_steps_per_second': train_result.metrics.get('train_steps_per_second', 0)
            })

            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_results = self.trainer.evaluate(eval_dataset=tokenized_datasets['test'])

            # Log test metrics
            test_metrics = {f'test_{k}': v for k, v in test_results.items()}
            mlflow.log_metrics(test_metrics)

            # Save the model
            model_path = os.path.join(output_dir, 'final_model')
            self.trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)

            # Log model to MLflow
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name="distillbert-sentiment"
            )

            # Generate and save detailed evaluation report
            evaluation_report = self.generate_evaluation_report(
                tokenized_datasets['test'],
                test_results
            )

            # Save evaluation report
            report_path = os.path.join(output_dir, 'evaluation_report.json')
            with open(report_path, 'w') as f:
                json.dump(evaluation_report, f, indent=2, default=str)

            mlflow.log_artifact(report_path)

            logger.info(f"Training completed. Model saved to {model_path}")

            return {
                'model_path': model_path,
                'train_metrics': train_result.metrics,
                'test_metrics': test_results,
                'evaluation_report': evaluation_report
            }

    def generate_evaluation_report(self, test_dataset: Dataset, test_results: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed evaluation report"""
        logger.info("Generating evaluation report...")

        # Get predictions for confusion matrix
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'test_size': len(test_dataset),
            'overall_metrics': test_results,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1.tolist(),
                'support': support.tolist()
            },
            'model_config': self.config
        }

        return report

    def validate_model_performance(self, evaluation_report: Dict[str, Any]) -> bool:
        """Validate if model meets performance criteria"""
        min_accuracy = self.config.get('validation', {}).get('min_accuracy', 0.85)
        min_f1 = self.config.get('validation', {}).get('min_f1', 0.80)

        accuracy = evaluation_report['overall_metrics']['eval_accuracy']
        f1 = evaluation_report['overall_metrics']['eval_f1']

        if accuracy < min_accuracy:
            logger.warning(f"Model accuracy {accuracy:.3f} below threshold {min_accuracy}")
            return False

        if f1 < min_f1:
            logger.warning(f"Model F1 score {f1:.3f} below threshold {min_f1}")
            return False

        logger.info(f"Model validation passed: accuracy={accuracy:.3f}, f1={f1:.3f}")
        return True


def create_sample_data(output_dir: str):
    """Create sample training data for testing"""
    np.random.seed(42)

    # Sample sentiment data
    positive_texts = [
        "I love this product!", "Amazing quality and fast delivery",
        "Great experience, highly recommended", "Fantastic service",
        "Excellent value for money", "Outstanding performance"
    ]

    negative_texts = [
        "Terrible product, waste of money", "Very disappointed with the quality",
        "Poor customer service", "Not worth the price",
        "Completely unsatisfied", "Worst purchase ever"
    ]

    # Create dataset
    texts = positive_texts * 100 + negative_texts * 100
    labels = [1] * 600 + [0] * 600

    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Split data
    total = len(texts)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    splits = {
        'train': (0, train_size),
        'validation': (train_size, train_size + val_size),
        'test': (train_size + val_size, total)
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, (start, end) in splits.items():
        df = pd.DataFrame({
            'text': texts[start:end],
            'labels': labels[start:end]
        })
        df.to_csv(os.path.join(output_dir, f'{split_name}.csv'), index=False)

    logger.info(f"Sample data created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train DistillBERT sentiment analysis model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-path', type=str, required=True, help='Output directory for model')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased', help='Model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--experiment-name', type=str, default='distillbert-sentiment', help='MLflow experiment name')
    parser.add_argument('--create-sample-data', action='store_true', help='Create sample data for testing')

    args = parser.parse_args()

    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data(args.data_path)
        logger.info("Sample data created. Run training again without --create-sample-data flag.")
        return

    # Training configuration
    config = {
        'model': {
            'name': args.model_name,
            'num_labels': 2
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'logging_steps': 10,
            'eval_steps': 500,
            'save_steps': 500
        },
        'validation': {
            'min_accuracy': 0.85,
            'min_f1': 0.80
        },
        'experiment_name': args.experiment_name,
        'use_wandb': False,
        'mlflow_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    }

    # Initialize trainer and run training
    trainer = ModelTrainer(config)

    try:
        results = trainer.train(args.data_path, args.output_path)

        # Validate model performance
        if trainer.validate_model_performance(results['evaluation_report']):
            logger.info("✅ Model training completed successfully and passed validation!")
        else:
            logger.error("❌ Model training completed but failed validation criteria")
            exit(1)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()