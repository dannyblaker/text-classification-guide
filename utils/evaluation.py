"""
Evaluation Metrics and Utilities

This module provides comprehensive evaluation tools for text classification.
Understanding model performance through various metrics is crucial for:
- Comparing different models
- Identifying weaknesses
- Making informed decisions
- Communicating results
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter
import warnings


class ClassificationEvaluator:
    """
    Comprehensive evaluation for classification models.

    Provides various metrics and visualizations to understand
    model performance from different angles.
    """

    def __init__(self, y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None):
        """
        Initialize evaluator with predictions and ground truth.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional list of all possible labels
        """
        self.y_true = y_true
        self.y_pred = y_pred

        if labels is None:
            self.labels = sorted(list(set(y_true + y_pred)))
        else:
            self.labels = labels

    def accuracy(self) -> float:
        """
        Calculate overall accuracy.

        Accuracy = (Correct Predictions) / (Total Predictions)

        Good when classes are balanced, misleading with imbalanced data.

        Returns:
            Accuracy score between 0 and 1
        """
        correct = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == p)
        return correct / len(self.y_true)

    def precision_recall_f1(self, average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score.

        Precision: Of all positive predictions, how many were correct?
            Precision = TP / (TP + FP)

        Recall: Of all actual positives, how many did we find?
            Recall = TP / (TP + FN)

        F1-Score: Harmonic mean of precision and recall
            F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Args:
            average: 'micro', 'macro', 'weighted', or None
                - micro: Calculate globally (treats all classes equally)
                - macro: Calculate per class, then average (unweighted)
                - weighted: Calculate per class, weighted by support
                - None: Return per-class metrics

        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true,
            self.y_pred,
            labels=self.labels,
            average=average,
            zero_division=0
        )

        if average is None:
            # Return per-class metrics
            return {
                'precision': dict(zip(self.labels, precision)),
                'recall': dict(zip(self.labels, recall)),
                'f1': dict(zip(self.labels, f1)),
                'support': dict(zip(self.labels, support))
            }
        else:
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

    def confusion_matrix(self) -> pd.DataFrame:
        """
        Generate confusion matrix.

        Shows the distribution of predictions vs actual labels.
        Diagonal elements are correct predictions.
        Off-diagonal elements show confusion between classes.

        Returns:
            DataFrame with confusion matrix
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        return pd.DataFrame(
            cm,
            index=[f"True: {l}" for l in self.labels],
            columns=[f"Pred: {l}" for l in self.labels]
        )

    def classification_report(self) -> str:
        """
        Generate detailed classification report.

        Returns:
            String with formatted report
        """
        from sklearn.metrics import classification_report

        return classification_report(
            self.y_true,
            self.y_pred,
            labels=self.labels,
            zero_division=0
        )

    def per_class_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for each class separately.

        Returns:
            DataFrame with per-class metrics
        """
        metrics = self.precision_recall_f1(average=None)

        df = pd.DataFrame({
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Support': metrics['support']
        })

        return df

    def error_analysis(self, texts: Optional[List[str]] = None, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze misclassified examples.

        Args:
            texts: Optional original texts
            top_n: Number of errors to return

        Returns:
            DataFrame with error details
        """
        errors = []

        for idx, (true, pred) in enumerate(zip(self.y_true, self.y_pred)):
            if true != pred:
                error = {
                    'index': idx,
                    'true_label': true,
                    'pred_label': pred,
                }
                if texts:
                    error['text'] = texts[idx][:100]  # Truncate for display
                errors.append(error)

        df = pd.DataFrame(errors)
        return df.head(top_n) if len(df) > top_n else df

    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Get a summary of all key metrics.

        Returns:
            Dictionary with metric names and values
        """
        metrics = self.precision_recall_f1(average='weighted')

        return {
            'accuracy': self.accuracy(),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
            'num_samples': len(self.y_true),
            'num_classes': len(self.labels)
        }

    def print_evaluation_report(self):
        """Print comprehensive evaluation report."""
        print("="*80)
        print("CLASSIFICATION EVALUATION REPORT")
        print("="*80)

        # Overall metrics
        print("\n📊 OVERALL METRICS")
        print("-" * 80)
        summary = self.get_metrics_summary()
        for metric, value in summary.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title():20s}: {value:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title():20s}: {value}")

        # Per-class metrics
        print("\n📈 PER-CLASS METRICS")
        print("-" * 80)
        print(self.per_class_metrics().to_string())

        # Confusion matrix
        print("\n🔀 CONFUSION MATRIX")
        print("-" * 80)
        print(self.confusion_matrix().to_string())

        print("\n" + "="*80)


def calculate_baseline_accuracy(y_true: List[str]) -> float:
    """
    Calculate majority class baseline accuracy.

    This is the accuracy you'd get by always predicting
    the most common class. Your model should beat this!

    Args:
        y_true: True labels

    Returns:
        Baseline accuracy
    """
    most_common = Counter(y_true).most_common(1)[0]
    return most_common[1] / len(y_true)


def stratified_split(texts: List[str],
                     labels: List[str],
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple:
    """
    Split data while preserving label distribution.

    Args:
        texts: List of texts
        labels: List of labels
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )


def cross_validate_model(model, texts: List[str], labels: List[str], cv: int = 5) -> Dict:
    """
    Perform k-fold cross-validation.

    Args:
        model: sklearn-compatible model
        texts: List of texts
        labels: List of labels
        cv: Number of folds

    Returns:
        Dictionary with cross-validation scores
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    # Create pipeline if model doesn't handle text
    if not hasattr(model, 'transform'):
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', model)
        ])
    else:
        pipeline = model

    # Perform cross-validation
    accuracy_scores = cross_val_score(
        pipeline, texts, labels, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(
        pipeline, texts, labels, cv=cv, scoring='f1_weighted')

    return {
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std(),
        'accuracy_scores': accuracy_scores,
        'f1_scores': f1_scores
    }


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple model results.

    Args:
        results: Dictionary mapping model names to their metrics

    Returns:
        DataFrame with comparison
    """
    comparison = []

    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison.append(row)

    df = pd.DataFrame(comparison)

    # Sort by F1 score if available
    if 'f1_score' in df.columns:
        df = df.sort_values('f1_score', ascending=False)

    return df


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("EVALUATION METRICS EXAMPLES")
    print("="*80)

    # Sample predictions
    y_true = ['positive', 'negative', 'neutral', 'positive', 'negative',
              'neutral', 'positive', 'negative', 'positive', 'neutral']
    y_pred = ['positive', 'negative', 'positive', 'positive', 'negative',
              'neutral', 'neutral', 'negative', 'positive', 'neutral']

    # Create evaluator
    evaluator = ClassificationEvaluator(y_true, y_pred)

    # Print comprehensive report
    evaluator.print_evaluation_report()

    # Calculate baseline
    print("\n📊 BASELINE COMPARISON")
    print("-" * 80)
    baseline = calculate_baseline_accuracy(y_true)
    model_acc = evaluator.accuracy()
    print(f"Baseline (majority class):    {baseline:.4f}")
    print(f"Model accuracy:               {model_acc:.4f}")
    print(f"Improvement over baseline:    {(model_acc - baseline):.4f}")

    # Error analysis
    sample_texts = [
        "Great product!",
        "Terrible quality",
        "It's okay I guess",
        "Love it so much",
        "Don't buy this",
        "Average product",
        "Best ever!",
        "Worst purchase",
        "Highly recommend",
        "Nothing special"
    ]

    print("\n🔍 ERROR ANALYSIS")
    print("-" * 80)
    errors = evaluator.error_analysis(sample_texts, top_n=5)
    if len(errors) > 0:
        print(errors.to_string(index=False))
    else:
        print("No errors found!")

    print("\n" + "="*80)
    print("METRICS INTERPRETATION GUIDE")
    print("="*80)
    print("""
Metric      | What it measures              | When to prioritize
------------|-------------------------------|--------------------------------
Accuracy    | Overall correctness           | Balanced datasets
Precision   | Correctness of predictions    | False positives are costly
Recall      | Coverage of actual cases      | False negatives are costly
F1-Score    | Balance of precision/recall   | General performance measure

Examples:
- Spam Detection: Prioritize PRECISION (avoid marking real emails as spam)
- Disease Detection: Prioritize RECALL (don't miss actual cases)
- General Classification: Use F1-SCORE for balanced view
    """)
