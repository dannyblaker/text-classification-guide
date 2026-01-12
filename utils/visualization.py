"""
Visualization Utilities

This module provides visualization functions for text classification results.
Good visualizations help understand:
- Model performance
- Data distribution
- Feature importance
- Error patterns
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import Counter
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_label_distribution(labels: List[str], title: str = "Label Distribution"):
    """
    Plot distribution of labels in dataset.

    Useful for:
    - Identifying class imbalance
    - Understanding data composition
    - Deciding on sampling strategies

    Args:
        labels: List of labels
        title: Plot title
    """
    counter = Counter(labels)
    labels_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    label_names = [l[0] for l in labels_sorted]
    counts = [l[1] for l in labels_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    ax1.bar(range(len(label_names)), counts,
            color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Labels', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(label_names)))
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, count in enumerate(counts):
        ax1.text(i, count, str(count), ha='center', va='bottom')

    # Pie chart
    ax2.pie(counts, labels=label_names, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette('pastel'))
    ax2.set_title(f'{title} - Proportions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          labels: List[str],
                          normalize: bool = False,
                          title: str = "Confusion Matrix"):
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix (numpy array or DataFrame)
        labels: Label names
        normalize: Whether to normalize by row (true labels)
        title: Plot title
    """
    if isinstance(cm, pd.DataFrame):
        cm = cm.values

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_precision_recall_f1(metrics: Dict[str, Dict[str, float]],
                             title: str = "Per-Class Metrics"):
    """
    Plot precision, recall, and F1-score for each class.

    Args:
        metrics: Dictionary with precision, recall, f1 per class
        title: Plot title
    """
    # Extract data
    classes = list(metrics['precision'].keys())
    precision = [metrics['precision'][c] for c in classes]
    recall = [metrics['recall'][c] for c in classes]
    f1 = [metrics['f1'][c] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label='Precision',
           color='skyblue', edgecolor='navy', alpha=0.7)
    ax.bar(x, recall, width, label='Recall',
           color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax.bar(x + width, f1, width, label='F1-Score',
           color='lightgreen', edgecolor='darkgreen', alpha=0.7)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_history(history: Dict[str, List[float]],
                          title: str = "Training History"):
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss',
                     marker='o', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss',
                     marker='s', linewidth=2)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy',
                     marker='o', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'],
                     label='Validation Accuracy', marker='s', linewidth=2)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names: List[str],
                            importances: np.ndarray,
                            top_n: int = 20,
                            title: str = "Top Features"):
    """
    Plot feature importance scores.

    Args:
        feature_names: Names of features
        importances: Importance scores
        top_n: Number of top features to show
        title: Plot title
    """
    # Get top features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores, color='skyblue', edgecolor='navy', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_text_length_distribution(texts: List[str],
                                  labels: Optional[List[str]] = None,
                                  title: str = "Text Length Distribution"):
    """
    Plot distribution of text lengths.

    Args:
        texts: List of texts
        labels: Optional labels for grouping
        title: Plot title
    """
    lengths = [len(text.split()) for text in texts]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    axes[0].hist(lengths, bins=30, color='skyblue',
                 edgecolor='navy', alpha=0.7)
    axes[0].set_xlabel('Text Length (words)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{title} - Histogram', fontsize=14, fontweight='bold')
    axes[0].axvline(np.mean(lengths), color='red', linestyle='--',
                    label=f'Mean: {np.mean(lengths):.1f}')
    axes[0].axvline(np.median(lengths), color='green',
                    linestyle='--', label=f'Median: {np.median(lengths):.1f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Box plot by label if provided
    if labels:
        length_df = pd.DataFrame({'length': lengths, 'label': labels})
        length_df.boxplot(column='length', by='label', ax=axes[1])
        axes[1].set_xlabel('Label', fontsize=12)
        axes[1].set_ylabel('Text Length (words)', fontsize=12)
        axes[1].set_title(f'{title} - By Label',
                          fontsize=14, fontweight='bold')
        plt.sca(axes[1])
        plt.xticks(rotation=45, ha='right')
    else:
        # Just show box plot
        axes[1].boxplot(lengths)
        axes[1].set_ylabel('Text Length (words)', fontsize=12)
        axes[1].set_title(f'{title} - Box Plot',
                          fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison(results_df: pd.DataFrame,
                          metric: str = 'f1_score',
                          title: str = "Model Comparison"):
    """
    Compare multiple models on a metric.

    Args:
        results_df: DataFrame with model results
        metric: Metric to compare
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    models = results_df['Model'].tolist()
    scores = results_df[metric].tolist()

    colors = sns.color_palette('husl', len(models))
    bars = ax.bar(range(len(models)), scores, color=colors,
                  edgecolor='black', alpha=0.7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width()/2, score, f'{score:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def save_figure(fig, filename: str, dpi: int = 300):
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution in dots per inch
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"✓ Figure saved to {filename}")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("VISUALIZATION EXAMPLES")
    print("="*80)

    # Sample data
    labels = ['positive']*50 + ['negative']*30 + ['neutral']*20

    # Example 1: Label distribution
    print("\n1. Creating label distribution plot...")
    fig1 = plot_label_distribution(labels)
    plt.savefig('/tmp/label_distribution.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved to /tmp/label_distribution.png")
    plt.close()

    # Example 2: Confusion matrix
    print("\n2. Creating confusion matrix plot...")
    cm = np.array([[45, 3, 2], [5, 23, 2], [3, 4, 13]])
    fig2 = plot_confusion_matrix(cm, ['positive', 'negative', 'neutral'])
    plt.savefig('/tmp/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved to /tmp/confusion_matrix.png")
    plt.close()

    # Example 3: Per-class metrics
    print("\n3. Creating per-class metrics plot...")
    metrics = {
        'precision': {'positive': 0.85, 'negative': 0.77, 'neutral': 0.72},
        'recall': {'positive': 0.90, 'negative': 0.77, 'neutral': 0.65},
        'f1': {'positive': 0.87, 'negative': 0.77, 'neutral': 0.68}
    }
    fig3 = plot_precision_recall_f1(metrics)
    plt.savefig('/tmp/per_class_metrics.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved to /tmp/per_class_metrics.png")
    plt.close()

    # Example 4: Text length distribution
    print("\n4. Creating text length distribution plot...")
    sample_texts = ["short text"] * 20 + ["medium length text example"] * 30 + \
                   ["this is a much longer text with many more words"] * 25
    fig4 = plot_text_length_distribution(sample_texts, labels)
    plt.savefig('/tmp/text_length_distribution.png',
                dpi=150, bbox_inches='tight')
    print("   ✓ Saved to /tmp/text_length_distribution.png")
    plt.close()

    print("\n" + "="*80)
    print("Visualization examples created successfully!")
    print("Check /tmp/ directory for generated plots")
    print("="*80)
