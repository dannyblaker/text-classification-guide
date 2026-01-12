"""
Active Learning for Text Classification

Active learning is a semi-supervised machine learning approach where the algorithm
can interactively query a user (or oracle) to label new data points.

Key Concept:
Instead of randomly selecting examples to label, active learning chooses the
most informative examples - those that would improve the model the most.

Benefits:
- Reduces labeling effort by 50-90%
- Focuses human effort on difficult cases
- Achieves good performance with less data
- Ideal when labeling is expensive

Common Query Strategies:
1. Uncertainty Sampling: Select examples the model is least confident about
2. Query-by-Committee: Use ensemble disagreement
3. Expected Model Change: Select examples that would change model most
4. Diversity Sampling: Ensure diverse representation

This module demonstrates:
- Uncertainty sampling implementation
- Active learning loop
- Performance tracking
- Stopping criteria
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Callable
from collections import defaultdict
from utils.preprocessing import TextPreprocessor
from utils.evaluation import ClassificationEvaluator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ActiveLearner:
    """
    Active learning system for text classification.

    This implements uncertainty sampling, where the model queries
    the examples it's least confident about.
    """

    def __init__(self,
                 initial_model,
                 query_strategy: str = 'uncertainty',
                 batch_size: int = 1):
        """
        Initialize active learner.

        Args:
            initial_model: sklearn-compatible model
            query_strategy: 'uncertainty', 'margin', or 'entropy'
            batch_size: Number of examples to query per iteration
        """
        self.model = initial_model
        self.query_strategy = query_strategy
        self.batch_size = batch_size
        self.history = []

    def uncertainty_sampling(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Select examples with lowest prediction confidence.

        Returns indices of least confident predictions.

        Args:
            probabilities: Prediction probabilities (n_samples, n_classes)

        Returns:
            Array of indices sorted by uncertainty (most uncertain first)
        """
        # Get maximum probability for each example
        max_probs = np.max(probabilities, axis=1)

        # Lower probability = higher uncertainty
        uncertainty = 1 - max_probs

        # Return indices sorted by uncertainty (descending)
        return np.argsort(uncertainty)[::-1]

    def margin_sampling(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Select examples with smallest margin between top two predictions.

        Args:
            probabilities: Prediction probabilities

        Returns:
            Array of indices sorted by margin (smallest first)
        """
        # Sort probabilities for each example
        sorted_probs = np.sort(probabilities, axis=1)

        # Margin is difference between top two
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Return indices sorted by margin (ascending)
        return np.argsort(margin)

    def entropy_sampling(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Select examples with highest prediction entropy.

        Args:
            probabilities: Prediction probabilities

        Returns:
            Array of indices sorted by entropy (highest first)
        """
        # Calculate entropy for each example
        # Entropy = -sum(p * log(p))
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(probabilities *
                          np.log(probabilities + epsilon), axis=1)

        # Return indices sorted by entropy (descending)
        return np.argsort(entropy)[::-1]

    def query(self, X_pool: List[str], n_instances: int = None) -> List[int]:
        """
        Query the most informative instances from unlabeled pool.

        Args:
            X_pool: Unlabeled data pool
            n_instances: Number of instances to query (default: batch_size)

        Returns:
            Indices of instances to query
        """
        if n_instances is None:
            n_instances = self.batch_size

        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_pool)

        # Select based on strategy
        if self.query_strategy == 'uncertainty':
            indices = self.uncertainty_sampling(probabilities)
        elif self.query_strategy == 'margin':
            indices = self.margin_sampling(probabilities)
        elif self.query_strategy == 'entropy':
            indices = self.entropy_sampling(probabilities)
        else:
            raise ValueError(f"Unknown strategy: {self.query_strategy}")

        # Return top n
        return indices[:n_instances].tolist()

    def teach(self, X: List[str], y: List[str]):
        """
        Train/update model with new labeled examples.

        Args:
            X: Training texts
            y: Training labels
        """
        self.model.fit(X, y)

    def evaluate(self, X_test: List[str], y_test: List[str]) -> dict:
        """
        Evaluate current model performance.

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            Dictionary with metrics
        """
        predictions = self.model.predict(X_test)
        evaluator = ClassificationEvaluator(y_test, predictions)
        return evaluator.get_metrics_summary()


def simulate_active_learning(
    X_pool: List[str],
    y_pool: List[str],
    X_test: List[str],
    y_test: List[str],
    initial_size: int = 10,
    query_size: int = 5,
    max_iterations: int = 20,
    target_accuracy: float = 0.9
) -> dict:
    """
    Simulate active learning process.

    Args:
        X_pool: Unlabeled data pool
        y_pool: True labels (oracle)
        X_test: Test data
        y_test: Test labels
        initial_size: Initial training set size
        query_size: Number of examples to query per iteration
        max_iterations: Maximum iterations
        target_accuracy: Stop when this accuracy is reached

    Returns:
        Dictionary with learning history
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    print("="*80)
    print("ACTIVE LEARNING SIMULATION")
    print("="*80)

    # Create model
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=3000)),
        ('classifier', MultinomialNB())
    ])

    # Initialize with random samples
    np.random.seed(42)
    initial_indices = np.random.choice(
        len(X_pool), size=initial_size, replace=False)

    X_train = [X_pool[i] for i in initial_indices]
    y_train = [y_pool[i] for i in initial_indices]

    # Remove from pool
    remaining_indices = [i for i in range(
        len(X_pool)) if i not in initial_indices]
    X_remaining = [X_pool[i] for i in remaining_indices]
    y_remaining = [y_pool[i] for i in remaining_indices]

    # Create active learner
    learner = ActiveLearner(
        model, query_strategy='uncertainty', batch_size=query_size)

    # Train initial model
    learner.teach(X_train, y_train)

    # Track progress
    history = {
        'iteration': [0],
        'training_size': [len(X_train)],
        'accuracy': [],
        'f1_score': []
    }

    # Initial evaluation
    metrics = learner.evaluate(X_test, y_test)
    history['accuracy'].append(metrics['accuracy'])
    history['f1_score'].append(metrics['f1_score'])

    print(f"\nIteration 0: {len(X_train)} training samples")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")

    # Active learning loop
    for iteration in range(1, max_iterations + 1):
        # Check stopping criteria
        if metrics['accuracy'] >= target_accuracy:
            print(f"\n✓ Reached target accuracy of {target_accuracy:.3f}")
            break

        if len(X_remaining) < query_size:
            print(f"\n✓ Exhausted data pool")
            break

        # Query most informative examples
        query_indices = learner.query(X_remaining, n_instances=query_size)

        # Get labels from oracle (simulated)
        for idx in sorted(query_indices, reverse=True):
            X_train.append(X_remaining[idx])
            y_train.append(y_remaining[idx])
            X_remaining.pop(idx)
            y_remaining.pop(idx)

        # Retrain model
        learner.teach(X_train, y_train)

        # Evaluate
        metrics = learner.evaluate(X_test, y_test)

        # Record history
        history['iteration'].append(iteration)
        history['training_size'].append(len(X_train))
        history['accuracy'].append(metrics['accuracy'])
        history['f1_score'].append(metrics['f1_score'])

        print(f"\nIteration {iteration}: {len(X_train)} training samples")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(
            f"  Improvement: {(metrics['accuracy'] - history['accuracy'][-2]):.3f}")

    return history


def compare_active_vs_random():
    """Compare active learning vs random sampling."""
    from use_cases.sentiment_analysis import create_sample_dataset
    from utils.evaluation import stratified_split

    print("="*80)
    print("ACTIVE LEARNING vs RANDOM SAMPLING")
    print("="*80)

    # Get data
    texts, labels = create_sample_dataset()
    X_pool, X_test, y_pool, y_test = stratified_split(
        texts, labels, test_size=0.3)

    # Run active learning
    print("\n" + "="*80)
    print("ACTIVE LEARNING (Uncertainty Sampling)")
    print("="*80)
    active_history = simulate_active_learning(
        X_pool, y_pool, X_test, y_test,
        initial_size=5,
        query_size=3,
        max_iterations=10,
        target_accuracy=0.85
    )

    # Run random sampling for comparison
    print("\n" + "="*80)
    print("RANDOM SAMPLING (Baseline)")
    print("="*80)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=3000)),
        ('classifier', MultinomialNB())
    ])

    random_history = {
        'iteration': [],
        'training_size': [],
        'accuracy': [],
        'f1_score': []
    }

    # Simulate random sampling with same schedule as active learning
    for iteration, train_size in enumerate(active_history['training_size']):
        # Randomly sample train_size examples
        indices = np.random.choice(len(X_pool), size=train_size, replace=False)
        X_train = [X_pool[i] for i in indices]
        y_train = [y_pool[i] for i in indices]

        # Train and evaluate
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        evaluator = ClassificationEvaluator(y_test, predictions)
        metrics = evaluator.get_metrics_summary()

        random_history['iteration'].append(iteration)
        random_history['training_size'].append(train_size)
        random_history['accuracy'].append(metrics['accuracy'])
        random_history['f1_score'].append(metrics['f1_score'])

        print(
            f"Iteration {iteration}: {train_size} samples - Accuracy: {metrics['accuracy']:.3f}")

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    comparison_df = pd.DataFrame({
        'Training Size': active_history['training_size'],
        'Active Learning': active_history['accuracy'],
        'Random Sampling': random_history['accuracy']
    })

    print(comparison_df.to_string(index=False))

    # Calculate improvement
    final_active = active_history['accuracy'][-1]
    final_random = random_history['accuracy'][-1]
    improvement = final_active - final_random

    print(f"\n🏆 Active Learning Final Accuracy: {final_active:.3f}")
    print(f"📊 Random Sampling Final Accuracy: {final_random:.3f}")
    print(
        f"✨ Improvement: {improvement:.3f} ({improvement/final_random*100:.1f}%)")


if __name__ == "__main__":
    compare_active_vs_random()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - ACTIVE LEARNING")
    print("="*80)
    print("""
WHEN TO USE ACTIVE LEARNING:

1. Labeling is Expensive:
   - Expert annotation required
   - Time-consuming labeling process
   - Limited labeling budget

2. Large Unlabeled Dataset:
   - Lots of unlabeled data available
   - Want to select best examples to label
   - Can't afford to label everything

3. Iterative Development:
   - Building model incrementally
   - Can wait for labels between iterations
   - Want to maximize label efficiency

QUERY STRATEGIES:

1. Uncertainty Sampling:
   - Simplest approach
   - Queries least confident predictions
   - Works well in practice

2. Margin Sampling:
   - Queries examples near decision boundary
   - Good for binary classification
   - Considers top two classes

3. Entropy Sampling:
   - Measures prediction uncertainty
   - Good for multi-class problems
   - More computationally expensive

BEST PRACTICES:

1. Start Small:
   - Begin with diverse initial set
   - Use stratified sampling if possible
   - Aim for 5-10 examples per class

2. Batch Queries:
   - Query multiple examples at once
   - Reduces iteration overhead
   - Typical batch size: 5-10

3. Diversity:
   - Don't just focus on uncertainty
   - Ensure diverse representation
   - Consider clustering for diversity

4. Stopping Criteria:
   - Set target performance threshold
   - Monitor improvement rate
   - Stop if plateauing

5. Practical Considerations:
   - Balance query frequency with labeler availability
   - Consider labeler fatigue
   - Provide clear labeling guidelines

EXPECTED BENEFITS:

- 50-70% reduction in labeling effort
- Faster convergence to target performance
- Better use of expert time
- More efficient model development

LIMITATIONS:

- Requires probability estimates
- May miss rare classes
- Can be biased toward edge cases
- Needs iterative label collection
    """)
