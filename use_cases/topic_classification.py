"""
Topic Classification - Text Classification Use Case

Topic classification assigns documents to predefined categories based on their content.
It's one of the fundamental tasks in information retrieval and organization.

Common Applications:
- News article categorization
- Document management systems
- Content recommendation
- Academic paper classification
- Customer support ticket routing
- Product categorization

Key Challenges:
- Documents may belong to multiple topics
- Topic boundaries can be fuzzy
- Need sufficient training data per topic
- Handling imbalanced topic distributions

This module demonstrates:
1. Multi-class topic classification
2. Feature extraction techniques
3. Different classification algorithms
4. Handling imbalanced data
"""

from utils.visualization import plot_label_distribution
from utils.evaluation import ClassificationEvaluator, stratified_split
from utils.preprocessing import TextPreprocessor
import warnings
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


warnings.filterwarnings('ignore')


class TopicClassifier:
    """
    Multi-class topic classification system.
    """

    def __init__(self, method: str = 'nb'):
        """
        Initialize topic classifier.

        Args:
            method: 'nb' (Naive Bayes), 'svm', 'rf' (Random Forest), or 'mlp' (Neural Network)
        """
        self.method = method
        self.model = None
        self.label_encoder = None
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_urls=True,
            remove_html=True,
            remove_stopwords=True,
            remove_extra_whitespace=True
        )

    def train(self, texts: List[str], topics: List[str]):
        """
        Train topic classification model.

        Args:
            texts: Training texts
            topics: Training topic labels
        """
        print(f"Training {self.method.upper()} topic classifier...")

        # Preprocess texts
        processed_texts = [self.preprocessor.clean_text(t) for t in texts]

        # Create model based on method
        if self.method == 'nb':
            self._train_naive_bayes(processed_texts, topics)
        elif self.method == 'svm':
            self._train_svm(processed_texts, topics)
        elif self.method == 'rf':
            self._train_random_forest(processed_texts, topics)
        elif self.method == 'mlp':
            self._train_neural_network(processed_texts, topics)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        print("✓ Training complete")

    def _train_naive_bayes(self, texts: List[str], topics: List[str]):
        """Train Multinomial Naive Bayes classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])

        self.model.fit(texts, topics)

    def _train_svm(self, texts: List[str], topics: List[str]):
        """Train Support Vector Machine classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2
            )),
            ('classifier', LinearSVC(C=1.0, max_iter=1000))
        ])

        self.model.fit(texts, topics)

    def _train_random_forest(self, texts: List[str], topics: List[str]):
        """Train Random Forest classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            ))
        ])

        self.model.fit(texts, topics)

    def _train_neural_network(self, texts: List[str], topics: List[str]):
        """Train Multi-layer Perceptron classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2)
            )),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ))
        ])

        self.model.fit(texts, topics)

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict topics for texts.

        Args:
            texts: List of texts

        Returns:
            List of predicted topics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Preprocess
        processed_texts = [self.preprocessor.clean_text(t) for t in texts]

        # Predict
        predictions = self.model.predict(processed_texts)

        return predictions.tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict topic probabilities.

        Args:
            texts: List of texts

        Returns:
            Array of probabilities for each topic
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Check if model supports probability prediction
        if not hasattr(self.model.named_steps['classifier'], 'predict_proba'):
            raise ValueError(
                f"{self.method} doesn't support probability prediction")

        processed_texts = [self.preprocessor.clean_text(t) for t in texts]
        return self.model.predict_proba(processed_texts)

    def get_top_features_per_topic(self, n_features: int = 10) -> Dict[str, List[str]]:
        """
        Get most important features (words) for each topic.

        Args:
            n_features: Number of top features to return

        Returns:
            Dictionary mapping topics to their top features
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        vectorizer = self.model.named_steps['vectorizer']
        classifier = self.model.named_steps['classifier']

        feature_names = vectorizer.get_feature_names_out()

        # Get feature importance based on model type
        if hasattr(classifier, 'coef_'):
            # Linear models (SVM, Naive Bayes with class log probabilities)
            top_features = {}
            for idx, topic in enumerate(classifier.classes_):
                if len(classifier.coef_.shape) > 1:
                    coefficients = classifier.coef_[idx]
                else:
                    coefficients = classifier.coef_
                top_indices = coefficients.argsort()[-n_features:][::-1]
                top_features[topic] = [feature_names[i] for i in top_indices]
        elif hasattr(classifier, 'feature_importances_'):
            # Tree-based models
            importances = classifier.feature_importances_
            top_indices = importances.argsort()[-n_features:][::-1]
            # Same features for all topics in tree models
            top_features = {topic: [feature_names[i] for i in top_indices]
                            for topic in classifier.classes_}
        else:
            top_features = {}

        return top_features


def create_topic_dataset() -> Tuple[List[str], List[str]]:
    """Create sample topic classification dataset."""
    data = [
        # Technology
        ("New smartphone features impressive AI capabilities and longer battery life.", "technology"),
        ("Software update introduces bug fixes and performance improvements.", "technology"),
        ("Cloud computing adoption accelerates in enterprise environments.", "technology"),
        ("Latest laptop models feature improved processors and graphics.", "technology"),
        ("Cybersecurity threats evolve as hackers develop new techniques.", "technology"),
        ("Virtual reality headsets become more affordable and accessible.", "technology"),
        ("Machine learning algorithms improve recommendation systems.", "technology"),
        ("5G networks expand coverage across major cities worldwide.", "technology"),

        # Sports
        ("Local team wins championship in thrilling overtime match.", "sports"),
        ("Star athlete signs record-breaking contract with new team.", "sports"),
        ("Olympic games preparation intensifies as event approaches.", "sports"),
        ("Tennis tournament sees unexpected upsets in early rounds.", "sports"),
        ("Basketball playoffs feature intense rivalry matchups.", "sports"),
        ("Soccer world cup qualifier results surprise fans.", "sports"),
        ("Marathon runners prepare for upcoming race season.", "sports"),
        ("Baseball team clinches division title with strong finish.", "sports"),

        # Business
        ("Stock market reaches new highs amid economic optimism.", "business"),
        ("Company announces quarterly earnings exceeding expectations.", "business"),
        ("Merger between tech giants approved by regulatory authorities.", "business"),
        ("Startup secures major funding round from venture capitalists.", "business"),
        ("Retail sales show strong growth during holiday season.", "business"),
        ("Central bank adjusts interest rates to control inflation.", "business"),
        ("International trade negotiations yield positive results.", "business"),
        ("Corporate restructuring aims to improve operational efficiency.", "business"),

        # Health
        ("New medical treatment shows promising results in clinical trials.", "health"),
        ("Health officials recommend updated vaccination guidelines.", "health"),
        ("Research reveals connection between diet and disease prevention.", "health"),
        ("Mental health awareness programs expand in communities.", "health"),
        ("Fitness trends emphasize holistic wellness approaches.", "health"),
        ("Medical breakthroughs offer hope for rare disease patients.", "health"),
        ("Public health campaign promotes healthy lifestyle choices.", "health"),
        ("Hospital implements innovative patient care protocols.", "health"),

        # Politics
        ("Election results show shift in voter preferences.", "politics"),
        ("Government announces new policy initiatives for reform.", "politics"),
        ("Diplomatic talks aim to resolve international tensions.", "politics"),
        ("Legislative debate continues on proposed bill.", "politics"),
        ("Political leaders meet to discuss climate change strategies.", "politics"),
        ("Campaign season begins with candidates announcing platforms.", "politics"),
        ("Supreme court ruling impacts future legal precedents.", "politics"),
        ("International summit addresses global security concerns.", "politics"),

        # Entertainment
        ("New movie breaks box office records opening weekend.", "entertainment"),
        ("Music festival announces impressive lineup of performers.", "entertainment"),
        ("Television series finale draws record viewership.", "entertainment"),
        ("Award ceremony honors outstanding artistic achievements.", "entertainment"),
        ("Celebrity couple announces engagement to fans.", "entertainment"),
        ("Streaming platform releases highly anticipated original series.", "entertainment"),
        ("Concert tour sells out within minutes of tickets going on sale.", "entertainment"),
        ("Film festival showcases independent cinema from around world.", "entertainment"),
    ]

    texts, topics = zip(*data)
    return list(texts), list(topics)


def compare_topic_classifiers():
    """Compare different topic classification methods."""
    print("="*80)
    print("TOPIC CLASSIFICATION COMPARISON")
    print("="*80)

    # Create dataset
    texts, topics = create_topic_dataset()
    print(f"\nDataset: {len(texts)} samples")
    print(f"Topics: {sorted(set(topics))}")

    # Show topic distribution
    topic_counts = pd.Series(topics).value_counts()
    print("\nTopic distribution:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count}")

    # Split data
    X_train, X_test, y_train, y_test = stratified_split(
        texts, topics, test_size=0.25)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Test different methods
    methods = ['nb', 'svm', 'rf']
    results = {}

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Testing {method.upper()} method")
        print('='*80)

        try:
            classifier = TopicClassifier(method=method)
            classifier.train(X_train, y_train)

            # Predict
            predictions = classifier.predict(X_test)

            # Evaluate
            evaluator = ClassificationEvaluator(y_test, predictions)
            metrics = evaluator.get_metrics_summary()

            print(f"\nResults for {method}:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1_score']:.3f}")

            # Show per-class metrics
            print("\nPer-topic performance:")
            per_class = evaluator.per_class_metrics()
            print(per_class.to_string())

            results[method] = metrics

        except Exception as e:
            print(f"  ⚠ Error with {method}: {e}")
            continue

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[[
        'accuracy', 'precision', 'recall', 'f1_score']]
    print(comparison_df.to_string())


def demo_topic_classification():
    """Interactive demonstration of topic classification."""
    print("="*80)
    print("TOPIC CLASSIFICATION DEMO")
    print("="*80)

    # Create and train model
    texts, topics = create_topic_dataset()
    classifier = TopicClassifier(method='nb')
    classifier.train(texts, topics)

    # Test texts
    test_texts = [
        "Scientists discover new treatment for cancer using AI technology.",
        "Soccer team wins championship after dramatic penalty shootout.",
        "Stock prices surge following positive economic indicators.",
        "New smartphone release features revolutionary camera system.",
        "Government passes legislation on healthcare reform.",
        "Latest blockbuster movie earns millions in opening weekend.",
    ]

    print("\nClassifying sample texts:\n")

    predictions = classifier.predict(test_texts)

    for text, prediction in zip(test_texts, predictions):
        print(f"Text: \"{text[:70]}...\"" if len(
            text) > 70 else f"Text: \"{text}\"")
        print(f"Predicted Topic: {prediction.upper()}")
        print("-" * 80)

    # Show top features for each topic
    print("\n" + "="*80)
    print("TOP WORDS FOR EACH TOPIC")
    print("="*80)

    try:
        top_features = classifier.get_top_features_per_topic(n_features=5)
        for topic, features in sorted(top_features.items()):
            print(f"\n{topic.upper()}:")
            print(f"  {', '.join(features)}")
    except:
        print("Feature importance not available for this model type")


if __name__ == "__main__":
    # Run demonstration
    demo_topic_classification()

    print("\n" + "="*80)
    print("Running comprehensive comparison...")
    print("="*80)

    # Compare methods
    compare_topic_classifiers()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - TOPIC CLASSIFICATION")
    print("="*80)
    print("""
CHOOSING THE RIGHT ALGORITHM:

1. NAIVE BAYES:
   - Pros: Fast, works well with small datasets, probabilistic
   - Cons: Assumes feature independence
   - Use when: Fast baseline needed, limited training data

2. SUPPORT VECTOR MACHINES (SVM):
   - Pros: Effective in high dimensions, memory efficient
   - Cons: Doesn't scale well, no probability estimates
   - Use when: Accuracy is priority, dataset not too large

3. RANDOM FOREST:
   - Pros: Handles non-linear patterns, feature importance
   - Cons: Slower, requires more memory
   - Use when: Complex patterns, need interpretability

4. NEURAL NETWORKS:
   - Pros: Can learn complex patterns, scalable
   - Cons: Needs more data, harder to interpret
   - Use when: Large dataset, computational resources available

BEST PRACTICES:

1. Data Preprocessing:
   - Remove stopwords for topic classification
   - Use TF-IDF instead of simple word counts
   - Consider bigrams for better context

2. Feature Engineering:
   - Experiment with different n-gram ranges
   - Try different max_features values
   - Consider domain-specific features

3. Handling Imbalanced Data:
   - Use stratified splitting
   - Consider SMOTE or class weights
   - Evaluate using F1-score, not just accuracy

4. Model Selection:
   - Start with Naive Bayes as baseline
   - Try SVM for better performance
   - Use cross-validation for reliable estimates

5. Evaluation:
   - Look at per-class metrics
   - Analyze confusion matrix
   - Check for topic confusion patterns
    """)
