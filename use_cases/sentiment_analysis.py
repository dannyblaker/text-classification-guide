"""
Sentiment Analysis - Text Classification Use Case

Sentiment analysis is one of the most common text classification tasks.
It involves determining the emotional tone or opinion expressed in text.

Common Applications:
- Product review analysis
- Social media monitoring
- Customer feedback analysis
- Brand reputation management
- Market research

This module demonstrates multiple approaches:
1. Rule-based sentiment analysis
2. Traditional ML (Naive Bayes, Logistic Regression, SVM)
3. Pre-trained models (TextBlob, VADER)
4. Transfer learning with transformers
"""

from utils.visualization import plot_confusion_matrix, plot_label_distribution
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


class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis with multiple approaches.
    """

    def __init__(self, method: str = 'ml'):
        """
        Initialize sentiment analyzer.

        Args:
            method: 'rule-based', 'ml', 'textblob', 'vader', or 'transformer'
        """
        self.method = method
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_html=True,
            remove_extra_whitespace=True
        )

    def train(self, texts: List[str], labels: List[str]):
        """
        Train sentiment analysis model.

        Args:
            texts: Training texts
            labels: Training labels
        """
        if self.method == 'ml':
            self._train_ml_model(texts, labels)
        elif self.method == 'transformer':
            self._train_transformer_model(texts, labels)
        else:
            print(f"{self.method} doesn't require training")

    def _train_ml_model(self, texts: List[str], labels: List[str]):
        """Train traditional ML model."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        # Preprocess texts
        processed_texts = [self.preprocessor.clean_text(t) for t in texts]

        # Create pipeline
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])

        # Train
        print("Training ML model...")
        self.model.fit(processed_texts, labels)
        print("✓ Training complete")

    def _train_transformer_model(self, texts: List[str], labels: List[str]):
        """Train or fine-tune transformer model."""
        print("Transformer training not implemented in this demo")
        print("For production, use Hugging Face's transformers library")

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment for texts.

        Args:
            texts: List of texts

        Returns:
            List of predicted sentiments
        """
        if self.method == 'rule-based':
            return self._predict_rule_based(texts)
        elif self.method == 'textblob':
            return self._predict_textblob(texts)
        elif self.method == 'vader':
            return self._predict_vader(texts)
        elif self.method == 'ml':
            return self._predict_ml(texts)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _predict_rule_based(self, texts: List[str]) -> List[str]:
        """Simple rule-based sentiment prediction."""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'best', 'perfect', 'beautiful', 'awesome', 'brilliant',
            'outstanding', 'superb', 'terrific', 'fabulous'
        }

        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'hate',
            'disappointing', 'useless', 'waste', 'broken', 'failed', 'garbage',
            'pathetic', 'disgusting', 'annoying'
        }

        predictions = []
        for text in texts:
            words = set(text.lower().split())
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)

            if pos_count > neg_count:
                predictions.append('positive')
            elif neg_count > pos_count:
                predictions.append('negative')
            else:
                predictions.append('neutral')

        return predictions

    def _predict_textblob(self, texts: List[str]) -> List[str]:
        """Predict using TextBlob."""
        try:
            from textblob import TextBlob
        except ImportError:
            print("TextBlob not installed. Using rule-based fallback.")
            return self._predict_rule_based(texts)

        predictions = []
        for text in texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                predictions.append('positive')
            elif polarity < -0.1:
                predictions.append('negative')
            else:
                predictions.append('neutral')

        return predictions

    def _predict_vader(self, texts: List[str]) -> List[str]:
        """Predict using VADER (Valence Aware Dictionary and sEntiment Reasoner)."""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()
        except:
            print("VADER not available. Using rule-based fallback.")
            return self._predict_rule_based(texts)

        predictions = []
        for text in texts:
            scores = sia.polarity_scores(text)
            compound = scores['compound']

            if compound >= 0.05:
                predictions.append('positive')
            elif compound <= -0.05:
                predictions.append('negative')
            else:
                predictions.append('neutral')

        return predictions

    def _predict_ml(self, texts: List[str]) -> List[str]:
        """Predict using trained ML model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        processed_texts = [self.preprocessor.clean_text(t) for t in texts]
        return self.model.predict(processed_texts).tolist()


def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """Create sample sentiment dataset."""
    data = [
        # Positive examples
        ("This product is absolutely amazing! I love it!", "positive"),
        ("Excellent quality, exceeded my expectations.", "positive"),
        ("Best purchase I've made this year. Highly recommend!", "positive"),
        ("Fantastic service and great value for money.", "positive"),
        ("I'm so happy with this! Works perfectly.", "positive"),
        ("Outstanding! Everything I hoped for and more.", "positive"),
        ("Superb quality, arrived quickly, very satisfied.", "positive"),
        ("Love this! Will definitely buy again.", "positive"),
        ("Perfect! Exactly what I needed.", "positive"),
        ("Brilliant product, can't fault it at all.", "positive"),

        # Negative examples
        ("Terrible quality, broke after one day.", "negative"),
        ("Worst purchase ever. Complete waste of money.", "negative"),
        ("Very disappointed. Not as described.", "negative"),
        ("Horrible! Don't waste your time or money.", "negative"),
        ("Poor quality, would not recommend to anyone.", "negative"),
        ("Awful experience, very frustrated.", "negative"),
        ("Complete garbage. Save your money.", "negative"),
        ("Useless product, didn't work at all.", "negative"),
        ("Really bad quality, not worth the price.", "negative"),
        ("Pathetic. Regret buying this.", "negative"),

        # Neutral examples
        ("It's okay, nothing special.", "neutral"),
        ("Average product, does what it says.", "neutral"),
        ("Received the item as described.", "neutral"),
        ("It works, but could be better.", "neutral"),
        ("Decent quality for the price.", "neutral"),
        ("Neither good nor bad, just average.", "neutral"),
        ("It's fine, meets basic expectations.", "neutral"),
        ("Standard product, no complaints.", "neutral"),
        ("Acceptable quality, nothing more.", "neutral"),
        ("It's alright, what you'd expect.", "neutral"),
    ]

    texts, labels = zip(*data)
    return list(texts), list(labels)


def compare_sentiment_methods():
    """Compare different sentiment analysis approaches."""
    print("="*80)
    print("SENTIMENT ANALYSIS COMPARISON")
    print("="*80)

    # Create dataset
    texts, labels = create_sample_dataset()
    print(f"\nDataset: {len(texts)} samples")
    print(f"Labels: {set(labels)}")

    # Split data
    X_train, X_test, y_train, y_test = stratified_split(
        texts, labels, test_size=0.3)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Test different methods
    methods = ['rule-based', 'textblob', 'vader', 'ml']
    results = {}

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Testing {method.upper()} method")
        print('='*80)

        try:
            analyzer = SentimentAnalyzer(method=method)

            # Train if needed
            if method == 'ml':
                analyzer.train(X_train, y_train)

            # Predict
            predictions = analyzer.predict(X_test)

            # Evaluate
            evaluator = ClassificationEvaluator(y_test, predictions)
            metrics = evaluator.get_metrics_summary()

            print(f"\nResults for {method}:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1_score']:.3f}")

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

    # Best method
    best_method = comparison_df['f1_score'].idxmax()
    print(
        f"\n🏆 Best method: {best_method.upper()} (F1: {comparison_df.loc[best_method, 'f1_score']:.3f})")


def demo_sentiment_analysis():
    """Interactive demonstration of sentiment analysis."""
    print("="*80)
    print("SENTIMENT ANALYSIS DEMO")
    print("="*80)

    # Sample texts
    demo_texts = [
        "I absolutely love this product! It's amazing!",
        "Terrible quality, very disappointed.",
        "It's okay, nothing special.",
        "Best purchase ever! Highly recommend!",
        "Waste of money, don't buy this.",
        "Average product, does the job.",
    ]

    print("\nAnalyzing sample texts with different methods:\n")

    methods = ['rule-based', 'textblob', 'vader']

    for text in demo_texts:
        print(f"\nText: \"{text[:60]}...\"" if len(
            text) > 60 else f"\nText: \"{text}\"")
        print("-" * 80)

        for method in methods:
            try:
                analyzer = SentimentAnalyzer(method=method)
                prediction = analyzer.predict([text])[0]
                print(f"  {method.ljust(12)}: {prediction.upper()}")
            except:
                print(f"  {method.ljust(12)}: [Not available]")


if __name__ == "__main__":
    # Run demonstration
    demo_sentiment_analysis()

    print("\n" + "="*80)
    print("Running comprehensive comparison...")
    print("="*80)

    # Compare methods
    compare_sentiment_methods()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. RULE-BASED:
   - Pros: Simple, fast, interpretable
   - Cons: Limited coverage, misses context
   - Use when: Simple keywords work well

2. TEXTBLOB:
   - Pros: Easy to use, handles some context
   - Cons: Less accurate than specialized models
   - Use when: Quick prototyping needed

3. VADER:
   - Pros: Good for social media, handles intensifiers
   - Cons: May not work well for formal text
   - Use when: Analyzing social media or informal text

4. MACHINE LEARNING:
   - Pros: Learns from data, customizable
   - Cons: Requires training data
   - Use when: Have labeled data, need customization

5. TRANSFORMERS (not shown):
   - Pros: State-of-the-art performance
   - Cons: Slower, requires more resources
   - Use when: Accuracy is critical, resources available
    """)
