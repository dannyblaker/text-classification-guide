"""
Spam Detection - Text Classification Use Case

Spam detection is a critical application of text classification.
It involves identifying unwanted, malicious, or unsolicited messages.

Common Applications:
- Email filtering
- SMS filtering
- Comment moderation
- Social media spam detection
- Fraud prevention

Key Characteristics of Spam:
- Suspicious keywords (free, winner, urgent, click here)
- Excessive punctuation and capitalization
- URLs and shortened links
- Requests for personal information
- Poor grammar and spelling
- Urgency and pressure tactics

This module demonstrates:
1. Feature engineering for spam detection
2. Rule-based approaches
3. Traditional ML classifiers
4. Ensemble methods
"""

from utils.visualization import plot_confusion_matrix
from utils.evaluation import ClassificationEvaluator, stratified_split
from utils.preprocessing import TextPreprocessor
import warnings
import re
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


warnings.filterwarnings('ignore')


class SpamDetector:
    """
    Comprehensive spam detection system.
    """

    def __init__(self, method: str = 'ml'):
        """
        Initialize spam detector.

        Args:
            method: 'rule-based', 'ml', or 'ensemble'
        """
        self.method = method
        self.model = None
        self.spam_keywords = self._load_spam_keywords()

    def _load_spam_keywords(self) -> set:
        """Load common spam keywords."""
        return {
            # Money related
            'free', 'cash', 'money', 'prize', 'winner', 'billion', 'million',
            'dollar', 'claim', 'reward', 'bonus', 'income', 'profit', 'earn',

            # Urgency
            'urgent', 'immediate', 'act now', 'limited time', 'expires',
            'hurry', 'offer ends', 'today only', 'now',

            # Actions
            'click here', 'click now', 'click below', 'subscribe', 'unsubscribe',
            'download', 'install', 'verify', 'confirm', 'update',

            # Scam indicators
            'congratulations', 'selected', 'guaranteed', 'risk-free', 'no obligation',
            'as seen on', 'certified', 'official', 'authorized',

            # Personal info requests
            'password', 'credit card', 'bank account', 'social security',
            'ssn', 'pin', 'verification code',
        }

    def extract_spam_features(self, text: str) -> Dict[str, float]:
        """
        Extract spam-indicative features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        text_lower = text.lower()

        features = {
            # Keyword features
            'spam_keyword_count': sum(1 for kw in self.spam_keywords if kw in text_lower),
            'has_free': 1 if 'free' in text_lower else 0,
            'has_click': 1 if 'click' in text_lower else 0,
            'has_urgent': 1 if 'urgent' in text_lower else 0,
            'has_winner': 1 if any(w in text_lower for w in ['winner', 'won', 'prize']) else 0,

            # Character features
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),

            # URL features
            'url_count': len(re.findall(r'http[s]?://\S+|www\.\S+', text)),
            'has_short_url': 1 if any(short in text_lower for short in ['bit.ly', 'tinyurl', 'goo.gl']) else 0,

            # Special characters
            'special_char_count': len(re.findall(r'[$€£¥₹]', text)),
            'excessive_punctuation': 1 if re.search(r'[!?]{2,}', text) else 0,

            # Length features
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,

            # Email/phone patterns
            'has_email': 1 if re.search(r'\S+@\S+', text) else 0,
            'has_phone': 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text) else 0,
        }

        return features

    def train(self, texts: List[str], labels: List[str]):
        """
        Train spam detection model.

        Args:
            texts: Training texts
            labels: Training labels ('spam' or 'not_spam')
        """
        if self.method == 'ml':
            self._train_ml_model(texts, labels)
        elif self.method == 'ensemble':
            self._train_ensemble_model(texts, labels)
        else:
            print(f"{self.method} doesn't require training")

    def _train_ml_model(self, texts: List[str], labels: List[str]):
        """Train ML model with text vectorization."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.preprocessing import FunctionTransformer

        print("Training spam detection model...")

        # Create pipeline with both text and custom features
        self.model = Pipeline([
            ('features', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=1000, C=1.0))
        ])

        self.model.fit(texts, labels)
        print("✓ Training complete")

    def _train_ensemble_model(self, texts: List[str], labels: List[str]):
        """Train ensemble of multiple classifiers."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        print("Training ensemble model...")

        # Create base vectorizer
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)

        # Create ensemble
        clf1 = LogisticRegression(max_iter=1000)
        clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf3 = MultinomialNB()

        ensemble = VotingClassifier(
            estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3)],
            voting='soft'
        )

        ensemble.fit(X, labels)

        self.model = Pipeline([
            ('vectorizer', vectorizer),
            ('ensemble', ensemble)
        ])

        print("✓ Ensemble training complete")

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict spam/not_spam for texts.

        Args:
            texts: List of texts

        Returns:
            List of predictions
        """
        if self.method == 'rule-based':
            return self._predict_rule_based(texts)
        elif self.method in ['ml', 'ensemble']:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            return self.model.predict(texts).tolist()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _predict_rule_based(self, texts: List[str]) -> List[str]:
        """Rule-based spam prediction."""
        predictions = []

        for text in texts:
            features = self.extract_spam_features(text)

            # Scoring system
            spam_score = 0

            # Strong spam indicators
            if features['spam_keyword_count'] >= 3:
                spam_score += 3
            if features['excessive_punctuation']:
                spam_score += 2
            if features['url_count'] >= 2:
                spam_score += 2
            if features['has_short_url']:
                spam_score += 2
            if features['uppercase_ratio'] > 0.3:
                spam_score += 2

            # Moderate indicators
            if features['has_free']:
                spam_score += 1
            if features['has_urgent']:
                spam_score += 1
            if features['has_winner']:
                spam_score += 1
            if features['special_char_count'] > 2:
                spam_score += 1

            # Threshold
            if spam_score >= 4:
                predictions.append('spam')
            else:
                predictions.append('not_spam')

        return predictions

    def explain_prediction(self, text: str) -> Dict:
        """
        Explain why a text was classified as spam or not.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction and explanation
        """
        features = self.extract_spam_features(text)
        prediction = self.predict([text])[0]

        # Identify key factors
        spam_indicators = []

        if features['spam_keyword_count'] > 0:
            spam_indicators.append(
                f"{features['spam_keyword_count']} spam keywords found")
        if features['excessive_punctuation']:
            spam_indicators.append("Excessive punctuation")
        if features['url_count'] > 0:
            spam_indicators.append(f"{features['url_count']} URLs found")
        if features['uppercase_ratio'] > 0.3:
            spam_indicators.append(
                f"High uppercase ratio: {features['uppercase_ratio']:.1%}")
        if features['has_short_url']:
            spam_indicators.append("Contains shortened URL")

        return {
            'text': text[:100],
            'prediction': prediction,
            'spam_indicators': spam_indicators,
            'features': features
        }


def create_spam_dataset() -> Tuple[List[str], List[str]]:
    """Create sample spam detection dataset."""
    data = [
        # Spam examples
        ("URGENT! You've WON $1,000,000!!! Click here NOW to claim: http://bit.ly/fake", "spam"),
        ("Congratulations! You have been selected for a FREE iPhone! Act now!", "spam"),
        ("CLICK HERE for amazing discounts! Limited time offer! Don't miss out!!!", "spam"),
        ("Get rich FAST! Make $5000/day working from home. No experience needed!", "spam"),
        ("Your account will be SUSPENDED unless you verify now: http://fake-bank.com", "spam"),
        ("FREE MONEY! Click now to claim your cash prize! Winner Winner!", "spam"),
        ("Urgent: Update your password immediately by clicking here", "spam"),
        ("You've been approved for a $50,000 loan! No credit check! Apply now!", "spam"),
        ("LOSE WEIGHT FAST!!! Buy now and get 50% OFF! Limited stock!!!", "spam"),
        ("Claim your FREE gift card now! Click here before it expires!", "spam"),
        ("!!!WINNER!!! You won the lottery! Send us your bank details to claim!", "spam"),
        ("Make MILLIONS working from home! 100% guaranteed! No risk!", "spam"),
        ("URGENT: Your package is waiting! Click to track: http://fake-ups.com", "spam"),
        ("Free trial! No obligation! Cancel anytime! Click now!!!", "spam"),
        ("You've been selected for a special offer! Act fast! Limited time!", "spam"),

        # Not spam examples
        ("Hi, I'll be 10 minutes late for our meeting today.", "not_spam"),
        ("Thanks for your email. I've reviewed the document and will send feedback tomorrow.", "not_spam"),
        ("Meeting scheduled for Thursday at 2 PM in Conference Room B.", "not_spam"),
        ("Could you please send me the quarterly report when you have a chance?", "not_spam"),
        ("The project deadline has been extended to next Friday.", "not_spam"),
        ("Great presentation today! Let's discuss the next steps on Monday.", "not_spam"),
        ("Your order #12345 has been shipped and will arrive in 3-5 business days.", "not_spam"),
        ("Reminder: Team lunch tomorrow at noon at the Italian restaurant.", "not_spam"),
        ("Please find attached the invoice for last month's services.", "not_spam"),
        ("Thank you for your purchase. Your receipt is attached.", "not_spam"),
        ("The meeting notes from yesterday are now available on the shared drive.", "not_spam"),
        ("Can we reschedule our call to Wednesday afternoon?", "not_spam"),
        ("I've completed the analysis. The results look promising.", "not_spam"),
        ("Your subscription will renew next month. No action needed.", "not_spam"),
        ("Looking forward to seeing you at the conference next week.", "not_spam"),
    ]

    texts, labels = zip(*data)
    return list(texts), list(labels)


def demo_spam_detection():
    """Demonstrate spam detection with different methods."""
    print("="*80)
    print("SPAM DETECTION DEMO")
    print("="*80)

    # Create dataset
    texts, labels = create_spam_dataset()
    print(f"\nDataset: {len(texts)} samples")

    # Count spam vs not spam
    spam_count = labels.count('spam')
    not_spam_count = labels.count('not_spam')
    print(f"Spam: {spam_count}, Not Spam: {not_spam_count}")

    # Split data
    X_train, X_test, y_train, y_test = stratified_split(
        texts, labels, test_size=0.3)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Test different methods
    methods = ['rule-based', 'ml']
    results = {}

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Testing {method.upper()} method")
        print('='*80)

        detector = SpamDetector(method=method)

        # Train if needed
        if method != 'rule-based':
            detector.train(X_train, y_train)

        # Predict
        predictions = detector.predict(X_test)

        # Evaluate
        evaluator = ClassificationEvaluator(y_test, predictions)
        evaluator.print_evaluation_report()

        results[method] = evaluator.get_metrics_summary()

    # Show example explanations
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS WITH EXPLANATIONS")
    print("="*80)

    test_texts = [
        "URGENT! You've WON! Click here NOW!!!",
        "Hi, let's meet for coffee tomorrow.",
        "FREE MONEY! Limited time offer!",
    ]

    detector = SpamDetector(method='rule-based')

    for text in test_texts:
        print(f"\nText: \"{text}\"")
        print("-" * 80)
        explanation = detector.explain_prediction(text)
        print(f"Prediction: {explanation['prediction'].upper()}")
        if explanation['spam_indicators']:
            print("Spam indicators:")
            for indicator in explanation['spam_indicators']:
                print(f"  • {indicator}")
        else:
            print("No significant spam indicators found")


if __name__ == "__main__":
    demo_spam_detection()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - SPAM DETECTION")
    print("="*80)
    print("""
IMPORTANT FEATURES FOR SPAM DETECTION:

1. Keyword-based Features:
   - Common spam words (free, winner, urgent, click)
   - Financial terms (money, cash, prize)
   - Action words (click, download, verify)

2. Structural Features:
   - Excessive punctuation (!!!, ???)
   - High capitalization ratio
   - Presence of URLs, especially shortened ones
   - Special characters ($, €, £)

3. Content Features:
   - Requests for personal information
   - Unrealistic promises
   - Pressure and urgency tactics
   - Poor grammar and spelling

4. Metadata (when available):
   - Sender reputation
   - Time of day
   - Volume patterns

BEST PRACTICES:

1. Balance Precision and Recall:
   - High precision: Avoid false positives (legitimate emails marked as spam)
   - High recall: Catch most spam
   - Usually prioritize precision for email, recall for comments

2. Regular Updates:
   - Spammers constantly evolve tactics
   - Update keyword lists regularly
   - Retrain models with new data

3. Multi-layer Defense:
   - Combine rule-based and ML approaches
   - Use ensemble methods
   - Implement feedback mechanisms

4. User Feedback:
   - Allow users to report spam/not spam
   - Use feedback to improve model
   - Monitor false positive rate
    """)
