"""
Automatic Text Labeling Methods

This module demonstrates various automatic and semi-automatic labeling techniques.
These methods can significantly reduce manual labeling effort while maintaining
reasonable quality, especially useful for:
- Large-scale datasets
- Initial data exploration
- Bootstrapping training sets
- Weak supervision scenarios

Techniques covered:
1. Rule-based labeling
2. Keyword matching
3. Pattern-based classification
4. Weak supervision with labeling functions
5. Zero-shot classification with pre-trained models
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Callable, Optional, Tuple
from collections import defaultdict
import warnings


class RuleBasedLabeler:
    """
    Rule-based text labeling using keywords and patterns.

    This is the simplest form of automatic labeling. Works well when:
    - Clear keywords indicate specific labels
    - Domain has well-defined terminology
    - High precision is more important than recall

    Example use cases:
    - Spam detection (keywords like "free money", "click here")
    - Topic classification (domain-specific terms)
    - Simple sentiment analysis (positive/negative words)
    """

    def __init__(self):
        self.rules = defaultdict(list)
        self.default_label = "unknown"

    def add_keyword_rule(self, label: str, keywords: List[str], case_sensitive: bool = False):
        """
        Add a rule that labels text based on keyword presence.

        Args:
            label: The label to assign
            keywords: List of keywords that indicate this label
            case_sensitive: Whether to match case exactly
        """
        for keyword in keywords:
            rule = {
                'type': 'keyword',
                'pattern': keyword if case_sensitive else keyword.lower(),
                'case_sensitive': case_sensitive,
                'label': label
            }
            self.rules[label].append(rule)

    def add_regex_rule(self, label: str, pattern: str):
        """
        Add a rule based on regular expression matching.

        Args:
            label: The label to assign
            pattern: Regex pattern to match
        """
        rule = {
            'type': 'regex',
            'pattern': re.compile(pattern, re.IGNORECASE),
            'label': label
        }
        self.rules[label].append(rule)

    def add_count_rule(self, label: str, keywords: List[str], threshold: int):
        """
        Add a rule that requires a minimum number of keyword matches.

        Args:
            label: The label to assign
            keywords: List of keywords to count
            threshold: Minimum number of matches required
        """
        rule = {
            'type': 'count',
            'keywords': [k.lower() for k in keywords],
            'threshold': threshold,
            'label': label
        }
        self.rules[label].append(rule)

    def label(self, text: str) -> str:
        """
        Label a single text using defined rules.

        Args:
            text: Text to label

        Returns:
            Predicted label
        """
        text_lower = text.lower()

        # Check all rules and collect matching labels with scores
        label_scores = defaultdict(int)

        for label, rules_list in self.rules.items():
            for rule in rules_list:
                if rule['type'] == 'keyword':
                    pattern = rule['pattern']
                    search_text = text if rule['case_sensitive'] else text_lower
                    if pattern in search_text:
                        label_scores[label] += 1

                elif rule['type'] == 'regex':
                    if rule['pattern'].search(text):
                        label_scores[label] += 1

                elif rule['type'] == 'count':
                    count = sum(
                        1 for kw in rule['keywords'] if kw in text_lower)
                    if count >= rule['threshold']:
                        label_scores[label] += count

        # Return label with highest score, or default
        if label_scores:
            return max(label_scores, key=label_scores.get)
        return self.default_label

    def label_dataset(self, texts: List[str]) -> List[str]:
        """
        Label multiple texts.

        Args:
            texts: List of texts to label

        Returns:
            List of predicted labels
        """
        return [self.label(text) for text in texts]


class LabelingFunction:
    """
    A labeling function for weak supervision.

    Inspired by the Snorkel framework, labeling functions (LFs) are
    simple functions that provide noisy labels. Multiple LFs can be
    combined to create training data without manual annotation.

    Key concepts:
    - Each LF votes for a label or abstains
    - LFs can conflict (vote for different labels)
    - A label aggregation method resolves conflicts
    - Coverage and accuracy vary per LF
    """

    def __init__(self, name: str, labeling_func: Callable, labels: List[str]):
        """
        Initialize a labeling function.

        Args:
            name: Descriptive name for this LF
            labeling_func: Function that takes text and returns label or None
            labels: List of possible labels
        """
        self.name = name
        self.func = labeling_func
        self.labels = labels
        self.abstain_label = None

    def apply(self, text: str) -> Optional[str]:
        """Apply the labeling function to text."""
        return self.func(text)


class WeakSupervisionLabeler:
    """
    Combine multiple labeling functions to create training data.

    This approach is useful when:
    - Manual labeling is expensive
    - You have domain knowledge to write heuristics
    - You can tolerate some label noise
    - You have multiple weak signals to combine
    """

    def __init__(self, labels: List[str]):
        self.labels = labels
        self.labeling_functions = []

    def add_labeling_function(self, lf: LabelingFunction):
        """Add a labeling function to the ensemble."""
        self.labeling_functions.append(lf)

    def apply_lfs(self, text: str) -> Dict[str, Optional[str]]:
        """
        Apply all labeling functions to a text.

        Args:
            text: Text to label

        Returns:
            Dictionary mapping LF names to their predictions
        """
        return {lf.name: lf.apply(text) for lf in self.labeling_functions}

    def label_with_majority_vote(self, text: str) -> str:
        """
        Label text using majority vote from all LFs.

        Args:
            text: Text to label

        Returns:
            Most common label (excluding abstentions)
        """
        votes = self.apply_lfs(text)
        valid_votes = [v for v in votes.values() if v is not None]

        if not valid_votes:
            return "unknown"

        # Count votes
        vote_counts = defaultdict(int)
        for vote in valid_votes:
            vote_counts[vote] += 1

        return max(vote_counts, key=vote_counts.get)

    def label_dataset(self, texts: List[str], method: str = 'majority') -> Tuple[List[str], pd.DataFrame]:
        """
        Label a dataset using labeling functions.

        Args:
            texts: List of texts to label
            method: Aggregation method ('majority' or 'unanimous')

        Returns:
            Tuple of (labels, vote_matrix)
        """
        labels = []
        vote_data = []

        for text in texts:
            if method == 'majority':
                label = self.label_with_majority_vote(text)
            elif method == 'unanimous':
                votes = self.apply_lfs(text)
                valid_votes = [v for v in votes.values() if v is not None]
                # Only assign label if all LFs agree
                label = valid_votes[0] if valid_votes and len(
                    set(valid_votes)) == 1 else "unknown"
            else:
                raise ValueError(f"Unknown method: {method}")

            labels.append(label)
            vote_data.append(self.apply_lfs(text))

        vote_matrix = pd.DataFrame(vote_data)
        return labels, vote_matrix

    def analyze_lfs(self, texts: List[str], true_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze the performance of labeling functions.

        Args:
            texts: List of texts
            true_labels: Optional ground truth labels for evaluation

        Returns:
            DataFrame with LF statistics
        """
        stats = []

        for lf in self.labeling_functions:
            predictions = [lf.apply(text) for text in texts]
            coverage = sum(
                1 for p in predictions if p is not None) / len(predictions)

            stat = {
                'name': lf.name,
                'coverage': coverage,
                'abstentions': sum(1 for p in predictions if p is None)
            }

            # If ground truth is available, compute accuracy
            if true_labels:
                correct = sum(1 for p, t in zip(predictions, true_labels)
                              if p is not None and p == t)
                non_abstain = sum(1 for p in predictions if p is not None)
                stat['accuracy'] = correct / \
                    non_abstain if non_abstain > 0 else 0.0

            stats.append(stat)

        return pd.DataFrame(stats)


class ZeroShotClassifier:
    """
    Zero-shot classification using pre-trained models.

    This approach leverages large language models that can classify text
    without any training examples, using natural language label descriptions.

    Advantages:
    - No training data needed
    - Works for new domains immediately
    - Can use arbitrary label descriptions

    Limitations:
    - Requires compute resources
    - May not match domain-specific performance
    - Slower than simpler methods
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize zero-shot classifier.

        Args:
            model_name: Hugging Face model name
        """
        try:
            from transformers import pipeline
            self.classifier = pipeline(
                "zero-shot-classification", model=model_name)
            self.available = True
        except ImportError:
            warnings.warn(
                "transformers not installed. Zero-shot classification unavailable.")
            self.available = False

    def classify(self,
                 text: str,
                 candidate_labels: List[str],
                 hypothesis_template: str = "This text is about {}.") -> Dict:
        """
        Classify text using zero-shot learning.

        Args:
            text: Text to classify
            candidate_labels: Possible labels
            hypothesis_template: Template for generating hypotheses

        Returns:
            Dictionary with labels and scores
        """
        if not self.available:
            raise RuntimeError("Zero-shot classifier not available")

        result = self.classifier(
            text,
            candidate_labels,
            hypothesis_template=hypothesis_template
        )
        return result

    def label_dataset(self,
                      texts: List[str],
                      candidate_labels: List[str],
                      threshold: float = 0.5) -> List[str]:
        """
        Label multiple texts with confidence threshold.

        Args:
            texts: List of texts
            candidate_labels: Possible labels
            threshold: Minimum confidence to assign label

        Returns:
            List of predicted labels
        """
        labels = []

        for text in texts:
            result = self.classify(text, candidate_labels)
            # Assign label if confidence exceeds threshold
            if result['scores'][0] >= threshold:
                labels.append(result['labels'][0])
            else:
                labels.append('uncertain')

        return labels


# Example usage and demonstrations
if __name__ == "__main__":
    print("="*80)
    print("AUTOMATIC TEXT LABELING EXAMPLES")
    print("="*80)

    # Sample data
    sample_texts = [
        "Get FREE money now! Click here to claim your prize!",
        "Meeting scheduled for tomorrow at 2 PM",
        "URGENT: Your account will be suspended unless you verify",
        "Thanks for your email. I'll review the document.",
        "Congratulations! You've won $1,000,000!!!",
        "Please find the attached report for your review",
    ]

    # Example 1: Rule-based labeling for spam detection
    print("\n1. RULE-BASED SPAM DETECTION")
    print("-" * 80)

    spam_labeler = RuleBasedLabeler()
    spam_labeler.default_label = "not_spam"

    # Add spam keywords
    spam_labeler.add_keyword_rule(
        label="spam",
        keywords=["free money", "click here",
                  "congratulations", "won", "prize", "urgent"],
        case_sensitive=False
    )

    # Add regex rule for excessive punctuation
    spam_labeler.add_regex_rule("spam", r"[!]{2,}")

    for text in sample_texts:
        label = spam_labeler.label(text)
        print(f"[{label.upper():10s}] {text[:60]}")

    # Example 2: Weak supervision with labeling functions
    print("\n2. WEAK SUPERVISION APPROACH")
    print("-" * 80)

    # Define labeling functions
    def lf_free_keyword(text):
        return "spam" if "free" in text.lower() else None

    def lf_urgent_keyword(text):
        return "spam" if "urgent" in text.lower() else None

    def lf_excessive_punctuation(text):
        return "spam" if text.count("!") >= 2 else None

    def lf_professional_language(text):
        professional_words = ["meeting", "schedule",
                              "review", "attached", "regards"]
        return "not_spam" if any(word in text.lower() for word in professional_words) else None

    # Create weak supervision labeler
    ws_labeler = WeakSupervisionLabeler(labels=["spam", "not_spam"])

    # Add labeling functions
    ws_labeler.add_labeling_function(
        LabelingFunction("lf_free", lf_free_keyword, ["spam", "not_spam"])
    )
    ws_labeler.add_labeling_function(
        LabelingFunction("lf_urgent", lf_urgent_keyword, ["spam", "not_spam"])
    )
    ws_labeler.add_labeling_function(
        LabelingFunction("lf_punctuation", lf_excessive_punctuation, [
                         "spam", "not_spam"])
    )
    ws_labeler.add_labeling_function(
        LabelingFunction("lf_professional", lf_professional_language, [
                         "spam", "not_spam"])
    )

    # Label dataset
    labels, vote_matrix = ws_labeler.label_dataset(
        sample_texts, method='majority')

    print("\nLabeling Function Vote Matrix:")
    print(vote_matrix.to_string())

    print("\nFinal Labels:")
    for text, label in zip(sample_texts, labels):
        print(f"[{label.upper():10s}] {text[:60]}")

    # Analyze labeling functions
    print("\nLabeling Function Analysis:")
    lf_stats = ws_labeler.analyze_lfs(sample_texts)
    print(lf_stats.to_string(index=False))

    # Example 3: Keyword-based sentiment labeling
    print("\n3. KEYWORD-BASED SENTIMENT LABELING")
    print("-" * 80)

    sentiment_texts = [
        "I absolutely love this product! It's amazing!",
        "Terrible quality, completely disappointed.",
        "It's okay, nothing special.",
        "Best purchase I've ever made!",
        "Waste of money, do not buy.",
    ]

    sentiment_labeler = RuleBasedLabeler()
    sentiment_labeler.default_label = "neutral"

    # Positive keywords
    sentiment_labeler.add_keyword_rule(
        "positive",
        ["love", "amazing", "best", "excellent", "great", "perfect", "wonderful"]
    )

    # Negative keywords
    sentiment_labeler.add_keyword_rule(
        "negative",
        ["terrible", "awful", "worst", "disappointed", "waste", "horrible", "bad"]
    )

    for text in sentiment_texts:
        label = sentiment_labeler.label(text)
        print(f"[{label.upper():10s}] {text}")

    print("\n" + "="*80)
    print("COMPARISON OF METHODS")
    print("="*80)
    print("""
Method                  | Pros                          | Cons
-----------------------|-------------------------------|---------------------------
Rule-Based             | Simple, fast, interpretable   | Low coverage, brittle
Weak Supervision       | Combines multiple signals     | Requires LF engineering
Zero-Shot              | No training data needed       | Slower, requires GPU
Pattern Matching       | Works for structured text     | Limited to patterns
    """)

    print("\n" + "="*80)
    print("BEST PRACTICES")
    print("="*80)
    print("""
1. Start Simple: Begin with rule-based methods for quick baseline
2. Iterate: Refine rules based on error analysis
3. Combine Methods: Use ensemble of multiple approaches
4. Validate: Always check automatic labels on sample data
5. Monitor: Track labeling quality over time
6. Document: Keep clear records of labeling logic
    """)
