#!/usr/bin/env python3
"""
Text Classification Repository Demo

This script demonstrates all the key features of the text classification repository.
Run this to see examples of:
- Manual and automatic labeling
- Sentiment analysis
- Spam detection
- Topic classification
- Active learning

Usage:
    python demo_all.py
"""

import sys
import os

print("="*80)
print("TEXT CLASSIFICATION REPOSITORY - COMPLETE DEMO")
print("="*80)
print("\nThis demo will showcase all major features of the repository.")
print("Each section can be run independently by navigating to the respective files.\n")


def demo_section(title, description):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)
    print(f"{description}\n")
    input("Press Enter to continue...")


# Demo 1: Manual Labeling
demo_section(
    "1. MANUAL LABELING",
    "Manual labeling tools for human annotation with quality control."
)

try:
    print("Running manual labeling example...")
    from labeling.manual_labeling import (
        LabelingGuidelines,
        AnnotationValidator
    )

    # Create guidelines
    guidelines = LabelingGuidelines("Demo Task")
    guidelines.guidelines['task_description'] = "Classify text sentiment"
    guidelines.add_label_definition(
        "positive",
        "Text expresses positive sentiment",
        ["Great!", "Love it!"]
    )
    print("✓ Created annotation guidelines")

    # Inter-annotator agreement
    annotator1 = ['pos', 'neg', 'neu', 'pos', 'neg']
    annotator2 = ['pos', 'neg', 'neg', 'pos', 'neg']

    validator = AnnotationValidator()
    kappa = validator.calculate_cohen_kappa(annotator1, annotator2)
    print(f"✓ Cohen's Kappa (agreement): {kappa:.3f}")

    print("\n→ See labeling/manual_labeling.py for interactive annotation")

except Exception as e:
    print(f"⚠ Error: {e}")

# Demo 2: Automatic Labeling
demo_section(
    "2. AUTOMATIC LABELING",
    "Rule-based and weak supervision methods for automatic text labeling."
)

try:
    print("Running automatic labeling example...")
    from labeling.automatic_labeling import RuleBasedLabeler

    # Create labeler
    labeler = RuleBasedLabeler()
    labeler.add_keyword_rule("spam", ["free", "winner", "urgent"])

    # Test
    texts = [
        "URGENT! You've WON free money!",
        "Meeting at 2 PM tomorrow"
    ]

    predictions = labeler.label_dataset(texts)
    for text, pred in zip(texts, predictions):
        print(f"  '{text[:40]}...' → {pred}")

    print("\n→ See labeling/automatic_labeling.py for more methods")

except Exception as e:
    print(f"⚠ Error: {e}")

# Demo 3: Sentiment Analysis
demo_section(
    "3. SENTIMENT ANALYSIS",
    "Multiple approaches to sentiment classification."
)

try:
    print("Running sentiment analysis demo...")
    from use_cases.sentiment_analysis import SentimentAnalyzer

    # Test different methods
    test_text = "This is absolutely amazing! I love it!"

    for method in ['rule-based', 'textblob', 'vader']:
        try:
            analyzer = SentimentAnalyzer(method=method)
            pred = analyzer.predict([test_text])[0]
            print(f"  {method:12s}: {pred}")
        except:
            print(f"  {method:12s}: [Not available]")

    print("\n→ Run use_cases/sentiment_analysis.py for full comparison")

except Exception as e:
    print(f"⚠ Error: {e}")

# Demo 4: Spam Detection
demo_section(
    "4. SPAM DETECTION",
    "Identify and filter spam messages with feature extraction."
)

try:
    print("Running spam detection demo...")
    from use_cases.spam_detection import SpamDetector

    detector = SpamDetector(method='rule-based')

    test_texts = [
        "URGENT! Click here for FREE money NOW!!!",
        "Thanks for your email, see you at the meeting."
    ]

    for text in test_texts:
        pred = detector.predict([text])[0]
        print(f"  [{pred:8s}] {text[:50]}")

    print("\n→ Run use_cases/spam_detection.py for detailed analysis")

except Exception as e:
    print(f"⚠ Error: {e}")

# Demo 5: Topic Classification
demo_section(
    "5. TOPIC CLASSIFICATION",
    "Categorize documents into predefined topics."
)

try:
    print("Running topic classification demo...")
    from use_cases.topic_classification import TopicClassifier, create_topic_dataset

    # Get data
    texts, topics = create_topic_dataset()

    print(f"  Dataset: {len(texts)} samples")
    print(f"  Topics: {len(set(topics))} categories")
    print(f"  Categories: {', '.join(sorted(set(topics)))}")

    print("\n→ Run use_cases/topic_classification.py to train and compare models")

except Exception as e:
    print(f"⚠ Error: {e}")

# Demo 6: Text Preprocessing
demo_section(
    "6. TEXT PREPROCESSING",
    "Clean and normalize text data for classification."
)

try:
    print("Running preprocessing demo...")
    from utils.preprocessing import TextPreprocessor

    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_punctuation=False
    )

    text = "Check out this AMAZING product at http://example.com !!!"
    cleaned = preprocessor.clean_text(text)

    print(f"  Original: {text}")
    print(f"  Cleaned:  {cleaned}")

    print("\n→ See utils/preprocessing.py for all preprocessing options")

except Exception as e:
    print(f"⚠ Error: {e}")

# Demo 7: Evaluation Metrics
demo_section(
    "7. EVALUATION METRICS",
    "Comprehensive evaluation tools for classification models."
)

try:
    print("Running evaluation demo...")
    from utils.evaluation import ClassificationEvaluator

    y_true = ['pos', 'neg', 'neu', 'pos', 'neg', 'neu']
    y_pred = ['pos', 'neg', 'pos', 'pos', 'neg', 'neu']

    evaluator = ClassificationEvaluator(y_true, y_pred)
    metrics = evaluator.get_metrics_summary()

    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")

    print("\n→ See utils/evaluation.py for detailed metrics and reports")

except Exception as e:
    print(f"⚠ Error: {e}")

# Demo 8: Active Learning
demo_section(
    "8. ACTIVE LEARNING",
    "Reduce labeling effort by querying the most informative examples."
)

try:
    print("Active learning reduces labeling effort by 50-70%!")
    print("It queries the examples the model is least confident about.")
    print("\nKey benefits:")
    print("  • Fewer labels needed for good performance")
    print("  • Focuses human effort on difficult cases")
    print("  • Ideal when labeling is expensive")

    print("\n→ Run labeling/active_learning.py to see active vs random sampling")

except Exception as e:
    print(f"⚠ Error: {e}")

# Final Summary
print("\n" + "="*80)
print("REPOSITORY OVERVIEW")
print("="*80)

print("""
📁 REPOSITORY STRUCTURE:

labeling/                       # Text labeling methods
  ├── manual_labeling.py       # Interactive annotation tools
  ├── automatic_labeling.py    # Rule-based & weak supervision
  ├── active_learning.py       # Active learning strategies
  └── annotation_guidelines.md # Best practices

use_cases/                      # Classification examples
  ├── sentiment_analysis.py    # Sentiment classification
  ├── spam_detection.py        # Spam filtering
  └── topic_classification.py  # Topic categorization

utils/                          # Utility modules
  ├── preprocessing.py         # Text cleaning & normalization
  ├── evaluation.py            # Metrics & evaluation tools
  ├── visualization.py         # Plotting functions
  └── data_loader.py           # Data loading utilities

📚 KEY CONCEPTS COVERED:

1. Labeling Methods
   - Manual annotation with quality control
   - Rule-based automatic labeling
   - Weak supervision
   - Active learning

2. Classification Tasks
   - Sentiment analysis (positive/negative/neutral)
   - Spam detection (spam/not spam)
   - Topic classification (multi-class)

3. Algorithms
   - Rule-based approaches
   - Naive Bayes
   - Support Vector Machines (SVM)
   - Random Forest
   - Logistic Regression

4. Best Practices
   - Text preprocessing strategies
   - Feature engineering
   - Model evaluation
   - Handling imbalanced data

🚀 NEXT STEPS:

1. Read GETTING_STARTED.md for installation and setup
2. Run individual use case examples:
   - python use_cases/sentiment_analysis.py
   - python use_cases/spam_detection.py
   - python use_cases/topic_classification.py

3. Explore labeling methods:
   - python labeling/automatic_labeling.py
   - python labeling/active_learning.py

4. Try with your own data:
   - Use utils/data_loader.py to load your datasets
   - Apply preprocessing with utils/preprocessing.py
   - Evaluate with utils/evaluation.py

📖 DOCUMENTATION:

- README.md: Project overview and introduction
- GETTING_STARTED.md: Step-by-step guide
- Code docstrings: Detailed API documentation
- Comments: Inline explanations

💡 TIPS:

- Start with simple rule-based approaches
- Use stratified train/test splits
- Evaluate with multiple metrics
- Consider class imbalance
- Iterate and improve based on errors

Happy Learning! 🎓
""")

print("="*80)
print("Demo complete! Check the individual files for more details.")
print("="*80)
