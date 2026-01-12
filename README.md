# Text Classification: A Comprehensive Educational Guide

Welcome to this educational repository on **Text Classification**! This project covers the fundamental concepts, techniques, and practical implementations of text classification in Natural Language Processing (NLP).

[![A Danny Blaker project badge](https://github.com/dannyblaker/dannyblaker.github.io/blob/main/danny_blaker_project_badge.svg)](https://github.com/dannyblaker/)

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Text Labeling Methods](#text-labeling-methods)
3. [Common Use Cases](#common-use-cases)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Examples](#examples)
7. [Additional Resources](#additional-resources)

## Introduction

[⬆ Back to top](#-table-of-contents)

**Text Classification** is the task of assigning predefined categories or labels to text documents. It's one of the most fundamental tasks in NLP and has numerous real-world applications.

### What You'll Learn

- Different approaches to labeling text data
- Manual vs. automatic labeling techniques
- Implementation of various text classification use cases
- Traditional ML and modern deep learning approaches
- Best practices and evaluation metrics

## Text Labeling Methods

[⬆ Back to top](#-table-of-contents)

### Manual Labeling

Manual labeling involves human annotators assigning labels to text data. This approach:
- **Pros**: High accuracy, domain-specific expertise, handles nuances
- **Cons**: Time-consuming, expensive, potential for bias
- **Best for**: Small datasets, complex tasks, establishing ground truth

**Approaches covered:**
- Simple annotation workflows
- Inter-annotator agreement
- Label quality validation
- Annotation guidelines

### Automatic Labeling

Automatic labeling uses algorithms to assign labels without human intervention:

1. **Rule-Based Methods**
   - Keyword matching
   - Regular expressions
   - Pattern-based classification

2. **Weak Supervision**
   - Labeling functions
   - Snorkel framework concepts
   - Programmatic labeling

3. **Transfer Learning**
   - Pre-trained models
   - Zero-shot classification
   - Few-shot learning

4. **Active Learning**
   - Uncertainty sampling
   - Query strategies
   - Human-in-the-loop

## Common Use Cases

[⬆ Back to top](#-table-of-contents)

### 1. Sentiment Analysis
Determine the emotional tone of text (positive, negative, neutral).

**Applications:**
- Product reviews
- Social media monitoring
- Customer feedback analysis
- Brand reputation management

### 2. Spam Detection
Identify unwanted or malicious messages.

**Applications:**
- Email filtering
- SMS filtering
- Comment moderation
- Fraud detection

### 3. Topic Classification
Categorize documents into predefined topics.

**Applications:**
- News categorization
- Document organization
- Content recommendation
- Academic paper classification

### 4. Intent Classification
Understand user intent in conversational AI.

**Applications:**
- Chatbots
- Virtual assistants
- Customer service automation

### 5. Language Detection
Identify the language of a text.

**Applications:**
- Multilingual content routing
- Translation services
- Content filtering

## Project Structure

[⬆ Back to top](#-table-of-contents)

```
text_classification/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── labeling/
│   ├── manual_labeling.py            # Manual annotation examples
│   ├── automatic_labeling.py         # Rule-based and weak supervision
│   ├── active_learning.py            # Active learning implementation
│   └── annotation_guidelines.md      # Best practices for labeling
├── use_cases/
│   ├── sentiment_analysis.py         # Sentiment classification
│   ├── spam_detection.py             # Spam filtering
│   ├── topic_classification.py       # Topic categorization
│   └── intent_classification.py      # Intent detection
├── utils/
│   ├── data_loader.py                # Data loading utilities
│   ├── preprocessing.py              # Text preprocessing
│   ├── evaluation.py                 # Metrics and evaluation
│   └── visualization.py              # Plotting and visualization
├── data/
│   ├── sample_reviews.csv            # Sample sentiment data
│   ├── sample_spam.csv               # Sample spam data
│   └── sample_news.csv               # Sample topic data
└── notebooks/
    ├── 01_introduction.ipynb         # Introduction to text classification
    ├── 02_manual_labeling.ipynb      # Manual labeling tutorial
    ├── 03_automatic_labeling.ipynb   # Automatic labeling tutorial
    └── 04_complete_pipeline.ipynb    # End-to-end pipeline
```

## Getting Started

[⬆ Back to top](#-table-of-contents)

### Prerequisites

- Python 3.8 or higher
- Virtual environment (already created in `.venv`)

### Installation

1. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
```

### Quick Start

Run the sentiment analysis example:
```bash
python use_cases/sentiment_analysis.py
```

Run the spam detection example:
```bash
python use_cases/spam_detection.py
```

## Examples

[⬆ Back to top](#-table-of-contents)

### Example 1: Simple Sentiment Classification

```python
from use_cases.sentiment_analysis import SentimentClassifier

# Create classifier
classifier = SentimentClassifier()

# Train on sample data
classifier.train()

# Predict sentiment
text = "This product is amazing! I love it!"
sentiment = classifier.predict(text)
print(f"Sentiment: {sentiment}")  # Output: positive
```

### Example 2: Rule-Based Spam Detection

```python
from labeling.automatic_labeling import RuleBasedLabeler

# Create labeler
labeler = RuleBasedLabeler()

# Add spam rules
labeler.add_rule("contains", ["free money", "click here", "winner"])

# Label text
text = "Congratulations! Click here to claim your free money!"
is_spam = labeler.label(text)
print(f"Is Spam: {is_spam}")  # Output: True
```

### Example 3: Manual Labeling Interface

```python
from labeling.manual_labeling import ManualLabeler

# Create labeling interface
labeler = ManualLabeler(labels=["positive", "negative", "neutral"])

# Start labeling session
labeler.label_dataset("data/sample_reviews.csv", output="labeled_data.csv")
```

## 📊 Evaluation Metrics

[⬆ Back to top](#-table-of-contents)

All examples include comprehensive evaluation:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive coverage
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## 🛠️ Technologies Used

[⬆ Back to top](#-table-of-contents)

- **scikit-learn**: Traditional ML algorithms
- **transformers**: Pre-trained language models
- **spaCy**: NLP preprocessing
- **pandas**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **nltk**: Text processing utilities

## 📚 Learning Path

[⬆ Back to top](#-table-of-contents)

1. **Start with basics**: Read through the manual labeling examples
2. **Understand automation**: Explore automatic labeling techniques
3. **Practice with use cases**: Implement sentiment analysis, spam detection
4. **Advanced topics**: Dive into transfer learning and active learning
5. **Build your own**: Create a custom classifier for your domain

## Additional Resources

[⬆ Back to top](#-table-of-contents)

- [scikit-learn Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [spaCy Text Classification](https://spacy.io/usage/training#textcat)
- [Active Learning Literature Survey](https://minds.wisconsin.edu/handle/1793/60660)