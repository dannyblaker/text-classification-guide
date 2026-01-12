# Text Classification - Quick Reference

## Installation (One-Time Setup)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## Quick Start Commands

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run complete demo
python demo_all.py

# Run specific use cases (now with standalone examples!)
python use_cases/sentiment_analysis.py
python use_cases/spam_detection.py
python use_cases/topic_classification.py

# Run labeling examples
python labeling/automatic_labeling.py
python labeling/active_learning.py
```

## Code Snippets

### 1. Sentiment Analysis

```python
from use_cases.sentiment_analysis import SentimentAnalyzer

# Create analyzer
analyzer = SentimentAnalyzer(method='ml')

# Train
analyzer.train(train_texts, train_labels)

# Predict
predictions = analyzer.predict(["I love this product!"])
# Output: ['positive']
```

### 2. Spam Detection

```python
from use_cases.spam_detection import SpamDetector

# Create detector
detector = SpamDetector(method='rule-based')

# Predict
result = detector.predict(["FREE MONEY! Click here NOW!!!"])
# Output: ['spam']

# Get explanation
explanation = detector.explain_prediction("FREE MONEY!")
# Shows: spam keywords, excessive punctuation, etc.
```

### 3. Topic Classification

```python
from use_cases.topic_classification import TopicClassifier

# Create classifier
classifier = TopicClassifier(method='nb')  # Naive Bayes

# Train
classifier.train(texts, topics)

# Predict
predictions = classifier.predict(["New smartphone released today"])
# Output: ['technology']

# Get top features per topic
top_features = classifier.get_top_features_per_topic(n_features=10)
```

### 4. Text Preprocessing

```python
from utils.preprocessing import TextPreprocessor

# Create preprocessor
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_stopwords=True
)

# Clean text
clean_text = preprocessor.clean_text("Check http://example.com!!!")
# Output: "check"
```

### 5. Rule-Based Labeling

```python
from labeling.automatic_labeling import RuleBasedLabeler

# Create labeler
labeler = RuleBasedLabeler()

# Add rules
labeler.add_keyword_rule("spam", ["free", "winner", "urgent"])
labeler.add_regex_rule("spam", r"[!]{3,}")

# Label
labels = labeler.label_dataset(texts)
```

### 6. Weak Supervision

```python
from labeling.automatic_labeling import WeakSupervisionLabeler, LabelingFunction

# Define labeling functions
def lf_free(text):
    return "spam" if "free" in text.lower() else None

def lf_urgent(text):
    return "spam" if "urgent" in text.lower() else None

# Create labeler
ws_labeler = WeakSupervisionLabeler(labels=["spam", "not_spam"])
ws_labeler.add_labeling_function(LabelingFunction("lf_free", lf_free, ["spam", "not_spam"]))
ws_labeler.add_labeling_function(LabelingFunction("lf_urgent", lf_urgent, ["spam", "not_spam"]))

# Label with majority vote
labels, vote_matrix = ws_labeler.label_dataset(texts)
```

### 7. Model Evaluation

```python
from utils.evaluation import ClassificationEvaluator

# Create evaluator
evaluator = ClassificationEvaluator(y_true, y_pred)

# Get metrics
metrics = evaluator.get_metrics_summary()
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")

# Print full report
evaluator.print_evaluation_report()

# Get confusion matrix
cm = evaluator.confusion_matrix()

# Analyze errors
errors = evaluator.error_analysis(texts, top_n=10)
```

### 8. Data Loading

```python
from utils.data_loader import DataLoader

# Load from CSV
texts, labels = DataLoader.load_csv('data.csv', 
                                     text_column='text',
                                     label_column='label')

# Save to CSV
DataLoader.save_csv(texts, labels, 'output.csv')

# Create train/test split files
DataLoader.create_train_test_files(texts, labels, 
                                    output_dir='data_split',
                                    test_size=0.2)
```

### 9. Active Learning

```python
from labeling.active_learning import simulate_active_learning

# Run simulation
history = simulate_active_learning(
    X_pool=unlabeled_texts,
    y_pool=true_labels,  # Oracle
    X_test=test_texts,
    y_test=test_labels,
    initial_size=10,
    query_size=5,
    max_iterations=20,
    target_accuracy=0.9
)

# Access results
print(f"Final accuracy: {history['accuracy'][-1]:.3f}")
```

### 10. Visualization

```python
from utils.visualization import (
    plot_label_distribution,
    plot_confusion_matrix,
    plot_precision_recall_f1
)
import matplotlib.pyplot as plt

# Plot label distribution
fig = plot_label_distribution(labels)
plt.savefig('label_dist.png')

# Plot confusion matrix
fig = plot_confusion_matrix(cm, labels=['pos', 'neg', 'neu'])
plt.savefig('confusion_matrix.png')

# Plot per-class metrics
metrics = {
    'precision': {'pos': 0.85, 'neg': 0.77, 'neu': 0.72},
    'recall': {'pos': 0.90, 'neg': 0.77, 'neu': 0.65},
    'f1': {'pos': 0.87, 'neg': 0.77, 'neu': 0.68}
}
fig = plot_precision_recall_f1(metrics)
plt.savefig('metrics.png')
```

## Common Patterns

### Pattern 1: Train-Evaluate-Improve Loop

```python
# 1. Load data
texts, labels = DataLoader.load_csv('data.csv')

# 2. Preprocess
preprocessor = TextPreprocessor(lowercase=True, remove_stopwords=True)
clean_texts = preprocessor.preprocess_dataset(texts)

# 3. Split
from utils.evaluation import stratified_split
X_train, X_test, y_train, y_test = stratified_split(clean_texts, labels)

# 4. Train
classifier = TopicClassifier(method='nb')
classifier.train(X_train, y_train)

# 5. Evaluate
predictions = classifier.predict(X_test)
evaluator = ClassificationEvaluator(y_test, predictions)
evaluator.print_evaluation_report()

# 6. Analyze errors and iterate
errors = evaluator.error_analysis(X_test, top_n=20)
print(errors)
```

### Pattern 2: Compare Multiple Models

```python
from utils.evaluation import compare_models

results = {}
for method in ['nb', 'svm', 'rf']:
    classifier = TopicClassifier(method=method)
    classifier.train(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    evaluator = ClassificationEvaluator(y_test, predictions)
    results[method] = evaluator.get_metrics_summary()

# Compare
comparison = compare_models(results)
print(comparison)
```

### Pattern 3: Incremental Labeling

```python
# Start with small labeled set
labeled_texts = []
labeled_labels = []

# Add labels incrementally
for text in unlabeled_texts[:10]:
    label = input(f"Label for '{text}': ")
    labeled_texts.append(text)
    labeled_labels.append(label)
    
    # Train with current data
    classifier.train(labeled_texts, labeled_labels)
    
    # Check performance
    if len(labeled_texts) % 5 == 0:
        predictions = classifier.predict(test_texts)
        evaluator = ClassificationEvaluator(test_labels, predictions)
        print(f"Accuracy with {len(labeled_texts)} labels: {evaluator.accuracy():.3f}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Activate venv: `source .venv/bin/activate` |
| spaCy model not found | Run: `python -m spacy download en_core_web_sm` |
| NLTK data missing | Run: `python -c "import nltk; nltk.download('all')"` |
| Poor accuracy | Check preprocessing, balance, features |
| Out of memory | Reduce max_features in TfidfVectorizer |

## Performance Tips

1. **Preprocessing**: Remove stopwords for topic classification, keep for sentiment
2. **Features**: Use TF-IDF with bigrams (1,2) for best results
3. **Algorithms**: Start with Naive Bayes, then try SVM
4. **Evaluation**: Use F1-score for imbalanced data, not accuracy
5. **Active Learning**: Can reduce labeling by 50-70%

## File Structure Quick Map

```
labeling/           → Text labeling methods
use_cases/          → Classification examples  
utils/              → Helper functions
README.md           → Full documentation
GETTING_STARTED.md  → Setup guide
demo_all.py         → Complete demo
```

## Get Help

- Check docstrings: `help(SentimentAnalyzer)`
- Run examples: Each file has `if __name__ == "__main__"` demos
- Read comments: Inline explanations throughout code
- See README.md: Comprehensive documentation
