# Getting Started with Text Classification

This guide will walk you through setting up and running the text classification examples.

## Installation

### 1. Activate Virtual Environment

The virtual environment is already created in the `.venv` folder. Activate it:

```bash
# On Linux/Mac
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Required NLP Models

```bash
# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('vader_lexicon')"
```

## Quick Start Examples

### Example 1: Sentiment Analysis

```bash
python use_cases/sentiment_analysis.py
```

This will:
- Compare different sentiment analysis methods (rule-based, TextBlob, VADER, ML)
- Show accuracy metrics for each approach
- Demonstrate predictions on sample texts

### Example 2: Spam Detection

```bash
python use_cases/spam_detection.py
```

This will:
- Train spam detection models
- Show feature importance
- Explain why texts are classified as spam

### Example 3: Topic Classification

```bash
python use_cases/topic_classification.py
```

This will:
- Classify texts into topics (technology, sports, business, etc.)
- Compare different algorithms (Naive Bayes, SVM, Random Forest)
- Show top keywords for each topic

## Interactive Examples

### Manual Text Labeling

```python
from labeling.manual_labeling import ManualLabeler
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'text': [
        "Great product!",
        "Terrible quality",
        "It's okay"
    ]
})

# Create labeler
labeler = ManualLabeler(labels=['positive', 'negative', 'neutral'])

# Start labeling (interactive)
# labeled_data = labeler.label_dataset(data, text_column='text', output_path='labeled.csv')
```

### Automatic Labeling

```python
from labeling.automatic_labeling import RuleBasedLabeler

# Create rule-based labeler
labeler = RuleBasedLabeler()
labeler.default_label = "not_spam"

# Add rules
labeler.add_keyword_rule(
    label="spam",
    keywords=["free money", "click here", "urgent"],
    case_sensitive=False
)

# Label texts
texts = [
    "Get FREE money now!",
    "Meeting at 2 PM tomorrow"
]

predictions = labeler.label_dataset(texts)
print(predictions)  # ['spam', 'not_spam']
```

### Custom Classifier

```python
from use_cases.sentiment_analysis import SentimentAnalyzer
from utils.evaluation import ClassificationEvaluator

# Create and train classifier
analyzer = SentimentAnalyzer(method='ml')

# Training data
train_texts = [
    "I love this!",
    "Terrible experience",
    "It's okay"
]
train_labels = ['positive', 'negative', 'neutral']

analyzer.train(train_texts, train_labels)

# Predict
test_texts = ["This is amazing!", "Not good"]
predictions = analyzer.predict(test_texts)

print(predictions)  # ['positive', 'negative']
```

## Project Structure

```
text_classification/
├── README.md                          # Main documentation
├── GETTING_STARTED.md                 # This file
├── requirements.txt                   # Dependencies
│
├── labeling/                          # Text labeling methods
│   ├── manual_labeling.py            # Manual annotation tools
│   ├── automatic_labeling.py         # Automatic labeling methods
│   └── annotation_guidelines.md      # Annotation best practices
│
├── use_cases/                         # Classification examples
│   ├── sentiment_analysis.py         # Sentiment classification
│   ├── spam_detection.py             # Spam filtering
│   └── topic_classification.py       # Topic categorization
│
└── utils/                             # Utility modules
    ├── preprocessing.py               # Text preprocessing
    ├── evaluation.py                  # Metrics and evaluation
    ├── visualization.py               # Plotting functions
    └── data_loader.py                 # Data loading utilities
```

## Common Tasks

### Task 1: Train a Custom Sentiment Classifier

```python
from use_cases.sentiment_analysis import SentimentAnalyzer
from utils.data_loader import DataLoader
from utils.evaluation import ClassificationEvaluator

# Load your data
texts, labels = DataLoader.load_csv('your_data.csv', 
                                    text_column='review', 
                                    label_column='sentiment')

# Create and train
analyzer = SentimentAnalyzer(method='ml')
analyzer.train(texts, labels)

# Evaluate
predictions = analyzer.predict(texts)
evaluator = ClassificationEvaluator(labels, predictions)
evaluator.print_evaluation_report()
```

### Task 2: Preprocess Text Data

```python
from utils.preprocessing import TextPreprocessor

# Create preprocessor with desired options
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_punctuation=False,
    remove_urls=True,
    remove_stopwords=True
)

# Clean text
text = "Check out this AMAZING product at http://example.com !!!"
cleaned = preprocessor.clean_text(text)
print(cleaned)  # "check amazing product"
```

### Task 3: Compare Multiple Models

```python
from use_cases.topic_classification import TopicClassifier
from utils.evaluation import ClassificationEvaluator, compare_models

# Train different models
models = {}
methods = ['nb', 'svm', 'rf']

for method in methods:
    classifier = TopicClassifier(method=method)
    classifier.train(train_texts, train_labels)
    predictions = classifier.predict(test_texts)
    
    evaluator = ClassificationEvaluator(test_labels, predictions)
    models[method] = evaluator.get_metrics_summary()

# Compare
comparison = compare_models(models)
print(comparison)
```

### Task 4: Visualize Results

```python
from utils.visualization import (
    plot_label_distribution,
    plot_confusion_matrix,
    plot_precision_recall_f1
)
from utils.evaluation import ClassificationEvaluator
import matplotlib.pyplot as plt

# Plot label distribution
fig = plot_label_distribution(labels)
plt.show()

# Plot confusion matrix
evaluator = ClassificationEvaluator(true_labels, predictions)
cm = evaluator.confusion_matrix()
fig = plot_confusion_matrix(cm.values, labels=evaluator.labels)
plt.show()
```

## Tips and Best Practices

### 1. Data Preprocessing

- **For sentiment analysis**: Keep punctuation (e.g., "!!!" indicates excitement)
- **For topic classification**: Remove stopwords aggressively
- **For spam detection**: Keep special characters and URLs as features

### 2. Model Selection

- **Start simple**: Begin with rule-based or Naive Bayes
- **Iterate**: Gradually add complexity based on performance
- **Validate**: Always use cross-validation for reliable estimates

### 3. Handling Imbalanced Data

```python
from sklearn.utils import class_weight
import numpy as np

# Compute class weights
classes = np.unique(labels)
weights = class_weight.compute_class_weight('balanced', 
                                            classes=classes, 
                                            y=labels)
class_weights = dict(zip(classes, weights))

# Use in model training
# model = LogisticRegression(class_weight=class_weights)
```

### 4. Evaluation Metrics

- **Accuracy**: Use when classes are balanced
- **Precision**: Prioritize when false positives are costly
- **Recall**: Prioritize when false negatives are costly
- **F1-Score**: Use for balanced view of precision and recall

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Make sure virtual environment is activated and dependencies are installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: spaCy model not found

**Solution**: Download the required model:
```bash
python -m spacy download en_core_web_sm
```

### Issue: NLTK data not found

**Solution**: Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

### Issue: Poor model performance

**Solutions**:
1. Check data quality and preprocessing
2. Ensure sufficient training data
3. Verify class balance
4. Try different algorithms
5. Tune hyperparameters

## Next Steps

1. **Explore the examples**: Run each use case to understand different approaches
2. **Try your own data**: Use the data loading utilities to work with your datasets
3. **Experiment with features**: Modify preprocessing and feature extraction
4. **Compare methods**: Test different algorithms on your specific task
5. **Read the documentation**: Check docstrings for detailed parameter descriptions

## Additional Resources

- [scikit-learn Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [spaCy Documentation](https://spacy.io/)
- [NLTK Book](https://www.nltk.org/book/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## Getting Help

If you encounter issues:
1. Check the docstrings in the code
2. Review the example usage in `if __name__ == "__main__"` blocks
3. Examine the error messages carefully
4. Ensure all dependencies are properly installed

Happy Learning! 🎓
