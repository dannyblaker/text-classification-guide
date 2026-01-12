# Text Classification Repository - Navigation Index

## 🚀 Start Here

**New to this repository?** Follow this path:

1. 📖 Read [README.md](README.md) - Project overview and concepts
2. ⚙️ Follow [GETTING_STARTED.md](GETTING_STARTED.md) - Installation and setup
3. 🎮 Run `python demo_all.py` - Interactive demonstration
4. 📚 Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Code snippets and commands

## 📁 Files by Purpose

### Documentation (Read These)
| File | Purpose | When to Read |
|------|---------|--------------|
| [README.md](README.md) | Main documentation, concepts, overview | First - understand the project |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Setup, installation, first steps | Second - get running |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Code snippets, quick commands | While coding - copy/paste examples |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Complete project inventory | For reference - see everything |
| [labeling/annotation_guidelines.md](labeling/annotation_guidelines.md) | Manual labeling best practices | When doing manual annotation |

### Executable Scripts (Run These)
| File | Description | Command |
|------|-------------|---------|
| [demo_all.py](demo_all.py) | Complete repository demo | `python demo_all.py` |
| [use_cases/sentiment_analysis.py](use_cases/sentiment_analysis.py) | Sentiment classification examples | `python use_cases/sentiment_analysis.py` |
| [use_cases/spam_detection.py](use_cases/spam_detection.py) | Spam detection demo | `python use_cases/spam_detection.py` |
| [use_cases/topic_classification.py](use_cases/topic_classification.py) | Topic classification demo | `python use_cases/topic_classification.py` |
| [labeling/automatic_labeling.py](labeling/automatic_labeling.py) | Automatic labeling demo | `python labeling/automatic_labeling.py` |
| [labeling/active_learning.py](labeling/active_learning.py) | Active learning demo | `python labeling/active_learning.py` |
| [labeling/manual_labeling.py](labeling/manual_labeling.py) | Manual labeling examples | `python labeling/manual_labeling.py` |

### Library Modules (Import These)
| Module | Main Classes/Functions | Use For |
|--------|------------------------|---------|
| [utils/preprocessing.py](utils/preprocessing.py) | `TextPreprocessor` | Cleaning and normalizing text |
| [utils/evaluation.py](utils/evaluation.py) | `ClassificationEvaluator` | Computing metrics, analysis |
| [utils/visualization.py](utils/visualization.py) | `plot_confusion_matrix`, etc. | Creating plots and charts |
| [utils/data_loader.py](utils/data_loader.py) | `DataLoader` | Loading/saving datasets |

## 🎯 Find What You Need

### I want to...

#### Learn Text Classification Basics
→ Start with [README.md](README.md) sections:
  - Introduction to Text Classification
  - Text Labeling Methods
  - Common Use Cases

#### Set Up the Project
→ Follow [GETTING_STARTED.md](GETTING_STARTED.md):
  - Installation steps
  - Quick start examples
  - Troubleshooting

#### Build a Sentiment Analyzer
→ Check:
1. [use_cases/sentiment_analysis.py](use_cases/sentiment_analysis.py) - Complete implementation
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Code snippets
3. Run: `python use_cases/sentiment_analysis.py`

#### Detect Spam Messages
→ Check:
1. [use_cases/spam_detection.py](use_cases/spam_detection.py) - Full spam detector
2. Focus on `extract_spam_features()` for feature engineering
3. Run: `python use_cases/spam_detection.py`

#### Classify Topics
→ Check:
1. [use_cases/topic_classification.py](use_cases/topic_classification.py) - Multi-class classifier
2. Compare algorithms (NB, SVM, RF)
3. Run: `python use_cases/topic_classification.py`

#### Label Data Efficiently
→ Check:
1. [labeling/active_learning.py](labeling/active_learning.py) - Reduce labeling by 50-70%
2. [labeling/automatic_labeling.py](labeling/automatic_labeling.py) - Rule-based labeling
3. [labeling/manual_labeling.py](labeling/manual_labeling.py) - Manual annotation tools

#### Preprocess Text
→ Use [utils/preprocessing.py](utils/preprocessing.py):
```python
from utils.preprocessing import TextPreprocessor
preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
clean_text = preprocessor.clean_text("Your text here")
```

#### Evaluate Models
→ Use [utils/evaluation.py](utils/evaluation.py):
```python
from utils.evaluation import ClassificationEvaluator
evaluator = ClassificationEvaluator(y_true, y_pred)
evaluator.print_evaluation_report()
```

#### Visualize Results
→ Use [utils/visualization.py](utils/visualization.py):
```python
from utils.visualization import plot_confusion_matrix
fig = plot_confusion_matrix(confusion_matrix, labels)
```

#### Work with Data
→ Use [utils/data_loader.py](utils/data_loader.py):
```python
from utils.data_loader import DataLoader
texts, labels = DataLoader.load_csv('data.csv')
```

## 📊 Repository Structure Map

```
Root
├── 📄 Documentation          → README.md, GETTING_STARTED.md, etc.
├── 🐍 Demo Script           → demo_all.py
├── 📦 Dependencies          → requirements.txt
│
├── 📂 labeling/             → How to label text data
│   ├── manual_labeling.py   → Human annotation
│   ├── automatic_labeling.py → Rule-based/weak supervision  
│   └── active_learning.py   → Smart labeling (50-70% reduction)
│
├── 📂 use_cases/            → Real-world applications
│   ├── sentiment_analysis.py → Positive/negative/neutral
│   ├── spam_detection.py    → Spam/not spam
│   └── topic_classification.py → Technology/sports/etc.
│
└── 📂 utils/                → Helper functions
    ├── preprocessing.py     → Text cleaning
    ├── evaluation.py        → Metrics and analysis
    ├── visualization.py     → Plots and charts
    └── data_loader.py       → Data I/O
```

## 🎓 Learning Paths

### Path 1: Beginner (2-3 hours)
1. Read README.md introduction
2. Follow GETTING_STARTED.md setup
3. Run `python demo_all.py`
4. Run `python use_cases/sentiment_analysis.py`
5. Try code snippets from QUICK_REFERENCE.md

### Path 2: Practitioner (4-6 hours)
1. Complete Beginner path
2. Run all use case examples
3. Study preprocessing options
4. Understand evaluation metrics
5. Try with your own small dataset

### Path 3: Advanced (1-2 days)
1. Complete Practitioner path
2. Study active learning implementation
3. Implement custom classifiers
4. Compare multiple algorithms
5. Build production pipeline

### Path 4: Researcher (Flexible)
1. Deep dive into specific methods
2. Study algorithm implementations
3. Modify and experiment
4. Add new features or algorithms
5. Benchmark on standard datasets

## 🔍 Quick Search

**Need something specific? Search for:**

| Topic | File | Keyword |
|-------|------|---------|
| Preprocessing | utils/preprocessing.py | `TextPreprocessor` |
| Metrics | utils/evaluation.py | `ClassificationEvaluator` |
| Plots | utils/visualization.py | `plot_` |
| Sentiment | use_cases/sentiment_analysis.py | `SentimentAnalyzer` |
| Spam | use_cases/spam_detection.py | `SpamDetector` |
| Topics | use_cases/topic_classification.py | `TopicClassifier` |
| Manual labels | labeling/manual_labeling.py | `ManualLabeler` |
| Auto labels | labeling/automatic_labeling.py | `RuleBasedLabeler` |
| Active learning | labeling/active_learning.py | `ActiveLearner` |

## 💻 Common Commands

```bash
# Setup
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run demos
python demo_all.py
python use_cases/sentiment_analysis.py
python use_cases/spam_detection.py
python use_cases/topic_classification.py

# Run labeling examples
python labeling/automatic_labeling.py
python labeling/active_learning.py
python labeling/manual_labeling.py
```

## 📞 Getting Help

**Error or question?** Check:

1. **Installation issues** → [GETTING_STARTED.md](GETTING_STARTED.md) Troubleshooting section
2. **How to use X?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) code snippets
3. **What does this do?** → Check docstrings: `help(ClassName)`
4. **Example usage?** → Each .py file has `if __name__ == "__main__"` examples

## 📈 Repository Stats

- **Total Code**: ~4,100 lines of Python
- **Documentation**: ~40,000 words across 5 files
- **Modules**: 12 Python files (7 libraries + 5 demos)
- **Classes**: 15+ reusable classes
- **Functions**: 100+ documented functions
- **Examples**: 30+ runnable code examples
- **Use Cases**: 3 complete implementations

## ✅ Quick Checklist

Before you start, make sure you have:
- [ ] Activated virtual environment (`.venv`)
- [ ] Installed requirements (`pip install -r requirements.txt`)
- [ ] Downloaded spaCy model (`python -m spacy download en_core_web_sm`)
- [ ] Read README.md or GETTING_STARTED.md
- [ ] Tried running `python demo_all.py`

---

**Ready to start?** → Go to [GETTING_STARTED.md](GETTING_STARTED.md)

**Just want code?** → Go to [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**Want complete overview?** → Go to [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
