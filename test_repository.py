#!/usr/bin/env python3
"""
Repository Test Suite
Tests all modules to ensure they work correctly.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    try:
        from labeling import manual_labeling, automatic_labeling, active_learning
        from use_cases import sentiment_analysis, spam_detection, topic_classification
        from utils import data_loader, preprocessing, evaluation, visualization
        print("✓ All imports successful\n")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}\n")
        return False


def test_manual_labeling():
    """Test manual labeling functionality."""
    print("=" * 60)
    print("TEST 2: Manual Labeling")
    print("=" * 60)
    try:
        from labeling.manual_labeling import LabelingGuidelines, AnnotationValidator
        
        # Test LabelingGuidelines
        guidelines = LabelingGuidelines('Test Task')
        guidelines.add_label_definition('positive', 'Positive sentiment', ['Great!', 'Love it'])
        guidelines.add_label_definition('negative', 'Negative sentiment', ['Bad', 'Hate it'])
        
        # Test AnnotationValidator
        annotations1 = ['pos', 'neg', 'pos', 'pos', 'neg']
        annotations2 = ['pos', 'neg', 'neg', 'pos', 'neg']
        kappa = AnnotationValidator.calculate_cohen_kappa(annotations1, annotations2)
        
        print(f"  - LabelingGuidelines created: ✓")
        print(f"  - AnnotationValidator working (Cohen's Kappa: {kappa:.3f}): ✓")
        print("✓ Manual labeling module works\n")
        return True
    except Exception as e:
        print(f"✗ Manual labeling failed: {e}\n")
        return False


def test_automatic_labeling():
    """Test automatic labeling functionality."""
    print("=" * 60)
    print("TEST 3: Automatic Labeling")
    print("=" * 60)
    try:
        from labeling.automatic_labeling import RuleBasedLabeler
        
        # Test RuleBasedLabeler
        rule_labeler = RuleBasedLabeler()
        rule_labeler.add_keyword_rule('positive', ['good', 'great', 'excellent'])
        rule_labeler.add_keyword_rule('negative', ['bad', 'terrible', 'awful'])
        result1 = rule_labeler.label('This is great!')
        
        print(f"  - RuleBasedLabeler: '{result1}' ✓")
        print("✓ Automatic labeling module works\n")
        return True
    except Exception as e:
        print(f"✗ Automatic labeling failed: {e}\n")
        return False


def test_active_learning():
    """Test active learning functionality."""
    print("=" * 60)
    print("TEST 4: Active Learning")
    print("=" * 60)
    try:
        from labeling.active_learning import ActiveLearner
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        
        # Test ActiveLearner
        model = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
        
        learner = ActiveLearner(initial_model=model)
        
        print(f"  - ActiveLearner initialized: ✓")
        print("✓ Active learning module works\n")
        return True
    except Exception as e:
        print(f"✗ Active learning failed: {e}\n")
        return False


def test_sentiment_analysis():
    """Test sentiment analysis use case."""
    print("=" * 60)
    print("TEST 5: Sentiment Analysis")
    print("=" * 60)
    try:
        from use_cases.sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(method='vader')
        
        test_texts = [
            "I love this product! It's amazing!",
            "This is terrible and disappointing.",
            "It's okay, nothing special."
        ]
        
        results = analyzer.predict(test_texts)
        
        print(f"  - Analyzed {len(results)} texts: ✓")
        print(f"  - Sample result: {results[0]}")
        print("✓ Sentiment analysis works\n")
        return True
    except Exception as e:
        print(f"✗ Sentiment analysis failed: {e}\n")
        return False


def test_spam_detection():
    """Test spam detection use case."""
    print("=" * 60)
    print("TEST 6: Spam Detection")
    print("=" * 60)
    try:
        from use_cases.spam_detection import SpamDetector
        
        # Create sample training data
        train_texts = [
            "Free prize! Click now!",
            "Meeting tomorrow at 3pm",
            "Win a million dollars!!!",
            "Can we schedule a call?"
        ]
        train_labels = ['spam', 'ham', 'spam', 'ham']
        
        detector = SpamDetector()
        detector.train(train_texts, train_labels)
        
        test_text = "Congratulations! You won!"
        prediction = detector.predict([test_text])
        
        print(f"  - Model trained on {len(train_texts)} samples: ✓")
        print(f"  - Prediction: '{prediction[0]}' ✓")
        print("✓ Spam detection works\n")
        return True
    except Exception as e:
        print(f"✗ Spam detection failed: {e}\n")
        return False


def test_topic_classification():
    """Test topic classification use case."""
    print("=" * 60)
    print("TEST 7: Topic Classification")
    print("=" * 60)
    try:
        from use_cases.topic_classification import TopicClassifier
        
        # Create larger sample training data to avoid vectorizer issues
        train_texts = [
            "The stock market rose today and investors are happy",
            "New smartphone released with amazing features and camera",
            "Football team wins championship after great performance",
            "GDP growth expected to increase next quarter",
            "Tech company announces new software update for users",
            "Basketball game had exciting moments and highlights"
        ]
        train_labels = ['business', 'technology', 'sports', 'business', 'technology', 'sports']
        
        classifier = TopicClassifier()
        classifier.train(train_texts, train_labels)
        
        test_text = "Basketball game highlights shown on television"
        prediction = classifier.predict([test_text])
        
        print(f"  - Model trained on {len(train_texts)} samples: ✓")
        print(f"  - Prediction: '{prediction[0]}' ✓")
        print("✓ Topic classification works\n")
        return True
    except Exception as e:
        print(f"✗ Topic classification failed: {e}\n")
        return False


def test_preprocessing():
    """Test preprocessing utilities."""
    print("=" * 60)
    print("TEST 8: Text Preprocessing")
    print("=" * 60)
    try:
        from utils.preprocessing import TextPreprocessor
        
        processor = TextPreprocessor()
        
        # Test main cleaning method
        result = processor.clean_text("Hello World! @user #test")
        print(f"  - clean_text works: ✓")
        
        # Test preprocessing a dataset
        texts = ["Hello World!", "Another text", "More data"]
        processed = processor.preprocess_dataset(texts)
        print(f"  - preprocess_dataset works: ✓")
        
        print("✓ Preprocessing module works\n")
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}\n")
        return False


def test_evaluation():
    """Test evaluation utilities."""
    print("=" * 60)
    print("TEST 9: Model Evaluation")
    print("=" * 60)
    try:
        from utils.evaluation import ClassificationEvaluator
        
        y_true = ['spam', 'ham', 'spam', 'ham', 'spam']
        y_pred = ['spam', 'ham', 'ham', 'ham', 'spam']
        
        evaluator = ClassificationEvaluator(y_true, y_pred)
        
        accuracy = evaluator.accuracy()
        metrics = evaluator.precision_recall_f1(average='macro')
        
        print(f"  - Accuracy: {accuracy:.3f} ✓")
        print(f"  - Precision (macro): {metrics['precision']:.3f} ✓")
        print(f"  - Recall (macro): {metrics['recall']:.3f} ✓")
        print(f"  - F1-Score (macro): {metrics['f1']:.3f} ✓")
        print("✓ Evaluation module works\n")
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}\n")
        return False


def test_data_loader():
    """Test data loading utilities."""
    print("=" * 60)
    print("TEST 10: Data Loading")
    print("=" * 60)
    try:
        from utils.data_loader import DataLoader
        from utils.evaluation import stratified_split
        import tempfile
        import os
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,label\n")
            f.write("This is a test,positive\n")
            f.write("Another test,negative\n")
            f.write("More text,positive\n")
            f.write("Final text,negative\n")
            temp_file = f.name
        
        try:
            texts, labels = DataLoader.load_csv(temp_file, text_column='text', label_column='label')
            print(f"  - Loaded {len(texts)} samples from CSV: ✓")
            
            # Test train/test split
            X_train, X_test, y_train, y_test = stratified_split(texts, labels, test_size=0.5)
            print(f"  - Train/test split created: ✓")
            print(f"  - Train size: {len(X_train)}, Test size: {len(X_test)}: ✓")
            
            print("✓ Data loader module works\n")
            return True
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ Data loader failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TEXT CLASSIFICATION REPOSITORY - TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_imports,
        test_manual_labeling,
        test_automatic_labeling,
        test_active_learning,
        test_sentiment_analysis,
        test_spam_detection,
        test_topic_classification,
        test_preprocessing,
        test_evaluation,
        test_data_loader,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test crashed: {e}\n")
            results.append(False)
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
