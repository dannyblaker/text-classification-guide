"""
Data Loading Utilities

This module provides utilities for loading and managing text classification datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class DataLoader:
    """
    Utility class for loading text classification datasets.
    """

    @staticmethod
    def load_csv(filepath: str,
                 text_column: str = 'text',
                 label_column: str = 'label') -> Tuple[List[str], List[str]]:
        """
        Load dataset from CSV file.

        Args:
            filepath: Path to CSV file
            text_column: Name of column containing text
            label_column: Name of column containing labels

        Returns:
            Tuple of (texts, labels)
        """
        df = pd.read_csv(filepath)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV")

        texts = df[text_column].fillna('').tolist()
        labels = df[label_column].fillna('').tolist()

        return texts, labels

    @staticmethod
    def save_csv(texts: List[str],
                 labels: List[str],
                 filepath: str,
                 text_column: str = 'text',
                 label_column: str = 'label'):
        """
        Save dataset to CSV file.

        Args:
            texts: List of texts
            labels: List of labels
            filepath: Output file path
            text_column: Name for text column
            label_column: Name for label column
        """
        df = pd.DataFrame({
            text_column: texts,
            label_column: labels
        })

        df.to_csv(filepath, index=False)
        print(f"✓ Saved {len(df)} samples to {filepath}")

    @staticmethod
    def load_from_directory(directory: str,
                            file_pattern: str = '*.txt') -> Tuple[List[str], List[str]]:
        """
        Load dataset from directory structure where subdirectories are labels.

        Expected structure:
        directory/
          ├── label1/
          │   ├── file1.txt
          │   └── file2.txt
          └── label2/
              ├── file3.txt
              └── file4.txt

        Args:
            directory: Root directory path
            file_pattern: Pattern for text files

        Returns:
            Tuple of (texts, labels)
        """
        from pathlib import Path

        root_path = Path(directory)
        texts = []
        labels = []

        for label_dir in root_path.iterdir():
            if label_dir.is_dir():
                label = label_dir.name

                for file_path in label_dir.glob(file_pattern):
                    if file_path.is_file():
                        text = file_path.read_text(
                            encoding='utf-8', errors='ignore')
                        texts.append(text)
                        labels.append(label)

        return texts, labels

    @staticmethod
    def create_train_test_files(texts: List[str],
                                labels: List[str],
                                output_dir: str,
                                test_size: float = 0.2,
                                random_state: int = 42):
        """
        Split data and save to separate train/test files.

        Args:
            texts: List of texts
            labels: List of labels
            output_dir: Directory to save files
            test_size: Proportion of test set
            random_state: Random seed
        """
        from sklearn.model_selection import train_test_split
        from pathlib import Path

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save files
        DataLoader.save_csv(X_train, y_train, f"{output_dir}/train.csv")
        DataLoader.save_csv(X_test, y_test, f"{output_dir}/test.csv")

        print(
            f"✓ Created train ({len(X_train)}) and test ({len(X_test)}) sets")


def generate_sample_data(n_samples: int = 100,
                         n_classes: int = 3,
                         random_state: int = 42) -> Tuple[List[str], List[str]]:
    """
    Generate synthetic text classification data for testing.

    Args:
        n_samples: Number of samples to generate
        n_classes: Number of classes
        random_state: Random seed

    Returns:
        Tuple of (texts, labels)
    """
    np.random.seed(random_state)

    class_names = [f"class_{i}" for i in range(n_classes)]

    # Generate texts with class-specific keywords
    texts = []
    labels = []

    for _ in range(n_samples):
        class_idx = np.random.randint(0, n_classes)
        class_name = class_names[class_idx]

        # Add class-specific keywords
        keywords = [f"keyword_{class_idx}_{i}" for i in range(3)]
        common_words = ["the", "a", "is", "and", "of", "to", "in"]

        # Generate text
        words = keywords + list(np.random.choice(common_words, size=5))
        np.random.shuffle(words)
        text = " ".join(words)

        texts.append(text)
        labels.append(class_name)

    return texts, labels


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("DATA LOADING UTILITIES DEMO")
    print("="*80)

    # Generate sample data
    print("\n1. Generating sample data...")
    texts, labels = generate_sample_data(n_samples=50, n_classes=3)
    print(f"   Generated {len(texts)} samples with {len(set(labels))} classes")

    # Save to CSV
    print("\n2. Saving to CSV...")
    DataLoader.save_csv(texts, labels, '/tmp/sample_data.csv')

    # Load from CSV
    print("\n3. Loading from CSV...")
    loaded_texts, loaded_labels = DataLoader.load_csv('/tmp/sample_data.csv')
    print(f"   Loaded {len(loaded_texts)} samples")

    # Create train/test split
    print("\n4. Creating train/test split...")
    DataLoader.create_train_test_files(texts, labels, '/tmp/data_split')

    print("\n" + "="*80)
    print("✓ Data loading utilities demonstrated successfully!")
