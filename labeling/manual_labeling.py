"""
Manual Text Labeling Tools

This module provides tools and utilities for manual text annotation.
Manual labeling is essential for creating high-quality training data,
especially for domain-specific tasks or when automatic labeling isn't feasible.

Key Concepts:
- Human-in-the-loop annotation
- Inter-annotator agreement
- Quality control mechanisms
- Annotation guidelines
"""

import pandas as pd
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time


class ManualLabeler:
    """
    A simple command-line interface for manual text labeling.

    This class provides a basic framework for annotating text data
    with predefined labels. It's useful for:
    - Creating training datasets
    - Validating automatic labels
    - Handling edge cases

    Attributes:
        labels: List of possible labels
        labeled_data: List of labeled examples
        statistics: Annotation statistics
    """

    def __init__(self, labels: List[str]):
        """
        Initialize the manual labeler.

        Args:
            labels: List of valid labels (e.g., ['positive', 'negative', 'neutral'])
        """
        self.labels = labels
        self.labeled_data = []
        self.statistics = {
            'total_labeled': 0,
            'time_spent': 0,
            'label_distribution': {label: 0 for label in labels}
        }

    def label_text(self, text: str) -> Optional[str]:
        """
        Label a single text instance interactively.

        Args:
            text: The text to label

        Returns:
            The selected label, or None if skipped
        """
        print("\n" + "="*80)
        print(f"Text to label:\n{text}")
        print("="*80)
        print("\nAvailable labels:")
        for idx, label in enumerate(self.labels, 1):
            print(f"  {idx}. {label}")
        print(f"  {len(self.labels) + 1}. Skip")
        print(f"  {len(self.labels) + 2}. Quit")

        while True:
            try:
                choice = input("\nEnter your choice (number): ").strip()
                choice_num = int(choice)

                if 1 <= choice_num <= len(self.labels):
                    selected_label = self.labels[choice_num - 1]
                    print(f"✓ Labeled as: {selected_label}")
                    return selected_label
                elif choice_num == len(self.labels) + 1:
                    print("⊘ Skipped")
                    return None
                elif choice_num == len(self.labels) + 2:
                    print("Quitting labeling session...")
                    return "QUIT"
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def label_dataset(self,
                      data: pd.DataFrame,
                      text_column: str = 'text',
                      output_path: Optional[str] = None,
                      start_index: int = 0) -> pd.DataFrame:
        """
        Label an entire dataset interactively.

        Args:
            data: DataFrame containing texts to label
            text_column: Name of the column containing text
            output_path: Path to save labeled data (saves after each label)
            start_index: Index to start labeling from

        Returns:
            DataFrame with labeled data
        """
        start_time = time.time()

        print(f"\n📝 Starting labeling session")
        print(f"   Total items: {len(data)}")
        print(f"   Starting from index: {start_index}")
        print(f"   Text column: {text_column}\n")

        # Create a copy with a label column if it doesn't exist
        if 'label' not in data.columns:
            data['label'] = None

        for idx in range(start_index, len(data)):
            text = data.iloc[idx][text_column]

            # Show progress
            progress = (idx + 1 - start_index) / \
                (len(data) - start_index) * 100
            print(f"\nProgress: {idx + 1}/{len(data)} ({progress:.1f}%)")

            label = self.label_text(text)

            if label == "QUIT":
                print(
                    f"\n✓ Labeled {self.statistics['total_labeled']} items before quitting")
                break
            elif label is not None:
                data.at[idx, 'label'] = label
                self.statistics['total_labeled'] += 1
                self.statistics['label_distribution'][label] += 1

                # Save progress
                if output_path:
                    data.to_csv(output_path, index=False)

        # Update statistics
        self.statistics['time_spent'] = time.time() - start_time

        # Display summary
        self._display_summary()

        return data

    def _display_summary(self):
        """Display annotation statistics."""
        print("\n" + "="*80)
        print("📊 LABELING SUMMARY")
        print("="*80)
        print(f"Total items labeled: {self.statistics['total_labeled']}")
        print(f"Time spent: {self.statistics['time_spent']:.2f} seconds")
        if self.statistics['total_labeled'] > 0:
            avg_time = self.statistics['time_spent'] / \
                self.statistics['total_labeled']
            print(f"Average time per item: {avg_time:.2f} seconds")

        print("\nLabel Distribution:")
        for label, count in self.statistics['label_distribution'].items():
            percentage = (count / self.statistics['total_labeled']
                          * 100) if self.statistics['total_labeled'] > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")


class AnnotationValidator:
    """
    Tools for validating annotation quality and computing inter-annotator agreement.

    This class helps ensure consistency and quality in manual annotations.
    """

    @staticmethod
    def calculate_cohen_kappa(annotator1: List[str],
                              annotator2: List[str]) -> float:
        """
        Calculate Cohen's Kappa coefficient for inter-annotator agreement.

        Cohen's Kappa measures agreement between two annotators while
        accounting for chance agreement.

        Interpretation:
        - < 0: Poor agreement
        - 0.0-0.20: Slight agreement
        - 0.21-0.40: Fair agreement
        - 0.41-0.60: Moderate agreement
        - 0.61-0.80: Substantial agreement
        - 0.81-1.00: Almost perfect agreement

        Args:
            annotator1: Labels from first annotator
            annotator2: Labels from second annotator

        Returns:
            Cohen's Kappa coefficient
        """
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(annotator1, annotator2)

    @staticmethod
    def find_disagreements(annotator1: List[str],
                           annotator2: List[str],
                           texts: List[str]) -> List[Dict]:
        """
        Find cases where annotators disagreed.

        Args:
            annotator1: Labels from first annotator
            annotator2: Labels from second annotator
            texts: Original texts

        Returns:
            List of disagreement cases with details
        """
        disagreements = []

        for idx, (label1, label2, text) in enumerate(zip(annotator1, annotator2, texts)):
            if label1 != label2:
                disagreements.append({
                    'index': idx,
                    'text': text,
                    'annotator1_label': label1,
                    'annotator2_label': label2
                })

        return disagreements

    @staticmethod
    def majority_vote(annotations: List[List[str]]) -> List[str]:
        """
        Resolve multiple annotations using majority voting.

        Args:
            annotations: List of label lists from different annotators

        Returns:
            Final labels based on majority vote
        """
        from collections import Counter

        final_labels = []
        num_items = len(annotations[0])

        for idx in range(num_items):
            # Get labels from all annotators for this item
            labels_for_item = [ann[idx] for ann in annotations]
            # Find most common label
            most_common = Counter(labels_for_item).most_common(1)[0][0]
            final_labels.append(most_common)

        return final_labels


class LabelingGuidelines:
    """
    Class to create and manage annotation guidelines.

    Clear guidelines are essential for consistent high-quality annotations.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.guidelines = {
            'task_description': '',
            'labels': {},
            'examples': [],
            'edge_cases': [],
            'dos_and_donts': []
        }

    def add_label_definition(self, label: str, definition: str, examples: List[str]):
        """
        Add definition and examples for a label.

        Args:
            label: The label name
            definition: Clear definition of when to use this label
            examples: Example texts for this label
        """
        self.guidelines['labels'][label] = {
            'definition': definition,
            'examples': examples
        }

    def add_edge_case(self, case_description: str, recommendation: str):
        """
        Document how to handle edge cases.

        Args:
            case_description: Description of the edge case
            recommendation: How to handle it
        """
        self.guidelines['edge_cases'].append({
            'case': case_description,
            'recommendation': recommendation
        })

    def export_to_markdown(self, output_path: str):
        """
        Export guidelines to a markdown file.

        Args:
            output_path: Path to save the markdown file
        """
        content = [f"# Annotation Guidelines: {self.task_name}\n"]

        if self.guidelines['task_description']:
            content.append(
                f"## Task Description\n{self.guidelines['task_description']}\n")

        content.append("## Labels\n")
        for label, info in self.guidelines['labels'].items():
            content.append(f"### {label}\n")
            content.append(f"{info['definition']}\n")
            content.append("**Examples:**\n")
            for ex in info['examples']:
                content.append(f"- {ex}\n")
            content.append("\n")

        if self.guidelines['edge_cases']:
            content.append("## Edge Cases\n")
            for ec in self.guidelines['edge_cases']:
                content.append(f"**{ec['case']}**\n")
                content.append(f"→ {ec['recommendation']}\n\n")

        Path(output_path).write_text('\n'.join(content))
        print(f"✓ Guidelines exported to {output_path}")


# Example usage and demonstration
if __name__ == "__main__":
    print("="*80)
    print("MANUAL TEXT LABELING EXAMPLES")
    print("="*80)

    # Example 1: Create sample data
    print("\n1. Creating sample dataset...")
    sample_data = pd.DataFrame({
        'text': [
            "This product exceeded my expectations!",
            "Terrible quality, waste of money.",
            "It's okay, nothing special.",
            "Absolutely love it! Best purchase ever.",
            "Not recommended, broke after one day."
        ]
    })

    # Example 2: Create annotation guidelines
    print("\n2. Creating annotation guidelines...")
    guidelines = LabelingGuidelines("Sentiment Analysis")
    guidelines.guidelines['task_description'] = "Classify the sentiment of product reviews"

    guidelines.add_label_definition(
        label="positive",
        definition="Use when the review expresses satisfaction, happiness, or positive emotions",
        examples=["Great product!", "Exceeded expectations", "Love it!"]
    )

    guidelines.add_label_definition(
        label="negative",
        definition="Use when the review expresses dissatisfaction, disappointment, or negative emotions",
        examples=["Terrible quality", "Waste of money", "Broke immediately"]
    )

    guidelines.add_label_definition(
        label="neutral",
        definition="Use when the review is balanced, factual, or lacks clear emotion",
        examples=["It's okay", "As described", "Average product"]
    )

    guidelines.add_edge_case(
        "Mixed sentiment",
        "If both positive and negative aspects are present, choose the overall dominant sentiment"
    )

    guidelines.export_to_markdown("labeling/annotation_guidelines.md")

    # Example 3: Demonstrate inter-annotator agreement
    print("\n3. Calculating inter-annotator agreement...")
    annotator1 = ['positive', 'negative', 'neutral', 'positive', 'negative']
    annotator2 = ['positive', 'negative', 'negative', 'positive', 'negative']

    validator = AnnotationValidator()
    kappa = validator.calculate_cohen_kappa(annotator1, annotator2)
    print(f"   Cohen's Kappa: {kappa:.3f}")

    if kappa >= 0.61:
        print("   ✓ Substantial agreement - good annotation quality!")
    elif kappa >= 0.41:
        print("   ⚠ Moderate agreement - consider clarifying guidelines")
    else:
        print("   ⚠ Poor agreement - guidelines need improvement")

    # Example 4: Find disagreements
    disagreements = validator.find_disagreements(
        annotator1, annotator2, sample_data['text'].tolist()
    )

    if disagreements:
        print(f"\n   Found {len(disagreements)} disagreement(s):")
        for d in disagreements:
            print(f"   - Index {d['index']}: '{d['text'][:50]}...'")
            print(f"     Annotator 1: {d['annotator1_label']}")
            print(f"     Annotator 2: {d['annotator2_label']}")

    print("\n" + "="*80)
    print("To start an interactive labeling session, uncomment the following code:")
    print("="*80)
    print("""
    # labeler = ManualLabeler(labels=['positive', 'negative', 'neutral'])
    # labeled_df = labeler.label_dataset(
    #     sample_data,
    #     text_column='text',
    #     output_path='labeled_data.csv'
    # )
    """)
