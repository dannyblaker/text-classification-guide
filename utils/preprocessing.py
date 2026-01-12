"""
Text Preprocessing Utilities

This module provides common text preprocessing functions used in
text classification tasks. Proper preprocessing is crucial for:
- Improving model performance
- Reducing vocabulary size
- Normalizing text variations
- Handling noise in data
"""

import re
import string
from typing import List, Optional
import warnings


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline.

    This class provides various preprocessing steps that can be
    combined in a pipeline for text classification tasks.
    """

    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 remove_extra_whitespace: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_html: bool = True,
                 expand_contractions: bool = False,
                 remove_stopwords: bool = False,
                 lemmatize: bool = False,
                 stem: bool = False):
        """
        Initialize preprocessor with desired options.

        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric digits
            remove_extra_whitespace: Normalize whitespace
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_html: Remove HTML tags
            expand_contractions: Expand contractions (can't -> cannot)
            remove_stopwords: Remove common stopwords
            lemmatize: Lemmatize words (requires spaCy)
            stem: Stem words (requires NLTK)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem

        # Initialize tools if needed
        if self.remove_stopwords:
            try:
                import nltk
                nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words('english'))
            except:
                warnings.warn("NLTK stopwords not available")
                self.remove_stopwords = False

        if self.stem:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except:
                warnings.warn("NLTK stemmer not available")
                self.stem = False

        if self.lemmatize:
            try:
                import spacy
                self.nlp = spacy.load(
                    'en_core_web_sm', disable=['parser', 'ner'])
            except:
                warnings.warn("spaCy not available for lemmatization")
                self.lemmatize = False

    def clean_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        if self.remove_html:
            text = self._remove_html_tags(text)

        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Remove emails
        if self.remove_emails:
            text = self._remove_emails(text)

        # Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Lemmatize or stem
        if self.lemmatize:
            text = self._lemmatize_text(text)
        elif self.stem:
            text = self._stem_text(text)

        # Remove stopwords
        if self.remove_stopwords:
            text = self._remove_stopwords(text)

        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = ' '.join(text.split())

        return text.strip()

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r'\S+@\S+', '', text)

    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords]
        return ' '.join(filtered)

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize text using spaCy."""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def _stem_text(self, text: str) -> str:
        """Stem text using Porter stemmer."""
        words = text.split()
        stemmed = [self.stemmer.stem(w) for w in words]
        return ' '.join(stemmed)

    def preprocess_dataset(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of preprocessed texts
        """
        return [self.clean_text(text) for text in texts]


def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenization.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    return text.split()


def remove_special_characters(text: str, keep_punctuation: bool = False) -> str:
    """
    Remove special characters from text.

    Args:
        text: Input text
        keep_punctuation: Whether to keep punctuation marks

    Returns:
        Cleaned text
    """
    if keep_punctuation:
        pattern = r'[^a-zA-Z0-9\s\.,!?;:\'\"-]'
    else:
        pattern = r'[^a-zA-Z0-9\s]'

    return re.sub(pattern, '', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize multiple whitespaces to single space.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    return ' '.join(text.split())


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text.

    Args:
        text: Input text

    Returns:
        List of hashtags
    """
    return re.findall(r'#\w+', text)


def extract_mentions(text: str) -> List[str]:
    """
    Extract mentions (@username) from text.

    Args:
        text: Input text

    Returns:
        List of mentions
    """
    return re.findall(r'@\w+', text)


def truncate_text(text: str, max_length: int = 512, add_ellipsis: bool = True) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum character length
        add_ellipsis: Whether to add '...' at the end

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]

    if add_ellipsis:
        truncated = truncated.rsplit(' ', 1)[0] + '...'

    return truncated


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("TEXT PREPROCESSING EXAMPLES")
    print("="*80)

    # Sample text with various issues
    sample_text = """
    Hey @user! Check out this AMAZING product at http://example.com 🎉
    It's the BEST!!! Contact us at support@example.com
    <p>Price: $99.99</p> Can't believe it's this good! #awesome #deal
    """

    print("\nOriginal Text:")
    print(sample_text)

    # Example 1: Basic preprocessing
    print("\n1. BASIC PREPROCESSING")
    print("-" * 80)
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=False,
        remove_urls=True,
        remove_emails=True,
        remove_html=True
    )
    cleaned = preprocessor.clean_text(sample_text)
    print(cleaned)

    # Example 2: Aggressive preprocessing
    print("\n2. AGGRESSIVE PREPROCESSING")
    print("-" * 80)
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        remove_stopwords=True
    )
    cleaned = preprocessor.clean_text(sample_text)
    print(cleaned)

    # Example 3: Extract features
    print("\n3. FEATURE EXTRACTION")
    print("-" * 80)
    print(f"Hashtags: {extract_hashtags(sample_text)}")
    print(f"Mentions: {extract_mentions(sample_text)}")

    # Example 4: Different preprocessing for different tasks
    print("\n4. TASK-SPECIFIC PREPROCESSING")
    print("-" * 80)

    print("\nFor Sentiment Analysis (keep some punctuation for emphasis):")
    sentiment_prep = TextPreprocessor(
        lowercase=True,
        remove_punctuation=False,
        remove_urls=True,
        remove_html=True
    )
    print(sentiment_prep.clean_text(sample_text))

    print("\nFor Topic Classification (aggressive cleaning):")
    topic_prep = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_urls=True,
        remove_html=True,
        remove_stopwords=True
    )
    print(topic_prep.clean_text(sample_text))

    print("\n" + "="*80)
    print("PREPROCESSING GUIDELINES")
    print("="*80)
    print("""
Task                    | Recommended Preprocessing
------------------------|-------------------------------------------------------
Sentiment Analysis      | Keep punctuation (!), lowercase, remove URLs/HTML
Spam Detection          | Keep some special chars, lowercase, normalize
Topic Classification    | Aggressive: remove stopwords, lemmatize, lowercase
Intent Detection        | Minimal: lowercase, remove URLs, keep structure
Language Detection      | Minimal: just normalize whitespace
    """)
