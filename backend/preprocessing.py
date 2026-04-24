"""
Text Preprocessing Utilities
=============================
Standalone NLP preprocessing functions used both by the model
and for exploratory analysis / reporting.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure NLTK assets are available
for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def basic_clean(text: str) -> str:
    """Lowercase + remove URLs, mentions, hashtags, extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _stop_words and t.isalpha()]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def stem(tokens: list[str]) -> list[str]:
    return [_stemmer.stem(t) for t in tokens]


def full_pipeline(text: str, use_lemma: bool = True) -> str:
    """
    Complete preprocessing pipeline:
    clean → remove punct → tokenize → remove stopwords → lemmatize/stem → join
    """
    text   = basic_clean(text)
    text   = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens) if use_lemma else stem(tokens)
    return " ".join(tokens)


def batch_preprocess(texts: list[str]) -> list[str]:
    return [full_pipeline(t) for t in texts]
