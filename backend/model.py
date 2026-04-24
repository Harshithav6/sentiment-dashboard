"""
Sentiment Analysis Model using Hugging Face Transformers.

Primary model : cardiffnlp/twitter-roberta-base-sentiment-latest
               (fine-tuned RoBERTa — state-of-the-art for sentiment)
Fallback model: distilbert-base-uncased-finetuned-sst-2-english
               (faster, lighter, if GPU/memory is constrained)

The module exposes a single SentimentAnalyzer class that lazily loads
the model on first call so the Flask app starts instantly.
"""

import re
import nltk
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Download NLTK data (punkt for tokenization, stopwords for preprocessing)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


# ── Label mapping ────────────────────────────────────────────────────────────
# cardiffnlp model uses 0→Negative, 1→Neutral, 2→Positive
CARDIFF_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# distilbert SST-2 uses 0→Negative, 1→Positive (no neutral)
DISTILBERT_LABEL_MAP = {0: "Negative", 1: "Positive"}


class SentimentAnalyzer:
    """
    Wraps a Hugging Face transformer for sentiment classification.
    Provides text preprocessing, inference, and structured output.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_map = None
        self.model_name_used = None
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self):
        """Load model & tokenizer (call once at startup or on first use)."""
        if self._loaded:
            return
        print(f"⏳ Loading model: {self.MODEL_NAME} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
self.label_map = {0: "Negative", 1: "Positive"}
self.model_name_used = self.MODEL_NAME
self.model.eval()
self._loaded = True
        self.model.eval()
        self._loaded = True

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text string.

        Returns:
            {
                "sentiment"       : "Positive" | "Negative" | "Neutral",
                "confidence"      : float  (0–1),
                "positive_score"  : float,
                "negative_score"  : float,
                "neutral_score"   : float,
                "preprocessed"    : str,
            }
        """
        if not self._loaded:
            self.load()

        cleaned = self._preprocess(text)
        scores = self._predict(cleaned)

        # ── Map scores to canonical keys ──────────────────────────────────
        # Both models output probs; we always return all three keys.
        if self.model_name_used == self.MODEL_NAME:
            # 3-class: Negative(0), Neutral(1), Positive(2)
            neg, neu, pos = scores[0], scores[1], scores[2]
        else:
            # 2-class SST-2: Negative(0), Positive(1) → set neutral = 0
            neg, pos = scores[0], scores[1]
            neu = 0.0

        sentiment_idx = int(np.argmax(scores))
        sentiment = self.label_map[sentiment_idx]
        confidence = float(scores[sentiment_idx])

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_score": float(pos),
            "negative_score": float(neg),
            "neutral_score": float(neu),
            "preprocessed": cleaned,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze a list of texts and return list of result dicts."""
        return [self.analyze(t) for t in texts]

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(text: str) -> str:
        """
        Clean text before inference:
        - lowercase
        - remove URLs
        - replace @mentions and #hashtags (Twitter convention)
        - remove excessive punctuation / whitespace
        """
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "@URL", text)
        text = re.sub(r"@\w+", "@USER", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = re.sub(r"[^\w\s@.,!?']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _predict(self, text: str) -> np.ndarray:
        """Tokenize → forward pass → softmax probabilities."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()
        return softmax(logits.numpy())


# ── Module-level singleton ────────────────────────────────────────────────────
analyzer = SentimentAnalyzer()
