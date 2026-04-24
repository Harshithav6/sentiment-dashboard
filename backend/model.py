from textblob import TextBlob

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return {
            "sentiment": "Positive",
            "confidence": float(polarity),
            "positive_score": float(polarity),
            "negative_score": 0.0,
            "neutral_score": 0.0,
            "preprocessed": text.lower()
        }
    elif polarity < 0:
        return {
            "sentiment": "Negative",
            "confidence": float(abs(polarity)),
            "positive_score": 0.0,
            "negative_score": float(abs(polarity)),
            "neutral_score": 0.0,
            "preprocessed": text.lower()
        }
    else:
        return {
            "sentiment": "Neutral",
            "confidence": 1.0,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_score": 1.0,
            "preprocessed": text.lower()
        }
