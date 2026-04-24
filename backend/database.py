"""
Database models and initialization for Sentiment Analysis Dashboard.
Uses SQLAlchemy with SQLite (easily swappable to PostgreSQL/MySQL).
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'data', 'sentiment.db')
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Feedback(Base):
    """Stores customer feedback with sentiment analysis results."""
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    source = Column(String(100), default="manual")          # e.g., manual, api, csv
    sentiment = Column(String(20), nullable=False)          # Positive / Negative / Neutral
    confidence = Column(Float, nullable=False)
    positive_score = Column(Float, default=0.0)
    negative_score = Column(Float, default=0.0)
    neutral_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    category = Column(String(100), default="General")       # optional tagging

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "sentiment": self.sentiment,
            "confidence": round(self.confidence * 100, 2),
            "positive_score": round(self.positive_score * 100, 2),
            "negative_score": round(self.negative_score * 100, 2),
            "neutral_score": round(self.neutral_score * 100, 2),
            "category": self.category,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


def init_db():
    """Create all tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized.")


def get_db():
    """Dependency: get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
