"""
Seed Database with Sample Data
================================
Populates the database with realistic-looking customer feedback samples
so you can immediately see the dashboard in action.

Usage:
    python seed_data.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db, SessionLocal, Feedback
from model import analyzer
from datetime import datetime, timedelta
import random

SAMPLES = [
    # Positive
    ("The product quality exceeded my expectations. Will definitely order again!", "Product"),
    ("Super fast delivery! Arrived two days early and packaging was perfect.", "Delivery"),
    ("Customer support resolved my issue in under 10 minutes. Amazing service.", "Support"),
    ("Best purchase I've made this year. Great value for the price.", "Pricing"),
    ("The app is intuitive and beautifully designed. Love the new update!", "UX/UI"),
    ("Very impressed with the build quality. Feels premium and durable.", "Product"),
    ("Shipping was lightning fast and the item was exactly as described.", "Delivery"),
    ("The support team went above and beyond to help me. Truly exceptional.", "Support"),
    ("Fantastic product at a very competitive price. Highly recommended.", "Pricing"),
    ("Seamless checkout experience. The website is so easy to navigate.", "UX/UI"),
    # Negative
    ("The item arrived damaged and the packaging was completely crushed.", "Delivery"),
    ("Waited 3 weeks and still no update on my order. Very frustrating.", "Delivery"),
    ("The product stopped working after just 2 days. Terrible quality.", "Product"),
    ("Customer support kept passing me around and never resolved my issue.", "Support"),
    ("Way overpriced for what you actually get. Not worth it at all.", "Pricing"),
    ("The website crashed twice during checkout. Lost my order twice.", "UX/UI"),
    ("Received completely wrong item and no response from support.", "Support"),
    ("Quality is nothing like the product photos. Feel deceived.", "Product"),
    # Neutral
    ("Package arrived on time. Product does what it says, nothing more.", "General"),
    ("It's okay. Not as impressive as I expected but it works fine.", "Product"),
    ("Delivery was average. Took the standard time, no issues.", "Delivery"),
    ("Support answered my question but couldn't fully solve the problem.", "Support"),
    ("Price seems fair for the market. Nothing particularly special.", "Pricing"),
    ("The interface is functional but could use some design improvements.", "UX/UI"),
    ("Product is decent. I might order again if the price drops.", "Pricing"),
    ("The app is okay. Some features I expected are missing.", "UX/UI"),
]


def seed():
    init_db()
    analyzer.load()

    db = SessionLocal()
    now = datetime.utcnow()

    print(f"🌱 Seeding {len(SAMPLES)} sample records…")
    for i, (text, category) in enumerate(SAMPLES):
        result = analyzer.analyze(text)
        # Spread created_at over the past 30 days for trend chart
        offset = timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))
        fb = Feedback(
            text=text,
            source="seed",
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            positive_score=result["positive_score"],
            negative_score=result["negative_score"],
            neutral_score=result["neutral_score"],
            category=category,
            created_at=now - offset,
        )
        db.add(fb)
        print(f"  [{i+1:02d}] {result['sentiment']:8s} ({result['confidence']*100:.1f}%) — {text[:55]}…")

    db.commit()
    db.close()
    print(f"\n✅ {len(SAMPLES)} records seeded successfully!")


if __name__ == "__main__":
    seed()
