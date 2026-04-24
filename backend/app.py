"""
AI Customer Sentiment Analysis Dashboard — Flask Backend
=========================================================
Routes:
  POST /api/analyze          — analyze a single text
  POST /api/analyze/batch    — analyze multiple texts (JSON array)
  POST /api/analyze/csv      — upload CSV file for bulk analysis
  GET  /api/history          — paginated history from DB
  GET  /api/stats            — aggregate statistics for dashboard charts
  DELETE /api/history/<id>   — delete a single record
  GET  /api/export           — export history as CSV
  GET  /healthz              — health check
"""

import io
import csv
import os
import sys

# Add parent dir so `backend` imports work when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
import pandas as pd

from database import init_db, SessionLocal, Feedback
from model import analyzer

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static'),
)
CORS(app)  # Allow cross-origin requests (useful when frontend is on a different port)

# Initialize DB on startup
init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_db() -> Session:
    return SessionLocal()


def save_feedback(db: Session, text: str, result: dict, source: str = "manual", category: str = "General") -> Feedback:
    fb = Feedback(
        text=text,
        source=source,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        positive_score=result["positive_score"],
        negative_score=result["negative_score"],
        neutral_score=result["neutral_score"],
        category=category,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb


# ── Serve Frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/healthz")
def health():
    return jsonify({"status": "ok", "model": analyzer.model_name_used or "not loaded"})


@app.route("/api/analyze", methods=["POST"])
def analyze_single():
    """Analyze a single feedback text and persist it."""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required and must not be empty."}), 400

    category = data.get("category", "General")
    source = data.get("source", "manual")

    result = analyzer.analyze(text)

    db = get_db()
    fb = save_feedback(db, text, result, source=source, category=category)
    db.close()

    return jsonify({"record": fb.to_dict(), "raw_scores": result}), 201


@app.route("/api/analyze/batch", methods=["POST"])
def analyze_batch():
    """Analyze a JSON array of texts."""
    data = request.get_json(force=True)
    items = data if isinstance(data, list) else data.get("items", [])

    if not items:
        return jsonify({"error": "Provide a JSON array of {text, category?} objects."}), 400

    db = get_db()
    records = []
    for item in items:
        text = (item.get("text") or "").strip() if isinstance(item, dict) else str(item).strip()
        if not text:
            continue
        category = item.get("category", "General") if isinstance(item, dict) else "General"
        result = analyzer.analyze(text)
        fb = save_feedback(db, text, result, source="batch", category=category)
        records.append(fb.to_dict())
    db.close()

    return jsonify({"count": len(records), "records": records}), 201


@app.route("/api/analyze/csv", methods=["POST"])
def analyze_csv():
    """
    Upload a CSV file with a 'text' column (and optionally 'category').
    Returns analysis results and persists them to the DB.
    """
    if "file" not in request.files:
        return jsonify({"error": "Attach a CSV file under the key 'file'."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported."}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    if "text" not in df.columns:
        return jsonify({"error": "CSV must contain a 'text' column."}), 400

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]

    if len(df) > 500:
        return jsonify({"error": "CSV batch limit is 500 rows per request."}), 400

    db = get_db()
    records = []
    for _, row in df.iterrows():
        text = row["text"].strip()
        category = str(row.get("category", "General"))
        result = analyzer.analyze(text)
        fb = save_feedback(db, text, result, source="csv", category=category)
        records.append(fb.to_dict())
    db.close()

    return jsonify({"count": len(records), "records": records}), 201


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return paginated feedback history with optional filters."""
    page = int(request.args.get("page", 1))
    per_page = min(int(request.args.get("per_page", 20)), 100)
    sentiment_filter = request.args.get("sentiment")          # Positive|Negative|Neutral
    category_filter = request.args.get("category")
    search = request.args.get("search", "").strip()
    days = request.args.get("days")                           # last N days

    db = get_db()
    query = db.query(Feedback)

    if sentiment_filter:
        query = query.filter(Feedback.sentiment == sentiment_filter)
    if category_filter:
        query = query.filter(Feedback.category == category_filter)
    if search:
        query = query.filter(Feedback.text.ilike(f"%{search}%"))
    if days:
        since = datetime.utcnow() - timedelta(days=int(days))
        query = query.filter(Feedback.created_at >= since)

    total = query.count()
    records = (
        query.order_by(desc(Feedback.created_at))
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    db.close()

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "records": [r.to_dict() for r in records],
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Returns aggregated statistics for dashboard charts:
    - sentiment distribution (pie/donut)
    - daily sentiment trend (line chart, last 30 days)
    - average confidence per sentiment
    - top categories
    - recent records
    """
    days = int(request.args.get("days", 30))
    since = datetime.utcnow() - timedelta(days=days)

    db = get_db()

    # ── Overall distribution ──────────────────────────────────────────────
    dist_rows = (
        db.query(Feedback.sentiment, func.count(Feedback.id).label("count"))
        .group_by(Feedback.sentiment)
        .all()
    )
    distribution = {r.sentiment: r.count for r in dist_rows}
    total = sum(distribution.values())

    # ── Daily trend ───────────────────────────────────────────────────────
    trend_rows = (
        db.query(
            func.date(Feedback.created_at).label("date"),
            Feedback.sentiment,
            func.count(Feedback.id).label("count"),
        )
        .filter(Feedback.created_at >= since)
        .group_by(func.date(Feedback.created_at), Feedback.sentiment)
        .order_by("date")
        .all()
    )
    # Reshape into {date: {Positive: n, Negative: n, Neutral: n}}
    trend_map: dict[str, dict] = {}
    for row in trend_rows:
        d = str(row.date)
        if d not in trend_map:
            trend_map[d] = {"date": d, "Positive": 0, "Negative": 0, "Neutral": 0}
        trend_map[d][row.sentiment] = row.count
    trend = sorted(trend_map.values(), key=lambda x: x["date"])

    # ── Average confidence ────────────────────────────────────────────────
    conf_rows = (
        db.query(Feedback.sentiment, func.avg(Feedback.confidence).label("avg_conf"))
        .group_by(Feedback.sentiment)
        .all()
    )
    avg_confidence = {r.sentiment: round(float(r.avg_conf) * 100, 1) for r in conf_rows}

    # ── Top categories ────────────────────────────────────────────────────
    cat_rows = (
        db.query(Feedback.category, func.count(Feedback.id).label("count"))
        .group_by(Feedback.category)
        .order_by(desc("count"))
        .limit(10)
        .all()
    )
    categories = [{"category": r.category, "count": r.count} for r in cat_rows]

    # ── Recent 5 records ──────────────────────────────────────────────────
    recent = (
        db.query(Feedback)
        .order_by(desc(Feedback.created_at))
        .limit(5)
        .all()
    )
    db.close()

    return jsonify({
        "total": total,
        "distribution": distribution,
        "trend": trend,
        "avg_confidence": avg_confidence,
        "categories": categories,
        "recent": [r.to_dict() for r in recent],
    })


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_record(record_id: int):
    db = get_db()
    fb = db.query(Feedback).filter(Feedback.id == record_id).first()
    if not fb:
        db.close()
        return jsonify({"error": "Record not found."}), 404
    db.delete(fb)
    db.commit()
    db.close()
    return jsonify({"message": f"Record {record_id} deleted."})


@app.route("/api/export", methods=["GET"])
def export_csv():
    """Export all feedback history as a downloadable CSV."""
    db = get_db()
    records = db.query(Feedback).order_by(desc(Feedback.created_at)).all()
    db.close()

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["id", "text", "sentiment", "confidence", "positive_score",
                    "negative_score", "neutral_score", "category", "source", "created_at"],
    )
    writer.writeheader()
    for r in records:
        writer.writerow(r.to_dict())

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"sentiment_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load the model so the first request is fast
    #analyzer.load()
    app.run(host="0.0.0.0", port=5000, debug=False)
