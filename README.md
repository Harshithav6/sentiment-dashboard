# 🧠 SentimentIQ — AI Customer Sentiment Analysis Dashboard

**LaunchED Global Internship · Artificial Intelligence Major Capstone Project · Option 1**

---

## Overview

SentimentIQ is a full-stack AI application that automatically classifies customer feedback
as **Positive**, **Negative**, or **Neutral** using a state-of-the-art transformer model
(RoBERTa), and visualizes trends through an interactive real-time dashboard.

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Model** | `cardiffnlp/twitter-roberta-base-sentiment-latest` (Hugging Face Transformers) |
| **Preprocessing** | NLTK (tokenization, lemmatization, stopword removal) |
| **Backend API** | Python · Flask · SQLAlchemy (SQLite) |
| **Frontend** | Vanilla HTML/CSS/JS · Chart.js |
| **Deployment** | Gunicorn · Render / Railway / Heroku |

---

## Project Structure

```
sentiment_dashboard/
├── backend/
│   ├── app.py            ← Flask API (all routes)
│   ├── model.py          ← RoBERTa sentiment model wrapper
│   ├── database.py       ← SQLAlchemy models + DB setup
│   ├── preprocessing.py  ← NLP text preprocessing utilities
│   ├── evaluate.py       ← Evaluation & report generator
│   └── seed_data.py      ← Demo data seeder
├── frontend/
│   └── templates/
│       └── index.html    ← Full dashboard UI (served by Flask)
├── data/                 ← SQLite database (auto-created)
├── reports/              ← Generated evaluation plots & report
├── requirements.txt
├── Procfile              ← For Render/Heroku deployment
└── README.md
```

---

## Quick Start (Local)

### 1. Clone / extract the project
```bash
cd sentiment_dashboard
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Seed demo data (optional but recommended)
```bash
cd backend
python seed_data.py
cd ..
```
> ⚠️ First run downloads the model (~500 MB). Subsequent runs are instant.

### 5. Start the server
```bash
python backend/app.py
```

Open your browser at: **http://localhost:5000**

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Dashboard UI |
| `POST` | `/api/analyze` | Analyze a single feedback text |
| `POST` | `/api/analyze/batch` | Analyze JSON array of texts |
| `POST` | `/api/analyze/csv` | Upload CSV file for bulk analysis |
| `GET`  | `/api/history` | Paginated feedback history |
| `GET`  | `/api/stats` | Aggregated stats for charts |
| `DELETE` | `/api/history/<id>` | Delete a record |
| `GET`  | `/api/export` | Download full history as CSV |
| `GET`  | `/healthz` | Health check |

### Example: Analyze a single text
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product! Exceeded all my expectations.", "category": "Product"}'
```

### Example: Bulk JSON
```bash
curl -X POST http://localhost:5000/api/analyze/batch \
  -H "Content-Type: application/json" \
  -d '[{"text":"Great service!","category":"Support"},{"text":"Terrible experience","category":"Delivery"}]'
```

---

## Generate Evaluation Report

After populating data:
```bash
python backend/evaluate.py
```

This generates in `/reports/`:
- `evaluation_report.md` — metrics & insights
- `sentiment_distribution.png`
- `confidence_distribution.png`
- `confusion_matrix.png`

---

## Deployment on Render (Free Tier)

1. Push code to a GitHub repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Deploy!

### For Railway
```bash
railway init
railway up
```

### For Heroku
```bash
heroku create your-app-name
git push heroku main
```

---

## Features

- ✅ **Real-time sentiment inference** via RoBERTa transformer
- ✅ **Interactive dashboard** with donut chart, trend line, confidence bars
- ✅ **Single text analysis** with score breakdown
- ✅ **Bulk CSV upload** (up to 500 rows)
- ✅ **Filterable history** with search, sentiment filter, date range
- ✅ **CSV export** of all records
- ✅ **SQLite database** (zero config, swappable to PostgreSQL)
- ✅ **Evaluation script** with confusion matrix and classification report
- ✅ **One-command deployment** via Gunicorn/Procfile

---

## Model Details

The primary model is **`cardiffnlp/twitter-roberta-base-sentiment-latest`**:
- Based on RoBERTa, fine-tuned on ~124M tweets
- 3-class output: Positive, Negative, Neutral
- Handles informal language, slang, and domain-shifted text well

A lighter **DistilBERT** fallback is used if the primary model cannot be downloaded.

---

## Author

Built as part of the LaunchED Global AI Major Capstone Project.
