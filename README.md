# NewsLens AI

NewsLens AI is an end-to-end machine learning project for classifying news headlines and article descriptions into categories, exposing the model through a FastAPI backend, and presenting insights through a React + Vite frontend.

The system combines:
- ML-based news category prediction
- sentiment analysis on headline/description text
- recommendation and discovery features
- analytics dashboards for model and category trends
- prediction history and user preferences

## Project Overview

This project is designed as a full-stack AI demo and educational application. It trains a text classification model on a news dataset, stores prediction history locally, and serves the results through a web application.

## Features

- News category prediction using TF-IDF + classical ML models
- Support for headline-only and combined headline + description inputs
- Explainable prediction output via model explanation endpoints
- Sentiment analysis for article text
- Personalized recommendations based on category preferences and bookmarks
- Category, trend, keyword, and timeline analytics
- Local prediction history tracking in SQLite
- Modern React dashboard UI

## Architecture

The repository is organized into a few main areas:

- `ml/` — data processing, model training, evaluation, and artifacts
- `backend/` — FastAPI API service and supporting business logic
- `frontend/` — React + Vite web client
- `Data/` — raw and processed datasets plus local database storage
- `artifacts/` — trained model files and evaluation JSON outputs
- `tests/` — backend and ML validation tests

## Tech Stack

### Machine Learning
- Python
- pandas
- NumPy
- scikit-learn
- joblib
- matplotlib

### Backend
- FastAPI
- Uvicorn
- HTTPX
- VaderSentiment

### Frontend
- React
- Vite
- Axios
- React Router
- Recharts
- Lucide React

## Repository Structure

```text
news-lens/
├── .gitignore
├── requirements.txt
├── README.md
├── Data/
│   ├── raw/
│   ├── processed/
│   └── history.db
├── artifacts/
│   ├── model.joblib
│   ├── model_metadata.json
│   ├── metrics.json
│   ├── model_comparison.json
│   ├── classification_report.json
│   ├── confusion_matrix.json
│   └── plots/
├── backend/
│   ├── requirements.txt
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       ├── config.py
│       ├── database.py
│       ├── main.py
│       ├── model_service.py
│       ├── recommendation_service.py
│       ├── analytics_service.py
│       ├── sentiment_service.py
│       └── schemas.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── src/
│   └── package-lock.json
├── ml/
│   ├── __init__.py
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── explore_dataset.py
│   ├── plots.py
│   └── text.py
├── tests/
│   ├── test_api.py
│   └── test_ml_pipeline.py
└── .
```

## Data

The ML pipeline uses the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) as the source dataset. The repository contains:

- `Data/raw/` — raw source data files
- `Data/processed/` — cleaned and prepared data used for training
- `Data/history.db` — SQLite database used for prediction history, bookmarks, and user preferences

## Model Pipeline

The ML workflow is implemented in `ml/train.py` and uses:

- TF-IDF text vectorization
- a small candidate set of classifiers
- validation-based model selection using macro-F1 as the primary metric
- final test-set evaluation
- artifact export for future inference and analytics

The project supports multiple input modes:
- `headline`
- `combined` (headline + short description)

Model selection is designed to compare a few classical models such as:
- Multinomial Naive Bayes
- Linear SVM
- Logistic Regression

## Backend API

The FastAPI app in `backend/app/main.py` exposes endpoints for:

- health checks
- prediction
- explanation
- sentiment analysis
- recommendation retrieval
- analytics queries
- preference management
- bookmark management
- prediction history

Example API routes include:

- `GET /api/health`
- `POST /api/predict`
- `POST /api/explain`
- `POST /api/sentiment/predict`
- `GET /api/recommendations`
- `GET /api/analytics/metrics`
- `GET /api/history`

## Frontend

The frontend is a React dashboard application that allows users to:

- predict the category of a news item
- browse analytics and visual data
- explore recommendations
- review previous predictions and saved bookmarks
- manage preferences

The main app uses React Router with dashboard sections for:
- Dashboard
- Predict
- Analytics
- Discover
- History

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- Git

### 1) Clone the Repository

```bash
git clone https://github.com/Sanskruti-Kunjir/news-lens.git
cd news-lens
```

### 2) Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 3) Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Preparing the Dataset and Training the Model

If the dataset is present and preprocessed, you can run the ML pipeline as follows:

```bash
python -m ml.prepare_dataset
python -m ml.train
```

If you want to run a smaller debug sample:

```bash
python -m ml.train --sample 20000
```

This creates trained model artifacts under `artifacts/`.

## Running the Application

### Start the Backend

From the project root:

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs will be available at:

```text
http://127.0.0.1:8000/docs
```

### Start the Frontend

```bash
cd frontend
npm run dev
```

Then open the Vite app in the browser, usually at:

```text
http://localhost:5173
```

## Testing

Run the project test suite with:

```bash
pytest
```

The project includes test coverage for:
- API behavior
- ML pipeline correctness and output validation

## Environment Variables

The backend supports the following environment variables:

```bash
NEWSLENS_ARTIFACTS_DIR
NEWSLENS_DB_PATH
NEWSLENS_DATASET_PATH
NEWSLENS_CORS_ORIGINS
```

These allow overriding default storage locations and CORS settings for local or deployment use.

## Notes

- The ML model is trained with a local dataset and generated artifacts.
- The backend reads artifact files from `artifacts/` when serving analytics and prediction info.
- The app stores user-specific history and preferences in SQLite at `Data/history.db`.

## License

This project is currently provided without a formal license file. If needed, add an appropriate open-source license before publishing or distributing the project externally.

## Contributing

Contributions are welcome. Typical improvements include:

- better model tuning
- expanded article metadata handling
- richer analytics and visualizations
- improved UI/UX for prediction and recommendation flows
- stronger production readiness and deployment support

## Author

Sanskruti Kunjir

## Project Goal

NewsLens AI demonstrates how to combine machine learning, backend APIs, and web interfaces into a practical news classification and insight application.

---

Built for intelligent news analysis, recommendations, and category-aware exploration.
