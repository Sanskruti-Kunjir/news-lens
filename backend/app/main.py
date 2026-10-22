"""NewsLens AI - FastAPI application.

Run from the project root:
    python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app import config, database as db
from backend.app.model_service import ModelService
from backend.app.schemas import (
    PredictRequest, PredictResponse, ExplainRequest,
    SentimentRequest, PreferencesUpdate, BookmarkRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("newslens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db(config.db_path())

    # -- category model
    app.state.model, app.state.model_error = None, None
    try:
        app.state.model = ModelService(config.artifacts_dir())
        log.info("Model loaded: %s", app.state.model.metadata["model_name"])
    except (FileNotFoundError, ValueError) as exc:
        app.state.model_error = str(exc)
        log.warning("Model not loaded: %s", exc)

    # -- sentiment service
    app.state.sentiment, app.state.sentiment_error = None, None
    try:
        from backend.app.sentiment_service import SentimentService
        app.state.sentiment = SentimentService()
    except Exception as exc:
        app.state.sentiment_error = str(exc)
        log.warning("Sentiment not loaded: %s", exc)

    # -- recommendation service
    app.state.recommender, app.state.recommender_error = None, None
    try:
        from backend.app.recommendation_service import RecommendationService
        ds = config.dataset_path()
        if ds.exists():
            app.state.recommender = RecommendationService(ds)
        else:
            app.state.recommender_error = f"Dataset not found at {ds}"
            log.warning("Recommender not loaded: dataset missing")
    except Exception as exc:
        app.state.recommender_error = str(exc)
        log.warning("Recommender not loaded: %s", exc)

    # -- analytics service
    app.state.analytics = None
    try:
        from backend.app.analytics_service import AnalyticsService
        app.state.analytics = AnalyticsService(config.artifacts_dir(), config.dataset_path())
    except Exception as exc:
        log.warning("Analytics not loaded: %s", exc)

    yield


app = FastAPI(title="NewsLens AI", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins(), allow_methods=["*"], allow_headers=["*"])


def read_artifact(name: str):
    path = config.artifacts_dir() / name
    if not path.exists():
        raise HTTPException(503, f"{name} is not available yet. Run the ML scripts first (see README).")
    return json.loads(path.read_text(encoding="utf-8"))


# ======================================================================= health
@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.model is not None,
            "sentiment_loaded": request.app.state.sentiment is not None,
            "recommender_loaded": request.app.state.recommender is not None}


@app.get("/api/model-info")
def model_info(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, request.app.state.model_error or "Model not loaded.")
    m = model.metadata
    return {"model_name": m["model_name"], "model_version": m["model_version"], "params": m["params"],
            "input_mode": m["input_mode"], "supported_input_modes": model.supported_modes,
            "score_type": m["score_type"], "n_categories": m["n_categories"], "trained_at_utc": m["trained_at_utc"],
            "sample_size": m.get("sample_size"), "split": m["split"]}


# ======================================================================= predict
@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, request.app.state.model_error or "Model not loaded.")
    try:
        result = model.predict(req.headline, req.short_description, req.input_mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    # Sentiment (optional, never blocks the prediction)
    sentiment = None
    if request.app.state.sentiment:
        try:
            text = f"{req.headline} {req.short_description}".strip()
            sentiment = request.app.state.sentiment.analyze(text)
        except Exception:
            log.exception("Sentiment failed (non-blocking)")

    history_id = None
    try:
        history_id = db.insert(config.db_path(), req.headline, req.short_description, result, sentiment)
    except Exception:  # storing history must never break a prediction
        log.exception("Could not save prediction history")
    return {**result, "sentiment": sentiment, "history_id": history_id}


# ======================================================================= explain
@app.post("/api/explain")
def explain(req: ExplainRequest, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, request.app.state.model_error or "Model not loaded.")
    try:
        return model.explain_detailed(req.headline, req.description)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ==================================================================== sentiment
@app.post("/api/sentiment/predict")
def sentiment_predict(req: SentimentRequest, request: Request):
    svc = request.app.state.sentiment
    if svc is None:
        raise HTTPException(503, request.app.state.sentiment_error or "Sentiment analysis unavailable.")
    text = f"{req.headline} {req.description}".strip()
    return svc.analyze(text)


# ---- offline evaluation artifacts (read straight from the files train.py wrote)
@app.get("/api/analytics/metrics")
def metrics():
    return read_artifact("metrics.json")


@app.get("/api/analytics/model-comparison")
def model_comparison():
    return read_artifact("model_comparison.json")


@app.get("/api/analytics/classification-report")
def classification_report():
    return read_artifact("classification_report.json")


@app.get("/api/analytics/confusion-matrix")
def confusion_matrix():
    return read_artifact("confusion_matrix.json")


@app.get("/api/dataset/summary")
def dataset_summary():
    return read_artifact("dataset_summary.json")


# ============================================================== recommendations
@app.get("/api/recommendations")
def recommendations(request: Request, limit: int = Query(20, ge=1, le=100)):
    rec = request.app.state.recommender
    if rec is None:
        raise HTTPException(503, request.app.state.recommender_error or "Recommendation service unavailable.")
    prefs = db.get_preferences(config.db_path())
    bookmarks = db.list_bookmarks(config.db_path())
    bk_ids = [b["article_id"] for b in bookmarks]
    items = rec.get_recommendations(
        preferred_categories=prefs.get("preferred_categories"),
        bookmarked_ids=bk_ids or None,
        exclude_ids=bk_ids or None,
        limit=limit,
    )
    return {"items": items, "total": len(items),
            "preferences": prefs.get("preferred_categories", []),
            "available_categories": rec.categories}


@app.get("/api/recommendations/similar/{article_id}")
def similar_articles(article_id: int, request: Request, limit: int = Query(10, ge=1, le=50)):
    rec = request.app.state.recommender
    if rec is None:
        raise HTTPException(503, "Recommendation service unavailable.")
    return {"items": rec.get_similar(article_id, limit)}


# ================================================================== preferences
@app.get("/api/preferences")
def get_preferences():
    return db.get_preferences(config.db_path())


@app.put("/api/preferences")
def update_preferences(body: PreferencesUpdate):
    return db.upsert_preferences(config.db_path(), "local", body.preferred_categories)


# ==================================================================== bookmarks
@app.get("/api/bookmarks")
def get_bookmarks():
    return {"items": db.list_bookmarks(config.db_path())}


@app.post("/api/bookmarks")
def add_bookmark(body: BookmarkRequest):
    return db.add_bookmark(config.db_path(), "local", body.article_id, body.headline, body.category)


@app.delete("/api/bookmarks/{bookmark_id}")
def delete_bookmark(bookmark_id: int):
    if not db.remove_bookmark(config.db_path(), bookmark_id):
        raise HTTPException(404, "Bookmark not found.")
    return {"deleted": bookmark_id}


# ==================================================================== analytics
@app.get("/api/analytics/categories")
def analytics_categories(request: Request):
    svc = request.app.state.analytics
    if svc is None:
        raise HTTPException(503, "Analytics service unavailable.")
    return svc.get_categories()


@app.get("/api/analytics/trends")
def analytics_trends(request: Request):
    svc = request.app.state.analytics
    if svc is None:
        raise HTTPException(503, "Analytics service unavailable.")
    return svc.get_trends()


@app.get("/api/analytics/keywords")
def analytics_keywords(request: Request,
                       category: str | None = Query(None, max_length=100),
                       limit: int = Query(25, ge=1, le=100)):
    svc = request.app.state.analytics
    if svc is None:
        raise HTTPException(503, "Analytics service unavailable.")
    return svc.get_keywords(category=category, limit=limit)


@app.get("/api/analytics/timeline")
def analytics_timeline(request: Request):
    svc = request.app.state.analytics
    if svc is None:
        raise HTTPException(503, "Analytics service unavailable.")
    return svc.get_timeline()


@app.get("/api/analytics/sentiment-distribution")
def analytics_sentiment(request: Request):
    from backend.app.analytics_service import AnalyticsService
    return AnalyticsService.get_prediction_stats(config.db_path())


@app.get("/api/analytics/prediction-history")
def analytics_prediction_history():
    from backend.app.analytics_service import AnalyticsService
    return AnalyticsService.get_prediction_stats(config.db_path())


# ============================================================ prediction history
@app.get("/api/history")
def list_history(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100),
                 search: str | None = Query(None, max_length=200), category: str | None = Query(None, max_length=100),
                 order: str = Query("newest", pattern="^(newest|oldest)$")):
    items, total = db.list_history(config.db_path(), page, page_size, search, category, order == "newest")
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.get("/api/history/{record_id}")
def get_history(record_id: int):
    rec = db.get_one(config.db_path(), record_id)
    if rec is None:
        raise HTTPException(404, "Record not found.")
    return rec


@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
    if not db.delete_one(config.db_path(), record_id):
        raise HTTPException(404, "Record not found.")
    return {"deleted": record_id}


@app.delete("/api/history")
def clear_history():
    return {"deleted": db.clear(config.db_path())}
