"""Content-based news recommendation service using TF-IDF cosine similarity.

Algorithm
---------
1. Load a sample of the most recent articles from the cleaned dataset.
2. Build a TF-IDF matrix over headline + description text.
3. Category preference scoring: boost articles whose category matches the
   user's selected interests.
4. Bookmark similarity: for each bookmarked article compute cosine similarity
   to all candidates and add to the score.
5. Rank by combined score and return the top-N articles.

Memory: The TF-IDF matrix is sparse (~20-40 MB for 15 000 articles × 8 000
features). Cosine similarity is computed per-request; no dense all-pairs
matrix is stored.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("newslens.recommendations")

_MAX_ARTICLES = 15_000
_TFIDF = dict(max_features=8000, ngram_range=(1, 2), min_df=3,
               max_df=0.85, sublinear_tf=True, stop_words="english")


class RecommendationService:
    """Recommends articles based on category preferences and content
    similarity to bookmarked articles."""

    def __init__(self, dataset_path: Path | str, max_articles: int = _MAX_ARTICLES,
                 seed: int = 42):
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path, keep_default_na=False)

        # Prefer the most recent articles for relevance
        if "date" in df.columns:
            _d = pd.to_datetime(df["date"], errors="coerce")
            df = df.assign(_sort_date=_d).sort_values(
                "_sort_date", ascending=False, na_position="last"
            ).drop(columns=["_sort_date"])
        if len(df) > max_articles:
            df = df.head(max_articles)

        df = df.reset_index(drop=True)
        self._df = df
        self._categories = sorted(df["category"].unique().tolist())

        texts = (df["headline"].astype(str) + " " +
                 df.get("short_description", pd.Series("", index=df.index)).astype(str)
                 ).str.strip()
        self._tfidf = TfidfVectorizer(**_TFIDF)
        self._matrix = self._tfidf.fit_transform(texts)

        log.info("Recommendation service loaded: %d articles, %d features",
                 len(df), self._matrix.shape[1])

    # -- public properties ------------------------------------------------
    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    @property
    def article_count(self) -> int:
        return len(self._df)

    # -- recommendations --------------------------------------------------
    def get_recommendations(
        self,
        preferred_categories: list[str] | None = None,
        bookmarked_ids: list[int] | None = None,
        exclude_ids: list[int] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        n = len(self._df)
        scores = np.zeros(n, dtype=np.float64)
        has_prefs = bool(preferred_categories)
        has_bm = bool(bookmarked_ids)

        # 1. Category preference boost
        if has_prefs:
            upper = {c.upper() for c in preferred_categories}
            mask = self._df["category"].str.upper().isin(upper)
            scores[mask] += 3.0

        # 2. Bookmark similarity boost
        if has_bm:
            valid = [b for b in bookmarked_ids if 0 <= b < n]
            if valid:
                bk_vecs = self._matrix[valid]
                sims = cosine_similarity(bk_vecs, self._matrix).max(axis=0)
                scores += sims * 2.0

        # 3. Diversity noise
        rng = np.random.RandomState(42)
        scores += rng.random(n) * 0.1

        # 4. If no preferences, boost underrepresented categories
        if not has_prefs and not has_bm:
            inv = 1.0 / self._df["category"].map(self._df["category"].value_counts())
            scores += inv.values * 5.0

        exclude = set(exclude_ids or [])
        order = np.argsort(scores)[::-1]

        results: list[dict] = []
        for idx_np in order:
            idx = int(idx_np)
            if idx in exclude:
                continue
            row = self._df.iloc[idx]
            cat = str(row["category"])
            results.append({
                "article_id": idx,
                "headline": str(row["headline"]),
                "category": cat,
                "date": self._safe_date(row),
                "relevance_score": round(float(scores[idx]), 4),
                "reason": self._reason(cat, preferred_categories, has_bm),
            })
            if len(results) >= limit:
                break
        return results

    # -- similar articles -------------------------------------------------
    def get_similar(self, article_id: int, limit: int = 10) -> list[dict]:
        n = len(self._df)
        if article_id < 0 or article_id >= n:
            return []
        sims = cosine_similarity(
            self._matrix[article_id:article_id + 1], self._matrix
        )[0]
        order = np.argsort(sims)[::-1][1:limit + 1]
        return [
            {
                "article_id": int(i),
                "headline": str(self._df.iloc[int(i)]["headline"]),
                "category": str(self._df.iloc[int(i)]["category"]),
                "date": self._safe_date(self._df.iloc[int(i)]),
                "similarity_score": round(float(sims[i]), 4),
            }
            for i in order
        ]

    # -- single article ---------------------------------------------------
    def get_article(self, article_id: int) -> dict | None:
        if article_id < 0 or article_id >= len(self._df):
            return None
        row = self._df.iloc[article_id]
        return {
            "article_id": article_id,
            "headline": str(row["headline"]),
            "category": str(row["category"]),
            "short_description": str(row.get("short_description", "")),
            "date": self._safe_date(row),
        }

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _safe_date(row) -> str | None:
        v = row.get("date")
        if v is None or (isinstance(v, float) and v != v):
            return None
        s = str(v).strip()
        return s if s and s.lower() not in ("nat", "nan", "") else None

    @staticmethod
    def _reason(category: str, pref_cats: list[str] | None, has_bm: bool) -> str:
        if pref_cats and category.upper() in {c.upper() for c in pref_cats}:
            return f"Matches your preferred '{category}' category"
        if has_bm:
            return "Similar to articles you bookmarked"
        return "Recommended for content diversity"
