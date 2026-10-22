"""Sentiment analysis service using VADER (rule-based, no download needed).

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and
rule-based sentiment analysis tool specifically attuned to social media and
news text.  It requires no GPU, no model download, and works offline.

Limitations documented per the project spec:
- Rule-based, not a trained ML model.
- May be inaccurate for sarcasm, mixed emotions, or context-dependent language.
- Scores are intensity proportions, not calibrated probabilities.
"""
from __future__ import annotations

import logging

log = logging.getLogger("newslens.sentiment")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    log.warning("vaderSentiment not installed. Run: pip install vaderSentiment")


class SentimentService:
    """VADER-based sentiment analyzer for news headlines and descriptions."""

    MODEL_NAME = "VADER"
    MODEL_FULL = "VADER (Valence Aware Dictionary and sEntiment Reasoner)"

    def __init__(self):
        if not _AVAILABLE:
            raise ImportError(
                "vaderSentiment is not installed. Install with: pip install vaderSentiment"
            )
        self._analyzer = SentimentIntensityAnalyzer()
        log.info("Sentiment service loaded (VADER)")

    def analyze(self, text: str) -> dict:
        """Analyse sentiment of *text* and return label, scores, and metadata."""
        if not text or not text.strip():
            return self._empty_result("Empty text defaults to Neutral.")

        scores = self._analyzer.polarity_scores(text.strip())
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "sentiment": label,
            "model_name": self.MODEL_NAME,
            "scores": {
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
                "compound": round(compound, 4),
            },
            "is_pretrained": True,
            "note": (
                f"Compound score {compound:+.3f} mapped to {label}. "
                "Thresholds: >= 0.05 Positive, <= -0.05 Negative, else Neutral. "
                "Scores are rule-based intensity proportions, not calibrated probabilities."
            ),
            "limitations": self._limitations(),
        }

    # ------------------------------------------------------------------
    def _empty_result(self, note: str) -> dict:
        return {
            "sentiment": "Neutral",
            "model_name": self.MODEL_NAME,
            "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0},
            "is_pretrained": True,
            "note": note,
            "limitations": self._limitations(),
        }

    @staticmethod
    def _limitations() -> list[str]:
        return [
            "VADER is rule-based (lexicon + heuristics), not a trained ML model.",
            "May be inaccurate for sarcasm, mixed emotions, quotations, and context-dependent language.",
            "Designed primarily for social-media text; accuracy on formal news may vary.",
            "Scores are sentiment-intensity proportions, not calibrated probabilities.",
        ]
