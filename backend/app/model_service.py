"""Loads the trained pipeline once and serves predictions + explanations."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from ml.config import ARTIFACT_SCHEMA_VERSION
from ml.text import build_text


class ModelService:
    def __init__(self, art_dir: Path):
        meta_path, model_path = art_dir / "model_metadata.json", art_dir / "model.joblib"
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError("Model artifacts not found. Run `python -m ml.train` first.")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Model artifacts are incompatible with this version. Retrain with `python -m ml.train`.")
        self.pipeline = joblib.load(model_path)
        self.tfidf = self.pipeline.named_steps["tfidf"]
        self.clf = self.pipeline.named_steps["clf"]
        self.classes = [str(c) for c in self.clf.classes_]

    @property
    def supported_modes(self) -> list[str]:
        return self.metadata.get("supported_input_modes", [self.metadata["input_mode"]])

    def predict(self, headline: str, description: str, mode: str) -> dict:
        if mode not in self.supported_modes:
            raise ValueError(f"input_mode '{mode}' is not available. This model supports: {self.supported_modes}")
        text = build_text(headline, description, mode)
        start = time.perf_counter()
        X = self.tfidf.transform([text])
        if hasattr(self.clf, "predict_proba"):
            scores, score_type = self.clf.predict_proba(X)[0], "probability"
        else:
            scores, score_type = self.clf.decision_function(X)[0], "decision_score"
        latency_ms = (time.perf_counter() - start) * 1000
        order = np.argsort(scores)[::-1][:3]
        return {
            "predicted_category": self.classes[int(order[0])],
            "score_type": score_type,
            "candidates": [{"category": self.classes[int(i)], "score": float(scores[i])} for i in order],
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
            "input_mode": mode,
            "latency_ms": round(latency_ms, 3),
            "explanation": self._explain(X, int(order[0])),
        }

    def explain_detailed(self, headline: str, description: str) -> dict:
        """Full explanation with second-best category comparison."""
        text = build_text(headline, description, "combined")
        X = self.tfidf.transform([text])
        if hasattr(self.clf, "predict_proba"):
            scores = self.clf.predict_proba(X)[0]
        else:
            scores = self.clf.decision_function(X)[0]
        order = np.argsort(scores)[::-1]
        pred_idx = int(order[0])
        main_exp = self._explain(X, pred_idx)
        second_best = None
        if len(order) > 1:
            sec_idx = int(order[1])
            sec_exp = self._explain(X, sec_idx)
            second_best = {"category": self.classes[sec_idx],
                           "score": round(float(scores[sec_idx]), 4),
                           "terms": sec_exp["terms"]}
        return {
            "predicted_category": self.classes[pred_idx],
            "model_name": self.metadata["model_name"],
            "explanation_method": main_exp.get("method"),
            "important_terms": main_exp["terms"],
            "second_best": second_best,
            "limitations": [
                "Feature contributions indicate model behavior, not factual correctness.",
                main_exp.get("note", ""),
            ],
        }

    def _explain(self, X, class_idx: int) -> dict:
        """Per-word contribution = TF-IDF value x the model's weight for the predicted class."""
        names = self.tfidf.get_feature_names_out()
        clf = self.clf
        if hasattr(clf, "coef_") and clf.coef_.shape[0] == len(self.classes):
            weights = clf.coef_[class_idx]
            method = "TF-IDF value x class coefficient"
        elif hasattr(clf, "feature_log_prob_"):
            flp = clf.feature_log_prob_
            weights = flp[class_idx] - flp.mean(axis=0)
            method = "TF-IDF value x (class log-probability minus average over classes)"
        else:
            return {"method": None, "note": "This model type does not support word explanations.", "terms": []}
        row = X.tocoo()
        contribs = sorted(((names[j], float(v * weights[j])) for j, v in zip(row.col, row.data)), key=lambda t: -abs(t[1]))
        terms = [{"term": t, "contribution": round(c, 4), "direction": "supports" if c > 0 else "opposes"} for t, c in contribs[:8]]
        return {"method": method, "terms": terms,
                "note": "Describes how the trained model behaves on these words. It does not prove cause and effect."}
