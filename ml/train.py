"""Train, compare, select and evaluate the three required classifiers.

Usage (from the project root):
    python -m ml.train                       # full dataset, headline + description
    python -m ml.train --sample 20000        # quick debugging run
    python -m ml.train --input-mode headline

Method
------
1. Stratified 70/15/15 split (seed 42). Classes with fewer than MIN_PER_CLASS
   articles cannot be split reliably and are kept in the training set only.
2. Every candidate (algorithm + hyper-parameters) is a Pipeline(TF-IDF -> classifier)
   fitted on the TRAIN split only and scored on the VALIDATION split.
3. The best candidate is chosen by validation macro-F1 (ties: accuracy, then
   the simpler model: Naive Bayes < Linear SVM < Logistic Regression).
4. The chosen configuration is refitted on TRAIN + VALIDATION and evaluated
   ONCE on the untouched TEST split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml import config, plots
from ml.text import build_text_series

MIN_PER_CLASS = 10
SIMPLICITY_RANK = {"multinomial_nb": 0, "linear_svm": 1, "logistic_regression": 2}
SCORE_TYPE = {"multinomial_nb": "probability", "logistic_regression": "probability", "linear_svm": "decision_score"}

DEFAULT_TFIDF = {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.9, "sublinear_tf": True, "strip_accents": "unicode"}

DEFAULT_CANDIDATES = (
    [("multinomial_nb", {"alpha": a}) for a in (0.1, 0.3, 1.0)]
    + [("linear_svm", {"C": c}) for c in (0.1, 0.5, 1.0)]
    + [("logistic_regression", {"C": c}) for c in (1.0, 5.0)]
)


# ----------------------------------------------------------------- splitting
def split_data(df: pd.DataFrame, seed: int = config.SEED, val_size: float = 0.15, test_size: float = 0.15,
               label_col: str = "category") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Stratified train/validation/test split with a documented fallback."""
    counts = df[label_col].value_counts()
    rare = df[label_col].map(counts) < MIN_PER_CLASS
    rare_df, main_df = df[rare], df[~rare]
    rare_classes = sorted(counts[counts < MIN_PER_CLASS].index.tolist())

    def _split(stratify: bool):
        strat = main_df[label_col] if stratify else None
        trainval, test = train_test_split(main_df, test_size=test_size, stratify=strat, random_state=seed)
        strat2 = trainval[label_col] if stratify else None
        train, val = train_test_split(trainval, test_size=val_size / (1 - test_size), stratify=strat2, random_state=seed)
        return train, val, test

    try:
        train, val, test = _split(True)
        stratified = True
    except ValueError:
        train, val, test = _split(False)
        stratified = False

    train = pd.concat([train, rare_df])
    info = {
        "seed": seed,
        "stratified": stratified,
        "rare_classes_train_only": rare_classes,
        "train_size": int(len(train)),
        "validation_size": int(len(val)),
        "test_size": int(len(test)),
        "headline_overlap_train_test": int(len(set(train["headline"]) & set(test["headline"]))),
        "headline_overlap_train_validation": int(len(set(train["headline"]) & set(val["headline"]))),
    }
    return train, val, test, info


# ------------------------------------------------------------------ modelling
def build_classifier(name: str, params: dict, seed: int):
    if name == "multinomial_nb":
        return MultinomialNB(**params)
    if name == "linear_svm":
        return LinearSVC(**params, max_iter=5000, random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(**params, max_iter=300, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, params: dict, seed: int, tfidf_params: dict | None = None) -> Pipeline:
    tfidf = TfidfVectorizer(**(tfidf_params or DEFAULT_TFIDF))
    return Pipeline([("tfidf", tfidf), ("clf", build_classifier(name, params, seed))])


def evaluate(pipe: Pipeline, X, y) -> dict:
    """Return metrics plus per-article prediction latency (ms)."""
    n = min(len(X), 500)
    start = time.perf_counter()
    pipe.predict(X.iloc[:n])
    latency_ms = (time.perf_counter() - start) / max(n, 1) * 1000
    pred = pipe.predict(X)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "latency_ms_per_article": round(latency_ms, 4),
    }, pred


def selection_key(result: dict):
    """Lower sorts first: higher macro-F1, then accuracy, then simpler model."""
    return (-round(result["macro_f1"], 6), -round(result["accuracy"], 6), SIMPLICITY_RANK[result["model"]])


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def run_experiment(df: pd.DataFrame, out_dir: Path = config.ARTIFACTS_DIR, input_mode: str = "combined",
                   seed: int = config.SEED, sample: int | None = None, tfidf_params: dict | None = None,
                   candidates=None, make_plots: bool = True, dataset_fingerprint: str | None = None,
                   write_artifacts: bool = True) -> dict:
    """Full train -> validate -> select -> refit -> test pipeline. Returns a summary dict."""
    out_dir = Path(out_dir)
    candidates = list(candidates or DEFAULT_CANDIDATES)
    tfidf_params = tfidf_params or DEFAULT_TFIDF

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    train, val, test, split_info = split_data(df, seed=seed)
    X_train, y_train = build_text_series(train, input_mode), train["category"]
    X_val, y_val = build_text_series(val, input_mode), val["category"]
    X_test, y_test = build_text_series(test, input_mode), test["category"]

    # ---- validation comparison (TF-IDF is fitted inside the pipeline on TRAIN only)
    results = []
    for name, params in candidates:
        pipe = build_pipeline(name, params, seed, tfidf_params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_seconds = time.perf_counter() - t0
        metrics, _ = evaluate(pipe, X_val, y_val)
        results.append({
            "model": name, "params": params, "score_type": SCORE_TYPE[name],
            "train_seconds": round(train_seconds, 3),
            "convergence_warning": any("converge" in str(w.message).lower() for w in caught),
            **metrics,
        })
        print(f"[val] {name:20s} {params}  macro_f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")

    results.sort(key=selection_key)
    selected = results[0]
    best_per_algorithm = []
    for r in results:  # already sorted, first hit per algorithm is its best
        if all(b["model"] != r["model"] for b in best_per_algorithm):
            best_per_algorithm.append(r)

    # ---- refit winner on train+validation, evaluate ONCE on test
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    final = build_pipeline(selected["model"], selected["params"], seed, tfidf_params)
    t0 = time.perf_counter()
    final.fit(X_tv, y_tv)
    final_train_seconds = time.perf_counter() - t0
    test_metrics, y_pred = evaluate(final, X_test, y_test)
    labels = list(final.classes_)

    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    now = datetime.now(timezone.utc)
    version = "v" + now.strftime("%Y%m%d%H%M%S")
    metrics_out = {
        "evaluation_split": "test", "model": selected["model"], "params": selected["params"],
        "input_mode": input_mode, "score_type": SCORE_TYPE[selected["model"]],
        "train_seconds": round(final_train_seconds, 3), "test_size": int(len(test)),
        **test_metrics,
    }
    comparison = {
        "evaluation_split": "validation", "input_mode": input_mode,
        "selection_rule": "highest macro_f1; ties -> accuracy -> simpler model (NB < SVM < LR)",
        "selected": {"model": selected["model"], "params": selected["params"]},
        "best_per_algorithm": best_per_algorithm, "all_results": results,
    }
    metadata = {
        "artifact_schema_version": config.ARTIFACT_SCHEMA_VERSION,
        "model_version": version, "model_name": selected["model"], "params": selected["params"],
        "score_type": SCORE_TYPE[selected["model"]], "input_mode": input_mode,
        "supported_input_modes": [input_mode], "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in tfidf_params.items()},
        "trained_at_utc": now.isoformat(), "random_seed": seed,
        "dataset_fingerprint": dataset_fingerprint, "sample_size": sample,
        "n_categories": len(labels), "categories": labels,
        "split": {**split_info, "final_model_trained_on": "train+validation", "evaluated_on": "test"},
        "libraries": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }

    summary = {"selected": selected, "test_metrics": metrics_out, "metadata": metadata, "model": final}
    if not write_artifacts:
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_dir / "model.joblib")
    _dump(metadata, out_dir / "model_metadata.json")
    _dump(metrics_out, out_dir / "metrics.json")
    _dump(comparison, out_dir / "model_comparison.json")
    _dump(report, out_dir / "classification_report.json")
    _dump({"labels": labels, "matrix": cm.tolist()}, out_dir / "confusion_matrix.json")
    _dump({"label_to_index": {l: i for i, l in enumerate(labels)}}, out_dir / "label_mapping.json")
    if make_plots:
        plots.model_comparison(best_per_algorithm, out_dir / "plots" / "model_comparison.png")
        plots.confusion_matrix_plot(cm, labels, out_dir / "plots" / "confusion_matrix.png")

    print(f"\n[test] {selected['model']} {selected['params']}")
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"[test] {k:16s} {test_metrics[k]:.4f}")
    print(f"Artifacts saved to {out_dir}")
    return summary


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=config.PROCESSED_PATH)
    parser.add_argument("--out-dir", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--input-mode", choices=config.INPUT_MODES, default="combined")
    parser.add_argument("--sample", type=int, default=None, help="use a random subset (debugging only)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"{args.input} not found. Run `python -m ml.prepare_dataset` first.")
    df = pd.read_csv(args.input, keep_default_na=False)
    run_experiment(df, out_dir=args.out_dir, input_mode=args.input_mode, seed=args.seed, sample=args.sample,
                   make_plots=not args.no_plots, dataset_fingerprint=file_fingerprint(args.input))


if __name__ == "__main__":
    main()
