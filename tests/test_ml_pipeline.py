"""Unit tests for the ML pipeline using small SYNTHETIC fixtures.

These tests check that the code works. They say nothing about real-world
model performance on the HuffPost dataset.
"""
import json
import random

import joblib
import pandas as pd
import pytest

from ml import prepare_dataset as prep
from ml import train
from ml.text import build_text, build_text_series, normalize_text

TOPICS = {
    "SPORTS": ["team", "match", "coach", "league", "goal", "season"],
    "POLITICS": ["senate", "election", "vote", "policy", "president", "congress"],
    "TECH": ["software", "chip", "startup", "app", "cloud", "data"],
}
FAST_TFIDF = {"ngram_range": (1, 2), "min_df": 1, "max_df": 1.0, "sublinear_tf": True}
FAST_CANDIDATES = [("multinomial_nb", {"alpha": 1.0}), ("linear_svm", {"C": 1.0}), ("logistic_regression", {"C": 1.0})]


def make_raw(n=60):
    rng = random.Random(0)
    rows = []
    for cat, words in TOPICS.items():
        for i in range(n):
            rows.append({
                "category": cat,
                "headline": " ".join(rng.sample(words, 3)) + f" story {i}",
                "short_description": " ".join(rng.sample(words, 2)),
                "authors": "", "link": "", "date": "2020-01-15",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def clean_df():
    df, _ = prep.clean_dataset(make_raw())
    return df


# ------------------------------------------------------------------ loading
def test_load_jsonl_counts_malformed_lines(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('{"category":"A","headline":"x"}\nnot json\n\n[1,2]\n{"category":"B","headline":"y"}\n', encoding="utf-8")
    df, malformed = prep.load_jsonl(p)
    assert len(df) == 2
    assert malformed == 2


def test_load_jsonl_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        prep.load_jsonl(tmp_path / "nope.json")


def test_required_columns_validated():
    with pytest.raises(ValueError, match="headline"):
        prep.validate_columns(pd.DataFrame({"category": ["A"]}))


# ----------------------------------------------------------------- cleaning
def test_clean_handles_missing_and_duplicates():
    raw = pd.DataFrame([
        {"category": "A", "headline": "  Hello   world ", "short_description": "d", "date": "2020-01-01"},
        {"category": "A", "headline": "Hello world", "short_description": "d", "date": "2020-01-01"},  # duplicate after normalising
        {"category": "A", "headline": "", "short_description": "d", "date": "2020-01-01"},  # missing headline
        {"category": "", "headline": "No label", "short_description": "d", "date": "2020-01-01"},  # missing category
        {"category": "B", "headline": "Hello world", "short_description": "d", "date": "bad-date"},  # label conflict + invalid date
    ])
    clean, stats = prep.clean_dataset(raw)
    assert stats["original_records"] == 5
    assert stats["missing_headline"] == 1
    assert stats["missing_category"] == 1
    assert stats["exact_duplicates_removed"] == 1
    assert stats["headline_label_conflicts"] == 1
    assert stats["invalid_dates"] == 1
    assert stats["retained_records"] == len(clean) == 2
    assert stats["original_records"] - stats["removed_records"] == stats["retained_records"]


def test_merge_categories_is_optional():
    raw = pd.DataFrame([
        {"category": "WORLDPOST", "headline": "a", "short_description": ""},
        {"category": "THE WORLDPOST", "headline": "b", "short_description": ""},
    ])
    assert prep.clean_dataset(raw)[1]["n_categories"] == 2
    assert prep.clean_dataset(raw, merge_categories=True)[1]["n_categories"] == 1


# --------------------------------------------------------------------- text
def test_text_preparation_modes():
    assert normalize_text(None) == ""
    assert normalize_text("  a \n b ") == "a b"
    assert build_text("Headline", "Desc", "headline") == "Headline"
    assert build_text("Headline", "Desc", "combined") == "Headline Desc"
    assert build_text("Headline", "", "combined") == "Headline"
    with pytest.raises(ValueError):
        build_text("x", "y", "bogus")


def test_series_and_scalar_text_agree(clean_df):
    row = clean_df.iloc[0]
    assert build_text_series(clean_df, "combined").iloc[0] == build_text(row.headline, row.short_description, "combined")


# ----------------------------------------------------------------- splitting
def test_split_sizes_stratified_and_disjoint(clean_df):
    train_df, val_df, test_df, info = train.split_data(clean_df, seed=42)
    assert info["stratified"] is True
    assert len(train_df) + len(val_df) + len(test_df) == len(clean_df)
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(train_df.index).isdisjoint(val_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)
    assert abs(len(test_df) / len(clean_df) - 0.15) < 0.03
    assert set(test_df["category"]) == set(TOPICS)


def test_split_is_reproducible(clean_df):
    a = train.split_data(clean_df, seed=42)[2].index.tolist()
    b = train.split_data(clean_df, seed=42)[2].index.tolist()
    assert a == b


def test_rare_classes_go_to_train_only(clean_df):
    rare = pd.DataFrame([{"category": "RARE", "headline": f"odd {i}", "short_description": ""} for i in range(3)])
    df = pd.concat([clean_df, rare], ignore_index=True)
    train_df, val_df, test_df, info = train.split_data(df)
    assert info["rare_classes_train_only"] == ["RARE"]
    assert "RARE" not in set(val_df["category"]) | set(test_df["category"])
    assert (train_df["category"] == "RARE").sum() == 3


# ------------------------------------------------------- training / artifacts
def test_selection_tie_break_prefers_simpler_model():
    a = {"model": "logistic_regression", "macro_f1": 0.8, "accuracy": 0.8}
    b = {"model": "multinomial_nb", "macro_f1": 0.8, "accuracy": 0.8}
    c = {"model": "linear_svm", "macro_f1": 0.9, "accuracy": 0.5}
    assert [r["model"] for r in sorted([a, b, c], key=train.selection_key)] == ["linear_svm", "multinomial_nb", "logistic_regression"]


@pytest.mark.parametrize("mode", ["headline", "combined"])
def test_training_smoke_and_artifacts(clean_df, tmp_path, mode):
    summary = train.run_experiment(clean_df, out_dir=tmp_path, input_mode=mode, tfidf_params=FAST_TFIDF,
                                   candidates=FAST_CANDIDATES, make_plots=True, dataset_fingerprint="test")
    for name in ["model.joblib", "model_metadata.json", "metrics.json", "model_comparison.json",
                 "classification_report.json", "confusion_matrix.json", "label_mapping.json",
                 "plots/model_comparison.png", "plots/confusion_matrix.png"]:
        assert (tmp_path / name).exists(), name

    comparison = json.loads((tmp_path / "model_comparison.json").read_text())
    assert comparison["evaluation_split"] == "validation"
    assert {r["model"] for r in comparison["best_per_algorithm"]} == {"multinomial_nb", "linear_svm", "logistic_regression"}

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["evaluation_split"] == "test"
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    meta = json.loads((tmp_path / "model_metadata.json").read_text())
    assert meta["split"]["final_model_trained_on"] == "train+validation"
    assert meta["split"]["train_size"] + meta["split"]["validation_size"] + meta["split"]["test_size"] == len(clean_df)

    report = json.loads((tmp_path / "classification_report.json").read_text())
    assert all(cat in report for cat in TOPICS)
    assert report["SPORTS"]["support"] > 0

    cm = json.loads((tmp_path / "confusion_matrix.json").read_text())
    assert len(cm["matrix"]) == len(cm["labels"]) == 3
    assert sum(map(sum, cm["matrix"])) == metrics["test_size"]

    reloaded = joblib.load(tmp_path / "model.joblib")
    text = "senate election vote" if mode == "headline" else "senate election vote policy"
    assert reloaded.predict([text])[0] == "POLITICS"
    assert summary["test_metrics"]["model"] == meta["model_name"]


def test_test_split_never_seen_during_fit(clean_df):
    """Tokens that appear only in test rows must not enter the TF-IDF vocabulary."""
    df = clean_df.copy()
    # tag every row with a unique token so we can tell which split it ended up in
    df["headline"] = [f"{h} tok{i}x" for i, h in enumerate(df["headline"])]
    _, _, test_df, _ = train.split_data(df)  # same seed/data -> same split as run_experiment
    summary = train.run_experiment(df, tfidf_params=FAST_TFIDF, candidates=FAST_CANDIDATES, write_artifacts=False)
    vocab = summary["model"].named_steps["tfidf"].vocabulary_
    test_tokens = {f"tok{i}x" for i in test_df.index}
    seen_tokens = {t for t in vocab if t.startswith("tok") and " " not in t}
    assert test_tokens, "fixture should produce a non-empty test split"
    assert test_tokens.isdisjoint(seen_tokens)
    assert len(seen_tokens) == len(df) - len(test_df)
