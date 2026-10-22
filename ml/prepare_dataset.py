"""Load, validate and clean the HuffPost News Category Dataset.

Usage (from the project root):
    python -m ml.prepare_dataset
    python -m ml.prepare_dataset --merge-categories

The raw file is never modified. Outputs:
    data/processed/clean.csv
    data/processed/processing_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml import config
from ml.text import normalize_text

REQUIRED_COLUMNS = ["category", "headline"]


def load_jsonl(path) -> tuple[pd.DataFrame, int]:
    """Read a JSON Lines file. Returns (DataFrame, number of malformed lines)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download News_Category_Dataset_v3.json from "
            f"{config.DATASET_SOURCE} and place it in data/raw/."
        )
    records, malformed = [], 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                malformed += 1
    if not records:
        raise ValueError("No valid JSON records found. Is this a JSON Lines file?")
    return pd.DataFrame(records), malformed


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")


def clean_dataset(
    df: pd.DataFrame, merge_categories: bool = False, malformed_lines: int = 0
) -> tuple[pd.DataFrame, dict]:
    """Clean the dataset and return (clean_df, statistics).

    Rules (documented in the README):
    1. Normalise whitespace/Unicode in text fields; upper-case the category.
    2. Drop rows with an empty headline or category.
    3. Optionally merge near-duplicate category labels.
    4. Drop exact duplicates on (headline, short_description, category).
       Rows with the same headline but DIFFERENT labels are kept and counted
       as `headline_label_conflicts` because they may be genuine examples.
    5. Invalid dates are counted but the rows are kept (date is not a feature).
    """
    validate_columns(df)
    df = df.copy()
    stats: dict = {
        "dataset_source": config.DATASET_SOURCE,
        "original_records": int(len(df)),
        "malformed_lines_skipped": int(malformed_lines),
    }

    for col in ("headline", "short_description", "category"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(normalize_text)
    df["category"] = df["category"].str.upper()

    stats["missing_headline"] = int((df["headline"] == "").sum())
    stats["missing_category"] = int((df["category"] == "").sum())
    stats["missing_description"] = int((df["short_description"] == "").sum())

    df = df[(df["headline"] != "") & (df["category"] != "")]
    stats["valid_records_before_dedup"] = int(len(df))

    if merge_categories:
        before = df["category"].nunique()
        df["category"] = df["category"].replace(config.CATEGORY_MERGES)
        stats["categories_merged"] = True
        stats["categories_before_merge"] = int(before)
    else:
        stats["categories_merged"] = False

    n_before = len(df)
    df = df.drop_duplicates(subset=["headline", "short_description", "category"], keep="first")
    stats["exact_duplicates_removed"] = int(n_before - len(df))

    label_counts = df.groupby("headline")["category"].nunique()
    stats["headline_label_conflicts"] = int((label_counts > 1).sum())

    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    stats["invalid_dates"] = int(parsed.isna().sum())
    df["date"] = parsed

    keep = [c for c in ("category", "headline", "short_description", "authors", "link", "date") if c in df.columns]
    df = df[keep].reset_index(drop=True)

    stats["retained_records"] = int(len(df))
    stats["removed_records"] = stats["original_records"] - stats["retained_records"]
    stats["n_categories"] = int(df["category"].nunique())
    stats["category_counts"] = {k: int(v) for k, v in df["category"].value_counts().items()}
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=config.RAW_PATH)
    parser.add_argument("--output", type=Path, default=config.PROCESSED_PATH)
    parser.add_argument("--stats", type=Path, default=config.PROCESSING_STATS_PATH)
    parser.add_argument("--merge-categories", action="store_true", help="merge near-duplicate labels")
    args = parser.parse_args()

    raw, malformed = load_jsonl(args.input)
    clean, stats = clean_dataset(raw, merge_categories=args.merge_categories, malformed_lines=malformed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Original records : {stats['original_records']}")
    print(f"Malformed lines  : {stats['malformed_lines_skipped']}")
    print(f"Duplicates removed: {stats['exact_duplicates_removed']}")
    print(f"Invalid dates    : {stats['invalid_dates']}")
    print(f"Retained records : {stats['retained_records']}")
    print(f"Categories       : {stats['n_categories']}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
