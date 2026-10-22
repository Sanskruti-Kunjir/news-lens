"""Compute dataset statistics and plots from the cleaned dataset.

Usage (from the project root):
    python -m ml.explore_dataset

Outputs:
    artifacts/dataset_summary.json
    artifacts/plots/category_distribution.png
    artifacts/plots/length_distributions.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ml import config, plots


def summarize(df: pd.DataFrame, processing_stats: dict | None = None) -> dict:
    counts = df["category"].value_counts()
    headline_len = df["headline"].str.split().str.len()
    desc_len = df["short_description"].str.split().str.len()
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(pd.NaT, index=df.index)
    valid = dates.dropna()

    summary = {
        "scope": "full cleaned dataset (not a train/test split)",
        "total_records": int(len(df)),
        "n_categories": int(counts.size),
        "category_counts": {k: int(v) for k, v in counts.items()},
        "most_frequent_category": {"name": counts.index[0], "count": int(counts.iloc[0])},
        "least_frequent_category": {"name": counts.index[-1], "count": int(counts.iloc[-1])},
        "imbalance_ratio_max_to_min": round(float(counts.iloc[0] / counts.iloc[-1]), 2),
        "headline_length_words": _describe(headline_len),
        "description_length_words": _describe(desc_len),
        "empty_descriptions": int((df["short_description"] == "").sum()),
        "date_range": {
            "min": valid.min().date().isoformat() if len(valid) else None,
            "max": valid.max().date().isoformat() if len(valid) else None,
            "valid_dates": int(len(valid)),
            "invalid_dates": int(dates.isna().sum()),
        },
    }
    if len(valid) >= 2:
        tmp = pd.DataFrame({"category": df.loc[valid.index, "category"], "year": valid.dt.year, "month": valid.dt.to_period("M").astype(str)})
        summary["articles_by_year"] = {str(k): int(v) for k, v in tmp["year"].value_counts().sort_index().items()}
        summary["articles_by_month"] = {k: int(v) for k, v in tmp["month"].value_counts().sort_index().items()}
        pivot = tmp.groupby(["year", "category"]).size()
        summary["category_by_year"] = {str(y): {c: int(n) for c, n in pivot.loc[y].items()} for y in sorted(tmp["year"].unique())}
    if processing_stats:
        summary["processing"] = {k: v for k, v in processing_stats.items() if k != "category_counts"}
    return summary


def _describe(series: pd.Series) -> dict:
    s = series.dropna()
    return {"mean": round(float(s.mean()), 2), "median": float(s.median()), "min": int(s.min()),
            "max": int(s.max()), "p95": float(s.quantile(0.95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=config.PROCESSED_PATH)
    parser.add_argument("--out-dir", type=Path, default=config.ARTIFACTS_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"{args.input} not found. Run `python -m ml.prepare_dataset` first.")
    df = pd.read_csv(args.input, keep_default_na=False)
    stats = None
    if config.PROCESSING_STATS_PATH.exists():
        stats = json.loads(config.PROCESSING_STATS_PATH.read_text(encoding="utf-8"))

    summary = summarize(df, stats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plots.category_distribution(summary["category_counts"], args.out_dir / "plots" / "category_distribution.png")
    plots.length_distributions(df["headline"].str.split().str.len(), df["short_description"].str.split().str.len(),
                               args.out_dir / "plots" / "length_distributions.png")

    print(f"Records: {summary['total_records']}  Categories: {summary['n_categories']}")
    print(f"Most frequent : {summary['most_frequent_category']}")
    print(f"Least frequent: {summary['least_frequent_category']}")
    print(f"Imbalance ratio (max/min): {summary['imbalance_ratio_max_to_min']}")
    print(f"Saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
