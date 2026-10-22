"""Central paths and constants shared by all ML scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "raw" / "News_Category_Dataset_v3.json"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "clean.csv"
PROCESSING_STATS_PATH = PROCESSED_DIR / "processing_stats.json"

ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

SEED = 42
ARTIFACT_SCHEMA_VERSION = 1

DATASET_SOURCE = "https://www.kaggle.com/datasets/rmisra/news-category-dataset"

# Optional category merging (off by default; enable with --merge-categories).
# These pairs are near-duplicate labels in the HuffPost dataset. Check the
# categories that actually exist in your file before relying on this list.
CATEGORY_MERGES = {
    "THE WORLDPOST": "WORLD NEWS",
    "WORLDPOST": "WORLD NEWS",
    "ARTS": "ARTS & CULTURE",
    "CULTURE & ARTS": "ARTS & CULTURE",
    "GREEN": "ENVIRONMENT",
    "COLLEGE": "EDUCATION",
    "PARENTS": "PARENTING",
    "STYLE": "STYLE & BEAUTY",
    "TASTE": "FOOD & DRINK",
}

INPUT_MODES = ("headline", "combined")
