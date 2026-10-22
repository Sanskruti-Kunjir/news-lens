"""Environment-based configuration (read at call time so tests can override it)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    return Path(os.environ.get("NEWSLENS_ARTIFACTS_DIR", ROOT / "artifacts"))


def db_path() -> Path:
    return Path(os.environ.get("NEWSLENS_DB_PATH", ROOT / "data" / "history.db"))


def dataset_path() -> Path:
    return Path(os.environ.get("NEWSLENS_DATASET_PATH", ROOT / "data" / "processed" / "clean.csv"))


def cors_origins() -> list[str]:
    raw = os.environ.get("NEWSLENS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]
