"""Text helpers used identically at training time and inference time."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ml.config import INPUT_MODES

_WS = re.compile(r"\s+")


def normalize_text(value) -> str:
    """Return a clean string: NFKC-normalised, whitespace collapsed, never None."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value)
    return _WS.sub(" ", value).strip()


def build_text(headline: str, description: str = "", mode: str = "combined") -> str:
    """Build the model input for one article.

    mode="headline": headline only.
    mode="combined": headline + " " + short_description (headline alone if the
    description is empty).
    """
    if mode not in INPUT_MODES:
        raise ValueError(f"input mode must be one of {INPUT_MODES}, got {mode!r}")
    headline = normalize_text(headline)
    if mode == "headline":
        return headline
    return f"{headline} {normalize_text(description)}".strip()


def build_text_series(df: pd.DataFrame, mode: str = "combined") -> pd.Series:
    """Vectorised version of build_text for a DataFrame."""
    if mode not in INPUT_MODES:
        raise ValueError(f"input mode must be one of {INPUT_MODES}, got {mode!r}")
    if mode == "headline":
        return df["headline"].astype(str)
    return (df["headline"].astype(str) + " " + df["short_description"].astype(str)).str.strip()
