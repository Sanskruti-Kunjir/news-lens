"""Pydantic request/response schemas."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ==================================================================== predict
class PredictRequest(BaseModel):
    headline: str = Field(..., max_length=500)
    short_description: str = Field("", max_length=2000)
    input_mode: Literal["headline", "combined"] = "combined"

    @field_validator("headline")
    @classmethod
    def headline_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("headline must not be empty")
        return v


class Candidate(BaseModel):
    category: str
    score: float


class ExplanationTerm(BaseModel):
    term: str
    contribution: float
    direction: Literal["supports", "opposes"]


class Explanation(BaseModel):
    method: Optional[str] = None
    note: str = ""
    terms: list[ExplanationTerm] = []


class SentimentResult(BaseModel):
    sentiment: str
    model_name: str
    scores: dict
    is_pretrained: bool = True
    note: str = ""
    limitations: list[str] = []


class PredictResponse(BaseModel):
    predicted_category: str
    score_type: Literal["probability", "decision_score"]
    candidates: list[Candidate]
    model_name: str
    model_version: str
    input_mode: str
    latency_ms: float
    explanation: Explanation
    sentiment: Optional[SentimentResult] = None
    history_id: Optional[int] = None


# ==================================================================== explain
class ExplainRequest(BaseModel):
    headline: str = Field(..., max_length=500)
    description: str = Field("", max_length=2000)
    model_name: str = Field("", max_length=100)

    @field_validator("headline")
    @classmethod
    def headline_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("headline must not be empty")
        return v


# ================================================================== sentiment
class SentimentRequest(BaseModel):
    headline: str = Field(..., max_length=500)
    description: str = Field("", max_length=2000)

    @field_validator("headline")
    @classmethod
    def headline_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("headline must not be empty")
        return v


# ============================================================= recommendations
class PreferencesUpdate(BaseModel):
    preferred_categories: list[str] = Field(default_factory=list, max_length=50)


class BookmarkRequest(BaseModel):
    article_id: int
    headline: str = Field(..., max_length=500)
    category: Optional[str] = None
