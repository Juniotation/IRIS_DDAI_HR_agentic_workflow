from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator


def _coerce_string_list(value):
    """Accept either a JSON list of strings or a single string from the LLM.
    Gemini can occasionally return a natural-language string where the prompt
    requested an array. We normalize it locally before validation.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Preserve the model's evidence as one signal. A single string is
        # still valid evidence; we should not invent content just to satisfy
        # a minimum list length.
        return [text]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


class DynamicCriterion(BaseModel):
    criterion_id: str = Field(description="Short stable ID such as C01, C02")
    name: str = Field(min_length=2, max_length=80)
    definition: str = Field(min_length=20, max_length=500)
    evidence_to_look_for: List[str] = Field(min_length=1, max_length=8)
    weight: float = Field(gt=0, le=100)
    is_must_have: bool = False
    weight_reason: str = Field(default="", max_length=500)
    source_evidence: str = Field(default="", max_length=800)

    @field_validator("evidence_to_look_for", mode="before")
    @classmethod
    def coerce_evidence_list(cls, value):
        return _coerce_string_list(value)


class RecommendationRules(BaseModel):
    strong_fit_min: float = 80
    potential_fit_min: float = 65
    not_recommended_below: float = 65


class JobCriteria(BaseModel):
    job_title: str = Field(min_length=2)
    role_summary: str = Field(min_length=30)
    criteria: List[DynamicCriterion] = Field(min_length=3, max_length=8)
    must_have_requirements: List[str] = Field(min_length=1, max_length=10)
    nice_to_have_requirements: List[str] = Field(default_factory=list, max_length=10)
    scoring_exclusions: List[str] = Field(min_length=3, max_length=12)
    recommendation_rules: RecommendationRules

    @field_validator("must_have_requirements", "nice_to_have_requirements", "scoring_exclusions", mode="before")
    @classmethod
    def coerce_requirement_lists(cls, value):
        return _coerce_string_list(value)

    @field_validator("criteria")
    @classmethod
    def weights_must_sum_to_100(cls, value: List[DynamicCriterion]) -> List[DynamicCriterion]:
        total = sum(c.weight for c in value)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Criteria weights must sum to 100. Got {total:.2f}")
        ids = [c.criterion_id for c in value]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion_id values must be unique")
        return value


class CriterionWeightExplanation(BaseModel):
    criterion_id: str
    weight_reason: str = Field(min_length=10, max_length=500)
    source_evidence: str = Field(min_length=5, max_length=800)


class CriteriaWeightExplanations(BaseModel):
    explanations: List[CriterionWeightExplanation] = Field(min_length=3, max_length=8)


class CriterionWeight(BaseModel):
    criterion_id: str
    weight: float = Field(gt=0, le=100)


class CriteriaWeightAllocation(BaseModel):
    allocations: List[CriterionWeight] = Field(min_length=3, max_length=8)

    @field_validator("allocations")
    @classmethod
    def allocations_must_sum_to_100(cls, value: List[CriterionWeight]) -> List[CriterionWeight]:
        total = sum(item.weight for item in value)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Criteria weights must sum to 100. Got {total:.2f}")
        ids = [item.criterion_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion_id values must be unique")
        return value


class CriterionAssessment(BaseModel):
    criterion_id: str
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=20, max_length=800)
    evidence: str = Field(default="", max_length=800)


class CandidateEvaluation(BaseModel):
    criterion_assessments: List[CriterionAssessment] = Field(min_length=3, max_length=8)
    strengths: str = Field(min_length=20, max_length=1200)
    gaps: str = Field(min_length=20, max_length=1200)

    @field_validator("criterion_assessments", mode="before")
    @classmethod
    def coerce_assessments(cls, value):
        if isinstance(value, dict):
            from .response_adapter import normalize_candidate_evaluation
            return normalize_candidate_evaluation({"criterion_assessments": value})["criterion_assessments"]
        return value

    @field_validator("strengths", "gaps", mode="before")
    @classmethod
    def coerce_summary_text(cls, value):
        # Gemini may return either a prose string or a JSON list of bullets.
        if value is None:
            return "No evidence provided."
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            items = [str(x).strip() for x in value if str(x).strip()]
            return "\n".join(f"• {x}" for x in items)
        return str(value).strip()


class InterviewQuestions(BaseModel):
    questions: List[str] = Field(min_length=2, max_length=3)

    @field_validator("questions", mode="before")
    @classmethod
    def coerce_questions(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return value


class OutreachEmail(BaseModel):
    subject: str = Field(min_length=5, max_length=150)
    body: str = Field(min_length=80, max_length=3000)

    @field_validator("subject", "body", mode="before")
    @classmethod
    def coerce_email_text(cls, value):
        if isinstance(value, list):
            return "\n".join(str(x).strip() for x in value if str(x).strip())
        if value is None:
            return ""
        return str(value).strip()
