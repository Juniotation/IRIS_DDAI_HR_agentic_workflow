from __future__ import annotations

import json
import re
from typing import Any, Type

from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types

from .config import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_SCREEN_MODEL,
    GEMINI_SCREEN_THINKING_LEVEL,
    GEMINI_THINKING_LEVEL,
    get_setting,
)
from .models import (
    CandidateEvaluation,
    CriteriaEvidenceSources,
    CriteriaWeightAllocation,
    CriteriaWeightExplanations,
    InterviewQuestions,
    JobCriteria,
    OutreachEmail,
)
from .response_adapter import (
    normalize_candidate_evaluation,
    normalize_interview_questions,
    normalize_job_criteria,
    normalize_outreach_email,
)


def get_api_key() -> str | None:
    return get_setting("GEMINI_API_KEY") or get_setting("GOOGLE_API_KEY")


def get_client() -> genai.Client:
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY in .env for local development "
            "or Streamlit Secrets for deployment."
        )
    # One bounded SDK retry for transient capacity/rate errors + short timeout.
    # We intentionally do not add another application-level retry loop.
    http_options = types.HttpOptions(
        timeout=45000,
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=1.0,
            max_delay=3.0,
            exp_base=2.0,
            jitter=0.2,
            http_status_codes=[429, 503],
        ),
    )
    return genai.Client(api_key=key, http_options=http_options)


def _parse_json(text: str) -> Any:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Gemini occasionally emits an unquoted key with a stray leading dash,
        # for example: { "score": 65, -reason": "..." }. Repair only this
        # specific syntax defect; other malformed responses remain explicit.
        repaired = re.sub(
            r'([,{]\s*)-([A-Za-z_][A-Za-z0-9_]*)"',
            r'\1"\2"',
            cleaned,
        )
        if repaired != cleaned:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}. Raw response: {cleaned[:600]}") from exc


def _thinking_config(model: str) -> types.ThinkingConfig | None:
    # Gemini 3.x supports thinking_level. Flash-Lite supports minimal.
    if model.startswith("gemini-3"):
        level = GEMINI_SCREEN_THINKING_LEVEL if model == GEMINI_SCREEN_MODEL else GEMINI_THINKING_LEVEL
        return types.ThinkingConfig(thinking_level=level)
    return None


def _gemini_response_schema(schema: Type[Any]) -> dict[str, Any]:
    if not issubclass(schema, BaseModel):
        raise TypeError("Gemini structured output requires a Pydantic model schema.")

    json_schema = schema.model_json_schema()

    def normalize(node: Any) -> Any:
        if isinstance(node, dict):
            normalized = {
                key: normalize(value)
                for key, value in node.items()
                if key not in {"exclusiveMinimum", "exclusiveMaximum"}
            }
            if "exclusiveMinimum" in node:
                normalized["minimum"] = node["exclusiveMinimum"]
            if "exclusiveMaximum" in node:
                normalized["maximum"] = node["exclusiveMaximum"]
            return normalized
        if isinstance(node, list):
            return [normalize(value) for value in node]
        return node

    normalized_schema = normalize(json_schema)

    def has_open_object(node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("additionalProperties"):
                return True
            return any(has_open_object(value) for value in node.values())
        if isinstance(node, list):
            return any(has_open_object(value) for value in node)
        return False

    if has_open_object(normalized_schema):
        raise ValueError(
            f"{schema.__name__} contains an open-ended object, which Gemini structured output does not support."
        )
    return normalized_schema


def _json_call(model: str, prompt: str, schema: Type[Any], max_output_tokens: int | None = None):
    client = get_client()
    thinking = _thinking_config(model)
    cfg_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": _gemini_response_schema(schema),
        "max_output_tokens": max_output_tokens or GEMINI_MAX_OUTPUT_TOKENS,
    }
    if thinking is not None:
        cfg_kwargs["thinking_config"] = thinking

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        payload = getattr(response, "parsed", None)
        if payload is None:
            payload = _parse_json(response.text)
        elif isinstance(payload, BaseModel):
            payload = payload.model_dump()

        # Normalize common LLM shape variations before strict Pydantic validation.
        if schema is JobCriteria:
            payload = normalize_job_criteria(payload)
        elif schema is CandidateEvaluation:
            payload = normalize_candidate_evaluation(payload)
        elif schema is InterviewQuestions:
            payload = normalize_interview_questions(payload)
        elif schema is OutreachEmail:
            payload = normalize_outreach_email(payload)

        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(
                "Gemini returned valid JSON, but it did not match the expected schema. "
                f"Validation detail: {exc}"
            ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        message = str(exc)
        upper = message.upper()
        if "503" in upper or "UNAVAILABLE" in upper:
            raise RuntimeError(
                "Gemini is temporarily unavailable (503). The app stopped after a bounded retry. "
                "Wait briefly and try the same agent once more."
            ) from exc
        if "429" in upper or "RESOURCE_EXHAUSTED" in upper:
            raise RuntimeError(
                "Gemini rate/quota limit reached (429). Wait for the limit to reset before retrying."
            ) from exc
        raise RuntimeError(f"Gemini request failed: {message}") from exc


def agent_1_build_criteria(jd_text: str) -> JobCriteria:
    prompt = f"""
You are Agent 1: Job Criteria Architect in a Human-in-the-Loop recruiting workflow.

Read the job description and CREATE the assessment framework yourself.
Do not assume a fixed list of skills or metrics. The criteria must emerge from the actual JD.

Rules:
- Create 3 to 8 job-relevant criteria.
- Choose criterion names that match this specific JD.
- Assign weights based on importance in the JD; weights must sum to exactly 100.
- Mark must-have only when the JD clearly indicates the requirement is required.
- Give concise evidence signals a recruiter can verify in a CV.
- For each criterion, explain why its weight reflects its relative importance in this JD.
- Criteria names and definitions may summarize the JD in new wording, as long as their meaning is supported by it.
- For each criterion, provide one source_evidence excerpt from the JD that supports it. This field must be copied verbatim.
- Include scoring exclusions. Name, email, university, age, gender, nationality and other protected/irrelevant attributes must not drive mathematical scoring.
- Do not invent requirements that are not supported by the JD.
- Keep the response concise; this is a screening rubric, not a full essay.

Return ONE JSON object with exactly these keys:
job_title, role_summary, criteria, must_have_requirements, nice_to_have_requirements, scoring_exclusions, recommendation_rules.

Each criteria item must contain exactly:
criterion_id, name, definition, evidence_to_look_for, weight, is_must_have,
weight_reason, source_evidence.

IMPORTANT FORMAT CONTRACT:
- criteria MUST be a JSON array of 3-8 objects.
- Every criterion MUST include all eight fields listed above; no field is optional.
- evidence_to_look_for MUST be a JSON array of short strings (never a single string).
- weight_reason MUST briefly explain why the criterion deserves its assigned percentage relative to the other criteria, using the JD.
- source_evidence MUST identify the original JD wording supporting the criterion. Do not force the criterion name or definition to repeat the JD wording.
- must_have_requirements, nice_to_have_requirements, scoring_exclusions MUST be JSON arrays of strings.
- weight MUST be a number, and all criterion weights must sum to exactly 100.
- recommendation_rules MUST be a JSON object with numeric strong_fit_min, potential_fit_min, not_recommended_below.
- Do not wrap the JSON in markdown fences.

JOB DESCRIPTION:
{jd_text}
"""
    criteria = _json_call(GEMINI_MODEL, prompt, JobCriteria, max_output_tokens=1800)
    normalized_jd = re.sub(r"\s+", " ", jd_text).casefold()
    missing_explanations = any(
        not criterion.weight_reason.strip()
        or not criterion.source_evidence.strip()
        or re.sub(r"\s+", " ", criterion.source_evidence).strip().casefold() not in normalized_jd
        for criterion in criteria.criteria
    )
    if missing_explanations:
        criteria = enrich_criteria_weight_explanations(jd_text, criteria)

    for criterion in criteria.criteria:
        if not criterion.weight_reason.strip():
            raise ValueError(f"Agent 1 did not explain the weight for '{criterion.name}'.")
        source_evidence = re.sub(r"\s+", " ", criterion.source_evidence).strip().casefold()
        if not source_evidence or source_evidence not in normalized_jd:
            raise ValueError(
                f"Agent 1 did not provide a verbatim job-description excerpt for '{criterion.name}'."
            )
    return criteria


def enrich_criteria_weight_explanations(jd_text: str, criteria: JobCriteria) -> JobCriteria:
    criteria = JobCriteria.model_validate(criteria.model_dump())
    prompt = f"""
You are explaining an existing assessment framework using its original job description.
Do not change, recalculate, or suggest different criterion weights.

For every criterion below, provide:
- weight_reason: why its already-assigned percentage is appropriate relative to the other criteria, based only on the JD.
- source_evidence: an exact, verbatim excerpt from the JD that supports the assigned weight.

Return one JSON object with key "explanations", an array containing exactly one object per criterion.
Each object must have exactly: criterion_id, weight_reason, source_evidence.
Include every criterion_id exactly once. Do not use candidate information.

JOB DESCRIPTION:
{jd_text}

EXISTING CRITERIA (weights are fixed and must not be changed):
{criteria.model_dump_json(indent=2)}
"""
    result = _json_call(
        GEMINI_MODEL,
        prompt,
        CriteriaWeightExplanations,
        max_output_tokens=1200,
    )
    by_id = {item.criterion_id: item for item in result.explanations}
    criterion_ids = {item.criterion_id for item in criteria.criteria}
    if len(by_id) != len(result.explanations) or set(by_id) != criterion_ids:
        raise ValueError("Agent did not return exactly one weight explanation for each criterion.")

    normalized_jd = re.sub(r"\s+", " ", jd_text).casefold()
    updated_criteria = []
    needs_source_selection = False
    for criterion in criteria.criteria:
        explanation = by_id[criterion.criterion_id]
        normalized_source = re.sub(r"\s+", " ", explanation.source_evidence).strip().casefold()
        source_evidence = explanation.source_evidence
        if not normalized_source or normalized_source not in normalized_jd:
            needs_source_selection = True
            source_evidence = ""
        updated_criteria.append(
            criterion.model_copy(
                update={
                    "weight_reason": explanation.weight_reason,
                    "source_evidence": source_evidence,
                }
            )
        )
    enriched = criteria.model_copy(update={"criteria": updated_criteria})
    if needs_source_selection:
        enriched = _select_verbatim_criteria_sources(jd_text, enriched)
    return enriched


def _job_description_segments(jd_text: str) -> list[tuple[str, str]]:
    segments = []
    for line in jd_text.splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            remaining = sentence.strip()
            while remaining:
                if len(remaining) <= 700:
                    segments.append(remaining)
                    break
                split_at = remaining.rfind(" ", 0, 700)
                if split_at <= 0:
                    split_at = 700
                segments.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()
    if not segments and jd_text.strip():
        segments.append(jd_text.strip()[:700])
    return [(f"S{index:03}", text) for index, text in enumerate(segments, start=1)]


def _select_verbatim_criteria_sources(jd_text: str, criteria: JobCriteria) -> JobCriteria:
    segments = _job_description_segments(jd_text)
    if not segments:
        raise ValueError("The job description does not contain text to use as criterion evidence.")

    segment_text = "\n".join(f"{segment_id}: {text}" for segment_id, text in segments)
    criteria_text = criteria.model_dump_json(indent=2)
    prompt = f"""
You are grounding an existing assessment framework in its original job description.
Criteria names and definitions may summarize the JD in different wording. For each criterion,
select the single JD segment that best supports its meaning. Do not rewrite or invent JD text.
Choose a segment by its ID; the application will display the original segment verbatim.
Do not change any criterion, weight, or explanation.

Return one JSON object with key "sources", containing exactly one object per criterion.
Each object must have exactly: criterion_id, source_segment_id.
Use every criterion_id exactly once and only segment IDs included below.

CRITERIA:
{criteria_text}

JOB DESCRIPTION SEGMENTS:
{segment_text}
"""
    result = _json_call(
        GEMINI_MODEL,
        prompt,
        CriteriaEvidenceSources,
        max_output_tokens=800,
    )
    by_id = {item.criterion_id: item.source_segment_id for item in result.sources}
    expected_ids = {item.criterion_id for item in criteria.criteria}
    if len(by_id) != len(result.sources) or set(by_id) != expected_ids:
        raise ValueError("Agent did not select exactly one job-description source for each criterion.")

    source_by_segment_id = dict(segments)
    updated_criteria = []
    for criterion in criteria.criteria:
        segment_id = by_id[criterion.criterion_id]
        if segment_id not in source_by_segment_id:
            raise ValueError(
                f"Agent selected an unknown job-description segment '{segment_id}' "
                f"for criterion '{criterion.name}'."
            )
        updated_criteria.append(
            criterion.model_copy(
                update={"source_evidence": source_by_segment_id[segment_id]}
            )
        )
    return criteria.model_copy(update={"criteria": updated_criteria})


def redistribute_criterion_weights(
    jd_text: str,
    criteria: JobCriteria,
    fixed_weights: dict[str, float],
    proposed_weights: dict[str, float],
    change_reason: str,
) -> JobCriteria:
    if not change_reason.strip():
        raise ValueError("A reason is required before criterion weights can be changed.")
    if not fixed_weights:
        raise ValueError("At least one recruiter-edited criterion weight must be fixed.")

    by_id = {criterion.criterion_id: criterion for criterion in criteria.criteria}
    unknown_fixed_ids = set(fixed_weights) - set(by_id)
    if unknown_fixed_ids:
        raise ValueError(f"Unknown fixed criterion ID(s): {', '.join(sorted(unknown_fixed_ids))}.")
    if set(proposed_weights) != set(by_id):
        raise ValueError("Proposed weights must include every existing criterion exactly once.")
    if any(not 0 < float(weight) < 100 for weight in proposed_weights.values()):
        raise ValueError("Every proposed criterion weight must be greater than 0 and less than 100.")
    if any(not 0 < float(weight) < 100 for weight in fixed_weights.values()):
        raise ValueError(
            "Every recruiter-edited criterion weight must be greater than 0 and less than 100."
        )
    if any(
        abs(float(proposed_weights[criterion_id]) - float(weight)) > 0.01
        for criterion_id, weight in fixed_weights.items()
    ):
        raise ValueError("Fixed criterion weights must match the recruiter's proposed weights.")

    other_ids = [criterion_id for criterion_id in by_id if criterion_id not in fixed_weights]
    remaining_weight = 100.0 - sum(float(weight) for weight in fixed_weights.values())
    if not other_ids or remaining_weight <= 0:
        raise ValueError(
            "The recruiter-edited weights must leave a positive percentage for at least one other criterion."
        )

    prompt = f"""
You are the assessment framework agent. A recruiter has changed one or more
criterion weights and needs the remaining criteria rebalanced to keep the
complete framework at exactly 100%.

Use the job description, each criterion's definition, original weight rationale,
current weights, and the recruiter's reason. Keep every recruiter-edited criterion
weight exactly unchanged. Allocate the remaining percentage only among criteria
the recruiter did not edit, reflecting their relative importance in the role.

Return one JSON object with one key, allocations. It must be an array containing
exactly one object per criterion, each with criterion_id and weight. Include all
existing criterion IDs exactly once. All weights must sum to 100, and every
recruiter-edited criterion must retain its requested weight.

JOB DESCRIPTION:
{jd_text}

RECRUITER'S REASON:
{change_reason.strip()}

RECRUITER'S PROPOSED WEIGHTS:
{json.dumps(proposed_weights, ensure_ascii=False)}

RECRUITER-EDITED WEIGHTS TO KEEP FIXED:
{json.dumps(fixed_weights, ensure_ascii=False)}

EXISTING FRAMEWORK:
{criteria.model_dump_json(indent=2)}
"""
    allocation = _json_call(
        GEMINI_MODEL,
        prompt,
        CriteriaWeightAllocation,
        max_output_tokens=1000,
    )
    weights_by_id = {item.criterion_id: item.weight for item in allocation.allocations}
    if set(weights_by_id) != set(by_id):
        raise ValueError("Agent did not return exactly one weight for each existing criterion.")

    proposed_other_total = sum(weights_by_id[criterion_id] for criterion_id in other_ids)
    if proposed_other_total <= 0:
        raise ValueError("Agent did not allocate any weight to the other criteria.")
    weights_by_id.update(
        {criterion_id: float(weight) for criterion_id, weight in fixed_weights.items()}
    )
    for criterion_id in other_ids:
        weights_by_id[criterion_id] = (
            weights_by_id[criterion_id] / proposed_other_total
        ) * remaining_weight

    updated_criteria = [
        criterion.model_copy(update={"weight": weights_by_id[criterion.criterion_id]})
        for criterion in criteria.criteria
    ]
    return JobCriteria.model_validate(criteria.model_copy(update={"criteria": updated_criteria}).model_dump())


def agent_2_score_candidate(criteria: JobCriteria, candidate: dict[str, Any]) -> CandidateEvaluation:
    prompt = f"""
You are Agent 2: Candidate Screening & Explainability Agent.

Assess the candidate ONLY against the job-specific criteria created by Agent 1.

Fairness rules:
- Do not use full_name, email, university, age, gender, nationality, or other non-job-relevant personal attributes.
- GPA may be used only when the generated criteria make academic performance relevant.
- Do not invent skills, projects, experience, certifications, or achievements.
- Match evidence by meaning, not by identical wording. A candidate's "great in Excel" can support a requirement for strong spreadsheet skills; equivalent terminology is valid when the raw evidence supports the criterion.
- If no direct, criterion-relevant evidence appears in the candidate's raw data, score that criterion 0.
- Return exactly one criterion assessment for every criterion ID below.
- Score each criterion 0-100 and provide a concise evidence-based reason plus the raw candidate evidence supporting that score.
- A score above 0 requires direct, criterion-relevant evidence. Judge whether the candidate evidence is relevant semantically; it does not need to repeat the criterion's exact words. For evidence, copy one contiguous, verbatim excerpt from a single CANDIDATE EVIDENCE field so the recruiter can verify the raw wording. Do not rewrite, infer unsupported qualifications, or fabricate evidence. If no direct evidence supports the criterion, return score 0 and an empty evidence string.
- Do not calculate the overall weighted score.

Return ONE JSON object with exactly these top-level keys:
criterion_assessments, strengths, gaps.
Each criterion_assessments item must contain exactly: criterion_id, score, reason, evidence.
IMPORTANT FORMAT CONTRACT:
- criterion_assessments MUST be a JSON array.
- strengths MUST be one concise string describing the candidate's strongest relevant evidence.
- gaps MUST be one concise string describing missing or weaker relevant evidence.
- score MUST be an integer from 0 to 100.
- score MUST be 0 whenever no direct, criterion-relevant raw-data evidence is available.
- reason MUST be a single string, 1-3 sentences.
- A semantic match may use different wording from the criterion, but any score above 0 MUST have a verifiable verbatim excerpt from the raw candidate data in evidence. evidence MUST be one contiguous excerpt from a single raw candidate field, or an empty string when none is available.
- Do not wrap the JSON in markdown fences.

CRITERIA:
{criteria.model_dump_json(indent=2)}

CANDIDATE EVIDENCE:
{json.dumps(candidate, ensure_ascii=False, default=str)}
"""
    return _json_call(GEMINI_SCREEN_MODEL, prompt, CandidateEvaluation, max_output_tokens=1500)


def agent_3_interview_questions(criteria: JobCriteria, candidate: dict[str, Any], gaps: str) -> InterviewQuestions:
    prompt = f"""
You are Agent 3: Interview Design Agent.
Create 2-3 personalized interview questions that validate the most important evidence gaps from the AI screening.
Do not ask about protected characteristics or irrelevant personal details.
Return one JSON object with one key: questions.
FORMAT CONTRACT: questions MUST be a JSON array containing exactly 2-3 strings. Do not wrap the JSON in markdown fences.

CRITERIA:
{criteria.model_dump_json(indent=2)}
CANDIDATE:
{json.dumps(candidate, ensure_ascii=False, default=str)}
SCREENING GAPS:
{gaps}
"""
    return _json_call(GEMINI_MODEL, prompt, InterviewQuestions, max_output_tokens=800)


def agent_4_outreach_email(criteria: JobCriteria, candidate_name: str, candidate: dict[str, Any]) -> OutreachEmail:
    prompt = f"""
You are Agent 4: Candidate Outreach Agent.
Draft a concise professional interview invitation for a candidate already approved by a human recruiter.
Do not mention AI, internal scores, internal criteria, or protected attributes.
Do not invent facts. Include a clear call to action for arranging the next interview step.
Return one JSON object with exactly: subject, body.
FORMAT CONTRACT: subject MUST be one string; body MUST be one string. Do not wrap the JSON in markdown fences.

ROLE / JOB CONTEXT:
{criteria.model_dump_json(indent=2)}
CANDIDATE NAME:
{candidate_name}
CANDIDATE EVIDENCE:
{json.dumps(candidate, ensure_ascii=False, default=str)}
"""
    return _json_call(GEMINI_MODEL, prompt, OutreachEmail, max_output_tokens=1200)
