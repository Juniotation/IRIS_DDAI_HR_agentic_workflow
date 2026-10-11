from __future__ import annotations

import json
import re
from typing import Any


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Try JSON first for strings containing an encoded array.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [text]
    return [value]


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = _clean_text(value).replace(",", "")
    # Accept '25%' / '25 percent'.
    text = re.sub(r"\s*(%|percent)\s*$", "", text, flags=re.I)
    try:
        return float(text)
    except ValueError:
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _clean_text(value).lower()
    if text in {"true", "yes", "y", "1", "required", "must-have", "must have"}:
        return True
    if text in {"false", "no", "n", "0", "optional", "nice-to-have", "nice to have"}:
        return False
    return default


def _unwrap(payload: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(payload, dict):
        return payload
    # Accept common wrapper keys such as {"result": {...}} or {"data": {...}}.
    for key in keys:
        if key in payload and isinstance(payload[key], (dict, list)):
            return payload[key]
    return payload


def _normalize_criterion(raw: dict[str, Any], index: int) -> dict[str, Any]:
    criterion_id = raw.get("criterion_id", raw.get("criterionId", raw.get("id", f"C{index:02d}")))
    name = raw.get("name", raw.get("criterion", raw.get("title", f"Criterion {index}")))
    definition = raw.get("definition", raw.get("description", raw.get("rationale", "")))
    evidence = raw.get(
        "evidence_to_look_for",
        raw.get("evidence_to_look_for", raw.get("evidence", raw.get("evidence_signals", []))),
    )
    weight = raw.get("weight", raw.get("percentage", raw.get("priority_weight", 0)))
    must_have = raw.get("is_must_have", raw.get("must_have", raw.get("required", False)))
    return {
        "criterion_id": _clean_text(criterion_id) or f"C{index:02d}",
        "name": _clean_text(name),
        "definition": _clean_text(definition),
        "evidence_to_look_for": [_clean_text(x) for x in _as_list(evidence) if _clean_text(x)],
        "weight": _number(weight, 0.0),
        "is_must_have": _bool(must_have),
        "weight_reason": _clean_text(raw.get("weight_reason", "")),
        "source_evidence": _clean_text(raw.get("source_evidence", "")),
    }


def normalize_job_criteria(payload: Any) -> dict[str, Any]:
    payload = _unwrap(payload, ("result", "data", "job_criteria", "assessment_framework"))
    if not isinstance(payload, dict):
        raise ValueError("Job criteria response must be a JSON object.")

    raw_criteria = payload.get("criteria", payload.get("criteria_items", payload.get("assessment_criteria", [])))
    if isinstance(raw_criteria, dict):
        raw_criteria = list(raw_criteria.values())
    raw_criteria = _as_list(raw_criteria)

    criteria = []
    for i, raw in enumerate(raw_criteria, start=1):
        if isinstance(raw, str):
            raw = {"name": raw, "definition": raw, "evidence_to_look_for": [raw]}
        if isinstance(raw, dict):
            criteria.append(_normalize_criterion(raw, i))

    # Keep criterion IDs deterministic even when the LLM omits or duplicates IDs.
    used_ids = set()
    for i, item in enumerate(criteria, start=1):
        cid = item["criterion_id"]
        if not cid or cid in used_ids:
            item["criterion_id"] = f"C{i:02d}"
        used_ids.add(item["criterion_id"])

    # Weights express relative importance; normalize them to exactly 100 rather than
    # failing because the LLM returned e.g. 30/30/20/19 or 3/3/2/2.
    raw_weights = [float(item.get("weight") or 0.0) for item in criteria]
    total_weight = sum(raw_weights)
    if criteria and total_weight > 0:
        normalized = [w * 100.0 / total_weight for w in raw_weights]
        rounded = [round(w, 2) for w in normalized]
        correction = round(100.0 - sum(rounded[:-1]), 2) if len(rounded) > 1 else 100.0
        rounded[-1] = correction
        for item, weight in zip(criteria, rounded):
            item["weight"] = weight

    rules = payload.get("recommendation_rules", payload.get("recommendation_thresholds", {}))
    if isinstance(rules, str):
        try:
            parsed = json.loads(rules)
            rules = parsed if isinstance(parsed, dict) else {}
        except Exception:
            rules = {}
    if not isinstance(rules, dict):
        rules = {}
    rules = {
        "strong_fit_min": _number(rules.get("strong_fit_min", rules.get("strong_fit", 80)), 80),
        "potential_fit_min": _number(rules.get("potential_fit_min", rules.get("potential_fit", 65)), 65),
        "not_recommended_below": _number(rules.get("not_recommended_below", 65), 65),
    }

    return {
        "job_title": _clean_text(payload.get("job_title", payload.get("title", "Generated Job"))),
        "role_summary": _clean_text(payload.get("role_summary", payload.get("summary", "Generated assessment framework."))),
        "criteria": criteria,
        "must_have_requirements": [_clean_text(x) for x in _as_list(payload.get("must_have_requirements", payload.get("must_have", []))) if _clean_text(x)],
        "nice_to_have_requirements": [_clean_text(x) for x in _as_list(payload.get("nice_to_have_requirements", payload.get("nice_to_have", []))) if _clean_text(x)],
        "scoring_exclusions": [_clean_text(x) for x in _as_list(payload.get("scoring_exclusions", payload.get("exclusions", []))) if _clean_text(x)],
        "recommendation_rules": rules,
    }


def _normalize_assessment_item(raw: Any, key_hint: str | None = None) -> dict[str, Any]:
    if isinstance(raw, dict):
        cid = raw.get("criterion_id", raw.get("criterionId", raw.get("id", raw.get("criterion", key_hint))))
        score = raw.get("score", raw.get("rating", raw.get("value", 0)))
        reason = raw.get("reason", raw.get("justification", raw.get("explanation", raw.get("evidence", ""))))
        evidence = raw.get("evidence", raw.get("raw_data", raw.get("raw_evidence", "")))
    else:
        cid = key_hint
        score = raw
        reason = "No detailed reason provided by model."
        evidence = ""
    score_num = _number(score, 0.0)
    score_int = max(0, min(100, int(round(score_num or 0))))
    return {
        "criterion_id": _clean_text(cid),
        "score": score_int,
        "reason": _clean_text(reason),
        "evidence": _clean_text(evidence),
    }


def normalize_candidate_evaluation(payload: Any) -> dict[str, Any]:
    payload = _unwrap(payload, ("result", "data", "evaluation", "candidate_evaluation"))
    if not isinstance(payload, dict):
        raise ValueError("Candidate evaluation response must be a JSON object.")

    raw_assessments = payload.get(
        "criterion_assessments",
        payload.get("criteria_assessments", payload.get("assessments", [])),
    )
    items: list[dict[str, Any]] = []
    if isinstance(raw_assessments, dict):
        for key, value in raw_assessments.items():
            items.append(_normalize_assessment_item(value, key_hint=str(key)))
    else:
        for raw in _as_list(raw_assessments):
            if isinstance(raw, dict):
                items.append(_normalize_assessment_item(raw))

    def summary(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(f"• {_clean_text(x)}" for x in value if _clean_text(x))
        if isinstance(value, tuple):
            return "\n".join(f"• {_clean_text(x)}" for x in value if _clean_text(x))
        if isinstance(value, dict):
            return "\n".join(f"• {k}: {_clean_text(v)}" for k, v in value.items())
        return _clean_text(value)

    return {
        "criterion_assessments": items,
        "strengths": summary(payload.get("strengths", payload.get("key_strengths", "No strengths stated."))),
        "gaps": summary(payload.get("gaps", payload.get("risks", payload.get("development_gaps", "No gaps stated.")))),
    }


def normalize_interview_questions(payload: Any) -> dict[str, Any]:
    payload = _unwrap(payload, ("result", "data", "interview_questions"))
    if isinstance(payload, dict):
        value = payload.get("questions", payload.get("interview_questions", payload.get("question", [])))
    else:
        value = payload
    return {"questions": [_clean_text(x) for x in _as_list(value) if _clean_text(x)]}


def normalize_outreach_email(payload: Any) -> dict[str, Any]:
    payload = _unwrap(payload, ("result", "data", "email", "outreach_email"))
    if not isinstance(payload, dict):
        raise ValueError("Outreach email response must be a JSON object.")

    def text_or_join(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(_clean_text(x) for x in value if _clean_text(x))
        return _clean_text(value)

    return {
        "subject": text_or_join(payload.get("subject", payload.get("email_subject", "Thara Energy Interview Invitation"))),
        "body": text_or_join(payload.get("body", payload.get("email_body", payload.get("message", "")))),
    }
