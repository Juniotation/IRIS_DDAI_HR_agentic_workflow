from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from .config import SCREENING_DELAY_SECONDS, SCREENING_WORKERS
from .models import JobCriteria
from .storage import (
    candidate_df, clear_candidate_screening_errors, clear_job_outputs, create_job, save_downstream,
    write_assessment, write_result, write_screening_error,
)


class RecommendationThresholds(Protocol):
    strong_fit_min: float
    potential_fit_min: float


def new_job_id() -> str:
    return "JOB-" + uuid.uuid4().hex[:8].upper()


def weighted_score(criteria: JobCriteria, assessments) -> float:
    by_id = {a.criterion_id: a.score for a in assessments}
    return round(sum(by_id[c.criterion_id] * c.weight / 100 for c in criteria.criteria), 1)


def _has_verifiable_evidence(evidence: str, candidate: dict[str, object]) -> bool:
    excerpt = str(evidence or "").strip()
    return bool(excerpt) and any(
        excerpt in str(value)
        for value in candidate.values()
        if value is not None
    )


def recommendation(score: float, rules: RecommendationThresholds | dict[str, float]) -> str:
    def threshold(name: str, default: float) -> float:
        if isinstance(rules, dict):
            value = rules[name] if name in rules else default
        else:
            value = getattr(rules, name, default)
        return float(value)

    strong = threshold("strong_fit_min", 80)
    potential = threshold("potential_fit_min", 65)
    if score >= strong:
        return "Strong Fit"
    if score >= potential:
        return "Potential Fit"
    return "Not Recommended"


def run_job_screening(
    job_id: str,
    criteria: JobCriteria,
    progress_callback=None,
    candidate_ids: set[str] | None = None,
):
    # Import the Gemini SDK only when an AI workflow is actually executed.
    # This keeps deterministic scoring/tests usable without the optional AI runtime.
    from .gemini_agents import agent_2_score_candidate

    df = candidate_df()
    if candidate_ids is not None:
        requested_ids = {str(candidate_id) for candidate_id in candidate_ids}
        if not requested_ids:
            raise ValueError("At least one candidate ID is required when retrying screening errors.")
        available_ids = set(df["candidate_id"].astype(str))
        missing_ids = requested_ids - available_ids
        if missing_ids:
            raise ValueError(f"Candidates were not found in the database: {sorted(missing_ids)}")
        df = df[df["candidate_id"].astype(str).isin(requested_ids)]
    total = len(df)
    if total == 0:
        raise ValueError("Candidate database is empty. Upload a CSV or load the mock candidate DB first.")

    if candidate_ids is None:
        clear_job_outputs(job_id)
    else:
        clear_candidate_screening_errors(job_id, sorted(requested_ids))
    expected_ids = {c.criterion_id for c in criteria.criteria}
    criterion_by_id = {c.criterion_id: c for c in criteria.criteria}

    def evaluate_one(row):
        candidate = {
            "degree": row["degree"], "gpa": row["gpa"],
            "skills_summary": row["skills_summary"], "experience_summary": row["experience_summary"],
        }
        evaluation = agent_2_score_candidate(criteria, candidate)
        name_to_id = {c.name.strip().lower(): c.criterion_id for c in criteria.criteria}
        canonical_items, seen = [], set()
        for item in evaluation.criterion_assessments:
            raw = item.criterion_id.strip()
            canonical = raw if raw in expected_ids else name_to_id.get(raw.lower())
            if not canonical:
                raise ValueError(f"Unknown criterion '{raw}'. Expected IDs={sorted(expected_ids)}")
            if canonical in seen:
                raise ValueError(f"Duplicate criterion '{canonical}'.")
            seen.add(canonical)
            item.criterion_id = canonical
            canonical_items.append(item)
        missing = expected_ids - seen
        if missing:
            raise ValueError(f"Omitted criteria: {sorted(missing)}")
        for item in canonical_items:
            if not _has_verifiable_evidence(item.evidence, candidate):
                item.score = 0
                item.evidence = ""
                item.reason = (
                    "No verifiable verbatim evidence excerpt was supplied from the "
                    "candidate's raw data; score set to 0."
                )
        score = weighted_score(criteria, canonical_items)
        rec = recommendation(score, criteria.recommendation_rules)
        if SCREENING_DELAY_SECONDS:
            time.sleep(SCREENING_DELAY_SECONDS)
        return str(row["candidate_id"]), evaluation, canonical_items, score, rec

    completed = errors = 0
    with ThreadPoolExecutor(max_workers=min(SCREENING_WORKERS, total)) as pool:
        futures = {pool.submit(evaluate_one, row): str(row["candidate_id"]) for _, row in df.iterrows()}
        for future in as_completed(futures):
            cid = futures[future]
            completed += 1
            try:
                candidate_id, evaluation, canonical_items, score, rec = future.result()
                for item in canonical_items:
                    c = criterion_by_id[item.criterion_id]
                    write_assessment(
                        job_id,
                        candidate_id,
                        c.criterion_id,
                        c.name,
                        c.weight,
                        item.score,
                        item.reason,
                        item.evidence,
                    )
                write_result(job_id, candidate_id, score, rec, evaluation.strengths, evaluation.gaps)
            except Exception as exc:
                errors += 1
                write_screening_error(job_id, cid, str(exc))
            finally:
                if progress_callback:
                    progress_callback(completed, total, cid, errors)
    return {"total": total, "completed": completed - errors, "errors": errors}


def create_job_from_jd(jd_text: str, source_name: str):
    from .gemini_agents import agent_1_build_criteria
    criteria = agent_1_build_criteria(jd_text)
    job_id = new_job_id()
    create_job(job_id, criteria.job_title, jd_text, criteria.model_dump(), source_name)
    return job_id, criteria


def approve_and_generate(job_id: str, candidate_id: str, criteria: JobCriteria, candidate_name: str):
    from .gemini_agents import agent_3_interview_questions, agent_4_outreach_email
    from .storage import results_df
    df = candidate_df()
    row_match = df[df["candidate_id"].astype(str) == str(candidate_id)]
    if row_match.empty:
        raise ValueError(f"Candidate {candidate_id} was not found in the candidate database.")
    row = row_match.iloc[0]
    candidate = {
        "degree": row["degree"], "gpa": row["gpa"],
        "skills_summary": row["skills_summary"], "experience_summary": row["experience_summary"],
    }
    result = results_df(job_id)
    match = result[result["candidate_id"].astype(str) == str(candidate_id)]
    if match.empty:
        raise ValueError("Candidate has no screening result yet.")
    gap_text = str(match.iloc[0]["gaps"])
    questions = agent_3_interview_questions(criteria, candidate, gap_text)
    email = agent_4_outreach_email(criteria, candidate_name, candidate)
    save_downstream(
        job_id,
        candidate_id,
        questions.questions,
        email.subject,
        email.body,
        audit_changes=False,
    )
    return questions, email
