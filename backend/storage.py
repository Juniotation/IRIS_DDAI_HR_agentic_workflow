from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH, DEFAULT_CANDIDATE_CSV, DATA_DIR

REQUIRED_CANDIDATE_COLUMNS = [
    "candidate_id", "full_name", "email", "degree", "university", "gpa",
    "skills_summary", "experience_summary",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_assessment_evidence_column(con: sqlite3.Connection) -> None:
    assessment_cols = {
        row[1] for row in con.execute("PRAGMA table_info(assessments)").fetchall()
    }
    if "evidence" not in assessment_cols:
        con.execute("ALTER TABLE assessments ADD COLUMN evidence TEXT NOT NULL DEFAULT ''")


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A short busy timeout prevents transient "database is locked" errors when
    # Streamlit reruns and a background screening worker touch SQLite close together.
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=15000")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL,
                degree TEXT,
                university TEXT,
                gpa REAL,
                skills_summary TEXT,
                experience_summary TEXT,
                extra_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_title TEXT NOT NULL,
                jd_text TEXT NOT NULL,
                criteria_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_name TEXT
            );

            CREATE TABLE IF NOT EXISTS criteria_weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                criterion_id TEXT NOT NULL,
                criterion_name TEXT NOT NULL,
                old_weight REAL,
                new_weight REAL NOT NULL,
                reason TEXT NOT NULL,
                source TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS assessments (
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                criterion_id TEXT NOT NULL,
                criterion_name TEXT NOT NULL,
                weight REAL NOT NULL,
                score INTEGER NOT NULL,
                score_reason TEXT NOT NULL,
                evidence TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (job_id, candidate_id, criterion_id)
            );

            CREATE TABLE IF NOT EXISTS candidate_results (
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                overall_score REAL NOT NULL,
                ai_recommendation TEXT NOT NULL,
                strengths TEXT NOT NULL,
                gaps TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                bias_audit_notes TEXT NOT NULL,
                case_status TEXT NOT NULL DEFAULT 'screened',
                screening_error TEXT DEFAULT '',
                PRIMARY KEY (job_id, candidate_id)
            );

            CREATE TABLE IF NOT EXISTS hitl_actions (
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                human_decision TEXT NOT NULL,
                recruiter_notes TEXT,
                human_reviewed_at TEXT NOT NULL,
                PRIMARY KEY (job_id, candidate_id)
            );

            CREATE TABLE IF NOT EXISTS downstream_outputs (
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                generated_interview_questions TEXT,
                generated_outreach_subject TEXT,
                generated_outreach_body TEXT,
                email_send_status TEXT NOT NULL DEFAULT 'still not send',
                email_sent_at TEXT,
                PRIMARY KEY (job_id, candidate_id)
            );

            CREATE TABLE IF NOT EXISTS candidate_edit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                field_key TEXT NOT NULL,
                old_value TEXT NOT NULL,
                new_value TEXT NOT NULL,
                changed_at TEXT NOT NULL
            );
            """
        )

        # Lightweight migrations for databases from V2.x.
        result_cols = {row[1] for row in con.execute("PRAGMA table_info(candidate_results)").fetchall()}
        if "case_status" not in result_cols and "status" in result_cols:
            con.execute("ALTER TABLE candidate_results RENAME COLUMN status TO case_status")
            result_cols = {row[1] for row in con.execute("PRAGMA table_info(candidate_results)").fetchall()}
        if "screening_error" not in result_cols:
            con.execute("ALTER TABLE candidate_results ADD COLUMN screening_error TEXT DEFAULT ''")
        _ensure_assessment_evidence_column(con)

        # Normalize legacy lifecycle values once. The app now uses case_status semantics.
        con.execute("UPDATE candidate_results SET case_status='screened' WHERE lower(case_status)='screened'")
        con.execute("UPDATE candidate_results SET case_status='screen error' WHERE lower(case_status)='screening error'")
        con.execute("UPDATE candidate_results SET case_status='approved' WHERE lower(case_status)='shortlisted'")
        con.execute("UPDATE candidate_results SET case_status='rejected' WHERE lower(case_status)='rejected'")
        con.execute("UPDATE candidate_results SET case_status='review' WHERE lower(case_status)='under review'")

        # Normalize legacy downstream email values for the new UI wording.
        con.execute("UPDATE downstream_outputs SET email_send_status='sent' WHERE lower(email_send_status)='sent'")
        con.execute("UPDATE downstream_outputs SET email_send_status='still not send' WHERE lower(email_send_status) IN ('not sent','still not send')")
        con.execute("UPDATE downstream_outputs SET email_send_status='failed' WHERE lower(email_send_status)='failed'")

        jobs = con.execute(
            "SELECT job_id, criteria_json, created_at FROM jobs"
        ).fetchall()
        for job in jobs:
            has_history = con.execute(
                "SELECT 1 FROM criteria_weight_history WHERE job_id=? LIMIT 1",
                (job["job_id"],),
            ).fetchone()
            if not has_history:
                _record_initial_criteria_history(
                    con,
                    job["job_id"],
                    json.loads(job["criteria_json"]),
                    job["created_at"],
                )


def clear_job_outputs(job_id: str | None) -> None:
    if not job_id:
        return
    with connect() as con:
        con.execute("DELETE FROM downstream_outputs WHERE job_id=?", (job_id,))
        con.execute("DELETE FROM hitl_actions WHERE job_id=?", (job_id,))
        con.execute("DELETE FROM assessments WHERE job_id=?", (job_id,))
        con.execute("DELETE FROM candidate_results WHERE job_id=?", (job_id,))


def clear_candidate_screening_errors(job_id: str, candidate_ids: list[str]) -> None:
    if not candidate_ids:
        return
    with connect() as con:
        placeholders = ",".join("?" for _ in candidate_ids)
        parameters = [job_id, *candidate_ids]
        con.execute(
            f"DELETE FROM assessments WHERE job_id=? AND candidate_id IN ({placeholders})",
            parameters,
        )
        con.execute(
            f"""
            DELETE FROM candidate_results
            WHERE job_id=? AND candidate_id IN ({placeholders})
                AND case_status='screen error'
            """,
            parameters,
        )


def validate_candidate_df(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in df.columns]


def import_candidates(df: pd.DataFrame, replace: bool = False) -> int:
    missing = validate_candidate_df(df)
    if missing:
        raise ValueError("Missing required candidate columns: " + ", ".join(missing))
    if df["candidate_id"].duplicated().any():
        raise ValueError("candidate_id values must be unique.")

    imported = 0
    with connect() as con:
        if replace:
            con.execute("DELETE FROM candidates")
        for _, row in df.iterrows():
            extra = {c: row[c] for c in df.columns if c not in REQUIRED_CANDIDATE_COLUMNS}
            extra = {k: (None if pd.isna(v) else v) for k, v in extra.items()}
            con.execute(
                """
                INSERT INTO candidates(candidate_id, full_name, email, degree, university, gpa, skills_summary, experience_summary, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    full_name=excluded.full_name,
                    email=excluded.email,
                    degree=excluded.degree,
                    university=excluded.university,
                    gpa=excluded.gpa,
                    skills_summary=excluded.skills_summary,
                    experience_summary=excluded.experience_summary,
                    extra_json=excluded.extra_json
                """,
                (
                    str(row["candidate_id"]), str(row["full_name"]), str(row["email"]),
                    str(row["degree"]), str(row["university"]),
                    float(row["gpa"]) if pd.notna(row["gpa"]) else None,
                    str(row["skills_summary"]), str(row["experience_summary"]),
                    json.dumps(extra, ensure_ascii=False, default=str),
                ),
            )
            imported += 1
    return imported


def ensure_seed_candidates() -> int:
    with connect() as con:
        count = con.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]
    if count:
        return count
    return import_candidates(pd.read_csv(DEFAULT_CANDIDATE_CSV))


def candidate_df() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM candidates ORDER BY candidate_id", con)


def create_job(job_id: str, job_title: str, jd_text: str, criteria: dict, source_name: str) -> None:
    with connect() as con:
        created_at = utc_now()
        con.execute(
            "INSERT INTO jobs(job_id, job_title, jd_text, criteria_json, created_at, source_name) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, job_title, jd_text, json.dumps(criteria, ensure_ascii=False), created_at, source_name),
        )
        _record_initial_criteria_history(con, job_id, criteria, created_at)


def _record_initial_criteria_history(
    con: sqlite3.Connection,
    job_id: str,
    criteria: dict,
    created_at: str,
) -> None:
    for criterion in criteria.get("criteria", []):
        con.execute(
            """
            INSERT INTO criteria_weight_history
            (job_id, criterion_id, criterion_name, old_weight, new_weight, reason, source, changed_at)
            VALUES (?, ?, ?, NULL, ?, ?, 'ai_generated', ?)
            """,
            (
                job_id,
                criterion["criterion_id"],
                criterion["name"],
                float(criterion["weight"]),
                "Initial weight generated by AI Agent 1.",
                created_at,
            ),
        )


def update_job_criteria(job_id: str, criteria: dict) -> None:
    with connect() as con:
        cursor = con.execute(
            "UPDATE jobs SET criteria_json=? WHERE job_id=?",
            (json.dumps(criteria, ensure_ascii=False), job_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Job {job_id} was not found while saving its assessment criteria.")


def save_job_criteria_weight_change(
    job_id: str,
    criteria: dict,
    reason: str,
    source: str,
) -> None:
    if not reason.strip():
        raise ValueError("A reason is required before criterion weights can be changed.")
    if source not in {"manual", "ai_redistribution"}:
        raise ValueError(f"Invalid criterion weight change source '{source}'.")

    criteria_items = criteria.get("criteria", [])
    new_by_id = {
        item["criterion_id"]: item
        for item in criteria_items
    }
    if len(new_by_id) != len(criteria_items) or not criteria_items:
        raise ValueError("Assessment criteria must have unique IDs before weights can be saved.")
    total_weight = sum(float(item["weight"]) for item in criteria_items)
    if abs(total_weight - 100.0) > 0.01:
        raise ValueError(f"Criteria weights must sum to 100. Got {total_weight:.2f}.")

    with connect() as con:
        job = con.execute(
            "SELECT criteria_json FROM jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if job is None:
            raise ValueError(f"Job {job_id} was not found while saving criterion weights.")

        old_items = json.loads(job["criteria_json"]).get("criteria", [])
        old_by_id = {item["criterion_id"]: item for item in old_items}
        if set(old_by_id) != set(new_by_id):
            raise ValueError("Criterion IDs cannot be added or removed while changing weights.")

        con.execute(
            "UPDATE jobs SET criteria_json=? WHERE job_id=?",
            (json.dumps(criteria, ensure_ascii=False), job_id),
        )
        for criterion_id, new_item in new_by_id.items():
            old_item = old_by_id[criterion_id]
            old_weight = float(old_item["weight"])
            new_weight = float(new_item["weight"])
            if abs(old_weight - new_weight) <= 0.000001:
                continue
            con.execute(
                """
                INSERT INTO criteria_weight_history
                (job_id, criterion_id, criterion_name, old_weight, new_weight, reason, source, changed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    criterion_id,
                    new_item["name"],
                    old_weight,
                    new_weight,
                    reason.strip(),
                    source,
                    utc_now(),
                ),
            )
            con.execute(
                "UPDATE assessments SET weight=? WHERE job_id=? AND criterion_id=?",
                (new_weight, job_id, criterion_id),
            )

        rules = criteria.get("recommendation_rules", {})
        strong_threshold = float(rules.get("strong_fit_min", 80))
        potential_threshold = float(rules.get("potential_fit_min", 65))
        assessment_rows = con.execute(
            """
            SELECT candidate_id, criterion_id, score
            FROM assessments
            WHERE job_id=?
            """,
            (job_id,),
        ).fetchall()
        scores_by_candidate: dict[str, dict[str, int]] = {}
        for assessment in assessment_rows:
            scores_by_candidate.setdefault(assessment["candidate_id"], {})[
                assessment["criterion_id"]
            ] = int(assessment["score"])
        expected_ids = set(new_by_id)
        for candidate_id, scores_by_id in scores_by_candidate.items():
            if set(scores_by_id) != expected_ids:
                continue
            overall_score = round(
                sum(
                    scores_by_id[criterion_id] * float(item["weight"]) / 100
                    for criterion_id, item in new_by_id.items()
                ),
                1,
            )
            recommendation = (
                "Strong Fit" if overall_score >= strong_threshold
                else "Potential Fit" if overall_score >= potential_threshold
                else "Not Recommended"
            )
            con.execute(
                """
                UPDATE candidate_results
                SET overall_score=?, ai_recommendation=?
                WHERE job_id=? AND candidate_id=?
                """,
                (overall_score, recommendation, job_id, candidate_id),
            )


def criteria_weight_history(
    job_id: str,
    criterion_id: str | None = None,
) -> list[dict[str, Any]]:
    with connect() as con:
        if criterion_id is None:
            rows = con.execute(
                """
                SELECT criterion_name, criterion_id, old_weight, new_weight, reason, source, changed_at
                FROM criteria_weight_history
                WHERE job_id=?
                ORDER BY id
                """,
                (job_id,),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT criterion_name, criterion_id, old_weight, new_weight, reason, source, changed_at
                FROM criteria_weight_history
                WHERE job_id=? AND criterion_id=?
                ORDER BY id
                """,
                (job_id, criterion_id),
            ).fetchall()
        return [dict(row) for row in rows]


def get_job(job_id: str) -> sqlite3.Row | None:
    with connect() as con:
        return con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()


def list_jobs() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT job_id, job_title, created_at, source_name FROM jobs ORDER BY created_at DESC", con)


def write_assessment(job_id: str, candidate_id: str, criterion_id: str, criterion_name: str, weight: float, score: int, reason: str, evidence: str = "") -> None:
    with connect() as con:
        _ensure_assessment_evidence_column(con)
        con.execute(
            """
            INSERT OR REPLACE INTO assessments(job_id,candidate_id,criterion_id,criterion_name,weight,score,score_reason,evidence)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (job_id, candidate_id, criterion_id, criterion_name, weight, score, reason, evidence),
        )


def write_result(job_id: str, candidate_id: str, overall_score: float, recommendation: str, strengths: str, gaps: str) -> None:
    audit = (
        "Mathematical score uses only job-relevant candidate evidence and the job-specific criteria. "
        "Name, email and university are excluded from scoring. Missing evidence is not invented."
    )
    with connect() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO candidate_results
            (job_id,candidate_id,overall_score,ai_recommendation,strengths,gaps,evaluated_at,bias_audit_notes,case_status,screening_error)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (job_id, candidate_id, float(overall_score), recommendation, strengths, gaps, utc_now(), audit, "screened", ""),
        )


def write_screening_error(job_id: str, candidate_id: str, error_message: str) -> None:
    audit = (
        "Screening did not complete for this candidate. No score was produced. "
        "The candidate must be retried or reviewed manually; no score is inferred from the failure."
    )
    with connect() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO candidate_results
            (job_id,candidate_id,overall_score,ai_recommendation,strengths,gaps,evaluated_at,bias_audit_notes,case_status,screening_error)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (job_id, candidate_id, 0.0, "Screening Error", "", "Screening error — no AI score available.", utc_now(), audit, "screen error", str(error_message)[:2000]),
        )


def results_df(job_id: str) -> pd.DataFrame:
    """Return one candidate-level workbench view, including HITL and downstream state."""
    with connect() as con:
        return pd.read_sql_query(
            """
            SELECT
                r.job_id,
                r.candidate_id,
                c.full_name,
                c.email,
                c.degree,
                c.university,
                c.gpa,
                c.skills_summary,
                c.experience_summary,
                r.overall_score,
                r.ai_recommendation,
                r.strengths,
                r.gaps,
                r.evaluated_at,
                r.bias_audit_notes,
                r.case_status AS case_status,
                r.screening_error,
                COALESCE(h.recruiter_notes, '') AS recruiter_note,
                h.human_decision,
                h.human_reviewed_at,
                COALESCE(d.generated_interview_questions, '') AS interview_question,
                COALESCE(d.generated_outreach_subject, '') AS outreach_subject,
                COALESCE(d.generated_outreach_body, '') AS outreach_email,
                COALESCE(d.email_send_status, 'still not send') AS email_status,
                d.email_sent_at
            FROM candidate_results r
            JOIN candidates c ON c.candidate_id=r.candidate_id
            LEFT JOIN hitl_actions h ON h.job_id=r.job_id AND h.candidate_id=r.candidate_id
            LEFT JOIN downstream_outputs d ON d.job_id=r.job_id AND d.candidate_id=r.candidate_id
            WHERE r.job_id=?
            ORDER BY
                CASE WHEN r.case_status='screen error' THEN 0 WHEN r.case_status='screened' THEN 1 WHEN r.case_status='review' THEN 2 WHEN r.case_status='approved' THEN 3 ELSE 4 END,
                r.overall_score DESC
            """,
            con, params=(job_id,),
        )


def assessments_df(job_id: str, candidate_id: str) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM assessments WHERE job_id=? AND candidate_id=? ORDER BY weight DESC",
            con, params=(job_id, candidate_id),
        )


def set_status(job_id: str, candidate_id: str, case_status: str, human_decision: str, recruiter_notes: str) -> None:
    allowed = {"screened", "screen error", "approved", "rejected", "review"}
    if case_status not in allowed:
        raise ValueError(f"Invalid case_status '{case_status}'. Allowed: {sorted(allowed)}")
    with connect() as con:
        previous = con.execute(
            "SELECT human_decision, recruiter_notes FROM hitl_actions WHERE job_id=? AND candidate_id=?",
            (job_id, candidate_id),
        ).fetchone()
        _record_candidate_edit(
            con, job_id, candidate_id, "next_action",
            str(previous["human_decision"] or "") if previous else "",
            human_decision,
        )
        _record_candidate_edit(
            con, job_id, candidate_id, "recruiter_note",
            str(previous["recruiter_notes"] or "") if previous else "",
            recruiter_notes or "",
        )
        con.execute(
            """
            INSERT INTO hitl_actions(job_id,candidate_id,human_decision,recruiter_notes,human_reviewed_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(job_id,candidate_id) DO UPDATE SET
                human_decision=excluded.human_decision,
                recruiter_notes=excluded.recruiter_notes,
                human_reviewed_at=excluded.human_reviewed_at
            """,
            (job_id, candidate_id, human_decision, recruiter_notes, utc_now()),
        )
        con.execute(
            "UPDATE candidate_results SET case_status=? WHERE job_id=? AND candidate_id=?",
            (case_status, job_id, candidate_id),
        )


def _record_candidate_edit(
    con: sqlite3.Connection,
    job_id: str,
    candidate_id: str,
    field_key: str,
    old_value: str,
    new_value: str,
) -> None:
    if old_value == new_value:
        return
    con.execute(
        """
        INSERT INTO candidate_edit_history
        (job_id, candidate_id, field_key, old_value, new_value, changed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, candidate_id, field_key, old_value, new_value, utc_now()),
    )


def candidate_edit_history(job_id: str, candidate_id: str) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute(
            """
            SELECT field_key, old_value, new_value, changed_at
            FROM candidate_edit_history
            WHERE job_id=? AND candidate_id=?
            ORDER BY changed_at DESC, id DESC
            """,
            (job_id, candidate_id),
        ).fetchall()
    return [dict(row) for row in rows]


def hitl_for(job_id: str, candidate_id: str) -> sqlite3.Row | None:
    with connect() as con:
        return con.execute("SELECT * FROM hitl_actions WHERE job_id=? AND candidate_id=?", (job_id, candidate_id)).fetchone()


def save_downstream(
    job_id: str,
    candidate_id: str,
    questions: list[str] | str,
    subject: str,
    body: str,
    preserve_email_status: bool = True,
    audit_changes: bool = True,
) -> None:
    question_text = questions if isinstance(questions, str) else "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    with connect() as con:
        existing = con.execute(
            """
            SELECT generated_interview_questions, generated_outreach_subject,
                   generated_outreach_body, email_send_status, email_sent_at
            FROM downstream_outputs WHERE job_id=? AND candidate_id=?
            """,
            (job_id, candidate_id),
        ).fetchone()
        status = existing["email_send_status"] if existing and preserve_email_status else "still not send"
        sent_at = existing["email_sent_at"] if existing and preserve_email_status else None
        if audit_changes:
            _record_candidate_edit(
                con, job_id, candidate_id, "interview_question",
                str(existing["generated_interview_questions"] or "") if existing else "",
                question_text,
            )
            _record_candidate_edit(
                con, job_id, candidate_id, "outreach_email_subject",
                str(existing["generated_outreach_subject"] or "") if existing else "",
                subject,
            )
            _record_candidate_edit(
                con, job_id, candidate_id, "outreach_email",
                str(existing["generated_outreach_body"] or "") if existing else "",
                body,
            )
        con.execute(
            """
            INSERT INTO downstream_outputs
            (job_id,candidate_id,generated_interview_questions,generated_outreach_subject,generated_outreach_body,email_send_status,email_sent_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(job_id,candidate_id) DO UPDATE SET
                generated_interview_questions=excluded.generated_interview_questions,
                generated_outreach_subject=excluded.generated_outreach_subject,
                generated_outreach_body=excluded.generated_outreach_body,
                email_send_status=excluded.email_send_status,
                email_sent_at=excluded.email_sent_at
            """,
            (job_id, candidate_id, question_text, subject, body, status, sent_at),
        )


def downstream_for(job_id: str, candidate_id: str) -> sqlite3.Row | None:
    with connect() as con:
        return con.execute("SELECT * FROM downstream_outputs WHERE job_id=? AND candidate_id=?", (job_id, candidate_id)).fetchone()


def mark_email(job_id: str, candidate_id: str, status: str, sent_at: str | None) -> None:
    with connect() as con:
        con.execute(
            "UPDATE downstream_outputs SET email_send_status=?, email_sent_at=? WHERE job_id=? AND candidate_id=?",
            (status, sent_at, job_id, candidate_id),
        )
