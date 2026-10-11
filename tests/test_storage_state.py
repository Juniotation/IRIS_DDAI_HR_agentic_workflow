import json
import sqlite3
from pathlib import Path

import pandas as pd

import backend.storage as storage


def test_legacy_assessments_gain_evidence_column_without_losing_rows(tmp_path, monkeypatch):
    db_path = Path(tmp_path) / "workbench.db"
    monkeypatch.setattr(storage, "DATA_DIR", Path(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE assessments (
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                criterion_id TEXT NOT NULL,
                criterion_name TEXT NOT NULL,
                weight REAL NOT NULL,
                score INTEGER NOT NULL,
                score_reason TEXT NOT NULL,
                PRIMARY KEY (job_id, candidate_id, criterion_id)
            )
            """
        )
        con.execute(
            """
            INSERT INTO assessments
            VALUES ('JOB', 'CAND-01', 'C01', 'Analysis', 100, 80, 'Good analysis')
            """
        )

    storage.init_db()

    assessment = storage.assessments_df("JOB", "CAND-01").iloc[0]
    assert assessment["score_reason"] == "Good analysis"
    assert assessment["evidence"] == ""


def test_write_assessment_migrates_legacy_table_before_insert(tmp_path, monkeypatch):
    db_path = Path(tmp_path) / "workbench.db"
    monkeypatch.setattr(storage, "DATA_DIR", Path(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE assessments (
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                criterion_id TEXT NOT NULL,
                criterion_name TEXT NOT NULL,
                weight REAL NOT NULL,
                score INTEGER NOT NULL,
                score_reason TEXT NOT NULL,
                PRIMARY KEY (job_id, candidate_id, criterion_id)
            )
            """
        )

    storage.write_assessment(
        "JOB", "CAND-01", "C01", "Analysis", 100, 80, "Good analysis",
        "Built a reporting dashboard.",
    )

    assessment = storage.assessments_df("JOB", "CAND-01").iloc[0]
    assert assessment["score_reason"] == "Good analysis"
    assert assessment["evidence"] == "Built a reporting dashboard."


def test_candidate_workbench_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", Path(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", Path(tmp_path) / "workbench.db")
    storage.init_db()
    storage.import_candidates(pd.DataFrame([{
        "candidate_id": "C01", "full_name": "Test Candidate", "email": "test@example.com",
        "degree": "B.Eng.", "university": "Mock University", "gpa": 3.5,
        "skills_summary": "Python, analytics", "experience_summary": "Mock internship",
    }]))

    storage.write_result("JOB", "C01", 82, "Strong Fit", "Strong evidence", "Validate depth")
    storage.set_status("JOB", "C01", "approved", "Approve", "Looks strong")
    storage.save_downstream("JOB", "C01", "1. Question", "Subject", "Body")

    row = storage.results_df("JOB").iloc[0]
    assert row["case_status"] == "approved"
    assert row["recruiter_note"] == "Looks strong"
    assert row["interview_question"] == "1. Question"
    assert row["email_status"] == "still not send"

    storage.mark_email("JOB", "C01", "sent", "now")
    assert storage.results_df("JOB").iloc[0]["email_status"] == "sent"


def test_candidate_edit_history_records_changed_fields_only(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", Path(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", Path(tmp_path) / "workbench.db")
    storage.init_db()
    storage.import_candidates(pd.DataFrame([{
        "candidate_id": "C01", "full_name": "Test Candidate", "email": "test@example.com",
        "degree": "B.Eng.", "university": "Mock University", "gpa": 3.5,
        "skills_summary": "Python, analytics", "experience_summary": "Mock internship",
    }]))
    storage.write_result("JOB", "C01", 82, "Strong Fit", "Strong evidence", "Validate depth")

    storage.set_status("JOB", "C01", "approved", "Approve", "Review SQL experience.")
    storage.set_status("JOB", "C01", "approved", "Approve", "Review SQL experience.")
    storage.save_downstream(
        "JOB", "C01", "Generated question", "Generated subject", "Generated email",
        audit_changes=False,
    )
    storage.save_downstream(
        "JOB", "C01", "Edited question", "Edited subject", "Edited email",
    )

    history = storage.candidate_edit_history("JOB", "C01")
    assert {entry["field_key"] for entry in history} == {
        "next_action",
        "recruiter_note",
        "interview_question",
        "outreach_email_subject",
        "outreach_email",
    }
    assert len(history) == 5
    assert all(entry["changed_at"] for entry in history)
    assert {
        (entry["field_key"], entry["old_value"], entry["new_value"])
        for entry in history
    } == {
        ("next_action", "", "Approve"),
        ("recruiter_note", "", "Review SQL experience."),
        ("interview_question", "Generated question", "Edited question"),
        ("outreach_email_subject", "Generated subject", "Edited subject"),
        ("outreach_email", "Generated email", "Edited email"),
    }


def test_update_job_criteria_persists_enrichment(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", Path(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", Path(tmp_path) / "workbench.db")
    storage.init_db()
    storage.create_job("JOB", "Analyst", "Original job description", {}, "jd.txt")

    storage.update_job_criteria("JOB", {"criteria": [{"weight_reason": "Role-critical skill"}]})

    assert storage.get_job("JOB")["criteria_json"] == (
        '{"criteria": [{"weight_reason": "Role-critical skill"}]}'
    )


def test_criteria_weight_changes_are_audited_and_reweight_existing_results(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "DATA_DIR", Path(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", Path(tmp_path) / "workbench.db")
    storage.init_db()
    criteria = {
        "recommendation_rules": {"strong_fit_min": 80, "potential_fit_min": 65},
        "criteria": [
            {"criterion_id": "C01", "name": "Analysis", "weight": 50},
            {"criterion_id": "C02", "name": "Modeling", "weight": 30},
            {"criterion_id": "C03", "name": "Communication", "weight": 20},
        ],
    }
    storage.create_job("JOB", "Analyst", "Job description", criteria, "jd.txt")
    storage.import_candidates(pd.DataFrame([{
        "candidate_id": "CAND-01", "full_name": "Test Candidate",
        "email": "test@example.com", "degree": "B.Eng.",
        "university": "Mock University", "gpa": 3.5,
        "skills_summary": "Analysis", "experience_summary": "Experience",
    }]))
    storage.write_assessment(
        "JOB", "CAND-01", "C01", "Analysis", 50, 100, "Strong",
        "Built a forecasting dashboard using SQL.",
    )
    storage.write_assessment("JOB", "CAND-01", "C02", "Modeling", 30, 0, "No evidence")
    storage.write_assessment("JOB", "CAND-01", "C03", "Communication", 20, 0, "No evidence")
    storage.write_result("JOB", "CAND-01", 50, "Not Recommended", "Evidence", "Gaps")

    updated = {
        "recommendation_rules": {"strong_fit_min": 80, "potential_fit_min": 65},
        "criteria": [
            {"criterion_id": "C01", "name": "Analysis", "weight": 40},
            {"criterion_id": "C02", "name": "Modeling", "weight": 35},
            {"criterion_id": "C03", "name": "Communication", "weight": 25},
        ],
    }
    storage.save_job_criteria_weight_change(
        "JOB", updated, "Modeling is more critical for this opening.", "manual"
    )

    assert storage.get_job("JOB")["criteria_json"] == json.dumps(updated, ensure_ascii=False)
    assessments = storage.assessments_df("JOB", "CAND-01")
    assert assessments["weight"].tolist() == [40, 35, 25]
    assert assessments.iloc[0]["evidence"] == "Built a forecasting dashboard using SQL."
    assert storage.results_df("JOB").iloc[0]["overall_score"] == 40

    history = storage.criteria_weight_history("JOB", "C02")
    assert [(entry["old_weight"], entry["new_weight"]) for entry in history] == [
        (None, 30.0),
        (30.0, 35.0),
    ]
    assert history[-1]["reason"] == "Modeling is more critical for this opening."
    assert history[-1]["source"] == "manual"


def test_legacy_job_gets_initial_criteria_history_once(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", Path(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", Path(tmp_path) / "workbench.db")
    storage.init_db()
    criteria = {
        "criteria": [
            {"criterion_id": "C01", "name": "Analysis", "weight": 50},
            {"criterion_id": "C02", "name": "Modeling", "weight": 30},
            {"criterion_id": "C03", "name": "Communication", "weight": 20},
        ],
    }
    storage.create_job("JOB", "Analyst", "Job description", criteria, "jd.txt")

    with storage.connect() as con:
        con.execute("DELETE FROM criteria_weight_history WHERE job_id='JOB'")

    storage.init_db()
    storage.init_db()

    assert len(storage.criteria_weight_history("JOB", "C01")) == 1
    assert storage.criteria_weight_history("JOB", "C01")[0]["old_weight"] is None
