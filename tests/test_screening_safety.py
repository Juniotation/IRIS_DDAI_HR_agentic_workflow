from types import SimpleNamespace

import pandas as pd

from backend import gemini_agents, workflow
from backend.models import (
    CandidateEvaluation,
    CriterionAssessment,
    DynamicCriterion,
    JobCriteria,
)
from backend.workflow import recommendation, weighted_score


def make_criteria():
    return JobCriteria(
        job_title="Test Role",
        role_summary="A role that needs analytical and communication capability for business work.",
        criteria=[
            DynamicCriterion(criterion_id="C01", name="Python", definition="Ability to use Python for relevant analysis.", evidence_to_look_for=["Python projects"], weight=60),
            DynamicCriterion(criterion_id="C02", name="Communication", definition="Ability to communicate findings clearly to stakeholders.", evidence_to_look_for=["presentations"], weight=20),
            DynamicCriterion(criterion_id="C03", name="Problem Solving", definition="Ability to solve relevant business problems.", evidence_to_look_for=["case work"], weight=20),
        ],
        must_have_requirements=["Relevant experience"],
        nice_to_have_requirements=[],
        scoring_exclusions=["name", "email", "university"],
        recommendation_rules={"strong_fit_min": 80, "potential_fit_min": 65, "not_recommended_below": 65},
    )


def test_weighted_score_and_recommendation():
    c = make_criteria()
    a = [CriterionAssessment(criterion_id="C01", score=90, reason="Strong Python project evidence."), CriterionAssessment(criterion_id="C02", score=70, reason="Good presentation and stakeholder evidence."), CriterionAssessment(criterion_id="C03", score=70, reason="Good problem-solving evidence.")]
    assert weighted_score(c, a) == 82.0
    assert recommendation(82.0, c.recommendation_rules) == "Strong Fit"


def test_recommendation_accepts_reloaded_rules_model():
    rules = SimpleNamespace(strong_fit_min=80, potential_fit_min=65)

    assert recommendation(82.0, rules) == "Strong Fit"
    assert recommendation(70.0, rules) == "Potential Fit"


def test_agent_two_allows_semantic_matches_but_requires_raw_evidence(monkeypatch):
    captured = {}
    expected = CandidateEvaluation(
        criterion_assessments=[
            CriterionAssessment(
                criterion_id="C01",
                score=80,
                reason="Excel proficiency supports the spreadsheet requirement.",
                evidence="Great in Excel",
            ),
            CriterionAssessment(
                criterion_id="C02",
                score=0,
                reason="No direct communication evidence was provided.",
                evidence="",
            ),
            CriterionAssessment(
                criterion_id="C03",
                score=0,
                reason="No direct problem-solving evidence was provided.",
                evidence="",
            ),
        ],
        strengths="The candidate reports strong Excel ability.",
        gaps="Communication and problem-solving evidence is not available.",
    )

    def fake_json_call(_model, prompt, _schema, max_output_tokens=None):
        captured["prompt"] = prompt
        return expected

    monkeypatch.setattr(gemini_agents, "_json_call", fake_json_call)
    criteria = make_criteria()
    criteria.criteria[0].name = "Spreadsheet Skills"
    criteria.criteria[0].definition = "Ability to use spreadsheet software effectively for relevant analysis."
    candidate = {
        "degree": "B.Eng.",
        "gpa": 3.5,
        "skills_summary": "Great in Excel",
        "experience_summary": "Relevant internship experience",
    }

    result = gemini_agents.agent_2_score_candidate(criteria, candidate)

    assert result.criterion_assessments[0].score == 80
    assert result.criterion_assessments[0].evidence == "Great in Excel"
    assert "Match evidence by meaning, not by identical wording." in captured["prompt"]
    assert "does not need to repeat the criterion's exact words" in captured["prompt"]
    assert "verbatim excerpt from a single CANDIDATE EVIDENCE field" in captured["prompt"]


def test_screening_keeps_score_for_semantic_match_with_verifiable_raw_excerpt(monkeypatch):
    criteria = make_criteria()
    criteria.criteria[0].name = "Spreadsheet Skills"
    criteria.criteria[0].definition = "Ability to use spreadsheet software effectively for relevant analysis."
    candidate = {
        "candidate_id": "C01",
        "degree": "B.Eng.",
        "gpa": 3.5,
        "skills_summary": "Great in Excel",
        "experience_summary": "Relevant internship experience",
    }
    candidates = pd.DataFrame([candidate])
    expected = CandidateEvaluation(
        criterion_assessments=[
            CriterionAssessment(
                criterion_id="C01",
                score=80,
                reason="Excel proficiency supports the spreadsheet requirement.",
                evidence="Great in Excel",
            ),
            CriterionAssessment(
                criterion_id="C02",
                score=0,
                reason="No direct communication evidence was provided.",
                evidence="",
            ),
            CriterionAssessment(
                criterion_id="C03",
                score=0,
                reason="No direct problem-solving evidence was provided.",
                evidence="",
            ),
        ],
        strengths="The candidate reports strong Excel ability.",
        gaps="Communication and problem-solving evidence is not available.",
    )
    saved_assessments = []
    saved_results = []
    monkeypatch.setattr(workflow, "candidate_df", lambda: candidates)
    monkeypatch.setattr(workflow, "clear_job_outputs", lambda _job_id: None)
    monkeypatch.setattr(
        workflow,
        "write_assessment",
        lambda _job_id, _candidate_id, criterion_id, _name, _weight, score, reason, evidence:
            saved_assessments.append((criterion_id, score, reason, evidence)),
    )
    monkeypatch.setattr(
        workflow,
        "write_result",
        lambda _job_id, _candidate_id, score, rec, *_args: saved_results.append((score, rec)),
    )
    monkeypatch.setattr(workflow, "write_screening_error", lambda *_args: None)
    monkeypatch.setattr(workflow, "SCREENING_DELAY_SECONDS", 0)
    monkeypatch.setattr(gemini_agents, "agent_2_score_candidate", lambda *_args: expected)

    result = workflow.run_job_screening("JOB", criteria)

    assert result == {"total": 1, "completed": 1, "errors": 0}
    assert saved_assessments[0][1] == 80
    assert saved_assessments[0][3] == "Great in Excel"
    assert saved_results == [(48.0, "Not Recommended")]


def test_screening_retry_only_evaluates_requested_candidates(monkeypatch):
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "degree": "B.Eng.",
                "gpa": 3.5,
                "skills_summary": f"Skills for {candidate_id}",
                "experience_summary": "Relevant internship experience",
            }
            for candidate_id in ("C01", "C02")
        ]
    )
    evaluated_ids = []
    cleared_errors = []
    written_results = []
    written_assessments = []
    overall_scores = []
    monkeypatch.setattr(workflow, "candidate_df", lambda: candidates)
    def fail_full_clear(_job_id):
        raise AssertionError("Full screening outputs must be preserved during a retry.")

    monkeypatch.setattr(workflow, "clear_job_outputs", fail_full_clear)
    monkeypatch.setattr(
        workflow,
        "clear_candidate_screening_errors",
        lambda _job_id, candidate_ids: cleared_errors.extend(candidate_ids),
    )
    monkeypatch.setattr(
        workflow,
        "write_assessment",
        lambda _job_id, _candidate_id, criterion_id, _name, _weight, score, reason, evidence:
            written_assessments.append((criterion_id, score, reason, evidence)),
    )
    monkeypatch.setattr(
        workflow,
        "write_result",
        lambda _job_id, candidate_id, score, *_args: (
            written_results.append(candidate_id),
            overall_scores.append(score),
        ),
    )
    monkeypatch.setattr(workflow, "write_screening_error", lambda *_args: None)
    monkeypatch.setattr(workflow, "SCREENING_DELAY_SECONDS", 0)

    def score_candidate(_criteria, candidate):
        candidate_id = candidate["skills_summary"].replace("Skills for ", "")
        evaluated_ids.append(candidate_id)
        return CandidateEvaluation(
            criterion_assessments=[
                CriterionAssessment(
                    criterion_id=criterion_id,
                    score=75,
                    reason="Relevant evidence is present for this criterion.",
                    evidence=(
                        "Relevant internship experience"
                        if criterion_id == "C01"
                        else "Invented evidence not present in raw candidate data"
                        if criterion_id == "C03"
                        else ""
                    ),
                )
                for criterion_id in ("C01", "C02", "C03")
            ],
            strengths="Relevant skills and experience were described.",
            gaps="More detail would help validate the depth of experience.",
        )

    monkeypatch.setattr(gemini_agents, "agent_2_score_candidate", score_candidate)

    result = workflow.run_job_screening(
        "JOB",
        make_criteria(),
        candidate_ids={"C02"},
    )

    assert result == {"total": 1, "completed": 1, "errors": 0}
    assert evaluated_ids == ["C02"]
    assert cleared_errors == ["C02"]
    assert written_results == ["C02"]
    assert [item[1] for item in written_assessments] == [75, 0, 0]
    assert written_assessments[0][3] == "Relevant internship experience"
    assert all(not item[3] for item in written_assessments[1:])
    assert all("score set to 0" in item[2] for item in written_assessments[1:])
    assert overall_scores == [45.0]
