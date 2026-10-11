from types import SimpleNamespace
from google.genai import _transformers

from backend import gemini_agents
from backend.models import (
    CandidateEvaluation,
    CriteriaWeightAllocation,
    CriteriaWeightExplanations,
    CriterionAssessment,
    InterviewQuestions,
    JobCriteria,
    OutreachEmail,
)
from backend.response_adapter import (
    normalize_candidate_evaluation,
    normalize_interview_questions,
    normalize_job_criteria,
    normalize_outreach_email,
)
from backend.gemini_agents import _parse_json


def test_parse_json_repairs_dangling_dash_before_property_name():
    raw = '{"score": 65, -reason": "Relevant experience is limited."}'

    assert _parse_json(raw) == {
        "score": 65,
        "reason": "Relevant experience is limited.",
    }


def test_parse_json_still_rejects_unrelated_malformed_json():
    try:
        _parse_json('{"score": 65,, "reason": "Bad JSON"}')
    except RuntimeError as exc:
        assert "Gemini returned invalid JSON" in str(exc)
    else:
        raise AssertionError("Unrelated malformed JSON should not be silently repaired.")


def test_json_call_uses_structured_schema_and_accepts_parsed_model(monkeypatch):
    evaluation = CandidateEvaluation(
        criterion_assessments=[
            CriterionAssessment(criterion_id="C01", score=80, reason="Strong evidence from a relevant work project."),
            CriterionAssessment(criterion_id="C02", score=70, reason="Some evidence from relevant coursework and tools."),
            CriterionAssessment(criterion_id="C03", score=60, reason="Limited direct evidence of stakeholder communication."),
        ],
        strengths="Relevant project experience and strong analytical tools.",
        gaps="Limited evidence of stakeholder communication.",
    )
    captured = {}

    def generate_content(*, model, contents, config):
        captured["config"] = config
        return SimpleNamespace(parsed=evaluation)

    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(gemini_agents, "get_client", lambda: fake_client)
    monkeypatch.setattr(gemini_agents, "_thinking_config", lambda _model: None)

    result = gemini_agents._json_call("test-model", "Return evaluation", CandidateEvaluation)

    assert result == evaluation
    assert captured["config"].response_mime_type == "application/json"
    assert isinstance(captured["config"].response_schema, dict)
    assert "additionalProperties" not in str(captured["config"].response_schema)


def test_response_schemas_convert_to_gemini_api_schema():
    models = (
        CandidateEvaluation,
        JobCriteria,
        CriteriaWeightAllocation,
        CriteriaWeightExplanations,
        InterviewQuestions,
        OutreachEmail,
    )
    fake_gemini_client = SimpleNamespace(vertexai=False)

    for model in models:
        response_schema = gemini_agents._gemini_response_schema(model)
        converted = _transformers.t_schema(fake_gemini_client, response_schema)
        assert converted is not None


def test_all_gemini_response_models_are_accepted_as_structured_schemas():
    for schema in (
        CandidateEvaluation,
        JobCriteria,
        CriteriaWeightExplanations,
        InterviewQuestions,
        OutreachEmail,
    ):
        config = gemini_agents.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        )
        assert config.response_schema is schema


def test_job_criteria_accepts_string_evidence_and_aliases():
    p = normalize_job_criteria({
        "title": "Data Engineer",
        "summary": "A data engineering role requiring analytics and engineering capability.",
        "criteria": [
            {"id": "C01", "name": "Python", "definition": "Use Python for data tasks.", "evidence": "Python projects", "percentage": "40%"},
            {"criterion": "SQL", "description": "Use SQL for querying and analysis.", "evidence_signals": ["SQL projects"], "weight": 35},
            {
                "name": "Communication",
                "definition": "Communicate technical findings clearly to stakeholders.",
                "evidence_to_look_for": ["presentations"],
                "weight": 25,
                "weight_reason": "This is important for translating results into action.",
                "source_evidence": "Communicate technical findings clearly to stakeholders.",
            },
        ],
        "must_have": "Relevant experience",
        "exclusions": ["name", "email", "university"],
    })
    assert p["criteria"][0]["evidence_to_look_for"] == ["Python projects"]
    assert sum(c["weight"] for c in p["criteria"]) == 100
    assert p["criteria"][2]["weight_reason"] == "This is important for translating results into action."
    assert p["criteria"][2]["source_evidence"] == "Communicate technical findings clearly to stakeholders."


def test_candidate_evaluation_accepts_string_summaries_and_map_shape():
    p = normalize_candidate_evaluation({
        "assessments": {
            "C01": {
                "score": "90",
                "justification": "Strong evidence from projects.",
                "evidence": "Built Python automation for monthly reporting.",
            },
            "C02": {"rating": 80, "reason": "Good evidence from coursework and internship."},
            "C03": {"value": 70, "explanation": "Some evidence but depth is limited."},
        },
        "strengths": ["Python", "Analytical thinking"],
        "gaps": "Limited cloud experience.",
    })
    assert len(p["criterion_assessments"]) == 3
    assert p["criterion_assessments"][0]["score"] == 90
    assert p["criterion_assessments"][0]["evidence"] == "Built Python automation for monthly reporting."
    assert p["criterion_assessments"][1]["evidence"] == ""
    assert "Python" in p["strengths"]


def test_downstream_shapes():
    assert len(normalize_interview_questions({"question": "Tell me about a project."})["questions"]) == 1
    email = normalize_outreach_email({"subject": ["Interview invitation"], "body": ["Hello", "Please reply."]})
    assert email["subject"] == "Interview invitation"
    assert "Please reply" in email["body"]
