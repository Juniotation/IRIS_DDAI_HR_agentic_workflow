import pytest

from backend import gemini_agents
from backend.models import (
    CriteriaEvidenceSources,
    CriteriaWeightAllocation,
    CriteriaWeightExplanations,
    CriterionWeight,
    CriterionWeightExplanation,
    DynamicCriterion,
    JobCriteria,
)


def test_dynamic_criteria_weights():
    criteria = JobCriteria(
        job_title="Test Role",
        role_summary="A role that needs analytical and communication capability for business work.",
        criteria=[
            DynamicCriterion(criterion_id="C01", name="Python", definition="Ability to use Python for relevant analysis.", evidence_to_look_for=["Python projects", "automation"], weight=40),
            DynamicCriterion(criterion_id="C02", name="Analytics", definition="Ability to analyse data and produce useful insights.", evidence_to_look_for=["analysis", "dashboards"], weight=35),
            DynamicCriterion(criterion_id="C03", name="Communication", definition="Ability to explain findings clearly to stakeholders.", evidence_to_look_for=["presentations", "stakeholders"], weight=25),
        ],
        must_have_requirements=["Relevant experience"],
        nice_to_have_requirements=[],
        scoring_exclusions=["name", "email", "university"],
        recommendation_rules={"strong_fit_min": 80, "potential_fit_min": 65},
    )
    assert sum(c.weight for c in criteria.criteria) == 100


def test_job_criteria_gemini_schema_has_no_additional_properties():
    schema = JobCriteria.model_json_schema()
    found = []

    def find_additional_properties(value, path="$"):
        if isinstance(value, dict):
            if "additionalProperties" in value:
                found.append(path)
            for key, child in value.items():
                find_additional_properties(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                find_additional_properties(child, f"{path}[{index}]")

    find_additional_properties(schema)

    assert not found


def test_legacy_criteria_without_weight_explanation_remain_loadable():
    criterion = DynamicCriterion(
        criterion_id="C01",
        name="Python",
        definition="Ability to use Python for relevant analysis.",
        evidence_to_look_for=["Python projects"],
        weight=100,
    )
    assert criterion.weight_reason == ""
    assert criterion.source_evidence == ""


def test_agent_one_accepts_paraphrased_criteria_and_shows_verbatim_source(monkeypatch):
    source = "Strong SQL and Excel capability is required."
    jd_text = (
        "Strong SQL and Excel capability is required. "
        "Please communicate findings clearly to business stakeholders."
    )
    criteria = JobCriteria(
        job_title="Data Analyst",
        role_summary="Analyze supply chain data and communicate findings to business teams.",
        criteria=[
            DynamicCriterion(
                criterion_id="C01",
                name="SQL",
                definition="Ability to query datasets using SQL for analysis.",
                evidence_to_look_for=["SQL projects"],
                weight=50,
                weight_reason="SQL is explicitly required and central to the analysis work.",
                source_evidence=source,
            ),
            DynamicCriterion(
                criterion_id="C02",
                name="Excel",
                definition="Ability to use Excel for data preparation and analysis.",
                evidence_to_look_for=["Excel reports"],
                weight=30,
                weight_reason="Excel is explicitly required as a core working tool.",
                source_evidence=source,
            ),
            DynamicCriterion(
                criterion_id="C03",
                name="Communication",
                definition="Ability to communicate analytical findings clearly.",
                evidence_to_look_for=["Presentations"],
                weight=20,
                weight_reason="Communication supports adoption of analytical findings.",
                source_evidence="Explain results to company leaders.",
            ),
        ],
        must_have_requirements=["Strong SQL and Excel capability"],
        scoring_exclusions=["name", "email", "age"],
        recommendation_rules={"strong_fit_min": 80, "potential_fit_min": 65},
    )
    explanations = CriteriaWeightExplanations(
        explanations=[
            CriterionWeightExplanation(
                criterion_id="C01",
                weight_reason="SQL is explicitly required and central to the analysis work.",
                source_evidence=source,
            ),
            CriterionWeightExplanation(
                criterion_id="C02",
                weight_reason="Excel is explicitly required as a core working tool.",
                source_evidence=source,
            ),
            CriterionWeightExplanation(
                criterion_id="C03",
                weight_reason="Communication supports adoption of analytical findings.",
                source_evidence="Explain results to company leaders.",
            ),
        ]
    )
    evidence_sources = CriteriaEvidenceSources(
        sources=[
            {"criterion_id": "C01", "source_segment_id": "S001"},
            {"criterion_id": "C02", "source_segment_id": "S001"},
            {"criterion_id": "C03", "source_segment_id": "S002"},
        ]
    )
    responses = iter([criteria, explanations, evidence_sources])
    monkeypatch.setattr(
        gemini_agents,
        "_json_call",
        lambda *_args, **_kwargs: next(responses),
    )

    result = gemini_agents.agent_1_build_criteria(jd_text)

    assert result.criteria[2].name == "Communication"
    assert result.criteria[2].definition == "Ability to communicate analytical findings clearly."
    assert result.criteria[2].source_evidence == (
        "Please communicate findings clearly to business stakeholders."
    )
    assert result.criteria[2].source_evidence in jd_text


def test_agent_one_enriches_missing_weight_explanations(monkeypatch):
    jd_text = "SQL is required. Excel is required. Communicate findings to stakeholders."
    criteria = JobCriteria(
        job_title="Data Analyst",
        role_summary="Analyze business data and communicate findings to stakeholders.",
        criteria=[
            DynamicCriterion(
                criterion_id=f"C0{i}",
                name=name,
                definition=f"Ability to perform {name.lower()} tasks for the role.",
                evidence_to_look_for=[name],
                weight=weight,
            )
            for i, (name, weight) in enumerate(
                [("SQL", 50), ("Excel", 30), ("Communication", 20)], start=1
            )
        ],
        must_have_requirements=["SQL"],
        scoring_exclusions=["name", "email", "age"],
        recommendation_rules={"strong_fit_min": 80, "potential_fit_min": 65},
    )
    explanations = CriteriaWeightExplanations(
        explanations=[
            CriterionWeightExplanation(
                criterion_id="C01",
                weight_reason="SQL is explicitly required and central to the role.",
                source_evidence="SQL is required.",
            ),
            CriterionWeightExplanation(
                criterion_id="C02",
                weight_reason="Excel is also explicitly required for core work.",
                source_evidence="Excel is required.",
            ),
            CriterionWeightExplanation(
                criterion_id="C03",
                weight_reason="Communication supports delivery of useful findings.",
                source_evidence="Communicate findings to stakeholders.",
            ),
        ]
    )
    monkeypatch.setattr(gemini_agents, "_json_call", lambda *args, **kwargs: criteria)
    monkeypatch.setattr(
        gemini_agents,
        "enrich_criteria_weight_explanations",
        lambda *_: criteria.model_copy(
            update={
                "criteria": [
                    criterion.model_copy(
                        update={
                            "weight_reason": explanation.weight_reason,
                            "source_evidence": explanation.source_evidence,
                        }
                    )
                    for criterion, explanation in zip(
                        criteria.criteria, explanations.explanations
                    )
                ]
            }
        ),
    )

    result = gemini_agents.agent_1_build_criteria(jd_text)

    assert [criterion.weight for criterion in result.criteria] == [50, 30, 20]
    assert all(criterion.weight_reason for criterion in result.criteria)
    assert all(criterion.source_evidence for criterion in result.criteria)


def test_legacy_criteria_can_be_enriched_without_changing_weights(monkeypatch):
    criteria = JobCriteria(
        job_title="Data Analyst",
        role_summary="Analyze data and communicate findings to stakeholders.",
        criteria=[
            DynamicCriterion(
                criterion_id=f"C0{i}",
                name=name,
                definition=f"Ability to perform {name.lower()} tasks for the role.",
                evidence_to_look_for=[name],
                weight=weight,
            )
            for i, (name, weight) in enumerate(
                [("SQL", 50), ("Excel", 30), ("Communication", 20)], start=1
            )
        ],
        must_have_requirements=["SQL"],
        scoring_exclusions=["name", "email", "age"],
        recommendation_rules={"strong_fit_min": 80, "potential_fit_min": 65},
    )
    explanations = CriteriaWeightExplanations(
        explanations=[
            CriterionWeightExplanation(
                criterion_id="C01",
                weight_reason="SQL is central to the role's data analysis responsibilities.",
                source_evidence="SQL ability is a key responsibility.",
            ),
            CriterionWeightExplanation(
                criterion_id="C02",
                weight_reason="Excel supports the required preparation and reporting work.",
                source_evidence="Spreadsheet reporting is important.",
            ),
            CriterionWeightExplanation(
                criterion_id="C03",
                weight_reason="Communication ensures findings reach relevant stakeholders.",
                source_evidence="Share insights clearly.",
            ),
        ]
    )
    evidence_sources = CriteriaEvidenceSources(
        sources=[
            {"criterion_id": "C01", "source_segment_id": "S001"},
            {"criterion_id": "C02", "source_segment_id": "S002"},
            {"criterion_id": "C03", "source_segment_id": "S003"},
        ]
    )
    responses = iter([explanations, evidence_sources])
    monkeypatch.setattr(
        gemini_agents,
        "_json_call",
        lambda *_args, **_kwargs: next(responses),
    )

    enriched = gemini_agents.enrich_criteria_weight_explanations(
        "Strong SQL skills are required.\n"
        "Advanced Excel is needed for reporting.\n"
        "Communicate findings to stakeholders.",
        criteria,
    )

    assert [item.weight for item in enriched.criteria] == [50, 30, 20]
    assert all(item.weight_reason for item in enriched.criteria)
    assert all(item.source_evidence for item in enriched.criteria)
    assert [item.source_evidence for item in enriched.criteria] == [
        "Strong SQL skills are required.",
        "Advanced Excel is needed for reporting.",
        "Communicate findings to stakeholders.",
    ]
    assert all(not item.weight_reason for item in criteria.criteria)


def test_ai_weight_redistribution_keeps_recruiter_selected_weight(monkeypatch):
    criteria = JobCriteria(
        job_title="Data Analyst",
        role_summary="Analyze business data and communicate findings to stakeholders.",
        criteria=[
            DynamicCriterion(
                criterion_id="C01",
                name="SQL",
                definition="Ability to query datasets using SQL for analysis.",
                evidence_to_look_for=["SQL projects"],
                weight=50,
            ),
            DynamicCriterion(
                criterion_id="C02",
                name="Excel",
                definition="Ability to use Excel for data preparation and analysis.",
                evidence_to_look_for=["Excel reports"],
                weight=30,
            ),
            DynamicCriterion(
                criterion_id="C03",
                name="Communication",
                definition="Ability to communicate analytical findings clearly.",
                evidence_to_look_for=["Presentations"],
                weight=20,
            ),
        ],
        must_have_requirements=["SQL"],
        scoring_exclusions=["name", "email", "university"],
        recommendation_rules={"strong_fit_min": 80, "potential_fit_min": 65},
    )
    allocation = CriteriaWeightAllocation(
        allocations=[
            CriterionWeight(criterion_id="C01", weight=40),
            CriterionWeight(criterion_id="C02", weight=35),
            CriterionWeight(criterion_id="C03", weight=25),
        ]
    )
    monkeypatch.setattr(gemini_agents, "_json_call", lambda *_args, **_kwargs: allocation)

    redistributed = gemini_agents.redistribute_criterion_weights(
        "SQL, Excel, and communication are required.",
        criteria,
        {"C01": 40},
        {"C01": 40, "C02": 35, "C03": 25},
        "Excel work is more critical for this role.",
    )

    assert [item.weight for item in redistributed.criteria] == [40, 35, 25]
    assert [item.weight for item in criteria.criteria] == [50, 30, 20]

    allocation_with_changed_fixed_weight = CriteriaWeightAllocation(
        allocations=[
            CriterionWeight(criterion_id="C01", weight=45),
            CriterionWeight(criterion_id="C02", weight=30),
            CriterionWeight(criterion_id="C03", weight=25),
        ]
    )
    monkeypatch.setattr(
        gemini_agents,
        "_json_call",
        lambda *_args, **_kwargs: allocation_with_changed_fixed_weight,
    )
    redistributed = gemini_agents.redistribute_criterion_weights(
        "SQL, Excel, and communication are required.",
        criteria,
        {"C01": 40},
        {"C01": 40, "C02": 35, "C03": 25},
        "Excel work is more critical for this role.",
    )

    assert redistributed.criteria[0].weight == 40
    assert redistributed.criteria[1].weight == pytest.approx(32.7272727)
    assert redistributed.criteria[2].weight == pytest.approx(27.2727273)
    assert sum(item.weight for item in redistributed.criteria) == pytest.approx(100)


def test_ai_weight_redistribution_preserves_all_manually_edited_weights(monkeypatch):
    criteria = JobCriteria(
        job_title="Data Analyst",
        role_summary="Analyze business data and communicate findings to stakeholders.",
        criteria=[
            DynamicCriterion(
                criterion_id="C01",
                name="SQL",
                definition="Ability to query datasets using SQL for analysis.",
                evidence_to_look_for=["SQL projects"],
                weight=50,
            ),
            DynamicCriterion(
                criterion_id="C02",
                name="Excel",
                definition="Ability to use Excel for data preparation and analysis.",
                evidence_to_look_for=["Excel reports"],
                weight=30,
            ),
            DynamicCriterion(
                criterion_id="C03",
                name="Communication",
                definition="Ability to communicate analytical findings clearly.",
                evidence_to_look_for=["Presentations"],
                weight=20,
            ),
        ],
        must_have_requirements=["SQL"],
        scoring_exclusions=["name", "email", "university"],
        recommendation_rules={"strong_fit_min": 80, "potential_fit_min": 65},
    )
    allocation = CriteriaWeightAllocation(
        allocations=[
            CriterionWeight(criterion_id="C01", weight=45),
            CriterionWeight(criterion_id="C02", weight=35),
            CriterionWeight(criterion_id="C03", weight=20),
        ]
    )
    monkeypatch.setattr(gemini_agents, "_json_call", lambda *_args, **_kwargs: allocation)

    redistributed = gemini_agents.redistribute_criterion_weights(
        "SQL, Excel, and communication are required.",
        criteria,
        {"C01": 40, "C02": 30},
        {"C01": 40, "C02": 30, "C03": 30},
        "SQL and Excel need these revised percentages.",
    )

    assert [item.weight for item in redistributed.criteria] == [40, 30, 30]
