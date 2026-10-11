from __future__ import annotations

import hashlib
import html
import importlib
import io
import json

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode

from backend.config import GEMINI_MODEL, GEMINI_SCREEN_MODEL
from backend.emailer import send_email
from backend import gemini_agents, models, storage, workflow
from backend.jd_parser import extract_jd
from backend.models import JobCriteria
from backend.storage import (
    assessments_df,
    candidate_df,
    downstream_for,
    ensure_seed_candidates,
    get_job,
    import_candidates,
    init_db,
    list_jobs,
    mark_email,
    results_df,
    save_downstream,
    set_status,
)


st.set_page_config(page_title="Thara Energy | AI Recruiting Workbench", page_icon="⚡", layout="wide")
init_db()
ensure_seed_candidates()

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "criteria" not in st.session_state:
    st.session_state.criteria = None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = None
if "jd_name" not in st.session_state:
    st.session_state.jd_name = None
if "jd_file_signature" not in st.session_state:
    st.session_state.jd_file_signature = None
if "candidate_file_signature" not in st.session_state:
    st.session_state.candidate_file_signature = None
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None
if "ratio_dialog_criterion" not in st.session_state:
    st.session_state.ratio_dialog_criterion = None
if "ratio_dialog_version" not in st.session_state:
    st.session_state.ratio_dialog_version = 0
if "history_dialog_open" not in st.session_state:
    st.session_state.history_dialog_open = False
if "pending_weight_edit" not in st.session_state:
    st.session_state.pending_weight_edit = None
if "pending_candidate_upload" not in st.session_state:
    st.session_state.pending_candidate_upload = None
if "candidate_upload_widget_version" not in st.session_state:
    st.session_state.candidate_upload_widget_version = 0
if "candidate_overview_grid_version" not in st.session_state:
    st.session_state.candidate_overview_grid_version = 0


def api_ready() -> bool:
    return bool(gemini_agents.get_api_key())


def get_criteria_weight_enricher():
    reload_criteria_agent_modules()
    enricher = getattr(gemini_agents, "enrich_criteria_weight_explanations", None)
    if not callable(enricher):
        raise ImportError(
            "The criteria explanation agent is missing from backend.gemini_agents. "
            "Update the app files and restart Streamlit."
        )
    return enricher


def reload_criteria_agent_modules() -> None:
    importlib.reload(models)
    importlib.reload(gemini_agents)
    global JobCriteria
    JobCriteria = models.JobCriteria


def persist_job_criteria(job_id: str, criteria_data: dict) -> None:
    updater = getattr(storage, "update_job_criteria", None)
    if not callable(updater):
        importlib.reload(storage)
        updater = getattr(storage, "update_job_criteria", None)
    if not callable(updater):
        raise ImportError(
            "The job-criteria storage function is missing from backend.storage. "
            "Update the app files and restart Streamlit."
        )
    updater(job_id, criteria_data)


def persist_criteria_weight_change(
    job_id: str,
    criteria_data: dict,
    reason: str,
    source: str,
) -> None:
    saver = getattr(storage, "save_job_criteria_weight_change", None)
    if not callable(saver):
        importlib.reload(storage)
        saver = getattr(storage, "save_job_criteria_weight_change", None)
    if not callable(saver):
        raise ImportError(
            "The criterion-weight storage function is missing from backend.storage. "
            "Update the app files and restart Streamlit."
        )
    saver(job_id, criteria_data, reason, source)


def load_criteria_weight_history(job_id: str) -> list[dict]:
    reader = getattr(storage, "criteria_weight_history", None)
    if not callable(reader):
        importlib.reload(storage)
        reader = getattr(storage, "criteria_weight_history", None)
    if not callable(reader):
        raise ImportError(
            "The criterion-weight history function is missing from backend.storage. "
            "Update the app files and restart Streamlit."
        )
    return reader(job_id)


def load_existing_job(job_id: str):
    row = get_job(job_id)
    if not row:
        return
    st.session_state.job_id = job_id
    st.session_state.jd_text = row["jd_text"]
    st.session_state.jd_name = row["source_name"]
    st.session_state.criteria = JobCriteria.model_validate(json.loads(row["criteria_json"]))
    st.session_state.ratio_dialog_criterion = None
    st.session_state.history_dialog_open = False
    st.session_state.pending_weight_edit = None


def status_card(label: str, value: int, detail: str = "", height: int = 142):
    extra = (
        f'<div style="font-size:0.88rem;color:#6b7280;margin-top:12px;min-height:1.2rem">{detail}</div>'
        if detail
        else '<div style="min-height:1.2rem;margin-top:12px"></div>'
    )
    st.markdown(
        f"""
        <div style="box-sizing:border-box;background:#ffffff;border:1px solid #d1d5db;border-radius:4px;padding:16px;height:{height}px;display:flex;flex-direction:column">
          <div style="font-size:1rem;font-weight:500;color:#1f2937">{label}</div>
          <hr style="border:0;border-top:1px solid #e5e7eb;margin:14px 0">
          <div style="font-size:2rem;font-weight:600;color:#111827;line-height:1.1">{value}</div>
          {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Change criterion ratios", dismissible=False)
def change_criteria_ratios_dialog() -> None:
    criteria = st.session_state.criteria
    job_id = st.session_state.job_id
    if criteria is None or not job_id:
        st.error("Load or generate an assessment framework before changing ratios.")
        return

    st.caption(
        "Edit one or more percentages. A reason is required, and the final total must be 100%. "
        "AI redistribution keeps the percentages you edited unchanged."
    )
    dialog_version = st.session_state.ratio_dialog_version
    with st.form(f"change_criteria_ratios_{job_id}_{dialog_version}"):
        proposed_weights = {
            item.criterion_id: st.number_input(
                f"{item.name} percentage",
                min_value=0.01,
                max_value=99.99,
                value=float(item.weight),
                step=1.0,
                format="%.2f",
                key=f"ratio_{job_id}_{dialog_version}_{item.criterion_id}",
            )
            for item in criteria.criteria
        }
        reason = st.text_area(
            "Reason for changing the ratio",
            placeholder="Explain why this criterion's importance should change.",
            key=f"ratio_reason_{job_id}_{dialog_version}",
        )
        submitted = st.form_submit_button("Submit ratio changes", type="primary")

    if submitted:
        if not reason.strip():
            st.error("A reason is required before ratios can be changed.")
        elif all(
            abs(proposed_weights[item.criterion_id] - item.weight) <= 0.000001
            for item in criteria.criteria
        ):
            st.error("Change at least one criterion percentage.")
        else:
            total = sum(proposed_weights.values())
            fixed_weights = {
                item.criterion_id: proposed_weights[item.criterion_id]
                for item in criteria.criteria
                if abs(proposed_weights[item.criterion_id] - item.weight) > 0.000001
            }
            pending = {
                "job_id": job_id,
                "fixed_weights": fixed_weights,
                "weights": proposed_weights,
                "reason": reason.strip(),
            }
            if abs(total - 100.0) > 0.01:
                st.session_state.pending_weight_edit = pending
            else:
                try:
                    updated = criteria.model_copy(
                        update={
                            "criteria": [
                                item.model_copy(
                                    update={"weight": proposed_weights[item.criterion_id]}
                                )
                                for item in criteria.criteria
                            ]
                        }
                    )
                    updated = JobCriteria.model_validate(updated.model_dump())
                    persist_criteria_weight_change(
                        job_id,
                        updated.model_dump(),
                        reason,
                        "manual",
                    )
                    st.session_state.criteria = updated
                    st.session_state.pending_weight_edit = None
                    st.session_state.ratio_dialog_criterion = None
                    st.session_state.history_dialog_open = False
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save the criterion ratios: {exc}")

    pending = st.session_state.pending_weight_edit
    if pending and pending["job_id"] == job_id:
        total = sum(pending["weights"].values())
        st.warning(f"The proposed criteria total is {total:g}%, not 100%.")
        redistribute_col, reedit_col = st.columns(2)
        with redistribute_col:
            if st.button(
                "Allow AI to redistribute",
                type="primary",
                use_container_width=True,
                disabled=not api_ready(),
                key=f"redistribute_ratios_{job_id}",
            ):
                try:
                    job = get_job(job_id)
                    if job is None:
                        raise ValueError(f"Job {job_id} could not be found.")
                    with st.spinner("AI is rebalancing the other criteria..."):
                        importlib.reload(gemini_agents)
                        redistributed = gemini_agents.redistribute_criterion_weights(
                            jd_text=job["jd_text"],
                            criteria=criteria,
                            fixed_weights={
                                key: float(value)
                                for key, value in pending["fixed_weights"].items()
                            },
                            proposed_weights={
                                key: float(value)
                                for key, value in pending["weights"].items()
                            },
                            change_reason=str(pending["reason"]),
                        )
                        persist_criteria_weight_change(
                            job_id,
                            redistributed.model_dump(),
                            str(pending["reason"]),
                            "ai_redistribution",
                        )
                    st.session_state.criteria = redistributed
                    st.session_state.pending_weight_edit = None
                    st.session_state.ratio_dialog_criterion = None
                    st.session_state.history_dialog_open = False
                    st.rerun()
                except Exception as exc:
                    st.error(f"AI could not redistribute the criteria: {exc}")
        with reedit_col:
            if st.button(
                "Go back and re-edit",
                use_container_width=True,
                key=f"reedit_ratios_{job_id}",
            ):
                st.session_state.pending_weight_edit = None
                st.rerun()
        if not api_ready():
            st.caption("Configure the Gemini API to use AI redistribution.")

    if st.button("Close", key=f"close_ratio_changes_{job_id}"):
        st.session_state.pending_weight_edit = None
        st.session_state.ratio_dialog_criterion = None
        st.session_state.history_dialog_open = False
        st.session_state.ratio_dialog_version += 1
        st.rerun()


@st.dialog("Criteria ratio change history", dismissible=False)
def criteria_ratio_history_dialog() -> None:
    job_id = st.session_state.job_id
    if not job_id:
        st.error("Load an assessment framework to view its ratio history.")
        return
    try:
        history = load_criteria_weight_history(job_id)
    except Exception as exc:
        st.error(f"Could not load criteria ratio history: {exc}")
        return
    if not history:
        st.info("No criteria ratio history is available yet.")
    for entry in history:
        st.markdown(f"**{entry['criterion_name']}**")
        if entry["old_weight"] is None:
            st.write(f"AI-generated starting ratio: {entry['new_weight']:g}%")
        else:
            st.write(f"{entry['old_weight']:g}% → {entry['new_weight']:g}%")
        source = {
            "ai_generated": "Generated by Agent 1",
            "manual": "Changed by recruiter",
            "ai_redistribution": "Rebalanced by AI",
        }.get(entry["source"], entry["source"])
        st.caption(f"{source} · {entry['changed_at']}")
        st.write(entry["reason"])
        if entry is not history[-1]:
            st.divider()
    if st.button("Close", key=f"close_ratio_history_{job_id}"):
        st.session_state.ratio_dialog_criterion = None
        st.session_state.history_dialog_open = False
        st.rerun()


@st.dialog("Replace candidate database?")
def replace_candidate_database_dialog() -> None:
    pending = st.session_state.pending_candidate_upload
    if pending is None:
        return
    st.warning("There is already a candidate database. Do you want to upload and replace it with this new database?")
    st.caption(f"{pending['name']} · {len(pending['data'])} candidate records")
    yes_col, no_col = st.columns(2)
    with yes_col:
        if st.button("Yes", type="primary", use_container_width=True):
            try:
                from backend.storage import clear_job_outputs

                imported = import_candidates(pending["data"], replace=True)
                clear_job_outputs(st.session_state.job_id)
                st.session_state.candidate_file_signature = pending["signature"]
                st.session_state.pending_candidate_upload = None
                st.success(f"Candidate database replaced: {imported} records loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Candidate upload failed: {exc}")
    with no_col:
        if st.button("No", use_container_width=True):
            st.session_state.pending_candidate_upload = None
            st.session_state.candidate_upload_widget_version += 1
            st.info("Upload canceled. The existing candidate database was kept.")
            st.rerun()


# Sidebar
st.sidebar.title("⚡ Thara Energy")
st.sidebar.caption("AI Agentic Recruiting Workbench")
if api_ready():
    st.sidebar.success("Gemini API: Connected")
else:
    st.sidebar.warning("Gemini API: Not configured")
st.sidebar.caption(f"Criteria model: `{GEMINI_MODEL}`")
st.sidebar.caption(f"Screening model: `{GEMINI_SCREEN_MODEL}`")

jobs = list_jobs()
if not jobs.empty:
    job_options = ["— Start new job —"] + jobs["job_id"].tolist()
    selected_job = st.sidebar.selectbox("Previous job runs", job_options)
    if selected_job != "— Start new job —" and selected_job != st.session_state.job_id:
        load_existing_job(selected_job)
        st.rerun()

st.sidebar.divider()
st.sidebar.markdown("**Agentic flow**")
st.sidebar.markdown("① JD → Criteria & Adjustable Weights  \n② Candidate → Evidence-Based Scoring  \n③ Human Review & Approval  \n④ Interview Questions + Outreach Email Draft")

st.title("AI Agentic Recruiting Workbench")
st.caption("Reusable across job openings — assessment criteria are generated dynamically from each Job Description.")

# ============================================================================
# 1. JOB INPUT
# ============================================================================
st.header("1. Job Input")
st.write(
    "The Job Input section only defines the job and its AI-generated assessment framework. "
    "Candidate database is managed in Section 2: Candidate Funnel."
)

input_left, _ = st.columns([1.2, 1])
with input_left:
    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["txt", "md", "pdf"],
        help="Agent 1 reads this and creates the criteria + weights.",
    )
    if jd_file is not None:
        try:
            raw = jd_file.getvalue()
            signature = hashlib.sha256(raw).hexdigest()
            if signature != st.session_state.jd_file_signature:
                st.session_state.jd_text = extract_jd(jd_file)
                st.session_state.jd_name = jd_file.name
                st.session_state.jd_file_signature = signature
                st.success(f"Loaded: {jd_file.name}")
        except Exception as exc:
            st.error(str(exc))

if st.session_state.jd_text:
    with st.expander(f"📄 View Job Description — {st.session_state.jd_name}"):
        st.text(st.session_state.jd_text)

run_criteria = st.button(
    "🤖 Agent 1 — Analyze JD & Generate Criteria",
    type="primary",
    disabled=not bool(st.session_state.jd_text) or not api_ready(),
    use_container_width=True,
)
if run_criteria:
    with st.spinner("Agent 1 is analyzing the JD and designing a job-specific assessment framework..."):
        try:
            reload_criteria_agent_modules()
            job_id, criteria = workflow.create_job_from_jd(
                st.session_state.jd_text,
                st.session_state.jd_name or "uploaded_jd",
            )
            st.session_state.job_id = job_id
            st.session_state.criteria = criteria
            st.success(f"Criteria generated for **{criteria.job_title}** — {len(criteria.criteria)} dynamic criteria, weights total 100%.")
        except Exception as exc:
            st.error(f"Agent 1 failed: {exc}")

# The framework appears ONLY after Agent 1 has generated it.
criteria = st.session_state.criteria
if criteria:
    if any(
        not getattr(criterion, "weight_reason", "").strip()
        or not getattr(criterion, "source_evidence", "").strip()
        for criterion in criteria.criteria
    ):
        if api_ready() and st.session_state.jd_text:
            try:
                with st.spinner("Generating weight explanations from the original job description..."):
                    criteria = get_criteria_weight_enricher()(st.session_state.jd_text, criteria)
                    st.session_state.criteria = criteria
                    if st.session_state.job_id:
                        persist_job_criteria(st.session_state.job_id, criteria.model_dump())
            except Exception as exc:
                st.error(f"Could not generate the missing criteria explanations: {exc}")
        else:
            st.warning(
                "This saved framework is missing weight explanations. Configure the Gemini API "
                "and reload this job to generate them from its original job description."
            )

    st.markdown("### AI-Generated Assessment Framework")
    st.info(f"**{criteria.job_title}** — {criteria.role_summary}")

    st.markdown(
        """
        <style>
        .st-key-assessment-criteria {
            background-color: #f3f4f6;
            border-radius: 8px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="assessment-criteria"):
        criterion_cols = st.columns(min(4, len(criteria.criteria)))
        for i, criterion in enumerate(criteria.criteria):
            with criterion_cols[i % len(criterion_cols)]:
                st.metric(criterion.name, f"{criterion.weight:g}%")
                st.caption("Must-have" if criterion.is_must_have else "Weighted criterion")
                with st.expander("Definition / evidence"):
                    weight_reason = getattr(criterion, "weight_reason", "")
                    source_evidence = getattr(criterion, "source_evidence", "")
                    st.write(f"**Reason:** {weight_reason or 'Not available for this saved framework.'}")
                    st.write(f"**Raw Data (Job Description):** {source_evidence or 'Not available for this saved framework.'}")
                    st.write("**Looking for:** " + ", ".join(criterion.evidence_to_look_for))
        _, controls_column = st.columns([4, 1])
        with controls_column:
            if st.button(
                "Change ratio",
                key=f"change_ratio_{st.session_state.job_id}",
                use_container_width=True,
            ):
                st.session_state.pending_weight_edit = None
                st.session_state.history_dialog_open = False
                st.session_state.ratio_dialog_version += 1
                st.session_state.ratio_dialog_criterion = criteria.criteria[0].criterion_id
            if st.button(
                "View change history",
                key=f"criteria_history_{st.session_state.job_id}",
                use_container_width=True,
            ):
                st.session_state.ratio_dialog_criterion = None
                st.session_state.history_dialog_open = True

    if st.session_state.ratio_dialog_criterion:
        change_criteria_ratios_dialog()
    elif st.session_state.history_dialog_open:
        criteria_ratio_history_dialog()

    st.markdown("**Must-have requirements**")
    for x in criteria.must_have_requirements:
        st.write("✓ " + x)
    if criteria.nice_to_have_requirements:
        st.markdown("**Nice-to-have requirements**")
        for x in criteria.nice_to_have_requirements:
            st.write("· " + x)
    st.info("🛡️ **Scoring exclusions**\n\n" + "\n".join("• " + x for x in criteria.scoring_exclusions))

# ============================================================================
# 2. CANDIDATE FUNNEL
# ============================================================================
st.divider()
st.header("2. Candidate Funnel")
st.caption("This section is the single operational workspace: database → AI recommendation → case status → recruiter review → interview/outreach.")

# Candidate database portal moved here from Job Input.
with st.expander("🗄️ Candidate Database", expanded=True):
    current_candidates = candidate_df()
    current_candidate_count = len(current_candidates)
    db_left, db_right = st.columns([1, 2])
    with db_left:
        st.metric("Current candidate records", current_candidate_count)
    with db_right:
        candidate_file = st.file_uploader(
            "Upload candidate CSV",
            type=["csv"],
            help=(
                "Required columns: candidate_id, full_name, email, degree, university, "
                "gpa, skills_summary, experience_summary."
            ),
            key=f"candidate_csv_upload_{st.session_state.candidate_upload_widget_version}",
        )
        if candidate_file is not None:
            try:
                raw = candidate_file.getvalue()
                signature = hashlib.sha256(raw).hexdigest()
                if (
                    signature != st.session_state.candidate_file_signature
                    and st.session_state.pending_candidate_upload is None
                ):
                    uploaded_df = pd.read_csv(io.BytesIO(raw))
                    missing = [
                        column
                        for column in [
                            "candidate_id",
                            "full_name",
                            "email",
                            "degree",
                            "university",
                            "gpa",
                            "skills_summary",
                            "experience_summary",
                        ]
                        if column not in uploaded_df.columns
                    ]
                    if missing:
                        st.error("Missing columns: " + ", ".join(missing))
                    elif uploaded_df["candidate_id"].duplicated().any():
                        st.error("Candidate upload failed: candidate_id values must be unique.")
                    elif current_candidate_count:
                        st.session_state.pending_candidate_upload = {
                            "data": uploaded_df,
                            "name": candidate_file.name,
                            "signature": signature,
                        }
                    else:
                        try:
                            imported = import_candidates(uploaded_df, replace=True)
                            st.session_state.candidate_file_signature = signature
                            st.session_state.candidate_upload_widget_version += 1
                            st.success(f"Candidate database uploaded: {imported} records loaded.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Candidate upload failed: {exc}")
            except Exception as exc:
                st.error(f"Candidate upload failed: {exc}")

    if current_candidate_count:
        st.markdown("**Current database preview — first 5 rows**")
        preview_columns = [
            "candidate_id",
            "full_name",
            "email",
            "degree",
            "university",
            "gpa",
            "skills_summary",
            "experience_summary",
        ]
        st.dataframe(
            current_candidates[[column for column in preview_columns if column in current_candidates.columns]].head(5),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No candidate database is currently loaded.")

if st.session_state.pending_candidate_upload is not None:
    replace_candidate_database_dialog()

current_candidate_count = len(current_candidates)

if not st.session_state.job_id:
    st.warning("Upload a JD and run Agent 1 first. The candidate database can already be inspected above, but screening requires a job-specific framework.")
    st.stop()

results = results_df(st.session_state.job_id)

retry_candidate_ids = (
    set(results.loc[results["case_status"] == "screen error", "candidate_id"].astype(str))
    if not results.empty
    else set()
)

if criteria and current_candidate_count > 0 and (results.empty or retry_candidate_ids):
    if results.empty:
        screen_target_ids = None
        screen_count = current_candidate_count
        button_label = f"🚀 Agent 2 — Screen {screen_count} Candidates & Build Shortlist"
        st.success(f"Ready to screen **{screen_count} candidates** using the current AI-generated framework.")
    else:
        screen_target_ids = retry_candidate_ids
        screen_count = len(screen_target_ids)
        button_label = f"🔁 Retry Agent 2 for {screen_count} failed candidate(s)"
        st.warning(f"{screen_count} candidate(s) still need a completed AI assessment.")
    run_screen = st.button(
        button_label,
        type="primary",
        use_container_width=True,
        disabled=not api_ready(),
    )
    if run_screen:
        progress = st.progress(0, text="Starting candidate screening...")
        try:
            importlib.reload(storage)
            screening_workflow = importlib.reload(workflow)

            def cb(i, total, cid, errors):
                suffix = f" | {errors} errors" if errors else ""
                progress.progress(i / total, text=f"Agent 2: processed {i}/{total} — {cid}{suffix}")
            summary = screening_workflow.run_job_screening(
                st.session_state.job_id,
                criteria,
                cb,
                candidate_ids=screen_target_ids,
            )
            progress.progress(1.0, text="Screening complete")
            if summary["errors"]:
                st.warning(f"Screening completed with {summary['completed']} scored and {summary['errors']} screen-error candidate(s).")
            else:
                st.success("Candidate assessment completed. Weighted scores were calculated deterministically by Python from the AI criterion scores.")
            st.rerun()
        except Exception as exc:
            st.error(f"Candidate screening failed: {exc}")

results = results_df(st.session_state.job_id)
if results.empty:
    st.info("No candidate screening results yet. Run Agent 2 after Agent 1 creates the assessment framework.")
    st.stop()

screen_errors = results[results["case_status"] == "screen error"]
if not screen_errors.empty:
    error_items = []
    for _, error_row in screen_errors.iterrows():
        raw_detail = error_row.get("screening_error")
        detail = "" if pd.isna(raw_detail) else str(raw_detail).strip()
        detail = detail or "No error details were recorded."
        if len(detail) > 240:
            detail = detail[:237].rstrip() + "..."
        raw_name = error_row.get("full_name")
        candidate_name = "" if pd.isna(raw_name) else str(raw_name).strip()
        candidate_label = (
            f"{error_row['candidate_id']} — {candidate_name}"
            if candidate_name
            else str(error_row["candidate_id"])
        )
        error_items.append(
            f"<li><strong>{html.escape(candidate_label)}:</strong> {html.escape(detail)}</li>"
        )
    st.markdown(
        f"""
        <div style="background:#FDECEC;border:1px solid #F4B4B4;border-radius:8px;padding:14px 18px;color:#8F1D24;margin-bottom:1rem">
          <div style="font-weight:700;margin-bottom:6px">Screening errors — {len(screen_errors)} candidate(s) could not be scored</div>
          <div style="margin-bottom:6px">Review the brief cause below. These candidates have no completed AI assessment.</div>
          <ul style="margin:0;padding-left:1.25rem">{"".join(error_items)}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Dashboard cards: explicitly separate AI Recommendation from case_status.
# --------------------------------------------------------------------------
st.markdown("### Funnel Dashboard")
strong_fit_count = int((results["ai_recommendation"] == "Strong Fit").sum())
potential_fit_count = int((results["ai_recommendation"] == "Potential Fit").sum())

st.markdown("#### AI recommendation")
recommendation_cols = st.columns(2)
with recommendation_cols[0]:
    status_card("Strong Fit", strong_fit_count)
with recommendation_cols[1]:
    status_card("Potential Fit", potential_fit_count)

st.markdown("#### Case status")
case_cols = st.columns(4)
with case_cols[0]:
    status_card("Approved", int((results["case_status"] == "approved").sum()))
with case_cols[1]:
    status_card("Rejected", int((results["case_status"] == "rejected").sum()))
with case_cols[2]:
    status_card("Review", int((results["case_status"] == "review").sum()))
with case_cols[3]:
    status_card("Screen Error", int((results["case_status"] == "screen error").sum()))

email_sent = int((results["email_status"] == "sent").sum())
email_pending = int((results["case_status"] == "approved").sum() - email_sent)
st.markdown("#### Email Status")
outreach_cols = st.columns(4)
with outreach_cols[0]:
    status_card("Email Sent", email_sent)
with outreach_cols[1]:
    status_card("Email Not Sent", max(0, email_pending), "Of approved candidates")

st.markdown("### Candidate Overview")
st.caption("Use the dropdowns to filter by status, search for text contained in any candidate field, or choose visible columns. Click a Candidate ID to open its Human-in-the-Loop review popup.")

display = results[[
    "candidate_id", "full_name", "degree", "overall_score", "ai_recommendation", "case_status",
    "recruiter_note", "interview_question", "outreach_email", "email_status",
]].copy()
display["overall_score"] = display.apply(
    lambda r: None if r["case_status"] == "screen error" else round(float(r["overall_score"]), 1), axis=1
)
display["email_status"] = display["email_status"].replace(
    {
        "sent": "Sent",
        "still not send": "Still not sent",
        "failed": "Failed",
    }
)
for column in ("interview_question", "outreach_email", "recruiter_note"):
    display[column] = display[column].fillna("").replace("", "—")

table_df = display.rename(columns={
    "candidate_id": "Candidate ID",
    "full_name": "Name",
    "degree": "Degree",
    "overall_score": "AI Score",
    "ai_recommendation": "AI Recommendation",
    "case_status": "Case Status",
    "recruiter_note": "Recruiter Note",
    "interview_question": "Interview Question",
    "outreach_email": "Outreach Email",
    "email_status": "Email Status",
})

with st.expander("Choose visible columns"):
    column_picker = st.columns(5)
    visible_columns = []
    for index, column in enumerate(table_df.columns):
        if column_picker[index % len(column_picker)].checkbox(
            column,
            value=True,
            key=f"candidate_column_visible_{column}",
        ):
            visible_columns.append(column)

filter_columns = st.columns(3)
email_status_options = sorted(results["email_status"].dropna().astype(str).unique())
case_status_options = sorted(table_df["Case Status"].dropna().astype(str).unique())
recommendation_options = sorted(table_df["AI Recommendation"].dropna().astype(str).unique())
with filter_columns[0]:
    email_status_filter = st.multiselect(
        "Email Status",
        email_status_options,
        default=email_status_options,
        format_func=lambda value: {
            "sent": "Sent",
            "still not send": "Still not sent",
            "failed": "Failed",
        }.get(value, value.title()),
        key="candidate_email_status_filter",
    )
with filter_columns[1]:
    case_status_filter = st.multiselect(
        "Case Status",
        case_status_options,
        default=case_status_options,
        format_func=lambda value: value.title(),
        key="candidate_case_status_filter",
    )
with filter_columns[2]:
    recommendation_filter = st.multiselect(
        "AI Recommendation",
        recommendation_options,
        default=recommendation_options,
        key="candidate_recommendation_filter",
    )

search_query = st.text_input(
    "Search candidates",
    placeholder="Type text contained in any candidate field",
    key="candidate_table_search",
)

if not visible_columns:
    st.info("Select at least one column to display the candidate table.")
else:
    row_filter = (
        results["email_status"].isin(email_status_filter)
        & table_df["Case Status"].isin(case_status_filter)
        & table_df["AI Recommendation"].isin(recommendation_filter)
    )
    if search_query:
        search_matches = table_df.astype("string").apply(
            lambda column: column.str.contains(
                search_query,
                case=False,
                regex=False,
                na=False,
            )
        ).any(axis=1)
        row_filter &= search_matches
    grid_df = table_df.loc[row_filter, visible_columns].copy()

    grid_builder = GridOptionsBuilder.from_dataframe(grid_df)
    grid_builder.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        minWidth=110,
        wrapText=False,
        autoHeight=False,
    )
    if "AI Score" in visible_columns:
        grid_builder.configure_column(
            "AI Score",
            type=["numericColumn"],
            filter="agNumberColumnFilter",
        )
    for column in ("Email Status", "Case Status", "AI Recommendation"):
        if column in visible_columns:
            grid_builder.configure_column(column, filter=False)
    if "Candidate ID" in visible_columns:
        grid_builder.configure_column(
            "Candidate ID",
            pinned="left",
            minWidth=150,
            cellStyle={
                "backgroundColor": "#EFF6FF",
                "color": "#1D4ED8",
                "fontWeight": "600",
                "cursor": "pointer",
                "border": "1px solid #BFDBFE",
                "borderRadius": "5px",
            },
        )
    grid_builder.configure_selection(selection_mode="single", use_checkbox=False)
    grid_builder.configure_grid_options(
        rowHeight=34,
        headerHeight=38,
        suppressRowClickSelection=False,
    )
    grid_response = AgGrid(
        grid_df,
        gridOptions=grid_builder.build(),
        height=430,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="streamlit",
        custom_css={
            ".ag-header": {"background-color": "#1D4ED8", "color": "#FFFFFF"},
            ".ag-header-cell": {"border-right": "1px solid #93C5FD !important"},
            ".ag-cell": {"border-right": "1px solid #E2E8F0 !important"},
            ".ag-row": {"border-bottom": "1px solid #E2E8F0 !important"},
        },
        key=f"candidate_overview_grid_{st.session_state.candidate_overview_grid_version}",
    )

    selected_rows = grid_response.get("selected_rows")
    if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        selected_candidate = selected_rows.iloc[0].get("Candidate ID")
        if selected_candidate is not None:
            st.session_state.selected_candidate = str(selected_candidate)
    elif isinstance(selected_rows, list) and selected_rows:
        selected_candidate = selected_rows[0].get("Candidate ID")
        if selected_candidate is not None:
            st.session_state.selected_candidate = str(selected_candidate)


# ============================================================================
# HUMAN-IN-THE-LOOP POPUP
# ============================================================================
def dismiss_candidate_dialog() -> None:
    st.session_state.selected_candidate = None
    st.session_state.candidate_overview_grid_version += 1


def open_candidate_dialog(candidate_id: str):
    @st.dialog(
        f"Candidate Review — {candidate_id}",
        width="large",
        on_dismiss=dismiss_candidate_dialog,
    )
    def _dialog():
        data = results_df(st.session_state.job_id)
        match = data[data["candidate_id"].astype(str) == str(candidate_id)]
        if match.empty:
            st.error("Candidate record is no longer available.")
            return

        row = match.iloc[0]
        st.caption(f"{row['full_name']} · {row['email']}")

        info_cols = st.columns(4)
        info_cols[0].metric("AI Score", "—" if row["case_status"] == "screen error" else f"{float(row['overall_score']):.1f}/100")
        info_cols[1].metric("AI Recommendation", row["ai_recommendation"])
        info_cols[2].metric("case_status", row["case_status"])
        info_cols[3].metric("Email", row["email_status"])

        p1, p2 = st.columns(2)
        with p1:
            st.write("**Degree:**", row["degree"])
            st.write("**GPA:**", row["gpa"])
            st.write("**University:**", row["university"])
        with p2:
            st.write("**Skills:**", row["skills_summary"])
            st.write("**Experience:**", row["experience_summary"])

        assessment = assessments_df(st.session_state.job_id, candidate_id)
        st.markdown("#### Dynamic criterion assessment")
        if assessment.empty:
            st.warning("No criterion assessment is available for this candidate.")
        else:
            for _, a in assessment.iterrows():
                with st.expander(f"{a['criterion_name']} — {int(a['score'])}/100 | Weight {a['weight']:g}%"):
                    st.write(f"**Reason:** {a['score_reason']}")
                    evidence = str(a.get("evidence", "") or "").strip()
                    if not evidence and int(a["score"]) == 0:
                        evidence = "No verifiable raw-data evidence; criterion score set to 0."
                    elif not evidence:
                        evidence = "Raw data was not captured for this screening."
                    st.write(
                        "**Evidence:** "
                        + evidence
                    )

        exp1, exp2 = st.columns(2)
        with exp1:
            st.success("**Strengths**\n\n" + str(row["strengths"] or "No strengths recorded."))
        with exp2:
            st.warning("**Gaps / validation points**\n\n" + str(row["gaps"] or "No gaps recorded."))
        st.info("🛡️ **Responsible AI Fair-Scoring Audit**\n\n" + str(row["bias_audit_notes"]))
        if row["case_status"] == "screen error":
            st.error("**Screening Error:** " + str(row["screening_error"] or "Unknown screening error"))

        hitl_notes_default = str(row["recruiter_note"] or "")
        with st.form(f"hitl_form_{candidate_id}"):
            st.markdown("#### Human decision")
            allowed = ["Reject", "Review"] if row["case_status"] == "screen error" else ["Approve", "Reject", "Review"]
            previous_decision = str(row["human_decision"] or "")
            default_index = allowed.index(previous_decision) if previous_decision in allowed else 0
            decision = st.radio("Next action", allowed, index=default_index, horizontal=True)
            notes = st.text_area("Recruiter note", value=hitl_notes_default, height=110)
            submit_decision = st.form_submit_button("Save Decision", type="primary", use_container_width=True)

        def render_edit_history() -> None:
            history_key = f"edit_history_open_{st.session_state.job_id}_{candidate_id}"
            if st.button("Edit history", key=f"edit_history_button_{candidate_id}"):
                st.session_state[history_key] = not st.session_state.get(history_key, False)
            if st.session_state.get(history_key, False):
                history = storage.candidate_edit_history(st.session_state.job_id, candidate_id)
                if history:
                    history_labels = {
                        "next_action": "Next action",
                        "recruiter_note": "Recruiter note",
                        "interview_question": "Interview question",
                        "outreach_email_subject": "Outreach email subject",
                        "outreach_email": "Outreach email",
                    }
                    history_df = pd.DataFrame(history)
                    history_df["field_key"] = history_df["field_key"].map(history_labels)
                    history_df = history_df.rename(
                        columns={
                            "changed_at": "Changed at (UTC)",
                            "field_key": "Field",
                            "old_value": "Previous value",
                            "new_value": "New value",
                        }
                    )
                    st.dataframe(history_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No edits have been recorded for this candidate.")

        if submit_decision:
            try:
                if decision == "Approve":
                    if not api_ready():
                        raise RuntimeError("Gemini API is not configured. Approval cannot generate downstream content without the API.")
                    with st.spinner("Agent 3 is generating interview questions and Agent 4 is drafting outreach..."):
                        workflow.approve_and_generate(
                            st.session_state.job_id,
                            candidate_id,
                            criteria,
                            row["full_name"],
                        )
                    set_status(st.session_state.job_id, candidate_id, "approved", "Approve", notes)
                    st.session_state.selected_candidate = candidate_id
                    st.rerun()
                elif decision == "Reject":
                    set_status(st.session_state.job_id, candidate_id, "rejected", "Reject", notes)
                    st.session_state.selected_candidate = candidate_id
                    st.rerun()
                else:
                    set_status(st.session_state.job_id, candidate_id, "review", "Review", notes)
                    st.session_state.selected_candidate = candidate_id
                    st.rerun()
            except Exception as exc:
                st.error(f"HITL workflow failed: {exc}")

        # Downstream editing is shown in the same popup after approval.
        refreshed = results_df(st.session_state.job_id)
        current = refreshed[refreshed["candidate_id"].astype(str) == str(candidate_id)]
        if not current.empty and current.iloc[0]["case_status"] == "approved":
            current = current.iloc[0]
            st.divider()
            st.markdown("### Interview & Outreach")
            st.caption("Edit the generated content if needed. Closing the popup does not send the email; sending is an explicit action.")
            out = downstream_for(st.session_state.job_id, candidate_id)
            if out:
                edited_questions = st.text_area(
                    "Interview question",
                    value=out["generated_interview_questions"] or "",
                    height=150,
                    key=f"questions_{candidate_id}",
                )
                edited_subject = st.text_input(
                    "Outreach email subject",
                    value=out["generated_outreach_subject"] or "",
                    key=f"subject_{candidate_id}",
                )
                edited_body = st.text_area(
                    "Outreach email",
                    value=out["generated_outreach_body"] or "",
                    height=220,
                    key=f"body_{candidate_id}",
                )
                save_col, send_col = st.columns(2)
                with save_col:
                    if st.button("💾 Save Edits", key=f"save_output_{candidate_id}", use_container_width=True):
                        save_downstream(st.session_state.job_id, candidate_id, edited_questions, edited_subject, edited_body, preserve_email_status=True)
                        st.success("Interview question and outreach email saved.")
                        st.rerun()
                with send_col:
                    if str(out["email_send_status"]) == "sent":
                        st.success("✓ Finish sending")
                    else:
                        if st.button("📨 Send Email", key=f"send_email_{candidate_id}", type="primary", use_container_width=True):
                            save_downstream(st.session_state.job_id, candidate_id, edited_questions, edited_subject, edited_body, preserve_email_status=True)
                            ok, message = send_email(current["email"], edited_subject, edited_body)
                            from datetime import datetime
                            mark_email(
                                st.session_state.job_id,
                                candidate_id,
                                "sent" if ok else "failed",
                                datetime.now().astimezone().isoformat(timespec="seconds") if ok else None,
                            )
                            (st.success if ok else st.error)(message)
                            if ok:
                                dismiss_candidate_dialog()
                            st.rerun()
            render_edit_history()
        else:
            render_edit_history()

    _dialog()


if st.session_state.selected_candidate:
    open_candidate_dialog(st.session_state.selected_candidate)

st.caption(f"Current job: {st.session_state.job_id} | API: {'Connected' if api_ready() else 'Not configured'}")
