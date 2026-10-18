# Thara Energy — AI Agentic Recruiting Workbench

A Streamlit prototype web application for AI-assisted recruitment, featuring dynamic job criteria generation, evidence-based candidate screening, weighted scoring, human review and approval, interview-question generation, and outreach email drafting.

## Main features

1. **Job Input** now contains only JD input + the **AI-Generated Assessment Framework**, shown after Agent 1 runs.
2. **Candidate Database** previews the first five rows and asks for Yes/No confirmation before replacing an existing database.
3. Funnel cards clearly separate **AI Recommendation** from **case_status**.
4. Lifecycle field is now `case_status` with values:
   - `screened`
   - `screen error`
   - `approved`
   - `rejected`
   - `review`
5. Candidate overview includes **Recruiter Note**, **Interview Question**, **Outreach Email**, and **Email Status**.
6. Candidate overview uses a compact, fixed-height grid with a frozen header, sortable/filterable column headers, full-table contains search, categorical dropdown filters for Email Status, Case Status, and AI Recommendation, and checkboxes for choosing visible fields. Clicking the styled Candidate ID action opens the **Human-in-the-Loop popup** with the candidate's full screening evidence and action buttons.
7. Approve generates interview questions + outreach email; both are editable in the same popup.
8. Recruiter notes and all actions persist to SQLite, and `st.rerun()` refreshes the dashboard immediately after changes.
9. Email is explicit: generated content is not sent automatically. In default `mock` mode, clicking Send Email marks it sent without sending a real external email.
10. Existing V2.x SQLite databases are migrated/normalized when the app starts.
11. Assessment criterion names and definitions may summarize the job description in new wording. Criterion details explain each assigned weight and show the original job-description excerpt selected as supporting raw data; weights are based on the role, not candidate profiles. Older saved frameworks are enriched from their original job description without changing their weights.
12. Recruiters can change criterion percentages using the controls at the bottom-right of the framework panel. A reason is required. If the total is not 100%, Agent 1 redistributes only criteria whose percentages the recruiter did not edit, preserving every manually edited percentage; the recruiter can also return and edit ratios. Ratio and history dialogs stay open when clicking outside and close only through their in-dialog controls. Closing an unsubmitted ratio edit discards it without changing saved ratios. Each criterion retains its AI-generated starting ratio and a persistent change history, and existing candidate scores are recalculated from saved criterion scores.

## Run locally

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
4. Run:

```bash
streamlit run app.py
```

The default candidate database is `data/thara_energy_candidates_db.csv`. The SQLite workbench database is created at `data/recruiting_workbench.db`.

## Email safety

`EMAIL_DELIVERY_MODE=mock` is the default. It does **not** send an external email. To use SMTP intentionally, configure all SMTP variables and change the mode to `smtp`.

## Core workflow

**Agent 1:** JD → dynamic assessment framework

**Agent 2:** candidate evidence → criterion scores/reasons based on semantic relevance (equivalent wording is allowed), with a verbatim raw-data excerpt for recruiter verification (criteria without a verifiable evidence excerpt score 0) → deterministic weighted score including all criteria → AI recommendation

**Human:** candidate popup → approve/reject/review + recruiter note

**Agents 3 & 4:** approved candidate → tailored interview questions + outreach draft

**Recruiter:** edit → explicitly send email or close without sending

## Storage / hosted-environment note

The prototype stores workflow state in SQLite so recruiter actions survive normal
Streamlit reruns and page interactions. The database location can be overridden with
`DATABASE_PATH` (or `DATA_DIR`). SQLite is appropriate for this prototype and local
testing, but it is **not a production persistence layer on Streamlit Community Cloud**:
its local filesystem can be recreated when the app is restarted/rebuilt. For the this
prototype, this is acceptable; a production version should move candidate/workflow
state to a managed database such as PostgreSQL or another persistent datastore.

The app also enables SQLite WAL mode and a busy timeout to reduce transient locking
when Streamlit reruns and screening work happen close together.

## Dependency / deployment notes

All runtime third-party imports are declared in `requirements.txt`. The repository
must not contain `.env`, Streamlit secrets, or a runtime `.db` file. For Streamlit
Community Cloud, configure `GEMINI_API_KEY` through the app's Secrets settings.
