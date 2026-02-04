import os
import pandas as pd
import streamlit as st

from src.config import Settings
from src.io_utils import load_resume_file, safe_filename, ensure_dir
from src.ranker import rank_candidates
from src.explain import generate_explanation

st.set_page_config(page_title="AI Resume Ranker", layout="wide")
settings = Settings()

# Session state defaults
if "has_results" not in st.session_state:
    st.session_state.has_results = False
if "results" not in st.session_state:
    st.session_state.results = []
if "df" not in st.session_state:
    st.session_state.df = None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "raw_text_map" not in st.session_state:
    st.session_state.raw_text_map = {}
if "explanations" not in st.session_state:
    st.session_state.explanations = {}  # key: candidate_id -> explanation text
if "jd_skills" not in st.session_state:
    st.session_state.jd_skills = []


st.title("AI Resume Screening & Candidate Ranking")
st.caption("Upload resumes + paste a job description → get ranked, explainable results with bias checks.")

# Sidebar: Inputs
with st.sidebar:
    st.header("1) Job Description")
    jd_text = st.text_area("Paste the Job Description here", height=220, value=st.session_state.jd_text)

    st.header("2) Upload Resumes")
    files = st.file_uploader(
        "Upload PDF/DOCX/TXT resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    st.header("3) Settings")
    st.write("Scoring weights (edit in src/config.py):")
    st.write(f"- Embedding similarity: {settings.w_embed}")
    st.write(f"- Skill overlap: {settings.w_skill}")
    st.write(f"- Experience heuristic: {settings.w_exp}")
    st.write(f"Bias flag threshold (delta): {settings.bias_delta_flag}")

    col1, col2 = st.columns(2)
    run_btn = col1.button("Run Ranking", type="primary", use_container_width=True)
    reset_btn = col2.button("Reset", use_container_width=True)

# Helpers
def _save_uploaded_file(uploaded) -> str:
    ensure_dir("data/uploads")
    path = os.path.join("data/uploads", safe_filename(uploaded.name))
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


# Reset logic
if reset_btn:
    st.session_state.has_results = False
    st.session_state.results = []
    st.session_state.df = None
    st.session_state.jd_text = ""
    st.session_state.raw_text_map = {}
    st.session_state.explanations = {}
    st.rerun()

# Run ranking logic (store in session_state)
if run_btn:
    if not jd_text.strip():
        st.error("Please paste a Job Description first.")
        st.stop()
    if not files:
        st.error("Please upload at least 1 resume.")
        st.stop()

    st.session_state.jd_text = jd_text

    from src.text_utils import extract_sections, tokenize_skills, extract_jd_relevant_block

    jd_text_n = jd_text.strip()
    jd_sections = extract_sections(jd_text_n)
    jd_block = jd_sections.get("skills", "") or extract_jd_relevant_block(jd_text_n)
    st.session_state.jd_skills = tokenize_skills(jd_block)


    resume_items = []
    raw_text_map = {}

    with st.spinner("Reading resumes..."):
        for f in files:
            path = _save_uploaded_file(f)
            text = load_resume_file(path)
            resume_items.append((f.name, text))
            raw_text_map[f.name] = text

    with st.spinner("Scoring + ranking candidates..."):
        results = rank_candidates(jd_text=jd_text, resumes=resume_items, settings=settings)

    df = pd.DataFrame([{
        "Candidate": r.candidate_id,
        "Resume File": r.filename,
        "Overall Score": r.score,
        "Embed Similarity": r.score_embed,
        "Skill Match": r.score_skill,
        "Exp Score": r.score_exp,
        "Years Exp (guess)": r.years_exp_guess,
        "Bias Flagged": r.bias_flagged,
        "Bias Δ (orig - masked)": r.bias_score_delta,
        "Sensitive Detected": ", ".join(r.bias_sensitive_found.keys()) if r.bias_sensitive_found else ""
    } for r in results])

    st.session_state.results = results
    st.session_state.df = df
    st.session_state.raw_text_map = raw_text_map
    st.session_state.has_results = True
    st.session_state.explanations = {}  # reset explanations for new run

# Display (uses session_state)
if st.session_state.has_results:
    st.subheader("Ranked Candidates")
    results = st.session_state.results
    df = st.session_state.df

    st.subheader("Ranked Candidates")

    with st.expander("Detected Job Description Skills (used for matching)", expanded=False):
        if st.session_state.jd_skills:
         st.write(", ".join(st.session_state.jd_skills))
        else:
            st.info("No JD skills detected. Try adding a 'Requirements' or 'Qualifications' section in the JD.")




    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export CSV + download button
    ensure_dir("outputs")
    csv_path = os.path.join("outputs", "ranked_candidates.csv")
    df.to_csv(csv_path, index=False)

    st.download_button(
        "Download ranked_candidates.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_candidates.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Candidate Drill-Down")

    pick = st.selectbox("Select a candidate", options=[r.candidate_id for r in results])
    chosen = next(r for r in results if r.candidate_id == pick)

    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"### {chosen.candidate_id} — {chosen.filename}")
        st.metric("Overall Score", chosen.score)

        st.write("**Matched skills:**")
        st.write(", ".join(chosen.matched_skills) if chosen.matched_skills else "None detected")

        st.write("**Missing/unclear skills:**")
        st.write(", ".join(chosen.missing_skills[:12]) if chosen.missing_skills else "None")

        st.write("**Evidence snippets:**")
        if chosen.evidence_snippets:
            for s in chosen.evidence_snippets:
                st.code(s)
        else:
            st.info("No direct snippet matches found. (Often caused by resume formatting.)")

    with right:
        st.markdown("### Bias & Transparency")

        if chosen.bias_sensitive_found:
            st.write("Sensitive patterns detected (for reviewer awareness):")
            st.json(chosen.bias_sensitive_found)
        else:
            st.write("No sensitive patterns detected by the scanner.")

        st.write(f"Score delta after masking sensitive info: **{chosen.bias_score_delta}**")

        if chosen.bias_flagged:
            st.warning("Flagged: Score changes meaningfully when sensitive info is removed → review recommended.")
        else:
            st.success("Not flagged by delta threshold.")

        st.markdown("### Recruiter-Friendly Explanation (LLM)")

        # Show existing explanation if already generated
        if chosen.candidate_id in st.session_state.explanations:
            st.markdown(st.session_state.explanations[chosen.candidate_id])

        if st.button("Generate Explanation"):
            with st.spinner("Generating explanation..."):
                explanation = generate_explanation(
                    jd_text=st.session_state.jd_text,
                    matched_skills=chosen.matched_skills,
                    missing_skills=chosen.missing_skills,
                    evidence_snippets=chosen.evidence_snippets,
                    bias_sensitive_found=chosen.bias_sensitive_found,
                    settings=settings
                )
            st.session_state.explanations[chosen.candidate_id] = explanation
            st.rerun()

else:
    st.info("Paste a job description and upload resumes, then click **Run Ranking**.")
