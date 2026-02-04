# AI Resume Screening & Candidate Ranking System

An AI-powered system that screens resumes, matches candidates to job descriptions, ranks them, and generates explainable hiring rationales with basic bias checks for transparency.

## Why this exists
Recruiters review hundreds of resumes per role. Screening can be slow, inconsistent, and vulnerable to bias. This tool produces ranked, explainable shortlists and flags cases where sensitive information may affect ranking.

## Key Features
- Upload resumes (PDF/DOCX/TXT) + paste a Job Description
- Hybrid ranking:
  - Embedding similarity (resume ↔ JD)
  - Skill overlap (extracted JD skills)
  - Experience heuristic
- Explainability:
  - Score breakdown + matched/missing skills
  - Evidence snippets from resumes
  - LLM-generated recruiter-friendly explanation
- Bias awareness:
  - Detect sensitive attributes (regex-based)
  - Mask sensitive info and re-score
  - Flag meaningful score deltas for human review
- Export ranked results to CSV

## Tech Stack
Python, Streamlit, Pandas, OpenAI API, scikit-learn, pdfplumber, python-docx

## How it works (high-level)
Job Description → Skill extraction → Resume parsing → Hybrid scoring → Bias checks → Explanation → Export

## Run locally
1. Install dependencies  
   `pip install -r requirements.txt`

2. Create a `.env` file in the project root:
   `OPENAI_API_KEY="YOUR_KEY"`

3. Run:
   `streamlit run app.py`
