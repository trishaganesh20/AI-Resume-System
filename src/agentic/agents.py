import json
from typing import List, Tuple, Dict, Any

from src.config import Settings
from src.text_utils import extract_sections, tokenize_skills, extract_jd_relevant_block, normalize
from src.ranker import rank_candidates
from src.explain import generate_explanation
from src.openai_utils import get_client

def jd_skills_rule_agent(jd_text: str) -> List[str]:
    jd = normalize(jd_text)
    sections = extract_sections(jd)
    jd_block = sections.get("skills", "") or extract_jd_relevant_block(jd)
    return tokenize_skills(jd_block)

def jd_skills_llm_agent(jd_text: str, settings: Settings) -> List[str]:
    """
    Fallback agent: if rule extraction is weak, ask LLM to return a clean JSON skills list.
    """
    client = get_client()
    prompt = f"""
Extract job-relevant skills from the job description.

Return ONLY valid JSON:
{{"skills": ["skill1", "skill2", ...]}}

Rules:
- 8 to 20 skills
- short phrases (1–4 words)
- include tools/methods/domain skills
- exclude fluff like "strong communication"

JOB DESCRIPTION:
{jd_text}
""".strip()

    resp = client.chat.completions.create(
        model=settings.explanation_model,
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        skills = data.get("skills", [])
        if isinstance(skills, list):
            return [str(s).strip().lower() for s in skills if str(s).strip()]
    except Exception:
        pass
    return []

def ranking_agent(jd_text: str, resumes: List[Tuple[str, str]], settings: Settings):
    """
    Calls your existing ranker. Returns (results_list, df_ready_rows).
    """
    results = rank_candidates(jd_text=jd_text, resumes=resumes, settings=settings)

    rows = []
    for r in results:
        rows.append({
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
        })
    return results, rows

def explanation_agent(jd_text: str, candidate_result, settings: Settings) -> str:
    return generate_explanation(
        jd_text=jd_text,
        matched_skills=candidate_result.matched_skills,
        missing_skills=candidate_result.missing_skills,
        evidence_snippets=candidate_result.evidence_snippets,
        bias_sensitive_found=candidate_result.bias_sensitive_found,
        settings=settings,
    )