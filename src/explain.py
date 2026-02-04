from typing import Dict, List
from .openai_utils import get_client
from .config import Settings

def generate_explanation(
    jd_text: str,
    matched_skills: List[str],
    missing_skills: List[str],
    evidence_snippets: List[str],
    bias_sensitive_found: Dict[str, List[str]],
    settings: Settings
) -> str:
    """
    LLM explanation: recruiter-friendly bullets grounded in evidence.
    """
    client = get_client()

    prompt = f"""
You are an ATS assistant helping a recruiter understand a candidate-job match.
Write 6–10 concise bullets that are:
- specific (mention matched skills and relevant experience)
- honest about gaps (missing skills)
- grounded in the evidence snippets provided
- neutral and fair (do NOT mention age/gender/nationality/religion even if present)

JOB DESCRIPTION:
{jd_text}

EVIDENCE SNIPPETS FROM RESUME:
{chr(10).join("- " + s for s in (evidence_snippets or ["(No direct snippet matches found)"]))}

MATCHED SKILLS:
{", ".join(matched_skills) if matched_skills else "(none detected)"}

MISSING / UNCLEAR SKILLS:
{", ".join(missing_skills[:12]) if missing_skills else "(none)"}

SENSITIVE INFO DETECTED (for bias review only; do not reference in explanation):
{bias_sensitive_found}

Return only the bullets.
""".strip()

    resp = client.chat.completions.create(
        model=settings.explanation_model,
        messages=[
            {"role": "system", "content": "You produce fair, structured, recruiter-ready explanations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
