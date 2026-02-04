from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings
from .text_utils import extract_sections, tokenize_skills, find_years_experience, normalize, extract_jd_relevant_block
from .openai_utils import embed_texts
from .bias_utils import scan_and_mask_sensitive, bias_flag

@dataclass
class CandidateResult:
    candidate_id: str
    filename: str
    score: float
    score_embed: float
    score_skill: float
    score_exp: float
    years_exp_guess: float
    matched_skills: List[str]
    missing_skills: List[str]
    evidence_snippets: List[str]
    bias_sensitive_found: Dict[str, List[str]]
    bias_score_delta: float
    bias_flagged: bool

def _skill_score(jd_skills: List[str], resume_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    jd_set = set([s.lower() for s in jd_skills])
    rs_set = set([s.lower() for s in resume_skills])
    if not jd_set:
        return 0.0, [], []

    matched = sorted(list(jd_set.intersection(rs_set)))
    missing = sorted(list(jd_set.difference(rs_set)))
    score = len(matched) / max(1, len(jd_set))
    return score, matched, missing

def _evidence_snippets(resume_text: str, matched_skills: List[str], max_snips: int = 6) -> List[str]:
    """
    Pull line evidence where matched skill keywords appear.
    """
    lines = [ln.strip() for ln in normalize(resume_text).splitlines() if ln.strip()]
    snips: List[str] = []

    for skill in matched_skills:
        sk = skill.lower()
        for ln in lines:
            if sk in ln.lower():
                snips.append(ln)
                if len(snips) >= max_snips:
                    return snips

    return snips[:max_snips]

def rank_candidates(
    jd_text: str,
    resumes: List[Tuple[str, str]],  # (filename, raw_text)
    settings: Settings
) -> List[CandidateResult]:

    jd_text_n = normalize(jd_text)
    jd_sections = extract_sections(jd_text_n)
    jd_block = jd_sections.get("skills", "") or extract_jd_relevant_block(jd_text_n)
    jd_skills = tokenize_skills(jd_block)


    # Embed JD once
    jd_vec = embed_texts([jd_text_n], model=settings.embedding_model)[0]

    results: List[CandidateResult] = []

    for idx, (filename, r_text) in enumerate(resumes, start=1):
        r_text_n = normalize(r_text)

        # Bias scan + masked text
        scan = scan_and_mask_sensitive(r_text_n)

        # Embed resume (original + masked)
        r_vec = embed_texts([r_text_n], model=settings.embedding_model)[0]
        r_vec_masked = embed_texts([scan.masked_text], model=settings.embedding_model)[0]

        sim = float(cosine_similarity([jd_vec], [r_vec])[0][0])
        sim_masked = float(cosine_similarity([jd_vec], [r_vec_masked])[0][0])

        # Skills overlap
        r_sections = extract_sections(r_text_n)
        r_skills = tokenize_skills(r_sections.get("skills", "") or r_text_n)
        s_skill, matched, missing = _skill_score(jd_skills, r_skills)

        # Experience heuristic
        years = find_years_experience(r_text_n)
        s_exp = min(years / 8.0, 1.0)  # cap at 8 years

        # Weighted score
        score = settings.w_embed * sim + settings.w_skill * s_skill + settings.w_exp * s_exp
        score_masked = settings.w_embed * sim_masked + settings.w_skill * s_skill + settings.w_exp * s_exp
        delta = score - score_masked

        flagged = bias_flag(delta, settings.bias_delta_flag)

        evidence = _evidence_snippets(r_text_n, matched)

        results.append(
            CandidateResult(
                candidate_id=f"C{idx:03d}",
                filename=filename,
                score=round(score, 4),
                score_embed=round(sim, 4),
                score_skill=round(s_skill, 4),
                score_exp=round(s_exp, 4),
                years_exp_guess=years,
                matched_skills=matched,
                missing_skills=missing,
                evidence_snippets=evidence,
                bias_sensitive_found=scan.found,
                bias_score_delta=round(delta, 4),
                bias_flagged=flagged,
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results
