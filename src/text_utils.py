import re
from typing import Dict, List, Tuple

SECTION_HEADERS = [
    "summary",
    "skills",
    "experience",
    "work experience",
    "education",
    "projects",
    "certifications",
    "certification",
]

# Job description section cues (common ATS / LinkedIn patterns)
JD_SECTION_CUES = [
    "requirements",
    "qualifications",
    "what you will do",
    "what you'll do",
    "responsibilities",
    "preferred qualifications",
    "preferred",
    "minimum qualifications",
    "about you",
    "what we’re looking for",
    "what we're looking for",
    "skills",
]

# Stopwords / filler phrases to ignore in skill extraction
FILLER_PATTERNS = [
    r"\babout\b",
    r"\bteam\b",
    r"\bcompany\b",
    r"\bresponsible for\b",
    r"\bability to\b",
    r"\bstrong\b",
    r"\bexcellent\b",
    r"\bcommunication\b",  # keep if you want, but usually too generic
    r"\bfast[- ]paced\b",
    r"\bdetail[- ]oriented\b",
    r"\bself[- ]starter\b",
]

# A curated "common skills" list helps catch important items
# (You can expand this over time; recruiters LOVE seeing a thoughtful list)
COMMON_SKILLS = {
    "sql", "python", "excel", "tableau", "power bi", "looker",
    "analytics", "data analysis", "data visualization",
    "a/b testing", "ab testing", "experimentation",
    "product analytics", "cohort analysis", "funnel analysis",
    "stakeholder management", "requirements gathering",
    "user research", "jira", "confluence",
    "statistics", "hypothesis testing",
    "etl", "data pipelines",
    "machine learning", "nlp",
    "mysql", "postgresql", "bigquery", "snowflake",
    "aws", "gcp", "azure",
}

def normalize(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_sections(text: str) -> Dict[str, str]:
    """
    Lightweight section splitter based on common headers.
    Returns dict with 'full' always present.
    """
    t = normalize(text)
    lines = [ln.strip() for ln in t.splitlines()]

    header_idxs: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        low = re.sub(r"[^a-z ]", "", ln.lower()).strip()
        if low in SECTION_HEADERS:
            header_idxs.append((i, low))

    sections: Dict[str, str] = {"full": t}
    if not header_idxs:
        return sections

    header_idxs.sort(key=lambda x: x[0])

    for idx, (line_i, header) in enumerate(header_idxs):
        start = line_i + 1
        end = header_idxs[idx + 1][0] if idx + 1 < len(header_idxs) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        sections[header] = body

    return sections

def extract_jd_relevant_block(jd_text: str) -> str:
    """
    Pull the most relevant part of a job description for skills/requirements.
    Looks for sections like Requirements/Qualifications/Responsibilities.
    If not found, returns the full JD.
    """
    t = normalize(jd_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # Find a cue line index
    cue_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(cue in low for cue in JD_SECTION_CUES):
            cue_idx = i
            break

    if cue_idx is None:
        return t

    # Take a block after the cue (until next big header-ish line or end)
    block_lines = []
    for ln in lines[cue_idx:]:
        # stop if we hit another section header style line
        low = ln.lower()
        if ln.isupper() and len(ln) <= 40 and len(block_lines) > 10:
            break
        block_lines.append(ln)

        # cap block size so we don’t pull the entire JD
        if len(block_lines) >= 80:
            break

    return "\n".join(block_lines).strip()

def _clean_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"^[\-\*\u2022•\d\.\)\( ]+", "", token)  # remove bullets/numbering
    token = re.sub(r"\s+", " ", token)
    return token.strip()

def tokenize_skills(text: str) -> List[str]:
    """
    Improved skill extraction:
    - Pull bullet items / comma-separated phrases
    - Keep multi-word phrases
    - Filter filler phrases
    - Add COMMON_SKILLS detection
    """
    t = normalize(text).lower()

    # Break into lines first (good for bullet lists)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    candidates: List[str] = []

    for ln in lines:
        # Split common bullet list separators
        parts = re.split(r"[,\|;/]+", ln)
        for p in parts:
            p = _clean_token(p)
            if not p:
                continue
            # remove long sentence-y items
            if len(p) > 60:
                continue
            candidates.append(p)

    # Also detect common skills anywhere in the block
    for sk in COMMON_SKILLS:
        if sk in t:
            candidates.append(sk)

    # Filter filler/soft skills phrases
    filtered = []
    for c in candidates:
        if any(re.search(fp, c) for fp in FILLER_PATTERNS):
            # if it's purely filler, skip
            # BUT allow if it contains a hard skill keyword (sql/python/etc.)
            if not any(h in c for h in ["sql", "python", "excel", "tableau", "power bi", "looker"]):
                continue
        # keep reasonable length
        if 1 < len(c) <= 40:
            filtered.append(c)

    # Normalize (dedupe)
    seen = set()
    out = []
    for c in filtered:
        c2 = c.strip().lower()
        c2 = c2.replace("ab testing", "a/b testing")
        if c2 not in seen:
            seen.add(c2)
            out.append(c2)

    return out

def find_years_experience(text: str) -> float:
    """
    Heuristic: find patterns like '3 years', '2+ years'
    Return max found.
    """
    t = normalize(text).lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years", t)
    vals: List[float] = []
    for m in matches:
        try:
            vals.append(float(m))
        except ValueError:
            pass
    return max(vals) if vals else 0.0
