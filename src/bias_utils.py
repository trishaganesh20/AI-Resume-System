import re
from dataclasses import dataclass
from typing import Dict, List

# Simple rule-based scanning (portfolio-friendly)
SENSITIVE_PATTERNS = {
    "age": r"\b(\d{2})\s*(years old|yo)\b|\bdate of birth\b|\bdob\b",
    "gendered_terms": r"\b(he/him|she/her|they/them|female|male|woman|man)\b",
    "nationality_immigration": r"\b(us citizen|u\.s\. citizen|citizenship|visa|h1b|f1|opt|cpt|green card)\b",
    "religion": r"\b(christian|muslim|hindu|jewish|buddhist|sikh)\b",
    "marital_parental": r"\b(married|single|divorced|mother|father|kids|children|pregnan)\b",
}

@dataclass
class BiasScan:
    found: Dict[str, List[str]]
    masked_text: str

def scan_and_mask_sensitive(text: str) -> BiasScan:
    """
    1) Detect sensitive patterns
    2) Create masked text replacing hits with [REDACTED]
    """
    found: Dict[str, List[str]] = {}
    masked = text or ""

    for label, pat in SENSITIVE_PATTERNS.items():
        hits = re.findall(pat, masked, flags=re.IGNORECASE)
        flat_hits: List[str] = []
        for h in hits:
            if isinstance(h, tuple):
                flat_hits.append(" ".join([x for x in h if x]).strip())
            else:
                flat_hits.append(str(h).strip())
        flat_hits = [h for h in flat_hits if h]

        if flat_hits:
            # normalize + dedupe
            uniq = list(dict.fromkeys([h.lower() for h in flat_hits]))
            found[label] = uniq
            masked = re.sub(pat, "[REDACTED]", masked, flags=re.IGNORECASE)

    return BiasScan(found=found, masked_text=masked)

def bias_flag(delta: float, threshold: float) -> bool:
    """
    Flag when sensitive masking changes score enough that a reviewer should look.
    """
    return abs(delta) >= threshold
