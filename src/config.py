from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Embeddings model
    embedding_model: str = "text-embedding-3-small"

    # Explanation model (used for recruiter-facing explanations)
    explanation_model: str = "gpt-4o-mini"

    # Hybrid scoring weights
    w_embed: float = 0.55   # semantic similarity (resume ↔ JD)
    w_skill: float = 0.30   # skill overlap
    w_exp: float = 0.15     # experience heuristic

    # Bias flag threshold:
    # If score changes by >= this value after masking sensitive info → flag for review
    bias_delta_flag: float = 0.06
