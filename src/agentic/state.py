from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class AgenticState:
    jd_text: str = ""
    jd_skills: List[str] = field(default_factory=list)
    jd_skill_source: str = "rule"  # "rule" or "llm"

    # resumes: filename -> raw text
    resumes: Dict[str, str] = field(default_factory=dict)

    # results
    ranked_df: Any = None  # pandas DataFrame
    results_obj: Optional[Any] = None  # list of CandidateResult from ranker.py
    explanations: Dict[str, str] = field(default_factory=dict)

    # logs (what makes it "agentic" + easy to demo)
    events: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.events.append(msg)