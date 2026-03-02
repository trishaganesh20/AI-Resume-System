from typing import List, Tuple
import pandas as pd

from src.config import Settings
from src.agentic.state import AgenticState
from src.agentic.agents import (
    jd_skills_rule_agent,
    jd_skills_llm_agent,
    ranking_agent,
    explanation_agent,
)

class AgentOrchestrator:
    """
    Agentic pipeline:
    - rule skills agent
    - fallback LLM skills agent if quality is low
    - ranking agent
    - optional explanation agent for top K
    """
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(
        self,
        jd_text: str,
        resumes: List[Tuple[str, str]],
        auto_explain_top_k: int = 0,
    ) -> AgenticState:
        state = AgenticState(jd_text=jd_text)
        state.log("Planner: starting agentic pipeline...")

        # 1) Skills (rule)
        state.log("JD Skills Agent (rule): extracting skills from JD...")
        skills = jd_skills_rule_agent(jd_text)

        # 2) Decide fallback
        if len(skills) < 6:
            state.log(f"Planner: only {len(skills)} skills found → using LLM fallback.")
            llm_skills = jd_skills_llm_agent(jd_text, self.settings)
            if len(llm_skills) >= len(skills):
                skills = llm_skills
                state.jd_skill_source = "llm"
                state.log(f"JD Skills Agent (LLM): extracted {len(skills)} skills.")
            else:
                state.log("Planner: LLM fallback did not improve; keeping rule skills.")
        else:
            state.log(f"Planner: rule skills look good ({len(skills)} skills).")

        state.jd_skills = skills

        # 3) Ranking
        state.log(f"Ranking Agent: scoring {len(resumes)} resumes...")
        results, rows = ranking_agent(jd_text, resumes, self.settings)
        state.results_obj = results
        state.ranked_df = pd.DataFrame(rows)
        state.log("Ranking Agent: done.")

        # 4) Explanations (optional)
        if auto_explain_top_k and auto_explain_top_k > 0:
            k = min(auto_explain_top_k, len(results))
            state.log(f"Explanation Agent: generating explanations for top {k} candidates...")
            for r in results[:k]:
                state.explanations[r.candidate_id] = explanation_agent(jd_text, r, self.settings)
            state.log("Explanation Agent: done.")

        state.resumes = {fn: txt for fn, txt in resumes}
        state.log("Planner: pipeline complete.")
        return state