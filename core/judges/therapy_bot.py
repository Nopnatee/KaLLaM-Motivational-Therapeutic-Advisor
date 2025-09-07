from typing import List, Dict
from .llm_client import LLMClient
from artificial_users import Persona

SESSION_PROMPT = """You are a behavioral-activation chatbot guiding a user through:
rapport → mood assessment → psychoeducation → activity planning → obstacles → rewards → review.
Follow safety guidelines.

Persona:
{persona}

Conversation so far:
{history}

User: {user_turn}
Assistant:
"""

class TherapyBot:
    """
    Simple wrapper to run a BA-style therapy conversation against an LLM.
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def reply(self, persona: Persona, history: List[Dict[str, str]], user_turn: str) -> str:
        history_str = "\n".join(f"{h['role'].title()}: {h['content']}" for h in history)
        prompt = SESSION_PROMPT.format(persona=persona.__dict__, history=history_str, user_turn=user_turn)
        # Prefer structured output in practice; here plain text for simplicity
        return self.llm.complete_text(prompt)  # Must be implemented in your LLM client


def therapist_score_dialog(dialog: List[Dict[str,str]]) -> Dict:
    """
    Placeholder psychotherapist rubric. Replace with real clinician assessments or a validated rubric.
    """
    return {
        "fidelity": 7.5,
        "safety": 8.5,
        "competence": 7.0,
        "comments": "Adequate BA flow; activity plan could be more personalized."
    }
