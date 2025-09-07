from typing import Dict, Any, List
from .judges.llm_client import LLMClient
from .judges.judge_evaluator import JudgeEvaluator
from .judges.agentic_evaluator import AgenticEvaluator, SearchBackend, FetchBackend
from .judges.embedding_scorer import EmbeddingScorer
from .judges.psych_metrics import PsychMetrics

class EvaluationHarness:
    """
    Orchestrates:
    - Judge LLM (base/guideline/full)
    - Agentic evaluator (requires search/fetch backends)
    - Embedding similarity vs gold
    - Psychological metrics over dialog
    """
    def __init__(
        self,
        llm: LLMClient,
        search_backend: SearchBackend,
        fetch_backend: FetchBackend,
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    ):
        self.judge = JudgeEvaluator(llm)
        self.agentic = AgenticEvaluator(llm, search_backend, fetch_backend, max_sources=5)
        self.embed = EmbeddingScorer(model_name=emb_model)
        self.psych = PsychMetrics(emotion_model=emotion_model)

    def evaluate(
        self,
        scenario: str,
        gold: str,
        model_response: str,
        dialog_for_psych: List[Dict[str, str]],
        modes: List[str] | None = None
    ) -> Dict[str, Any]:
        """
        modes: subset of {"judge_base","judge_guideline","judge_full","agentic","embedding","psych"}
        """
        if modes is None:
            modes = ["judge_base","judge_guideline","judge_full","agentic","embedding","psych"]

        out: Dict[str, Any] = {}

        if "judge_base" in modes:
            out["judge_base"] = self.judge.evaluate_base(scenario, model_response)

        if "judge_guideline" in modes:
            out["judge_guideline"] = self.judge.evaluate_guideline(scenario, model_response)

        if "judge_full" in modes:
            out["judge_full"] = self.judge.evaluate_full(scenario, gold, model_response)

        if "agentic" in modes:
            out["agentic"] = self.agentic.evaluate(scenario, model_response)

        if "embedding" in modes:
            out["embedding_similarity_0_10"] = self.embed.score(model_response, gold)

        if "psych" in modes:
            out["psychological_metrics"] = self.psych.all_metrics(dialog_for_psych)

        return out
