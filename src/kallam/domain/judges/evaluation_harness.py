# core/eval/evaluation_harness.py
# Thin orchestrator that wraps MentalHealthMetricsWrapper and exposes a simple API
# for app code. You plug in:
#   - generate_fn(prompt, cfg) -> string   (your ChatbotManager or model client)
#   - judge call_fns returning JSON strings for base/guideline/full

import logging
from typing import Dict, Any, List, Optional, Callable

import pandas as pd

from .gold_metrics import (
    MentalHealthMetricsWrapper,
    JudgeConfig,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class EvaluationHarness:
    """
    One-call evaluation:
        harness = EvaluationHarness(benchmark_csv_path="data/mental_health_benchmark.csv")
        report = harness.run_full_evaluation(
            generate_fn=my_generate,
            judge_call_fns={"gpt-4o": judge_gpt4o_fn},  # name -> fn
            judge_modes=("base", "guideline", "full"),
            system_prompt="You are a supportive, safe mental-health assistant...",
            gen_cfg={"temperature": 0.0},
            extra_meta={"mut_model": "sealion-70b", "judges": ["gpt-4o"]},
        )

    Returns dict with summary and file paths; artifacts saved in runs/<run_id>/...
    """

    def __init__(
        self,
        benchmark_csv_path: str = "data/mental_health_benchmark.csv",
        run_dir: str = "runs",
        emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        emb_model_name_alt: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        seed: int = 42,
    ):
        self.wrapper = MentalHealthMetricsWrapper(
            benchmark_csv_path=benchmark_csv_path,
            run_dir=run_dir,
            emb_model_name=emb_model_name,
            emb_model_name_alt=emb_model_name_alt,
            seed=seed,
        )
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self.wrapper.load_benchmark()
            self._loaded = True

    def _build_judges(
        self,
        judge_call_fns: Dict[str, Callable[[str], str]],
        judge_modes: List[str],
    ) -> List[JudgeConfig]:
        judges: List[JudgeConfig] = []
        for name, call_fn in judge_call_fns.items():
            for mode in judge_modes:
                judges.append(JudgeConfig(name=name, mode=mode, call_fn=call_fn))
        return judges

    def run_full_evaluation(
        self,
        generate_fn: Callable[[str, Dict[str, Any]], str],
        judge_call_fns: Dict[str, Callable[[str], str]],
        judge_modes: List[str] = ("base", "guideline", "full"),
        system_prompt: Optional[str] = None,
        gen_cfg: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes:
          1) generation on benchmark
          2) judging with all judges/modes
          3) embedding similarity
          4) summary aggregation

        Returns the summary dict (also saved to runs/.../summary.json).
        """
        self.load()
        logger.info("Running generation step...")
        answers_df: pd.DataFrame = self.wrapper.run_generation(generate_fn, system_prompt, gen_cfg)

        logger.info("Preparing judges...")
        judges = self._build_judges(judge_call_fns, list(judge_modes))

        logger.info("Running judges...")
        judge_scores_df: pd.DataFrame = self.wrapper.run_judges(answers_df, judges)

        logger.info("Running embeddings...")
        emb_df: pd.DataFrame = self.wrapper.run_embeddings(answers_df)

        logger.info("Summarizing results...")
        summary = self.wrapper.summarize(
            answers_df=answers_df,
            judges_df=judge_scores_df,
            emb_df=emb_df,
            extra_meta=extra_meta,
        )
        return summary
