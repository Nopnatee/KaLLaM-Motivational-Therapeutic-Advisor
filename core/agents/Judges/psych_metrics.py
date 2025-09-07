# pip install transformers torch numpy
from typing import List, Dict
from collections import Counter
import numpy as np
from transformers import pipeline

class PsychMetrics:
    """
    Psychological metrics for dialog systems:
    - emotion_matching
    - emotional_entropy
    - linguistic_style_matching
    - empathy (proxy)
    - agreeableness (proxy)
    """
    def __init__(self, emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"):
        # For production, cache the pipeline and ensure device mapping
        self.emotion_clf = pipeline("text-classification", model=emotion_model, top_k=None)
        self.FUNCTION_WORDS = set("""
            i you he she it we they me him her us them my your his its our their
            a an the and but or so because although while if when
            to from of in on at by with for about as into like through after before over under
        """.split())
        self.EMPATHY_CUES = ["i’m here for you","that sounds hard","i understand","it makes sense","you’re not alone","thank you for sharing"]
        self.AGREEABLE_CUES = ["please","thank you","let’s","we can","would you like","could you"]

    # ---- internals ----
    def _emotion_label(self, text: str) -> str:
        preds = self.emotion_clf(text[:512])[0]
        return max(preds, key=lambda x: x["score"])["label"]

    def _style_vector(self, text: str) -> Counter:
        tokens = [t.lower() for t in text.split()]
        return Counter(t for t in tokens if t in self.FUNCTION_WORDS)

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        keys = list(set(a) | set(b))
        va = np.array([a[k] for k in keys], dtype=float)
        vb = np.array([b[k] for k in keys], dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
        return float(np.dot(va, vb) / denom)

    def _cue_rate(self, text: str, cues: List[str]) -> float:
        t = text.lower()
        return sum(c in t for c in cues) / max(1, len(cues))

    # ---- public metrics ----
    def emotion_matching(self, dialog: List[Dict[str,str]]) -> float:
        matches, total = 0, 0
        for i in range(len(dialog) - 1):
            if dialog[i]["role"] == "user" and dialog[i+1]["role"] == "assistant":
                u = self._emotion_label(dialog[i]["content"])
                a = self._emotion_label(dialog[i+1]["content"])
                total += 1
                if u == a:
                    matches += 1
        return round(matches / total, 3) if total else 0.0

    def emotional_entropy(self, dialog: List[Dict[str,str]]) -> float:
        emos = [self._emotion_label(t["content"]) for t in dialog if t["role"] == "assistant"]
        if not emos:
            return 0.0
        counts = np.array(list(Counter(emos).values()), dtype=float)
        p = counts / counts.sum()
        return round(float(-(p * np.log2(p + 1e-12)).sum()), 3)

    def linguistic_style_matching(self, dialog: List[Dict[str,str]]) -> float:
        sims, total = 0.0, 0
        for i in range(len(dialog) - 1):
            if dialog[i]["role"] == "user" and dialog[i+1]["role"] == "assistant":
                sims += self._cosine(self._style_vector(dialog[i]["content"]),
                                     self._style_vector(dialog[i+1]["content"]))
                total += 1
        return round(sims / total, 3) if total else 0.0

    def empathy(self, dialog: List[Dict[str,str]]) -> float:
        asst_text = " ".join(d["content"] for d in dialog if d["role"] == "assistant")
        return round(10 * self._cue_rate(asst_text, self.EMPATHY_CUES), 2)

    def agreeableness(self, dialog: List[Dict[str,str]]) -> float:
        asst_text = " ".join(d["content"] for d in dialog if d["role"] == "assistant")
        return round(10 * self._cue_rate(asst_text, self.AGREEABLE_CUES), 2)

    def all_metrics(self, dialog: List[Dict[str,str]]) -> Dict[str, float]:
        return {
            "emotion_matching": self.emotion_matching(dialog),
            "emotional_entropy": self.emotional_entropy(dialog),
            "style_matching": self.linguistic_style_matching(dialog),
            "empathy": self.empathy(dialog),
            "agreeableness": self.agreeableness(dialog),
        }
