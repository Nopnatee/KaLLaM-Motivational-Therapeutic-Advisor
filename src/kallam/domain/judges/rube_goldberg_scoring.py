# rube_goldberg_scoring.py
# A minimal, pluggable evaluator for dialog quality using psychologically
# grounded metrics inspired by "Psychological Metrics for Dialog System Evaluation" (arXiv:2305.14757).
#
# Works directly on the JSON exported by ChatbotManager.JsonExporter.
# No heavyweight deps required; ships with lexicon-based fallbacks and clean hooks for ML models.
#
# Usage:
#   python rube_goldberg_scoring.py /path/to/exported_session.json > scores.json
#
# Programmatic:
#   from rube_goldberg_scoring import score_session_json
#   result = score_session_json(json_obj_or_path)
#
# Output schema (top-level):
# {
#   "session_id": "...",
#   "message_count": N,
#   "per_turn": [ { "idx": i, "role": "assistant", "text": "...",
#                   "emotion": "...", "emotion_matching": 0/1/None,
#                   "lsm_turn": 0..1 }, ... ],
#   "aggregates": {
#       "emotional_entropy": 0..logK,
#       "emotion_matching_rate": 0..1,
#       "lsm_mean": 0..1,
#       "agreeableness": 0..1,
#       "empathy": 0..1
#   },
#   "notes": "...",
#   "version": "0.1.0"
# }

from __future__ import annotations
import json
import math
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import Counter, defaultdict

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Turn:
    role: str          # 'user' | 'assistant' | 'system'
    text: str
    ts: Optional[str] = None

@dataclass
class Dialog:
    session_id: Optional[str]
    turns: List[Turn]

# ----------------------------
# Utilities
# ----------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

def _norm_text(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.lower()

def _soft_div(a: float, b: float, eps: float=1e-9) -> float:
    return a / (b + eps)

def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return ent  # natural log base

# ----------------------------
# Simple lexicons / heuristics
# ----------------------------

# Emotions (toy set). Replace with a proper classifier if you want.
EMOTION_LEX = {
    "joy": {"happy","glad","great","love","awesome","nice","good","joy","delight"},
    "sad": {"sad","down","unhappy","depressed","blue","cry","upset"},
    "anger": {"mad","angry","furious","annoyed","irritated"},
    "fear": {"afraid","scared","fear","anxious","worried","panic"},
    "disgust": {"disgust","gross","nasty","revolting"},
    "surprise": {"surprised","shocked","wow","unexpected"},
    # fallback category will be "neutral"
}

# Function words subset for LSM (inspired by LIWC categories; minimal here).
# You can expand freely.
FUNC_WORD_CATS = {
    "articles": {"a","an","the"},
    "preps": {"of","in","to","for","with","on","at","from","by","about","as","into","like","through","after","over","between","out","against","during","without","before","under","around","among"},
    "aux": {"am","is","are","was","were","be","been","being","do","does","did","have","has","had","can","could","should","would","may","might","must","will","shall"},
    "pronouns": {"i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","her","its","our","their"},
    "conj": {"and","or","but","so","yet","for","nor"},
    "adverbs": {"very","really","just","too","quite","rather","so","not"},
}

# Agreeableness / Empathy cue lexicons (ridiculously small on purpose).
# Replace with ML models for production.
AGREEABLE_CUES_POS = {"please","thank","appreciate","let’s","together","understand","kind","support"}
AGREEABLE_CUES_NEG = {"blame","fault","stupid","useless","ignore","shut","must","obviously"}

EMPATHY_CUES = {
    "ack": {"i understand","i’m sorry to hear","that sounds","it makes sense","i hear you","i get that"},
    "validate": {"it’s valid","it’s okay","it’s understandable","your feelings"},
    "support": {"i’m here","let’s work","we can","resources","reach out","support line"},
}

def _contains_any(text: str, phrases: set[str]) -> bool:
    t = " " + _norm_text(text) + " "
    for p in phrases:
        if f" {p} " in t:
            return True
    return False

# ----------------------------
# Analyzer interfaces (pluggable)
# ----------------------------

class EmotionAnalyzer:
    def infer(self, text: str) -> str:
        """Return one of {joy,sad,anger,fear,disgust,surprise,neutral}."""
        t = set(_norm_text(text).split())
        best, score = "neutral", 0
        for emo, words in EMOTION_LEX.items():
            inter = len(t & words)
            if inter > score:
                best, score = emo, inter
        return best

class StyleAnalyzer:
    def category_freqs(self, text: str) -> Dict[str, float]:
        """Return normalized frequencies for function-word categories."""
        toks = _norm_text(text).split()
        n = max(1, len(toks))
        cats = {}
        for cat, vocab in FUNC_WORD_CATS.items():
            count = sum(1 for w in toks if w in vocab)
            cats[cat] = count / n
        return cats

class ProsocialAnalyzer:
    def agreeableness(self, dialog_text: str) -> float:
        t = _norm_text(dialog_text)
        pos = sum(1 for w in AGREEABLE_CUES_POS if f" {w} " in f" {t} ")
        neg = sum(1 for w in AGREEABLE_CUES_NEG if f" {w} " in f" {t} ")
        raw = pos - 0.7*neg
        # squish to 0..1
        return max(0.0, min(1.0, 0.5 + 0.1*raw))

    def empathy(self, dialog_text: str) -> float:
        t = _norm_text(dialog_text)
        hits = 0
        for bucket in EMPATHY_CUES.values():
            hits += sum(1 for phrase in bucket if f" {phrase} " in f" {t} ")
        return max(0.0, min(1.0, hits * 0.08))  # tiny scale, capped

# ----------------------------
# Core metrics
# ----------------------------

def emotional_entropy(assistant_emotions: List[str]) -> float:
    return _entropy_from_counts(Counter(assistant_emotions))

def emotion_matching(user_emotions: List[str], assistant_emotions: List[str]) -> Tuple[float, List[Optional[int]]]:
    """
    Compare user emotion at t with assistant emotion at t+1.
    Return (match_rate, turn_matches) where turn_matches[i] is 1/0/None for each assistant turn.
    """
    matches: List[Optional[int]] = []
    u_idx = -1
    last_user_emo = None
    for emo_u, emo_a in zip(_fill_user_sequence(user_emotions), assistant_emotions):
        if emo_u is None or emo_a is None:
            matches.append(None)
        else:
            matches.append(1 if emo_u == emo_a else 0)
    vals = [m for m in matches if m is not None]
    rate = sum(vals)/len(vals) if vals else 0.0
    return rate, matches

def _fill_user_sequence(user_emotions: List[str]) -> List[Optional[str]]:
    """
    Shift user emotions so index i aligns to assistant turn i (user at t vs assistant at t).
    In many chats, turns alternate; to be safer, we simply reuse last seen user emotion.
    """
    out: List[Optional[str]] = []
    last = None
    for emo in user_emotions:
        last = emo or last
        out.append(last)
    # If assistant has more turns, pad with last
    if out:
        last = out[-1]
    return out

def lsm_turn(u_freqs: Dict[str, float], a_freqs: Dict[str, float]) -> float:
    """
    Linguistic Style Matching per Gonzales et al.:
      LSM_i = 1 - |x_i - y_i| / (x_i + y_i + eps)
    Overall turn LSM is mean over categories observed.
    """
    vals = []
    for cat in FUNC_WORD_CATS.keys():
        x = u_freqs.get(cat, 0.0)
        y = a_freqs.get(cat, 0.0)
        vals.append(1.0 - abs(x - y) / (x + y + 1e-9))
    return sum(vals) / len(vals) if vals else 0.0

# ----------------------------
# JSON loader (robust to your exporter)
# ----------------------------

def _coerce_dialog(obj: Dict[str, Any]) -> Dialog:
    """
    Expect formats like:
      {
        "session": {... maybe has session_id ...},
        "messages": [
           {"role":"user","content":"...","timestamp":"..."},
           {"role":"assistant","content":"..."}
        ]
      }
    Or flat {"session_id": "...", "messages":[...]}
    """
    session_id = obj.get("session_id") or (obj.get("session") or {}).get("session_id")
    raw_msgs = obj.get("messages") or obj.get("data") or []
    turns: List[Turn] = []
    for m in raw_msgs:
        role = m.get("role") or ""
        text = m.get("translated_content") or m.get("content") or ""
        ts = m.get("timestamp")
        if role in {"user","assistant","system"} and text:
            turns.append(Turn(role=role, text=text, ts=ts))
    return Dialog(session_id=session_id, turns=turns)

def _load_json(path_or_obj: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(path_or_obj, (str, Path)):
        with open(path_or_obj, "r", encoding="utf-8") as f:
            return json.load(f)
    elif isinstance(path_or_obj, dict):
        return path_or_obj
    else:
        raise TypeError("path_or_obj must be a file path or a dict.")

# ----------------------------
# Public API
# ----------------------------

def score_session_json(path_or_obj: Union[str, Path, Dict[str, Any]],
                       emotion_analyzer: Optional[EmotionAnalyzer]=None,
                       style_analyzer: Optional[StyleAnalyzer]=None,
                       prosocial_analyzer: Optional[ProsocialAnalyzer]=None) -> Dict[str, Any]:
    """
    Main entry: load session JSON, compute metrics per assistant response and aggregates.
    """
    obj = _load_json(path_or_obj)
    dialog = _coerce_dialog(obj)

    emo = emotion_analyzer or EmotionAnalyzer()
    sty = style_analyzer or StyleAnalyzer()
    pro = prosocial_analyzer or ProsocialAnalyzer()

    # Extract aligned user/assistant sequences
    user_turns = [t for t in dialog.turns if t.role == "user"]
    asst_turns = [t for t in dialog.turns if t.role == "assistant"]

    # Emotions per turn
    user_emotions = [emo.infer(t.text) for t in user_turns]
    asst_emotions = [emo.infer(t.text) for t in asst_turns]

    # Emotion matching
    match_rate, match_vec = emotion_matching(user_emotions, asst_emotions)

    # LSM per assistant turn (paired to nearest preceding user turn)
    lsm_vals: List[float] = []
    per_turn_records: List[Dict[str, Any]] = []
    # Pair up to min length
    L = min(len(user_turns), len(asst_turns))
    for i in range(L):
        uf = sty.category_freqs(user_turns[i].text)
        af = sty.category_freqs(asst_turns[i].text)
        lsm_val = lsm_turn(uf, af)
        lsm_vals.append(lsm_val)

    # Emotional entropy over assistant emotions
    ee = emotional_entropy(asst_emotions)

    # Dialog-level agreeableness & empathy (assistant-only text)
    asst_text_concat = "\n".join([t.text for t in asst_turns])
    agree = pro.agreeableness(asst_text_concat)
    empath = pro.empathy(asst_text_concat)

    # Build per-turn records aligned to assistant turns
    for i, t in enumerate(asst_turns):
        rec = {
            "idx": i,
            "role": "assistant",
            "text": t.text,
            "timestamp": t.ts,
            "emotion": asst_emotions[i] if i < len(asst_emotions) else None,
            "emotion_matching": match_vec[i] if i < len(match_vec) else None,
            "lsm_turn": lsm_vals[i] if i < len(lsm_vals) else None,
        }
        per_turn_records.append(rec)

    result = {
        "session_id": dialog.session_id,
        "message_count": len(dialog.turns),
        "per_turn": per_turn_records,
        "aggregates": {
            "emotional_entropy": ee,
            "emotion_matching_rate": match_rate,
            "lsm_mean": sum(v for v in lsm_vals if v is not None)/max(1, len(lsm_vals)),
            "agreeableness": agree,
            "empathy": empath,
        },
        "notes": (
            "Lexicon-based heuristics used. For research-grade replication, plug in proper "
            "emotion/style/trait models via the analyzer interfaces."
        ),
        "version": "0.1.0",
    }
    return result

# ----------------------------
# CLI
# ----------------------------

def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python rube_goldberg_scoring.py <exported_session.json>", file=sys.stderr)
        return 2
    path = argv[1]
    out = score_session_json(path)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
