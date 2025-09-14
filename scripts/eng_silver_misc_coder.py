# -*- coding: utf-8 -*-
"""
BiMISC-style coding pipeline (SEA-LION edition)

Implements:
- Prompt template: task instruction + role-specific MISC manual + 2 examples/code + brief history
- Deterministic decoding (temperature=0)
- Multi-label outputs with a confidence gate (threshold)
- Fine-grained codes + optional mapping to AnnoMI coarse codes
- Metrics: Accuracy, Precision, Recall, Macro-F1 (multi-label)
- Robust JSON-only output enforcement and retry/backoff for API stability

Environment (.env):
  SEA_LION_API_KEY=...               # required
  SEA_LION_BASE_URL=https://api.sea-lion.ai/v1   # optional (default)
  SEA_LION_MODEL=aisingapore/Gemma-SEA-LION-v4-27B-IT   # optional (default)

Expected input dataset (JSONL):
  Each line: {
    "history": [{"role":"Client","text":"..."}, {"role":"Therapist","text":"..."} ...],
    "utterance_role": "Therapist" | "Client",
    "utterance_text": "..."
    # optional gold annotations:
    # "gold_fine": ["OQ", "SR", ...],
    # "gold_coarse": ["QS", "RF", ...]
  }

Output:
  - Writes silver annotations into each item:
      "silver_fine": [...], "silver_coarse": [...]
  - Saves JSONL to `save_path`
"""

from __future__ import annotations
import json
import os
import re
import time
import math
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable, Optional

import requests
from dotenv import load_dotenv
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ----------------------------
# Environment & logging
# ----------------------------

load_dotenv()

SEA_LION_API_KEY = os.getenv("SEA_LION_API_KEY") or ""
SEA_LION_BASE_URL = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
SEA_LION_MODEL = os.getenv("SEA_LION_MODEL", "aisingapore/Gemma-SEA-LION-v4-27B-IT")

if not SEA_LION_API_KEY:
    raise ValueError("Missing SEA_LION_API_KEY in environment/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("bimisc")

# ----------------------------
# MISC definitions (BiMISC + MISC 2.5 extended)
# ----------------------------

# -------- MISC decoding policy (production) --------
THRESHOLD = 0.60           # main decision boundary
BACKOFF_THRESHOLD = 0.40   # if nothing crosses THRESHOLD, allow top-1 if >= this
MAX_CODES_PER_UTT = 1      # MISC gold is 1 code/utterance for scoring

# Optional per-code thresholds (override the global; tweak later if needed)
PER_CODE_THRESHOLDS = {
    "ADW": 0.70, "RCW": 0.70, "CO": 0.65, "WA": 0.60,   # high cost of FP
    "CR": 0.55, "RF": 0.65, "ADP": 0.60, "RCP": 0.60,   # trickier semantics
    "FA": 0.50, "FI": 0.50, "ST": 0.50, "OQ": 0.55,     # easy stuff
    "CQ": 0.65,
}

# Accept BiMISC-era aliases from the model and normalize to MISC 2.5
ALIAS_MAP = {
    "SP": "SU",
    "STR": "ST",
    "WAR": "WA",
    "PS": "EC",
    "OP": "GI",
    "ASK": "FN",   # strict 2.5 folds client questions into FN
}

THERAPIST_CODES: Dict[str, str] = {
    "OQ": "Open Question",
    "CQ": "Closed Question",
    "SR": "Simple Reflection",
    "CR": "Complex Reflection",
    "ADP": "Advise with Permission",
    "ADW": "Advise without Permission",
    "AF": "Affirm",
    "CO": "Confront",
    "DI": "Direct",
    "EC": "Emphasize Control",
    "FA": "Facilitate",
    "FI": "Filler",
    "GI": "Giving Information",
    "SU": "Support",
    "ST": "Structure",
    "WA": "Warn",
    "RCP": "Raise Concern with Permission",
    "RCW": "Raise Concern without Permission",
    "RF": "Reframe",
}

CLIENT_CODES: Dict[str, str] = {
    "FN": "Follow/Neutral",

    # Change talk (toward change)
    "CM+": "Commitment toward change",
    "TS+": "Taking step toward change",
    "R+": "Reason for change",
    "O+": "Other change-intent",

    # Sustain talk (against change)
    "CM-": "Commitment against change",
    "TS-": "Taking step against change",
    "R-": "Reason against change",
    "O-": "Other sustain-intent",
}


# AnnoMI coarse mapping (MISC 2.5 → AnnoMI)
FINE_TO_COARSE: Dict[str, str] = {
    # Therapist → QS (Questions)
    "OQ": "QS", "CQ": "QS",

    # Therapist → RF (Reflections family)
    "SR": "RF", "CR": "RF", "RF": "RF",   # Reframe groups with reflections per its function

    # Therapist → TI (all other interventions/information)
    "ADP": "TI", "ADW": "TI",
    "AF": "TI",
    "CO": "TI",
    "DI": "TI",
    "EC": "TI",
    "FA": "TI",
    "FI": "TI",
    "GI": "TI",
    "SU": "TI",
    "ST": "TI",
    "WA": "TI",
    "RCP": "TI", "RCW": "TI",
    # No PS/OP in MISC 2.5; permission-seeking is EC, "opinions" without advice are GI. :contentReference[oaicite:1]{index=1}
    
    # Client → NT / CT / ST
    "FN": "NT",  # In MISC 2.5, client questions fall under FN → NT. :contentReference[oaicite:2]{index=2}
    "ASK": "NT", # If you keep this BiMISC convenience code, collapse to NT.
    "CM+": "CT", "TS+": "CT", "R+": "CT", "O+": "CT",
    "CM-": "ST", "TS-": "ST", "R-": "ST", "O-": "ST",
}

# ----------------------------
# Notes:
# ----------------------------
# - This schema follows MISC 2.5 (Houck et al., 2010 update) exactly:contentReference[oaicite:2]{index=2}.
# - BiMISC simplifies some categories:
#     • ADV = ADP + ADW
#     • SP = SU
#     • STR = ST
#     • Drops CO, RCP, RCW, RF
# - If your target is AnnoMI (QS, RF, TI, NT, CT, ST), BiMISC mapping is sufficient.
# - If you want strict gold-standard MISC 2.5 coding, you must use this full set.


# Minimal, role-specific examples (two per code)
    # Therapist examples: list of (lhs, rhs) where lhs includes "Client: ...\nTherapist:"
    # Client examples: list of plain strings
EXAMPLES = {
    "THERAPIST": {
        # Open Question: invites elaboration, not answerable with yes/no
        "OQ": [
            ("Client: I think I should cut down.\nTherapist:", "What makes cutting down important to you right now?"),
            ("Client: I'm torn about my meds.\nTherapist:", "How are you weighing the pros and cons of taking them?"),
            ("Client: I'm so pissed at myself right now.\nTherapist:", "Can you tell me more?")
        ],

        # Closed Question: seeks specific fact, yes/no, or detail
        "CQ": [
            ("Client: I missed my meds.\nTherapist:", "Did you miss them yesterday?"),
            ("Client: I might go tomorrow.\nTherapist:", "Will you go tomorrow?"),
        ],

        # Simple Reflection: repeats/rephrases client, adds little new meaning
        "SR": [
            ("Client: I'm overwhelmed.\nTherapist:", "You're feeling swamped by all this."),
            ("Client: It's been a lot lately.\nTherapist:", "It's been heavy and nonstop for you."),
        ],

        # Complex Reflection: adds significant meaning, emotion, or new framing
        "CR": [
            ("Client: Work drains me.\nTherapist:", "The stress at work is leaving you exhausted and irritable."),
            ("Client: I fail every time.\nTherapist:", "Each setback has been chipping away at your confidence."),
        ],

        # Advise with Permission (ADP): gives advice after asking or when client invites it
        "ADP": [
            ("Client: Could you suggest something?\nTherapist:", "You could try a 10-minute walk after dinner to get started."),
            ("Client: Is there a way to sleep better?\nTherapist:", "You might keep a fixed bedtime and avoid screens before bed."),
        ],

        # Advise without Permission (ADW): gives advice without first asking or invitation
        "ADW": [
            ("Client: My sleep is a mess.\nTherapist:", "You should start a sleep schedule and cut caffeine after noon."),
            ("Client: I have been stressed lately.\nTherapist:", "You could join a mindfulness class this week."),
        ],

        # Affirm: compliments, expresses confidence, or appreciates effort
        "AF": [
            ("Client: I booked an appointment.\nTherapist:", "That took initiative. Nice work."),
            ("Client: I told my partner.\nTherapist:", "That was brave and constructive."),
        ],

        # Confront: disagrees, criticizes, shames, judges, or argues
        "CO": [
            ("Client: I looked for a job this week.\nTherapist:", "Sure you did. Right."),
            ("Client: I don't think alcohol is a problem.\nTherapist:", "So you think there's nothing wrong at all?"),
        ],

        # Direct: commands or imperative language
        "DI": [
            ("Client: I keep skipping doses.\nTherapist:", "Set an alarm and take it tonight."),
            ("Client: I can't decide.\nTherapist:", "Call your clinic today."),
        ],

        # Emphasize Control: underscores client's autonomy, includes permission-seeking
        "EC": [
            ("Client: I'm unsure.\nTherapist:", "It's your call how you want to proceed."),
            ("Client: I don't like being told.\nTherapist:", "You're in charge, we'll go at your pace."),
            ("Client: Not sure about advice.\nTherapist:", "Is it okay if I share a suggestion?"),
        ],

        # Facilitate: short encouragers or backchannels ("mm-hmm", "okay")
        "FA": [
            ("Client: ...\nTherapist:", "Mm-hmm."),
            ("Client: I don't know.\nTherapist:", "Okay."),
        ],

        # Filler: small talk or pleasantries, not substantive
        "FI": [
            ("Therapist:", "Good morning."),
            ("Therapist:", "Nice to see you."),
        ],

        # Giving Information: factual, explanatory, or feedback statements
        "GI": [
            ("Client: What does this med do?\nTherapist:", "It lowers inflammation and pain."),
            ("Client: How often should I take it?\nTherapist:", "Once daily with food."),
        ],

        # Support: sympathetic or compassionate statements ("hug" not "praise")
        "SU": [
            ("Client: I feel alone.\nTherapist:", "That sounds really hard. I'm with you in this."),
            ("Client: I'm scared to slip.\nTherapist:", "It makes sense you'd feel worried about that."),
        ],

        # Structure: tells client what will happen in session, transitions topics
        "ST": [
            ("Therapist:", "First we'll review your week, then plan next steps."),
            ("Therapist:", "Let's switch to goals, then barriers, then actions."),
        ],

        # Warn: threat or prediction of negative consequence
        "WA": [
            ("Therapist:", "If you keep skipping insulin, you could end up hospitalized."),
            ("Therapist:", "Driving after drinking puts you at real risk of losing your license."),
        ],

        # Raise Concern with Permission (RCP): names a concern after asking or being invited
        "RCP": [
            ("Client: What do you think of that plan?\nTherapist:", "I'm concerned it might put you near old triggers."),
            ("Client: Is there anything I'm missing?\nTherapist:", "I'm a bit worried moving back could make staying sober harder."),
        ],

        # Raise Concern without Permission (RCW): expresses a concern without asking first
        "RCW": [
            ("Client: I'll hang with the same crowd.\nTherapist:", "I'm concerned that could pull you back into using."),
            ("Client: I'll just skip the dose if I forget.\nTherapist:", "That worries me given your recent symptoms."),
        ],

        # Reframe: changes the meaning or emotional valence of client's statement
        "RF": [
            ("Client: My husband keeps nagging me about meds.\nTherapist:", "He sounds really concerned about your health."),
            ("Client: I failed again.\nTherapist:", "Each attempt has taught you something you're using now."),
        ],
    },

    "CLIENT": {
        # Follow/Neutral: neutral info, history, or off-target statements
        "FN": ["Yeah.", "Okay.", "I usually drink 4–5 days a week.", "Mmm"],

        # Commitment to change (+) or sustain (–)
        "CM+": ["I'll cut down to two drinks tonight.", "I'm going to start tomorrow.", "I'll try."],
        "CM-": ["I won't commit to that right now.", "I'm not planning to stop."],

        # Taking steps toward change (+) or against change (–)
        "TS+": ["I tossed out my cigarettes yesterday.", "I set up my pillbox today."],
        "TS-": ["I bought another pack this morning.", "I skipped the appointment again."],

        # Reason for change (+) or reason against (–)
        "R+": ["It would help my kids if I quit.", "I want my energy back."],
        "R-": ["I need the drinks to sleep.", "It's the only way I relax."],

        # Other change intent (+) or sustain intent (–)
        "O+": ["I'm ready to change.", "This time I'm serious."],
        "O-": ["I'm not changing anything.", "This is just who I am."],
    },
}



# ----------------------------
# Prompt builder
# ----------------------------

def build_prompt(
    role: str,
    history: List[Tuple[str, str]],
    utterance_role: str,
    utterance_text: str,
    misc_manual: Dict[str, str],
    examples: Dict[str, List],
    history_window: int = 6,
) -> str:
    assert role in ("THERAPIST", "CLIENT") # Check dataset
    role_header = "Therapist" if role == "THERAPIST" else "Client"

    manual_lines = [f"- {code}: {desc}" for code, desc in misc_manual.items()]

    ex_lines: List[str] = []
    for code, pairs in examples.items():
        for ex in pairs[:2]:
            if role == "THERAPIST":
                lhs, rhs = ex  # tuple
                ex_lines.append(f"{code}:\n{lhs} {rhs}")
            else:
                text = ex if isinstance(ex, str) else (ex[0] if ex else "")
                ex_lines.append(f"{code}:\nClient: {text}")

    # Trim context
    hist = history[-history_window:] if history_window > 0 else history
    history_lines = [f"{r}: {t}" for r, t in hist]

    allowed = list(misc_manual.keys())

    json_guard = (
        "Return ONLY valid minified JSON. Do not include prose, preambles, or code fences."
    )

    return f"""You are performing Motivational Interviewing behavioral coding (MISC) for the last utterance.

Role to classify: {role_header}

MISC manual for {role_header}:
{chr(10).join(manual_lines)}

MISC examples for {role_header}:
{chr(10).join(ex_lines)}

Historical conversation (most recent last):
{chr(10).join(history_lines)}

Utterance for classification:
{utterance_role}: {utterance_text}

Task:
Identify ALL applicable fine-grained MISC codes for this utterance strictly from {allowed}.
Respond only in JSON with:
{{"codes":[{{"code":"<MISC>","confidence":<0..1>}},...],"notes":"<brief justification>"}}
Only include a code if confidence >= 0.50. Use calibrated confidence, not random.

{json_guard}
"""

# ----------------------------
# SEA-LION API helpers
# ----------------------------

def _format_messages(task_prompt: str) -> List[Dict[str, str]]:
    # System defines output discipline, user carries the concrete task
    return [
        {"role": "system", "content": "You are a strict grader that outputs only JSON."},
        {"role": "user", "content": task_prompt},
    ]

def _extract_first_json_blob(text: str) -> str:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{(?:[^{}]|(?R))*\}", s)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:200]}...")
    return m.group(0)

def _generate_response(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: int = 45,
    max_retries: int = 6,
) -> str: # type: ignore
    headers = {
        "Authorization": f"Bearer {SEA_LION_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    base = 1.2
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{SEA_LION_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == max_retries - 1:
                    resp.raise_for_status()
                sleep_s = (base ** attempt) * (1.0 + random.random() * 0.3)
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            content = (choices[0].get("message") or {}).get("content") or ""
            if not content.strip():
                raise ValueError("Empty content from model")
            return content
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            sleep_s = (base ** attempt) * (1.0 + random.random() * 0.3)
            time.sleep(sleep_s)

def call_llm(prompt: str, model: Optional[str] = None, temperature: float = 0.0) -> Dict[str, Any]:
    model = model or SEA_LION_MODEL
    messages = _format_messages(prompt)
    raw = _generate_response(messages, model=model, temperature=temperature)
    blob = _extract_first_json_blob(raw)
    data = json.loads(blob)

    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")

    codes = data.get("codes", [])
    if not isinstance(codes, list):
        raise ValueError("`codes` must be a list")

    norm = []
    for item in codes:
        if isinstance(item, dict) and "code" in item:
            code = str(item["code"]).strip()
            conf = float(item.get("confidence", 0))
            norm.append({"code": code, "confidence": conf})
    data["codes"] = norm

    data["notes"] = data.get("notes", "") if isinstance(data.get("notes", ""), str) else ""
    return data

# ----------------------------
# Multi-label decoding & mapping
# ----------------------------

def _norm_code(c: str) -> str:
    c = (c or "").strip().upper()
    return ALIAS_MAP.get(c, c)

# Can optionally get custom treshold
def _select_codes(
    llm_json: dict,
    allowed: set[str],
    *,
    max_k: int = MAX_CODES_PER_UTT,
    threshold: float = THRESHOLD,
    backoff: float = BACKOFF_THRESHOLD,
    per_code: dict[str, float] = PER_CODE_THRESHOLDS,
) -> list[str]:
    """Normalize -> threshold (with per-code overrides) -> pick top-k by confidence -> optional backoff."""
    raw = llm_json.get("codes", []) or []
    scored = []
    for it in raw:
        code = _norm_code(str(it.get("code", "")))
        if code and (not allowed or code in allowed):
            conf = float(it.get("confidence", 0.0))
            cut = per_code.get(code, threshold)
            if conf >= cut:
                scored.append((code, conf))

    # Sort by confidence desc, then by code for stability
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

    # Keep unique codes only
    seen = set()
    picked = []
    for code, conf in scored:
        if code not in seen:
            picked.append((code, conf))
            seen.add(code)
        if len(picked) >= max_k:
            break

    # Backoff: if nothing selected but there exists a candidate above backoff, take the best one
    if not picked and raw:
        best = max((( _norm_code(str(it.get("code",""))), float(it.get("confidence",0.0)) )
                   for it in raw if _norm_code(str(it.get("code",""))) in allowed),
                   key=lambda t: t[1], default=None)
        if best and best[1] >= backoff:
            picked = [best]

    return [c for c, _ in picked]

def decode_codes(llm_json: Dict[str, Any], allowed: Iterable[str]) -> List[str]:
    allowed_set = set(allowed)
    return _select_codes(llm_json, allowed_set)

def map_to_coarse(fine_codes: Iterable[str]) -> List[str]:
    return sorted(set(FINE_TO_COARSE[c] for c in fine_codes if c in FINE_TO_COARSE))

# ----------------------------
# Metrics (multi-label)
# ----------------------------

@dataclass
class Scores:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

def multilabel_scores(y_true: List[List[str]], y_pred: List[List[str]], label_set: List[str]) -> Scores:
    eps = 1e-9
    from collections import Counter
    tp, fp, fn = Counter(), Counter(), Counter()

    for true_labels, pred_labels in zip(y_true, y_pred):
        t, p = set(true_labels), set(pred_labels)
        for lab in label_set:
            if lab in p and lab in t:
                tp[lab] += 1
            elif lab in p and lab not in t:
                fp[lab] += 1
            elif lab not in p and lab in t:
                fn[lab] += 1

    precs, recs, f1s = [], [], []
    for lab in label_set:
        prec = tp[lab] / (tp[lab] + fp[lab] + eps)
        rec = tp[lab] / (tp[lab] + fn[lab] + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        precs.append(prec); recs.append(rec); f1s.append(f1)

    exact = sum(1 for t, p in zip(y_true, y_pred) if set(t) == set(p)) / max(len(y_true), 1)

    return Scores(
        accuracy=exact,
        precision_macro=sum(precs) / len(precs),
        recall_macro=sum(recs) / len(recs),
        f1_macro=sum(f1s) / len(f1s),
    )

# ----------------------------
# Runner
# ----------------------------

def run_bimisc(
    jsonl_path: str,
    request_coarse: bool = True,
    limit: int | None = None,
    save_path: str | None = None,
    history_window: int = 6,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(jsonl_path).expanduser().resolve()
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if limit is not None and i >= limit:
                break
            items.append(json.loads(line))

    preds_fine: List[List[str]] = []
    preds_coarse: List[List[str]] = []

    # Use tqdm for progress bar
    for idx, ex_item in enumerate(tqdm(items, desc="Processing items", unit="item")):
        # Role gating per utterance
        utt_role_text = str(ex_item.get("utterance_role", "")).strip().lower()
        role_key = "THERAPIST" if utt_role_text.startswith("ther") else "CLIENT"

        manual = THERAPIST_CODES if role_key == "THERAPIST" else CLIENT_CODES
        examples = EXAMPLES[role_key]
        allowed_codes = list(manual.keys())

        history = [(h["role"], h["text"]) for h in ex_item.get("history", [])]
        utter_text = ex_item.get("utterance_text", "")

        prompt = build_prompt(
            role=role_key,
            history=history,
            utterance_role=ex_item.get("utterance_role", ""),
            utterance_text=utter_text,
            misc_manual=manual,
            examples=examples,
            history_window=history_window,
        )

        llm_json = call_llm(prompt, model=model or SEA_LION_MODEL, temperature=0.0)
        fine_codes = decode_codes(llm_json, allowed=allowed_codes)
        ex_item["silver_fine"] = fine_codes
        preds_fine.append(fine_codes)

        if request_coarse:
            coarse_codes = map_to_coarse(fine_codes)
            ex_item["silver_coarse"] = coarse_codes
            preds_coarse.append(coarse_codes)

    if save_path:
        out_path = Path(save_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info("Silver-standard dataset written to %s", str(out_path))

    return {
        "n": len(items),
        "threshold": THRESHOLD,
        "role": "AUTO",
        "model": model or SEA_LION_MODEL,
        "preds_fine": preds_fine,
        "preds_coarse": preds_coarse if request_coarse else None,
    }

# ----------------------------
# CLI entry
# ----------------------------

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = REPO_ROOT / "dataset" / "test.jsonl"
    OUT_PATH = REPO_ROOT / "dataset" / "test_silver.jsonl"

    log.info("Run config: %s", json.dumps({
        "model": SEA_LION_MODEL,
        "temperature": 0.0,
        "threshold": THRESHOLD,
        "backoff": BACKOFF_THRESHOLD,
        "max_codes_per_utt": MAX_CODES_PER_UTT,
        "history_window": 6,
        "base_url": SEA_LION_BASE_URL,
    }, ensure_ascii=False))

    out = run_bimisc(
        jsonl_path=str(DATA_PATH),
        request_coarse=True,
        limit=500,
        save_path=str(OUT_PATH),
        history_window=6,
        model=SEA_LION_MODEL,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))