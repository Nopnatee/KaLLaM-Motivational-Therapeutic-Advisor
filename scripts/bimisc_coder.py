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
# MISC definitions
# ----------------------------

THERAPIST_CODES: Dict[str, str] = {
    "OQ": "Open question",
    "CQ": "Closed question",
    "SR": "Simple reflection",
    "CR": "Complex reflection",
    "ADV": "Advice",
    "AFF": "Affirm",
    "DIR": "Direct",
    "EC": "Emphasize control",
    "FA": "Facilitate",
    "FIL": "Filler",
    "GI": "Giving information",
    "SP": "Support",
    "STR": "Structure",
    "WAR": "Warn",
    "PS": "Permission seeking",
    "OP": "Opinion",
}

CLIENT_CODES: Dict[str, str] = {
    "FN": "Follow/Neutral",
    "ASK": "Ask",
    "CM+": "Commitment toward change",
    "TS+": "Taking step toward change",
    "R+": "Reason for change",
    "O+": "Other change-intent",
    "CM-": "Commitment against change",
    "TS-": "Taking step against change",
    "R-": "Reason against change",
    "O-": "Other sustain-intent",
}

# AnnoMI coarse mapping
FINE_TO_COARSE: Dict[str, str] = {
    # Therapist
    "OQ": "QS", "CQ": "QS",
    "SR": "RF", "CR": "RF",
    "ADV": "TI", "AFF": "TI", "DIR": "TI", "EC": "TI", "FA": "TI", "FIL": "TI",
    "GI": "TI", "SP": "TI", "STR": "TI", "WAR": "TI", "PS": "TI", "OP": "TI",
    # Client
    "FN": "NT", "ASK": "NT",
    "CM+": "CT", "TS+": "CT", "R+": "CT", "O+": "CT",
    "CM-": "ST", "TS-": "ST", "R-": "ST", "O-": "ST",
}

# Minimal, role-specific examples (two per code)
    # Therapist examples: list of (lhs, rhs) where lhs includes "Client: ...\nTherapist:"
    # Client examples: list of plain strings
EXAMPLES = {
    "THERAPIST": {
        "OQ": [
            ("Client: I think I should cut down.\nTherapist:", "What makes cutting down important now?"),
            ("Client: I'm torn about my meds.\nTherapist:", "What concerns you most about taking them?"),
        ],
        "CQ": [
            ("Client: I missed my meds.\nTherapist:", "Did you miss them yesterday?"),
            ("Client: I might go tomorrow.\nTherapist:", "Will you go tomorrow?"),
        ],
        "SR": [
            ("Client: I'm overwhelmed.\nTherapist:", "You’re feeling swamped by all this."),
            ("Client: It’s been a lot lately.\nTherapist:", "It’s been heavy and nonstop for you."),
        ],
        "CR": [
            ("Client: Work drains me.\nTherapist:", "The stress at work is leaving you exhausted and irritable."),
            ("Client: I fail every time.\nTherapist:", "It feels like each attempt chips away at your confidence."),
        ],
        "ADV": [
            ("Client: I want to manage stress.\nTherapist:", "Try a short daily walk to get started."),
            ("Client: My sleep is a mess.\nTherapist:", "Start with a consistent bedtime and no screens 30 minutes prior."),
        ],
        "AFF": [
            ("Client: I booked an appointment.\nTherapist:", "That took initiative. Good move."),
            ("Client: I told my partner.\nTherapist:", "That was brave and constructive."),
        ],
        "DIR": [
            ("Client: I keep skipping doses.\nTherapist:", "Set an alarm and take it tonight."),
            ("Client: I can’t decide.\nTherapist:", "Start with one small step today: call your clinic."),
        ],
        "EC": [
            ("Client: I’m unsure.\nTherapist:", "It’s your call how you want to proceed."),
            ("Client: I don’t like being told.\nTherapist:", "You’re in charge; we’ll follow your pace."),
        ],
        "FA": [
            ("Client: ...\nTherapist:", "Tell me more about that."),
            ("Client: I don’t know.\nTherapist:", "What comes to mind first about that situation?"),
        ],
        "FIL": [
            ("Therapist:", "Good morning!"),
            ("Therapist:", "Mhm, I see."),
        ],
        "GI": [
            ("Client: What does this med do?\nTherapist:", "It lowers inflammation and pain."),
            ("Client: How often should I take it?\nTherapist:", "Take it once daily with food."),
        ],
        "SP": [
            ("Client: I feel alone.\nTherapist:", "I’m here to support you as we sort this out."),
            ("Client: I’m scared to slip.\nTherapist:", "You’re not doing this alone; we’ll plan safety nets."),
        ],
        "STR": [
            ("Therapist:", "First we’ll review your week, then plan next steps."),
            ("Therapist:", "Let’s start with goals, then barriers, then actions."),
        ],
        "WAR": [
            ("Therapist:", "Uncontrolled sugar can damage your vision."),
            ("Therapist:", "Smoking increases your risk of heart disease."),
        ],
        "PS": [
            ("Therapist:", "Would it be okay if I share a tip on sleep?"),
            ("Therapist:", "Can I offer a suggestion on planning your doses?"),
        ],
        "OP": [
            ("Therapist:", "In my view, pacing yourself could help."),
            ("Therapist:", "I think tracking triggers would be useful."),
        ],
    },
    "CLIENT": {
        "FN": ["Yeah.", "Hmm, okay."],
        "ASK": ["What options do I have?", "Can you explain how that works?"],
        "CM+": ["I’ll cut down to two drinks tonight.", "I’m going to start tomorrow."],
        "TS+": ["I tossed out my cigarettes yesterday.", "I set up my pillbox today."],
        "R+": ["It would help my kids if I quit.", "I want my energy back."],
        "O+": ["I’m ready to change.", "This time I’m serious."],
        "CM-": ["I won’t commit to that right now.", "Not making promises."],
        "TS-": ["I bought another pack this morning.", "I skipped the appointment again."],
        "R-": ["I need the drinks to sleep.", "It’s the only way I relax."],
        "O-": ["I’m not changing anything.", "This is just who I am."],
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
    request_coarse: bool = True,
    history_window: int = 6,
) -> str:
    assert role in ("THERAPIST", "CLIENT")
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
    coarse_note = ""
    if request_coarse:
        coarse_note = (
            "\nAdditionally map selected fine-grained codes to the coarse set "
            "{'QS','RF','TI','NT','CT','ST'} using the known mapping."
        )

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
Only include a code if confidence >= 0.50. Use calibrated confidence, not random.{coarse_note}

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
) -> str:
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

def decode_codes(llm_json: Dict[str, Any], threshold: float = 0.50, allowed: Optional[Iterable[str]] = None) -> List[str]:
    allowed_set = set(allowed or [])
    out: List[str] = []
    for item in llm_json.get("codes", []):
        code = str(item.get("code", "")).strip()
        conf = float(item.get("confidence", 0))
        if (not allowed_set or code in allowed_set) and conf >= threshold:
            out.append(code)
    return sorted(set(out))

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
    role: str,
    request_coarse: bool = True,
    threshold: float = 0.50,
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

    role_key = "THERAPIST" if role.upper().startswith("THER") else "CLIENT"
    manual = THERAPIST_CODES if role_key == "THERAPIST" else CLIENT_CODES
    ex = EXAMPLES[role_key]
    allowed_codes = list(manual.keys())

    preds_fine: List[List[str]] = []
    preds_coarse: List[List[str]] = []

    for idx, ex_item in enumerate(items):
        history = [(h["role"], h["text"]) for h in ex_item.get("history", [])]
        utter_role = ex_item["utterance_role"]
        utter_text = ex_item["utterance_text"]

        prompt = build_prompt(
            role=role_key,
            history=history,
            utterance_role=utter_role,
            utterance_text=utter_text,
            misc_manual=manual,
            examples=ex,
            request_coarse=request_coarse,
            history_window=history_window,
        )

        llm_json = call_llm(prompt, model=model or SEA_LION_MODEL, temperature=0.0)
        fine_codes = decode_codes(llm_json, threshold=threshold, allowed=allowed_codes)
        preds_fine.append(fine_codes)
        ex_item["silver_fine"] = fine_codes

        if request_coarse:
            coarse_codes = map_to_coarse(fine_codes)
            preds_coarse.append(coarse_codes)
            ex_item["silver_coarse"] = coarse_codes

        if (idx + 1) % 50 == 0:
            log.info("Processed %d items...", idx + 1)

    if save_path:
        out_path = Path(save_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info("Silver-standard dataset written to %s", str(out_path))

    return {
        "n": len(items),
        "threshold": threshold,
        "role": role_key,
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
        "threshold": 0.50,
        "history_window": 6,
        "base_url": SEA_LION_BASE_URL,
    }, ensure_ascii=False))

    out = run_bimisc(
        jsonl_path=str(DATA_PATH),
        role="THERAPIST",
        request_coarse=True,
        threshold=0.50,
        limit=1,            # set an integer for smoke tests, e.g., 100
        save_path=str(OUT_PATH),
        history_window=6,
        model=SEA_LION_MODEL,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
