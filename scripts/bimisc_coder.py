# -*- coding: utf-8 -*-
"""
BiMISC-style coding pipeline.

What this mirrors from the paper:
- Prompt template with: task instruction + MISC manual + examples + brief history
- Zero temperature deterministic decoding
- Multi-label outputs with a simple confidence gate
- Fine-grained MISC codes + optional mapping to AnnoMI coarse codes
- Metrics: Accuracy, Precision, Recall, Macro-F1 (multi-label)

Swap `call_llm` with your actual client (OpenAI, Bedrock, SEA-Lion, whatever).
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable
from collections import defaultdict, Counter
import math
import random

# ----------------------------
# Fine-grained MISC codes (subset names and short descs per paper Table 2)
# ----------------------------

THERAPIST_CODES = {
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

CLIENT_CODES = {
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

# Mapping to coarse AnnoMI buckets (paper Table 3)
FINE_TO_COARSE = {
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

# Minimal examples (paper used two examples per code; keep placeholders concise)
EXAMPLES = {
    "THERAPIST": {
        "OQ": [("Client: I think I should cut down.\nTherapist:", "What makes cutting down important now?")],
        "CQ": [("Client: I missed my meds.\nTherapist:", "Did you miss them yesterday?")],
        "SR": [("Client: I'm overwhelmed.\nTherapist:", "You’re feeling swamped by all this.")],
        "CR": [("Client: Work drains me.\nTherapist:", "The stress at work is leaving you exhausted and irritable.")],
        "ADV": [("Client: I want to manage stress.\nTherapist:", "Try a short daily walk to get started.")],
        "AFF": [("Client: I booked an appointment.\nTherapist:", "That took initiative. Good move.")],
        "DIR": [("Client: I keep skipping doses.\nTherapist:", "Set an alarm and take it tonight.")],
        "EC":  [("Client: I’m unsure.\nTherapist:", "It’s your call how you want to proceed.")],
        "FA":  [("Client: …\nTherapist:", "Tell me more about that.")],
        "FIL": [("Therapist:", "Good morning!")],
        "GI":  [("Client: What does this med do?\nTherapist:", "It lowers inflammation and pain.")],
        "SP":  [("Client: I feel alone.\nTherapist:", "I’m here to support you as we sort this out.")],
        "STR": [("Therapist:", "First we’ll review your week, then plan next steps.")],
        "WAR": [("Therapist:", "Uncontrolled sugar can damage your vision.")],
        "PS":  [("Therapist:", "Would it be okay if I share a tip on sleep?")],
        "OP":  [("Therapist:", "In my view, pacing yourself could help.")],
    },
    "CLIENT": {
        "FN":  [("Client:", "Yeah."), ("Client:", "Hmm, okay.")],
        "ASK": [("Client:", "What options do I have?")],
        "CM+": [("Client:", "I’ll cut down to two drinks tonight.")],
        "TS+": [("Client:", "I tossed out my cigarettes yesterday.")],
        "R+":  [("Client:", "It would help my kids if I quit.")],
        "O+":  [("Client:", "I’m ready to change.")],
        "CM-": [("Client:", "I won’t commit to that right now.")],
        "TS-": [("Client:", "I bought another pack this morning.")],
        "R-":  [("Client:", "I need the drinks to sleep.")],
        "O-":  [("Client:", "I’m not changing anything.")],
    },
}

# ----------------------------
# Prompt builder (matches paper structure)
# ----------------------------

def build_prompt(
    role: str,
    history: List[Tuple[str, str]],
    utterance_role: str,
    utterance_text: str,
    misc_manual: Dict[str, str],
    examples: Dict[str, List[Tuple[str, str]]],
    request_coarse: bool = True,
) -> str:
    assert role in ("THERAPIST", "CLIENT")
    role_header = "Therapist" if role == "THERAPIST" else "Client"

    manual_lines = []
    for code, desc in misc_manual.items():
        manual_lines.append(f"- {code}: {desc}")

    ex_lines = []
    for code, pairs in examples.items():
        for (lhs, rhs) in pairs[:2]:  # up to 2 examples like the paper
            if role == "THERAPIST":
                ex_lines.append(f"{code}:\n{lhs} {rhs}")
            else:
                # client examples are just client lines
                ex_lines.append(f"{code}:\n{lhs}")

    history_lines = []
    for r, t in history[-6:]:  # brief context, trim to last 6
        history_lines.append(f"{r}: {t}")

    allowed = list(misc_manual.keys())
    coarse_note = ""
    if request_coarse:
        coarse_note = (
            "\nAdditionally map selected fine-grained codes to the coarse set "
            "{'QS','RF','TI','NT','CT','ST'} using the known mapping."
        )

    # Deterministic, structured JSON output to reduce parsing drama
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
Respond in JSON with:
{{
  "codes": [{{"code": "<MISC>", "confidence": <0..1>}}, ...],
  "notes": "<brief justification>"
}}
Only include a code if confidence >= 0.50. Use calibrated confidence, not random.{coarse_note}
"""

# ----------------------------
# Dummy LLM call (swap this)
# ----------------------------

def call_llm(prompt: str, model: str = "dummy-llm", temperature: float = 0.0) -> Dict[str, Any]:
    """
    Replace this with your real client call. Keep temperature=0 for reproducibility (paper).
    Expected return: dict with 'codes' as list of {code, confidence}.
    """
    # Extremely naive heuristic for the demo; replace immediately in real use.
    text = prompt.lower()
    picked = []
    def add(code, conf):
        picked.append({"code": code, "confidence": conf})

    # Toy rules
    if "?" in text or "what" in text or "how" in text:
        add("OQ", 0.78)
    if "did you" in text or "did they" in text:
        add("CQ", 0.74)
    if "tell me more" in text:
        add("FA", 0.72)
    if "it’s your call" in text or "your choice" in text:
        add("EC", 0.66)
    if "good job" in text or "well done" in text:
        add("AFF", 0.7)
    if "client:" in prompt.split("Utterance for classification:")[-1].lower():
        # pretend client final utterance
        if "yeah" in text or "okay" in text or "hmm" in text:
            add("FN", 0.8)
        if "i will" in text or "i'll" in text:
            add("CM+", 0.75)
        if "i bought" in text or "i smoked" in text:
            add("TS-", 0.76)

    return {"codes": picked, "notes": "dummy"}

# ----------------------------
# Multi-label thresholding and mapping
# ----------------------------

def decode_codes(llm_json: Dict[str, Any], threshold: float = 0.50) -> List[str]:
    out = []
    for item in llm_json.get("codes", []):
        if float(item.get("confidence", 0)) >= threshold:
            out.append(item["code"])
    # Deduplicate
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
    # Per-label precision/recall/F1 then macro average
    eps = 1e-9
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for true_labels, pred_labels in zip(y_true, y_pred):
        t = set(true_labels)
        p = set(pred_labels)
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

    # Subset accuracy (exact set match)
    exact = sum(1 for t, p in zip(y_true, y_pred) if set(t) == set(p)) / max(len(y_true), 1)

    return Scores(accuracy=exact,
                  precision_macro=sum(precs)/len(precs),
                  recall_macro=sum(recs)/len(recs),
                  f1_macro=sum(f1s)/len(f1s))

# ----------------------------
# Runner
# ----------------------------

def run_bimisc(jsonl_path: str,
               role: str,
               request_coarse: bool = True,
               threshold: float = 0.50) -> Dict[str, Any]:
    """
    jsonl schema expectation per item:
    {
      "history": [{"role": "Therapist"|"Client", "text": "..."}],  # optional, short window
      "utterance_role": "Therapist"|"Client",
      "utterance_text": "...",
      "gold_fine": ["OQ","SR",...],           # optional but needed for metrics
      "gold_coarse": ["QS","RF",...]          # optional
    }
    """

    path = Path(jsonl_path)
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    role_key = "THERAPIST" if role.upper().startswith("THER") else "CLIENT"
    manual = THERAPIST_CODES if role_key == "THERAPIST" else CLIENT_CODES
    ex = EXAMPLES[role_key]

    preds_fine, golds_fine = [], []
    preds_coarse, golds_coarse = [], []

    for ex_item in items:
        history = [(h["role"], h["text"]) for h in ex_item.get("history", [])]
        utter_role = ex_item["utterance_role"]
        utter_text = ex_item["utterance_text"]

        prompt = build_prompt(role=role_key,
                              history=history,
                              utterance_role=utter_role,
                              utterance_text=utter_text,
                              misc_manual=manual,
                              examples=ex,
                              request_coarse=request_coarse)

        llm_json = call_llm(prompt, model="dummy-llm", temperature=0.0)
        fine_codes = decode_codes(llm_json, threshold=threshold)
        preds_fine.append(fine_codes)

        if "gold_fine" in ex_item:
            golds_fine.append(ex_item["gold_fine"])

        if request_coarse:
            preds_coarse.append(map_to_coarse(fine_codes))
            if "gold_coarse" in ex_item:
                golds_coarse.append(ex_item["gold_coarse"])

    results = {"n": len(items), "threshold": threshold, "role": role_key, "preds_fine": preds_fine}

    # Metrics if gold present
    if golds_fine:
        labels_fine = sorted(manual.keys())
        results["metrics_fine"] = multilabel_scores(golds_fine, preds_fine, labels_fine).__dict__

    if request_coarse and golds_coarse:
        labels_coarse = sorted(set(FINE_TO_COARSE[c] for c in FINE_TO_COARSE))
        results["metrics_coarse"] = multilabel_scores(golds_coarse, preds_coarse, labels_coarse).__dict__

    return results

if __name__ == "__main__":
    # Example run with your uploaded file
    # Replace role with "CLIENT" to evaluate client codes.
    out = run_bimisc(jsonl_path="../dataset/test.jsonl", role="THERAPIST", request_coarse=True, threshold=0.50)
    print(json.dumps(out, ensure_ascii=False, indent=2))
