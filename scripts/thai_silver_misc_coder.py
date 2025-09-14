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
    "OQ": "การใช้คำถามปลายเปิด",
    "CQ": "การใช้คำถามปลายปิด",
    "SR": "การสะท้อนอย่างเรียบง่าย",
    "CR": "การสะท้อนอย่างซับซ้อน",
    "ADP": "การให้คำแนะนำโดยได้รับอนุญาติ",
    "ADW": "การให้คำแนะนำโดยไม่ได้รับอนุญาติ",
    "AF": "การยืนยัน",
    "CO": "การประจันหน้า",
    "DI": "การตรงไปตรงมา",
    "EC": "การเน้นการควบคุม",
    "FA": "การอำนวยความสะดวก",
    "FI": "การใช้ประโยคตัวเติม",
    "GI": "การให้ข้อมูล",
    "SU": "การสนับสนุน",
    "ST": "การอยู่ในโครงสร้าง",
    "WA": "การเตือน",
    "RCP": "การเตือนอย่างได้รับอนุญาติ",
    "RCW": "การเตือนอย่างไม่ได้รับอนุญาติ",
    "RF": "การเปลี่ยนมุมมอง",
}

CLIENT_CODES: Dict[str, str] = {
    "FN": "การตามบทสนทนา",

    # Change talk (toward change)
    "CM+": "การลงมือในทางที่ดี",
    "TS+": "การพูดสู่ทางที่ดี",
    "R+": "การให้เหตุผลในการเปลี่ยนแปลงสู่ทางที่ดี",
    "O+": "การแสดงเจตนาในการเปลี่ยนแปลงที่ดีอื่นๆ",

    # Sustain talk (against change)
    "CM-": "การลงมือในทางที่ไม่ดี",
    "TS-": "การพูดสู่ทางที่ไม่ดี",
    "R-": "การให้เหตุผลในการเปลี่ยนแปลงสู่ทางที่ไม่ดี",
    "O-": "การแสดงเจตนาในการเปลี่ยนแปลงที่ไม่ดีอื่นๆ",
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
            ("Client: ผมว่าผมควรลดมันดูบ้าง\nTherapist:", "อะไรคือสิ่งที่ทำให้คุณคิดว่าการลดนั้นสำคัญ?"),
            ("Client: ชั้นเบื่อยาแล้ว\nTherapist:", "คุณคิดว่าข้อดีและข้อเสียของการกินยาคืออะไร?"),
            ("Client: ผมโคตรจะเบื่อตัวเองเลยหว่ะ.\nTherapist:", "เพราะอะไรหรอครับ?")
        ],

        # Closed Question: seeks specific fact, yes/no, or detail
        "CQ": [
            ("Client: ฉันลืมกินยา\nTherapist:", "คุณลืมกินเมื่อวานหรอคะ?"),
            ("Client: ผมอาจจะไปพรุ่งนี้\nTherapist:", "คุณจะไปพรุ่งนี้หรอครับ?"),
        ],

        # Simple Reflection: repeats/rephrases client, adds little new meaning
        "SR": [
            ("Client: ฉันรู้สึกเหนื่อย\nTherapist:", "คุณกำลังรู้สึกหนักอึ้งกับทุกสิ่งที่เกิดขึ้น"),
            ("Client: ที่ผ่านมามีเรื่องเยอะมาก\nTherapist:", "มันมีอะไรมากมายอย่างไม่หยุดหย่อนโถมเข้ามาหาคุณ"),
        ],

        # Complex Reflection: adds significant meaning, emotion, or new framing
        "CR": [
            ("Client: งานทำให้ชั้นเหนื่อย\nTherapist:", "ความเครียดในการทำงานทำให้คุณรู้สึกไม่เป็นตัวเอง"),
            ("Client: ผมล้มเหลวตลอด\nTherapist:", "ทุกความผิดพลาดค่อยๆกัดกินความมั่นใจของคุณ"),
        ],

        # Advise with Permission (ADP): gives advice after asking or when client invites it
        "ADP": [
            ("Client: ช่วยผมที\nTherapist:", "คุณลองไปเดินเล่นซัก 10 นาทีดูมั้ยหล่ะ"),
            ("Client: มีวิธีช่วยให้นอนง่ายขึ้นมั้ย?\nTherapist:", "คุณลองนอนเวลาเดิมและไม่ดูจอก่อนนอนครับ"),
        ],

        # Advise without Permission (ADW): gives advice without first asking or invitation
        "ADW": [
            ("Client: ผมนอนมั่วมาก\nTherapist:", "คุณควรปรับเวลานอนให้เป็นระบบนะครับ"),
            ("Client: ชั้นเครียดมากในช่วงนี้\nTherapist:", "คุณลองไปเข้ากิจกรรมผ่อนคลายต่างๆสิ"),
        ],

        # Affirm: compliments, expresses confidence, or appreciates effort
        "AF": [
            ("Client: ฉันนัดหมอแล้ว\nTherapist:", "ดีเลยครับ"),
            ("Client: ผมบอกแฟนผมแล้ว\nTherapist:", "คุณกล้าหาญมากค่ะ"),
        ],

        # Confront: disagrees, criticizes, shames, judges, or argues
        "CO": [
            ("Client: ชั้นเพิ่งหางานมาเมื่อสัปดาห์ที่แล้ว\nTherapist:", "จริงหรอครับ"),
            ("Client: ผมไม่คิดว่าปัญหามันอยู่ที่เหล้า\nTherapist:", "งั้นคุณจะบอกว่าไม่มีอะไรที่เป็นปัญหาเลยหรอคะ"),
        ],

        # Direct: commands or imperative language
        "DI": [
            ("Client: ผมชอบลืมทานยา\nTherapist:", "ตั้งนาฬิกาปลุกแล้วกินคืนนี้"),
            ("Client: ฉันตัดสินใจไม่ได้\nTherapist:", "โทรหาคลินิกวันนี้"),
        ],

        # Emphasize Control: underscores client's autonomy, includes permission-seeking
        "EC": [
            ("Client: ชั้นไม่มั่นใจ\nTherapist:", "จริงๆ ก็ขึ้นอยู่กับคุณว่าจะทำอย่างไร"),
            ("Client: ฉันไม่ชอบให้ใครมาบอก\nTherapist:", "คุณคือคนนำ เราจะเอาอย่างที่คุณว่าละกัน"),
            ("Client: ผมไม่ค่อยมั่นใจในคำแนะนำของคุณ\nTherapist:", "ไม่เป็นไร แต่ผมขอแนะนำอะไรอีกอย่างได้ไหม"),
        ],

        # Facilitate: short encouragers or backchannels ("mm-hmm", "okay")
        "FA": [
            ("Client: ...\nTherapist:", "อืม"),
            ("Client: ผมไม่รู้\nTherapist:", "โอเค"),
        ],

        # Filler: small talk or pleasantries, not substantive
        "FI": [
            ("Therapist:", "สวัสดีค่ะ"),
            ("Therapist:", "ยินดีที่ได้พบคุณครับ"),
        ],

        # Giving Information: factual, explanatory, or feedback statements
        "GI": [
            ("Client: ยาตัวนี้เอาไว้ทำอะไร?\nTherapist:", "ช่วยในการลดปวดและการบวมค่ะ"),
            ("Client: ฉันควรกินอย่างไร\nTherapist:", "วันละครั้งหลังทานข้าวครับ"),
        ],

        # Support: sympathetic or compassionate statements ("hug" not "praise")
        "SU": [
            ("Client: ผมรู้สึกเหมือนอยู่คนเดียว\nTherapist:", "นั่นฟังดูไม่ค่อยดีเลย ผมจะคอยรับฟังคุณเองครับ"),
            ("Client: ชั้นกลัวที่จะล้มเหลว\nTherapist:", "จริงๆนั่นเป็นเรื่องปกติมากเลยค่ะ"),
        ],

        # Structure: tells client what will happen in session, transitions topics
        "ST": [
            ("Therapist:", "เราจะเริ่มจากการทบทวนสัปดาห์ที่แล้ว และวางแผนสำหรับครั้งนี้กันค่ะ"),
            ("Therapist:", "เราจะเปลี่ยนเป้าหมาย แล้วค่อยข้อจำกัด แล้วจึงลงมือกันครับ"),
        ],

        # Warn: threat or prediction of negative consequence
        "WA": [
            ("Therapist:", "ถ้าคุณยังไม่ทานยาอีก คุณจะได้เข้าโรงพยาบาลสักวันแน่"),
            ("Therapist:", "การขับรถหลังดื่มจะลงเอยด้วยการเจ็บตัวและเสียใบขับขี่"),
        ],

        # Raise Concern with Permission (RCP): names a concern after asking or being invited
        "RCP": [
            ("Client: คุณคิดว่าไง?\nTherapist:", "ฉันกลัวว่ามันจะทำให้คุณได้รับการกระตุ้น Trigger"),
            ("Client: ผมพลาดอะไรไปรึเปล่า?\nTherapist:", "ฉันกังวลนิดหน่อยว่าการกลับมาอาจทำให้การเลิกเหล้าเป็นเรื่องยากขึ้น"),
        ],

        # Raise Concern without Permission (RCW): expresses a concern without asking first
        "RCW": [
            ("Client: ฉันจะไปกับแก้งเดิม\nTherapist:", "ผมไม่คิดว่านั่นเป็นไอเดียที่ดีนะครับ"),
            ("Client: ผมจะข้ามถ้าผมลืมทานยา\nTherapist:", "นั่นทำให้ฉันกังวลเกี่ยวกับอาการของคุณค่ะ"),
        ],

        # Reframe: changes the meaning or emotional valence of client's statement
        "RF": [
            ("Client: ผัวชั้นเอาแต่บอกให้ชั้นกินยา\nTherapist:", "เขาฟังดูเป็นห่วงคุณมากเลยนะครับ"),
            ("Client: ผมล้มเหลวอีกแล้ว\nTherapist:", "ทุกความพยายามสอนบางอย่างที่คุณใช้อยู่ในปัจจุบัน"),
        ],
    },

    "CLIENT": {
        # Follow/Neutral: neutral info, history, or off-target statements
        "FN": ["ได้", "เคร", "ผมดื่มนานๆครั้ว", "อืม"],

        # Commitment to change (+) or sustain (–)
        "CM+": ["งั้นผมจะลองลดดูละกัน", "ฉันจะเริ่มพรุ่งนี้", "ชั้นจะลองดู"],
        "CM-": ["ผมจะไม่ทำอะไรตอนนี้", "ชั้นไม่คิดที่จะเลิก"],

        # Taking steps toward change (+) or against change (–)
        "TS+": ["ผมทิ้งบุหรี่เมื่อวาน", "ฉันจัดยาเมื่อวาน"],
        "TS-": ["ฉันเพิ่งซื้อบุหรี่มา", "ชั้นบิดหมอเมื่อสัปดาห์ที่แล้ว"],

        # Reason for change (+) or reason against (–)
        "R+": ["มันน่าจะช่วยลูกผม ถ้าผมเลิก", "ชั้นอยากมีพลังอีกครั้ง"],
        "R-": ["ฉันต้องดื่มเพื่อที่จะนอน", "นั้นเป็นทางผ่่อนคลายเดียว"],

        # Other change intent (+) or sustain intent (–)
        "O+": ["ผมพร้อมเปลี่ยนตัวเองแล้ว", "ถึงเวลาเอาจริงแล้ว"],
        "O-": ["ชั้นจะไปเปลี่ยนอะไร", "สันดานคนเรามันเปลี่ยนไม่ได้"],
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

    return f"""คุณกำลังทำการเข้ารหัสพฤติกรรมของการสัมภาษณ์เชิงสร้างแรงบันดาลใจ (MISC) สำหรับคำพูดสุดท้าย.

บทบาทในการจำแนกประเภท: {role_header}

คู่มือ MISC สำหรับ {role_header}:
{chr(10).join(manual_lines)}

คู่มือ MISC สำหรับ {role_header}:
{chr(10).join(ex_lines)}

ประวัติการสนทนา (สุดท้ายใหม่สุด):
{chr(10).join(history_lines)}

ถ้อยคำที่ต้องการการจำแนกประเภท:
{utterance_role}: {utterance_text}

งานที่ต้องทำ:
ระบุรหัส MISC โดยละเอียดทั้งหมดที่เกี่ยวข้องสำหรับคำพูดนี้อย่างเคร่งครัดจาก {allowed}.
ตอบด้วยโครงสร้าง JSON เท่านั้นโดยโครงสร้างที่กำหนดให้รวมถึงระบุค่าความมั่นใจในคำตอบ (confidence) ห้ามสุ่มขึ้นมา:
{{"codes":[{{"code":"<MISC>","confidence":<0..1>}},...],"notes":"<brief justification>"}}

{json_guard}
"""

# ----------------------------
# SEA-LION API helpers
# ----------------------------

def _format_messages(task_prompt: str) -> List[Dict[str, str]]:
    # System defines output discipline, user carries the concrete task
    return [
        {"role": "system", "content": "คุณคือผู้ตัดสินจำแนกที่เคร่งครัดและตอบสนองด้วย JSON เท่านั้น"},
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
    DATA_PATH = REPO_ROOT / "data" / "orchestrated" / "pre_annotate.jsonl"
    OUT_PATH = REPO_ROOT / "data" / "orchestrated" / "post_annotate.jsonl"

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