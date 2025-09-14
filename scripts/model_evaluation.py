#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_evaluation.py  (MISC 2.5-aligned)

Roll-up evaluator for MISC silver annotations with MISC 2.5-compatible metrics.

Input JSONL items (minimum):
{
  "utterance_role": "Therapist" | "Client",
  "silver_fine": ["OQ","SR",...],        # fine codes per utterance (list)
  "silver_coarse": ["QS","RF",...]       # optional
}

Outputs a JSON report with:
- Counselor metrics: R/Q, %OQ, %CR, reflections_per100, questions_per100, info_per100,
  %MI-consistent (MICO / (MICO + MIIN)), MICO_per100, MIIN_per100
- Client metrics: CT, ST, %CT
- Coverage: fine and coarse code counts

Compatibility:
- Accepts strict MISC 2.5 tags:
    OQ, CQ, SR, CR, RF, ADP, ADW, AF, CO, DI, EC, FA, FI, GI, SU, ST, WA, RCP, RCW
  and maps common BiMISC-era aliases:
    SP->SU, STR->ST, WAR->WA, PS->EC, OP->GI
  Note: legacy "ADV" is ambiguous; we do NOT auto-split into ADP/ADW.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Iterable

# ---------- Helper / config ----------

def _safe_list(x) -> List[str]:
    return x if isinstance(x, list) else []

def per100(x: int, denom: int) -> float:
    return 100.0 * x / max(denom, 1)

# Normalize common aliases (BiMISC -> MISC 2.5)
ALIAS_MAP: Dict[str, str] = {
    "SP": "SU",
    "STR": "ST",
    "WAR": "WA",
    "PS": "EC",   # permission-seeking utterances are EC in MISC 2.5
    "OP": "GI",   # neutral opinions are treated as informational here
}

# MISC 2.5 counselor buckets
MISC25_MICO = {  # MI-consistent
    "AF", "ADP", "EC", "RCP", "SU",
    # Questions and Reflections are counted in MICO for %MIC:
    "OQ", "SR", "CR", "RF"
}
MISC25_MIIN = {  # MI-inconsistent
    "ADW", "CO", "DI", "RCW", "WA"
}
# Neutral/other counselor codes (not in MIC denominator)
NEUTRAL_COUNSELOR = {"CQ", "FA", "FI", "GI", "ST"}

# Client valence sets (BiMISC-style CT/ST; ASK folds into FN)
CLIENT_CT = {"CM+", "TS+", "R+", "O+", "D+", "A+", "N+"}
CLIENT_ST = {"CM-", "TS-", "R-", "O-", "D-", "A-", "N-"}

def normalize_codes(codes: Iterable[str]) -> List[str]:
    out = []
    for c in codes:
        c = c.strip().upper()
        c = ALIAS_MAP.get(c, c)
        out.append(c)
    return out

# ---------- Core evaluation ----------

def compute_misc_stats(
    jsonl_path: str,
    *,
    use_coarse: bool = True,
    fine_field: str = "silver_fine",
    coarse_field: str = "silver_coarse",
) -> Dict[str, Any]:
    path = Path(jsonl_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    n_items = 0
    thr_utt = 0
    cli_utt = 0

    thr_code_counts = Counter()
    cli_code_counts = Counter()
    coarse_counts_thr = Counter()
    coarse_counts_cli = Counter()

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue

            n_items += 1
            role = str(item.get("utterance_role", "")).strip().lower()
            is_thr = role.startswith("ther")
            is_cli = role.startswith("client")

            if is_thr: thr_utt += 1
            if is_cli: cli_utt += 1

            fine = normalize_codes(_safe_list(item.get(fine_field, [])))
            if is_thr:
                thr_code_counts.update(fine)
            elif is_cli:
                # Fold ASK into FN so strict 2.5 remains consistent
                fine = ["FN" if c == "ASK" else c for c in fine]
                cli_code_counts.update(fine)

            if use_coarse:
                coarse = _safe_list(item.get(coarse_field, []))
                if is_thr: coarse_counts_thr.update(coarse)
                if is_cli: coarse_counts_cli.update(coarse)

    # Counselor tallies
    n_OQ = thr_code_counts["OQ"]
    n_CQ = thr_code_counts["CQ"]
    n_SR = thr_code_counts["SR"]
    n_CR = thr_code_counts["CR"]
    n_RF = thr_code_counts["RF"]
    n_GI = thr_code_counts["GI"]

    n_Q = n_OQ + n_CQ
    n_R = n_SR + n_CR + n_RF  # reflections family includes RF

    # Core counselor ratios
    R_over_Q = (n_R / n_Q) if n_Q else 0.0
    pct_complex_reflection = (n_CR / (n_SR + n_CR)) if (n_SR + n_CR) else 0.0
    pct_open_questions = (n_OQ / n_Q) if n_Q else 0.0

    # Per-100 rates
    reflections_per100 = per100(n_R, thr_utt)
    questions_per100 = per100(n_Q, thr_utt)
    info_per100 = per100(n_GI, thr_utt)

    # MI-consistent vs MI-inconsistent (counselor)
    mico_n = sum(thr_code_counts[c] for c in MISC25_MICO)
    miin_n = sum(thr_code_counts[c] for c in MISC25_MIIN)
    mic_den = mico_n + miin_n
    pct_mi_consistent = (mico_n / mic_den) if mic_den else 0.0
    mico_per100 = per100(mico_n, thr_utt)
    miin_per100 = per100(miin_n, thr_utt)

    # Client talk balance
    ct = sum(cli_code_counts[c] for c in CLIENT_CT)
    st = sum(cli_code_counts[c] for c in CLIENT_ST)
    pct_ct = (ct / (ct + st)) if (ct + st) else 0.0

    report = {
        "psychometrics": {
            "n_items": n_items,
            "therapist_utts": thr_utt,
            "client_utts": cli_utt,

            # Counselor ratios
            "R_over_Q": R_over_Q,
            "pct_open_questions": pct_open_questions,
            "pct_complex_reflection": pct_complex_reflection,

            # Counselor rates
            "reflections_per100": reflections_per100,
            "questions_per100": questions_per100,
            "info_per100": info_per100,

            # MI-consistency (counselor)
            "pct_mi_consistent": pct_mi_consistent,
            "mico_per100": mico_per100,
            "miin_per100": miin_per100,

            # Client balance
            "client_CT": ct,
            "client_ST": st,
            "pct_CT_over_CT_plus_ST": pct_ct,
        },
        "coverage": {
            "therapist_code_counts": dict(thr_code_counts),
            "client_code_counts": dict(cli_code_counts),
        },
        "coarse_coverage": {
            "therapist": dict(coarse_counts_thr),
            "client": dict(coarse_counts_cli),
        } if use_coarse else None,
        "performance": None,
        "meta": {
            "alias_map_applied": bool(ALIAS_MAP),
            "mico_set": sorted(MISC25_MICO),
            "miin_set": sorted(MISC25_MIIN),
            "neutral_counselor_set": sorted(NEUTRAL_COUNSELOR),
            "client_ct_set": sorted(CLIENT_CT),
            "client_st_set": sorted(CLIENT_ST),
        },
    }
    return report


if __name__ == "__main__":
    # >>> EDIT THESE PATHS <<<
    IN_PATH = "dataset/test_silver.jsonl"
    OUT_PATH = "dataset/test_silver_report.json"

    # IN_PATH = "dataset/thai_test_silver.jsonl"
    # OUT_PATH = "dataset/thai_test_silver_report.json"

    stats = compute_misc_stats(IN_PATH, use_coarse=True)
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)

    Path(OUT_PATH).write_text(text, encoding="utf-8")
    print(f"\nReport written to {OUT_PATH}")
