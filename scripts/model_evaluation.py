#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_misc_report_simple.py

Standalone roll-up for MISC silver annotations.
- Input: JSONL with fields produced by your pipeline (at minimum):
    {
      "utterance_role": "Therapist" | "Client",
      "silver_fine": ["OQ","SR",...],
      "silver_coarse": ["QS","RF",...]   # optional
    }

- Output: prints JSON report and optionally writes to file.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

ADHERENT = {"AFF", "EC", "FA", "SP", "PS", "OQ", "SR", "CR"}
NONADHERENT = {"DIR", "WAR", "OP", "ADV"}
CLIENT_CT = {"CM+", "TS+", "R+", "O+"}
CLIENT_ST = {"CM-", "TS-", "R-", "O-"}

def _safe_list(x):
    return x if isinstance(x, list) else []

def per100(x: int, denom: int) -> float:
    return 100.0 * x / max(denom, 1)

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

            role = str(item.get("utterance_role", "")).lower()
            is_thr = role.startswith("ther")
            is_cli = role.startswith("client")

            if is_thr: thr_utt += 1
            if is_cli: cli_utt += 1

            fine = _safe_list(item.get(fine_field, []))
            if is_thr: thr_code_counts.update(fine)
            if is_cli: cli_code_counts.update(fine)

            if use_coarse:
                coarse = _safe_list(item.get(coarse_field, []))
                if is_thr: coarse_counts_thr.update(coarse)
                if is_cli: coarse_counts_cli.update(coarse)

    # Therapist stats
    n_OQ = thr_code_counts["OQ"]
    n_CQ = thr_code_counts["CQ"]
    n_SR = thr_code_counts["SR"]
    n_CR = thr_code_counts["CR"]
    n_GI = thr_code_counts["GI"]
    n_Q = n_OQ + n_CQ
    n_R = n_SR + n_CR

    R_over_Q = (n_R / n_Q) if n_Q else 0.0
    pct_complex_reflection = (n_CR / n_R) if n_R else 0.0
    pct_open_questions = (n_OQ / n_Q) if n_Q else 0.0

    reflections_per100 = per100(n_R, thr_utt)
    questions_per100 = per100(n_Q, thr_utt)
    info_per100 = per100(n_GI, thr_utt)

    n_adherent = sum(thr_code_counts[c] for c in ADHERENT)
    n_nonadherent = sum(thr_code_counts[c] for c in NONADHERENT)
    adherent_rate_per100 = per100(n_adherent, thr_utt)
    nonadherent_rate_per100 = per100(n_nonadherent, thr_utt)

    # Client talk balance
    ct = sum(cli_code_counts[c] for c in CLIENT_CT)
    st = sum(cli_code_counts[c] for c in CLIENT_ST)
    change_share = (ct / (ct + st)) if (ct + st) else 0.0

    return {
        "psychometrics": {
            "n_items": n_items,
            "therapist_utts": thr_utt,
            "client_utts": cli_utt,
            "R_over_Q": R_over_Q,
            "pct_complex_reflection": pct_complex_reflection,
            "pct_open_questions": pct_open_questions,
            "reflections_per100": reflections_per100,
            "questions_per100": questions_per100,
            "info_per100": info_per100,
            "adherent_rate_per100": adherent_rate_per100,
            "nonadherent_rate_per100": nonadherent_rate_per100,
            "change_share_CT_over_CT_plus_ST": change_share,
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
    }


if __name__ == "__main__":
    # >>> EDIT THESE PATHS <<<
    IN_PATH = "dataset/test_silver.jsonl"         # where your silver annotations live
    OUT_PATH = "dataset/test_silver_report.json"  # optional save

    stats = compute_misc_stats(IN_PATH, use_coarse=True)

    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)

    # write to file too
    Path(OUT_PATH).write_text(text, encoding="utf-8")
    print(f"\nReport written to {OUT_PATH}")
