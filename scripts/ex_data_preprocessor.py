# csv_to_bimisc.py
# One-pass converter: dataset CSV -> rolling-history BiMISC-style JSONL
# Usage:
#   python csv_to_bimisc.py --in dataset/test.csv --out dataset/converted_conversations/bimisc_pretest.jsonl --history 6
#
# Notes:
# - Works with your current train/valid/test schema (conv_id/utterance_idx/speaker_idx/utterance/...).
# - If the CSV lacks conv_id, everything becomes a single conversation.
# - Strips leading "User:", "Bot:", "Client:", "Therapist:", numeric "1:", "2:", and bracketed/parenthesized variants.

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "data" / "psychologist" / "test.csv"
OUT_PATH = REPO_ROOT / "data" / "psychologist" / "pre_annotate.jsonl"

# ----------------------------
# I/O args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str,
                default="dataset/test.csv", help="Input CSV path")
    ap.add_argument("--out", dest="out_path", type=str,
                default="dataset/bimisc_pretest.jsonl", help="Output JSONL path")
    ap.add_argument("--history", dest="history_window", type=int,
                    default=6, help="Rolling history window size")
    return ap.parse_args()


# ----------------------------
# Loaders (from dataset_to_jsonl.py semantics)
# ----------------------------
def load_train_valid(path: Path) -> pd.DataFrame:
    # Standard CSV loader with tolerant parsing
    return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")

def load_test_like(path: Path) -> pd.DataFrame:
    # Quirky loader for test.csv with messy commas (same heuristic from your script)
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0].split(",")
    rows, buf = [], ""
    for line in lines[1:]:
        buf = line if not buf else f"{buf} {line}"
        parts = buf.split(",")
        if len(parts) >= 8:
            fixed = parts[:7] + [",".join(parts[7:])]
            rows.append(fixed)
            buf = ""
    cols = header[:8] if len(header) >= 8 else [f"c{i}" for i in range(8)]
    return pd.DataFrame(rows, columns=cols)

def smart_load_csv(path: Path) -> pd.DataFrame:
    # If file name contains "test", use the special loader; else use standard
    name = path.name.lower()
    if "test" in name:
        return load_test_like(path)
    return load_train_valid(path)

# ----------------------------
# Cleaning (from dataset_to_jsonl.py)
# ----------------------------
def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["prompt","utterance","tags","context"]:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                               .str.replace("_comma_", ",", regex=False)
                               .str.replace("\r"," ", regex=False)
                               .str.replace("\n"," ", regex=False)
                               .str.strip())
    for col in ["utterance_idx","speaker_idx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

# ----------------------------
# Conversation assembler (from dataset_to_jsonl.py)
# ----------------------------
def _ensure_conv_id(df: pd.DataFrame) -> pd.DataFrame:
    cand_cols = ["conv_id","conversation_id","dialogue_id","episode_id","episode_idx"]
    found = next((c for c in cand_cols if c in df.columns), None)
    if found:
        return df.rename(columns={found: "conv_id"})
    df = df.copy()
    df["conv_id"] = 0
    return df

def transcript_from_conv(df_conv: pd.DataFrame) -> str:
    parts = []
    speaker = df_conv.get("speaker_idx")
    for _, r in df_conv.sort_values("utterance_idx", na_position="first").iterrows():
        who = "User" if (speaker is not None and r.get("speaker_idx", 0) == 0) else "Bot"
        utt = str(r.get("utterance","")).strip()
        parts.append(f"{who}: {utt}")
    return "\n".join(parts)

def build_conversation_only(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_conv_id(df)
    keep_cols = ["conv_id","utterance_idx","speaker_idx","utterance","context","prompt"]
    df2 = df[[c for c in keep_cols if c in df.columns]].copy()
    df2 = df2.sort_values(["conv_id","utterance_idx"])
    out_rows = []
    for conv_id, g in df2.groupby("conv_id"):
        conv_text = transcript_from_conv(g)
        out = {
            "conv_id": conv_id,
            "conversation": conv_text,
            "context": g["context"].iloc[0] if "context" in g.columns else None,
            "prompt":  g["prompt"].iloc[0]  if "prompt"  in g.columns else None,
        }
        out_rows.append(out)
    return pd.DataFrame(out_rows)

# ----------------------------
# Prefix stripping + turn parsing (from jsonl_to_proper.py)
# ----------------------------
PREFIX_RE = re.compile(
    r"""^\s*
        (?:
          (?:user|bot|client|therapist)     # named roles
          |[12]                              # numeric speaker ids
          |\[(?:user|bot|client|therapist)\] # bracketed roles
          |\((?:user|bot|client|therapist)\) # parenthesized roles
        )
        \s*[:)\]-]*\s*                       # trailing separators
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _strip_prefix(text: str) -> str:
    return PREFIX_RE.sub("", text).strip()

def _split_lines(conv_text: str) -> List[str]:
    return [ln.strip() for ln in re.split(r"\r?\n+", conv_text.strip()) if ln.strip()]

def parse_turns(conv_text: str) -> List[Tuple[str, str]]:
    lines = _split_lines(conv_text)
    turns: List[Tuple[str, str]] = []
    for i, ln in enumerate(lines):
        clean = _strip_prefix(ln)
        if not clean:
            continue
        role = "Client" if i % 2 == 0 else "Therapist"
        turns.append((role, clean))
    return turns

def yield_items(turns: List[Tuple[str, str]], history_window: int = 6) -> Iterable[Dict[str, Any]]:
    for i, (role, text) in enumerate(turns):
        hist = turns[max(0, i - history_window):i]
        yield {
            "history": [{"role": r, "text": t} for r, t in hist],
            "utterance_role": role,      # "Client" or "Therapist"
            "utterance_text": text,
        }

# ----------------------------
# End-to-end
# ----------------------------
def main():
    in_path = IN_PATH
    out_path = OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = smart_load_csv(in_path)
    df = clean_text(df)
    conv_df = build_conversation_only(df)

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for _, row in conv_df.iterrows():
            conv_text = (row.get("conversation") or "").strip()
            if not conv_text:
                continue
            turns = parse_turns(conv_text)
            for item in yield_items(turns, history_window=6):
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    print(f"{in_path} -> {out_path} | wrote {written} items")

if __name__ == "__main__":
    main()