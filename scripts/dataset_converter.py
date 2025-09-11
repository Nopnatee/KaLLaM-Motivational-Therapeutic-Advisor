import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent   # go up from scripts/
BASE = ROOT / "dataset"
OUT  = BASE / "converted_conversations"
OUT.mkdir(parents=True, exist_ok=True)   # parents=True fixes it


# ---------- Loaders ----------
def load_train_valid(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")

def load_test(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
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

# ---------- Cleaning ----------
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

# ---------- Helpers ----------
def _ensure_conv_id(df: pd.DataFrame) -> pd.DataFrame:
    cand_cols = ["conv_id","conversation_id","dialogue_id","episode_id","episode_idx"]
    found = next((c for c in cand_cols if c in df.columns), None)
    if found:
        df = df.rename(columns={found: "conv_id"})
        return df
    # fallback: everything one conversation
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

# ---------- Builder ----------
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

# ---------- Main ----------
train = clean_text(load_train_valid(BASE/"train.csv"))
valid = clean_text(load_train_valid(BASE/"valid.csv"))
test  = clean_text(load_test(BASE/"test.csv"))

train_co = build_conversation_only(train)
valid_co = build_conversation_only(valid)
test_co  = build_conversation_only(test)

# Save plain conversation datasets (no scores)
train_co.to_csv(OUT/"train.csv", index=False)
valid_co.to_csv(OUT/"valid.csv", index=False)
test_co.to_csv(OUT/"test.csv",  index=False)

train_co.to_json(OUT/"train.jsonl", lines=True, orient="records", force_ascii=False)
valid_co.to_json(OUT/"valid.jsonl", lines=True, orient="records", force_ascii=False)
test_co.to_json(OUT/"test.jsonl",  lines=True, orient="records", force_ascii=False)
