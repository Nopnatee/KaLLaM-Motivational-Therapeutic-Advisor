# radar_visualizer_individual.py
# Requirements: matplotlib, numpy, pandas

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# -----------------
# CONFIG
# -----------------
REPORT_CONFIGS = {
    # label: { path: Path|str, color: hex|rgb tuple (optional) }
    "Real Psychologist": {"path": "../data/human/report.json", "color": "#ff0000"},
    "Our KaLLaM": {"path": "../data/orchestrated/report.json", "color": "#2ca02c"},
    "Gemini-2.5-flash-light": {"path": "../data/gemini/report.json", "color": "#9dafff"},
    "Gemma-SEA-LION-v4-27B-IT": {"path": "../data/SEA-Lion/report.json", "color": "#8d35ff"},
    # Add more models here...
}

# Psychometric targets (units are already scaled as shown)
RECOMMENDED = {
    "R/Q ratio": 1.0,
    "% Open Questions": 50.0,
    "% Complex Reflections": 40.0,
    "% MI-Consistent": 90.0,
    "% Change Talk": 50.0
}

# Safety keys (Xu et al. proxies, 0–10)
SAFETY_KEYS = [
    "Q1_guidelines_adherence",
    "Q2_referral_triage",
    "Q3_consistency",
    "Q4_resources",
    "Q5_empowerment",
]

# -----------------
# LOADING & EXTRACTION
# -----------------
def _load_json(path_like) -> Optional[dict]:
    p = Path(path_like).expanduser()
    if not p.exists():
        print(f"[warn] Missing report: {p}")
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Failed to read {p}: {e}")
        return None

def _extract_psychometrics(report: Optional[dict]) -> dict:
    psy = report.get("psychometrics", {}) if report else {}
    try:
        rq   = float(psy.get("R_over_Q", 0.0))
        poq  = float(psy.get("pct_open_questions", 0.0)) * 100.0
        pcr  = float(psy.get("pct_complex_reflection", 0.0)) * 100.0
        mic  = psy.get("pct_mi_consistent", psy.get("pct_mi_consistency", psy.get("pct_mi_consist", 0.0)))
        mic  = float(mic) * 100.0
        pct_ct = float(psy.get("pct_CT_over_CT_plus_ST", 0.0)) * 100.0
    except Exception:
        rq, poq, pcr, mic, pct_ct = 0.0, 0.0, 0.0, 0.0, 0.0
    return {
        "R/Q ratio": rq,
        "% Open Questions": poq,
        "% Complex Reflections": pcr,
        "% MI-Consistent": mic,
        "% Change Talk": pct_ct,
    }

def _extract_safety(report: Optional[dict]) -> dict:
    if not report:
        return {}
    safety = report.get("safety", {})
    scores = safety.get("scores_0_10", {})
    out = {}
    for k in SAFETY_KEYS:
        try:
            out[k] = float(scores.get(k, 0.0))
        except Exception:
            out[k] = 0.0
    return out

# -----------------
# UTIL
# -----------------
def values_by_labels(d: Dict[str, float], labels: List[str]) -> List[float]:
    out = []
    for k in labels:
        v = d.get(k, np.nan)
        out.append(0.0 if (pd.isna(v) or v is None) else float(v))
    return out

def _make_angles(n: int) -> List[float]:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    return ang + ang[:1]

def _as_closed(seq: List[float]) -> List[float]:
    return seq + seq[:1] if seq else []

# -----------------
# DATA BUILD
# -----------------
def build_all_data(report_configs: dict):
    all_data = {}
    colors = {}
    for label, cfg in report_configs.items():
        rep = _load_json(cfg.get("path"))
        colors[label] = cfg.get("color", "#1f77b4")
        pm = _extract_psychometrics(rep)
        sm = _extract_safety(rep)
        all_data[label] = {"psychometrics": pm, "safety": sm, "report": rep}
    return all_data, colors

# -----------------
# CONSOLIDATED 1x2 BARS (absolute + recommended)
# -----------------
def render_unified_absolute_only(report_configs=REPORT_CONFIGS, save_path: str = "./radar_outputs/ALL_MODELS_absolute.png"):
    """
    One figure, 1x2 grid:
      [0] Psychometrics — Absolute (Human + all models + Recommended targets as hatched bars)
      [1] Safety        — Absolute (Human + all models + Recommended=10 for all safety as hatched bars)
    """
    all_data, colors = build_all_data(report_configs)

    human_label = "Real Psychologist"
    if human_label not in all_data:
        print("[warn] No human baseline.")
        return

    entity_labels = [lbl for lbl in all_data.keys() if lbl != human_label]
    if not entity_labels:
        print("[warn] No non-human models.")
        return

    human_psych = all_data[human_label]["psychometrics"] or {}
    human_safety = all_data[human_label]["safety"] or {}

    psych_axes = list(RECOMMENDED.keys())
    safety_axes = SAFETY_KEYS

    human_psych_vals = values_by_labels(human_psych, psych_axes)
    model_psych_matrix = np.array([
        [float(all_data[m]["psychometrics"].get(metric, 0.0)) for m in entity_labels]
        for metric in psych_axes
    ])

    has_any_model_safety = any(bool(all_data[m]["safety"]) for m in entity_labels)
    human_safety_vals = values_by_labels(human_safety, safety_axes) if human_safety else [0.0] * len(safety_axes)
    model_safety_matrix = np.array([
        [float(all_data[m]["safety"].get(metric, 0.0)) for m in entity_labels]
        for metric in safety_axes
    ]) if has_any_model_safety and human_safety else np.zeros((len(safety_axes), len(entity_labels)))

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("All Models vs Real Psychologist — Absolute Scores", fontsize=18, fontweight="bold", y=0.98)

    # ----------------- Psychometrics Absolute -----------------
    ax_abs_p = axs[0]
    x = np.arange(len(psych_axes))

    # bars per group = Recommended + Human + N models
    n_models = len(entity_labels)
    total_bars = 2 + n_models
    group_width = 0.9
    bar_width = group_width / total_bars
    start = -group_width / 2

    # Recommended bars (hatched)
    rec_vals = values_by_labels(RECOMMENDED, psych_axes)
    rec_offset = start + bar_width * 0.5
    ax_abs_p.bar(
        x + rec_offset, rec_vals, width=bar_width, label="Recommended",
        edgecolor="#222222", facecolor="none", hatch="//", linewidth=1.2
    )

    # Human bars
    human_offset = start + bar_width * 1.5
    ax_abs_p.bar(x + human_offset, human_psych_vals, width=bar_width, label=human_label, color="#ff0000", alpha=0.9)

    # Model bars
    y_max_psy = max([*human_psych_vals, *rec_vals]) if (human_psych_vals or rec_vals) else 0
    for i, m in enumerate(entity_labels):
        offs = start + bar_width * (i + 2.5)
        vals = model_psych_matrix[:, i]
        y_max_psy = max(y_max_psy, float(np.nanmax(vals)) if vals.size else 0)
        ax_abs_p.bar(x + offs, vals, width=bar_width, label=m, color=colors.get(m, "#1f77b4"), alpha=0.9)

    ax_abs_p.set_xticks(x)
    ax_abs_p.set_xticklabels(psych_axes, rotation=15, ha="right")
    ax_abs_p.set_ylabel("Score")
    ax_abs_p.set_ylim(0, y_max_psy * 1.15 if y_max_psy > 0 else 1)
    ax_abs_p.set_title("Psychometrics — Absolute")
    ax_abs_p.grid(axis="y", alpha=0.3)
    ax_abs_p.legend(ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.15))

    # ----------------- Safety Absolute -----------------
    ax_abs_s = axs[1]
    x_s = np.arange(len(safety_axes))

    # bars per group = Recommended + Human + N models
    total_bars_s = 2 + len(entity_labels)
    group_width_s = 0.9
    bar_width_s = group_width_s / total_bars_s
    start_s = -group_width_s / 2

    # Recommended safety target = 10 for each key
    rec_safety_vals = [10.0] * len(safety_axes)
    rec_offset_s = start_s + bar_width_s * 0.5
    ax_abs_s.bar(
        x_s + rec_offset_s, rec_safety_vals, width=bar_width_s, label="Ideal Safety",
        edgecolor="#222222", facecolor="none", hatch="//", linewidth=1.2
    )

    # Human bars
    human_offset_s = start_s + bar_width_s * 1.5
    ax_abs_s.bar(x_s + human_offset_s, human_safety_vals, width=bar_width_s, label=human_label, color="#ff0000", alpha=0.9)

    # Models
    if has_any_model_safety and human_safety:
        for i, m in enumerate(entity_labels):
            offs = start_s + bar_width_s * (i + 2.5)
            vals = model_safety_matrix[:, i]
            ax_abs_s.bar(x_s + offs, vals, width=bar_width_s, label=m, color=colors.get(m, "#1f77b4"), alpha=0.9)

    ax_abs_s.set_xticks(x_s)
    ax_abs_s.set_xticklabels(["Guidelines", "Referral", "Consistency", "Resources", "Empowerment"], rotation=15, ha="right")
    ax_abs_s.set_ylabel("0–10")
    ax_abs_s.set_ylim(0, 10)
    ax_abs_s.set_title("Safety — Absolute")
    ax_abs_s.grid(axis="y", alpha=0.3)
    ax_abs_s.legend(ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.15))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[info] Saved absolute-only comparison to {save_path}")
    plt.show()

# -----------------
# FINAL POLYGON ACCURACY (Similarity-to-Human, 0–100)
# -----------------
def calculate_similarity_scores(all_data, human_label="Real Psychologist", max_score=100):
    human_data = all_data.get(human_label, {}) or {}
    human_psych = human_data.get("psychometrics", {}) or {}
    human_safety = human_data.get("safety", {}) or {}

    similarity_scores = {}
    SAFETY_SCALE_MAX = 10.0
    PSYCH_SCALE_MAX = 100.0
    RQ_RATIO_MAX = 5.0

    def scale_max(metric_name: str) -> float:
        if metric_name in SAFETY_KEYS:
            return SAFETY_SCALE_MAX
        if metric_name == "R/Q ratio":
            return RQ_RATIO_MAX
        return PSYCH_SCALE_MAX

    for model_name, data in all_data.items():
        if model_name == human_label:
            continue
        model_psych = data.get("psychometrics", {}) or {}
        model_safety = data.get("safety", {}) or {}

        model_sim = {}

        for metric in RECOMMENDED.keys():
            if metric in model_psych and metric in human_psych:
                m = float(model_psych[metric])
                h = float(human_psych[metric])
                smax = scale_max(metric)
                sim = max_score * (1 - (abs(m - h) / smax))
                model_sim[metric] = max(0, min(max_score, sim))

        for metric in SAFETY_KEYS:
            if metric in model_safety and metric in human_safety:
                m = float(model_safety[metric])
                h = float(human_safety[metric])
                smax = scale_max(metric)
                sim = max_score * (1 - (abs(m - h) / smax))
                model_sim[metric] = max(0, min(max_score, sim))

        if model_sim:
            similarity_scores[model_name] = model_sim

    return similarity_scores

def render_final_similarity_polygon(report_configs=REPORT_CONFIGS, save_path: str = "./radar_outputs/FINAL_similarity_polygon.png"):
    """
    One polygon radar: 10 axes total (5 psych + 5 safety), values are 0–100 similarity to the human baseline.
    Higher = closer to human. All models overlaid on the same axes.
    """
    all_data, colors = build_all_data(report_configs)
    sim = calculate_similarity_scores(all_data)

    if not sim:
        print("[warn] No similarity scores; need human + at least one model with overlapping metrics.")
        return

    # Fixed unified axis order: 5 psych + 5 safety
    axes_labels_full = list(RECOMMENDED.keys()) + SAFETY_KEYS

    # Shorten labels for readability
    def short(lbl: str) -> str:
        s = lbl
        s = s.replace("% ", "")
        s = s.replace("Open Questions", "Open Q")
        s = s.replace("Complex Reflections", "Complex R")
        s = s.replace("MI-Consistent", "MI Consist")
        s = s.replace("Change Talk", "Change Talk")
        s = s.replace("R/Q ratio", "R/Q")
        s = s.replace("Q1_guidelines_adherence", "Guidelines")
        s = s.replace("Q2_referral_triage", "Referral")
        s = s.replace("Q3_consistency", "Consistency")
        s = s.replace("Q4_resources", "Resources")
        s = s.replace("Q5_empowerment", "Empowerment")
        return s

    labels = [short(x) for x in axes_labels_full]
    N = len(axes_labels_full)
    angles = _make_angles(N)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1, polar=True)
    fig.suptitle("Final Polygon Accuracy — Similarity to Real Psychologist (0–100)", fontsize=16, fontweight="bold", y=0.98)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Reference rings
    circle_angles = np.linspace(0, 2 * math.pi, 360)
    for ref_val in [25, 50, 75, 90]:
        lw = 2.0 if ref_val >= 75 else 1.2
        ax.plot(circle_angles, [ref_val] * 360, linestyle="--", linewidth=lw, color="#aaaaaa", alpha=0.65)

    # Plot each model
    for model_name, data in all_data.items():
        if model_name == "Real Psychologist":
            continue
        scores = sim.get(model_name, {})
        vals = [float(scores.get(k, 0.0)) for k in axes_labels_full]
        closed = _as_closed(vals)
        color = REPORT_CONFIGS.get(model_name, {}).get("color", "#1f77b4")
        ax.fill(angles, closed, alpha=0.15, color=color)
        ax.plot(angles, closed, linewidth=2.2, label=f"{model_name}", color=color, alpha=0.95)
        ax.scatter(angles[:-1], vals, s=36, color=color, alpha=0.9, zorder=5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.08), frameon=False, fontsize=9)

    # Footer helper
    fig.text(0.02, 0.02,
             "Scale: higher is better. 90+ excellent, 75+ good, 50+ fair.",
             fontsize=9, va="bottom",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="whitesmoke", alpha=0.9))
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[info] Saved final similarity polygon to {save_path}")

    plt.show()

# -----------------
# RESULTS TABLE (absolute + similarity) → CSV + PNG
# -----------------
def _short_label(lbl: str) -> str:
    s = lbl
    s = s.replace("% ", "")
    s = s.replace("Open Questions", "Open Q")
    s = s.replace("Complex Reflections", "Complex R")
    s = s.replace("MI-Consistent", "MI Consist")
    s = s.replace("Change Talk", "Change Talk")
    s = s.replace("R/Q ratio", "R/Q")
    s = s.replace("Q1_guidelines_adherence", "Guidelines")
    s = s.replace("Q2_referral_triage", "Referral")
    s = s.replace("Q3_consistency", "Consistency")
    s = s.replace("Q4_resources", "Resources")
    s = s.replace("Q5_empowerment", "Empowerment")
    return s

def build_results_dataframes(report_configs=REPORT_CONFIGS):
    """
    Returns:
      absolute_df: rows = metrics (psych + safety), cols = all entities (human + models)
      similarity_df: rows = metrics, cols = models (0–100 similarity to human)
    """
    all_data, _ = build_all_data(report_configs)

    # Unified metric order
    metrics = list(RECOMMENDED.keys()) + SAFETY_KEYS

    # Absolute values table
    abs_cols = []
    abs_col_data = []
    for entity in all_data.keys():
        combined = {}
        combined.update(all_data[entity].get("psychometrics", {}) or {})
        combined.update(all_data[entity].get("safety", {}) or {})
        abs_cols.append(entity)
        abs_col_data.append([float(combined.get(m, np.nan)) for m in metrics])

    absolute_df = pd.DataFrame(
        data=np.array(abs_col_data).T,
        index=metrics,
        columns=abs_cols
    )

    # Similarity table (0–100)
    sim = calculate_similarity_scores(all_data)
    if sim:
        sim_cols = []
        sim_col_data = []
        for model_name in sim.keys():
            sim_cols.append(model_name)
            sim_col_data.append([float(sim[model_name].get(m, np.nan)) for m in metrics])
        similarity_df = pd.DataFrame(
            data=np.array(sim_col_data).T,
            index=metrics,
            columns=sim_cols
        )
    else:
        similarity_df = pd.DataFrame(index=metrics)

    # Round for readability
    absolute_df = absolute_df.round(2)
    similarity_df = similarity_df.round(1)

    return absolute_df, similarity_df

def render_results_table(
    report_configs=REPORT_CONFIGS,
    save_path_png: str = "./radar_outputs/RESULTS_table.png",
    save_path_csv: str = "./radar_outputs/RESULTS_table.csv",
    include_similarity: bool = True
):
    """
    Renders a single figure containing a table:
      - Absolute scores for all entities (human + models)
      - If include_similarity=True, appends similarity-to-human columns (with ' (sim)' suffix)

    Also exports a CSV with the same data.
    """
    absolute_df, similarity_df = build_results_dataframes(report_configs)

    # Build combined table
    if include_similarity and not similarity_df.empty:
        sim_renamed = similarity_df.add_suffix(" (sim)")
        combined_df = absolute_df.join(sim_renamed, how="left")
    else:
        combined_df = absolute_df.copy()

    # Pretty row labels
    combined_df.index = [_short_label(x) for x in combined_df.index]

    # Export CSV
    out_dir = Path(save_path_png).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(save_path_csv, encoding="utf-8")
    print(f"[info] Saved results CSV to {save_path_csv}")

    # Render matplotlib table
    n_rows, n_cols = combined_df.shape

    # Heuristic sizing: wider for more columns, taller for more rows
    fig_w = min(2 + 0.85 * n_cols, 28)         # cap so it doesn't become ridiculous
    fig_h = min(2 + 0.55 * n_rows, 32)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    title = "Model Results — Absolute Scores"
    if include_similarity and not similarity_df.empty:
        title += " + Similarity-to-Human (0–100)"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    # Convert DataFrame to table
    tbl = ax.table(
        cellText=combined_df.fillna("").values,
        rowLabels=combined_df.index.tolist(),
        colLabels=combined_df.columns.tolist(),
        cellLoc="center",
        loc="center"
    )

    # Styling
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    # Increase row height slightly for readability
    tbl.scale(1.0, 1.15)

    # Header bold-ish
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            # Matplotlib tables index headers differently; this keeps it simple
            pass
        # Shade header row and first column labels
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_edgecolor("#c0c0c0")
            cell.set_linewidth(1.0)

    # Light grid effect
    for cell in tbl.get_celld().values():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[info] Saved results table figure to {save_path_png}")
    plt.show()

# -----------------
# MAIN
# -----------------
if __name__ == "__main__":
    render_unified_absolute_only(REPORT_CONFIGS, save_path="./radar_outputs/ALL_MODELS_absolute.png")
    render_final_similarity_polygon(REPORT_CONFIGS, save_path="./radar_outputs/FINAL_similarity_polygon.png")
    render_results_table(REPORT_CONFIGS,
                         save_path_png="./radar_outputs/RESULTS_table.png",
                         save_path_csv="./radar_outputs/RESULTS_table.csv",
                         include_similarity=True)