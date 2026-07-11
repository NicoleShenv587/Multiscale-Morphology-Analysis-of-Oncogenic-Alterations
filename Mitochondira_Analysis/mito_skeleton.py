#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# =========================================================
# GLOBAL STYLE SETTINGS: edit only this section
# =========================================================

# ---- font family ----
FONT_FAMILY = "Arial"

# ---- global font sizes ----
FONT_SIZE_GLOBAL = 10
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 10
FONT_SIZE_XTICK = 9
FONT_SIZE_YTICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_NOTE = 9
FONT_SIZE_SIG = 14

# ---- x label angle ----
X_LABEL_ROTATION = 0      # change angle here
X_LABEL_HA = "right"       # "right", "center", "left"

# ---- figure sizes ----
STACKED_FIG_W = 3
STACKED_FIG_H = 3
BOX_FIG_W = 1.5
BOX_FIG_H = 1.8

# ---- line thickness ----
LINE_WIDTH_AXIS = 1.2
LINE_WIDTH_BAR_ERROR = 1.2
LINE_WIDTH_BOX_EDGE = 1.2
LINE_WIDTH_WHISKER = 1.2
LINE_WIDTH_CAP = 1.2
LINE_WIDTH_MEDIAN = 1.8
LINE_WIDTH_SIG = 1.2
LINE_WIDTH_DOT_EDGE = 0.35

# ---- tick settings ----
TICK_WIDTH = 1.2
TICK_LENGTH_X = 0
TICK_LENGTH_Y = 4

# ---- object sizes ----
BAR_WIDTH = 0.65
BOX_WIDTH = 0.55
DOT_SIZE = 18
DOT_ALPHA = 0.9
BOX_ALPHA = 0.75
ERROR_CAPSIZE = 3

# ---- y padding for boxplot ----
SIG_Y_OFFSET = 0.06
SIG_H = 0.04
YLIM_BOTTOM_PAD = 0.08
YLIM_TOP_PAD = 0.18

# ---- random seed for dot jitter ----
RNG_SEED = 42

# =========================================================
# FORCE ARIAL EVERYWHERE
# =========================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY],
    "font.size": FONT_SIZE_GLOBAL,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
})

# =========================================================
# USER OPTIONS
# =========================================================
# Choose:
# "PPC_PCC"
# "NORMAL_CTNNB1"
COMPARISON_MODE = "NORMAL_CTNNB1"

CELL_ID_COL = "ImageName"
LENGTH_COL = "TotalSkeletonLength_um"

OUT_DIR = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/skeleton"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# FILE PATHS
# =========================================================
PPC_FILE = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/PPC_ExM.xlsx"
PCC_FILE = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/PCC_ExM.xlsx"

NORMAL_FILE = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/Mito_segmentation/mutation/normal_Exm.xlsx"
CTNNB1_FILE = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/Mito_segmentation/mutation/beta_Exm.xlsx"

# =========================================================
# COLORS
# =========================================================
# PPC = purple
PPC_COLORS = ["#eddcf6", "#c89be0", "#8c57b8"]
PPC_MAIN = "#c89be0"
PPC_EDGE = "#8c57b8"

# PCC = blue
PCC_COLORS = ["#d6ecfa", "#8dc3e6", "#3f8fc5"]
PCC_MAIN = "#9ecae1"
PCC_EDGE = "#3f8fc5"

# Normal = gray
NORMAL_COLORS = ["#d9d9d9", "#9e9e9e", "#4d4d4d"]
NORMAL_MAIN = "#7f7f7f"
NORMAL_EDGE = "#4d4d4d"

# CTNNB1 = pink
CTNNB1_COLORS = ["#f7d6e8", "#e377c2", "#c44e9b"]
CTNNB1_MAIN = "#e377c2"
CTNNB1_EDGE = "#c44e9b"

# =========================================================
# SELECT COMPARISON
# =========================================================
if COMPARISON_MODE == "PPC_PCC":
    LEFT_FILE = PPC_FILE
    RIGHT_FILE = PCC_FILE

    LEFT_LABEL = "PPC"
    RIGHT_LABEL = "PCC"

    LEFT_COLORS = PPC_COLORS
    RIGHT_COLORS = PCC_COLORS

    LEFT_MAIN = PPC_MAIN
    LEFT_EDGE = PPC_EDGE

    RIGHT_MAIN = PCC_MAIN
    RIGHT_EDGE = PCC_EDGE

    OUT_SVG_STACKED = os.path.join(OUT_DIR, "per_cell_mito_length_distribution_PPC_vs_PCC.svg")
    OUT_SVG_BOX = os.path.join(OUT_DIR, "per_cell_long_mito_fraction_box_PPC_vs_PCC.svg")

elif COMPARISON_MODE == "NORMAL_CTNNB1":
    LEFT_FILE = NORMAL_FILE
    RIGHT_FILE = CTNNB1_FILE

    LEFT_LABEL = "Normal"
    RIGHT_LABEL = "CTNNB1"

    LEFT_COLORS = NORMAL_COLORS
    RIGHT_COLORS = CTNNB1_COLORS

    LEFT_MAIN = NORMAL_MAIN
    LEFT_EDGE = NORMAL_EDGE

    RIGHT_MAIN = CTNNB1_MAIN
    RIGHT_EDGE = CTNNB1_EDGE

    OUT_SVG_STACKED = os.path.join(OUT_DIR, "per_cell_mito_length_distribution_Normal_vs_CTNNB1.svg")
    OUT_SVG_BOX = os.path.join(OUT_DIR, "per_cell_long_mito_fraction_box_Normal_vs_CTNNB1.svg")

else:
    raise ValueError("COMPARISON_MODE must be 'PPC_PCC' or 'NORMAL_CTNNB1'")

# =========================================================
# LOAD & PREPROCESS
# =========================================================
def preprocess(df, length_col):
    df = df.copy()
    df[length_col] = pd.to_numeric(df[length_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[length_col])
    df = df[df[length_col] > 0]
    return df

left_df = preprocess(pd.read_excel(LEFT_FILE), LENGTH_COL)
right_df = preprocess(pd.read_excel(RIGHT_FILE), LENGTH_COL)

# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================
required_cols = [CELL_ID_COL, LENGTH_COL]
for c in required_cols:
    if c not in left_df.columns:
        raise ValueError(f"Column '{c}' not found in LEFT file: {LEFT_FILE}")
    if c not in right_df.columns:
        raise ValueError(f"Column '{c}' not found in RIGHT file: {RIGHT_FILE}")

# =========================================================
# THRESHOLDS BASED ON POOLED DATA
# =========================================================
all_lengths = np.concatenate([
    left_df[LENGTH_COL].values,
    right_df[LENGTH_COL].values
])

q1, q2 = np.percentile(all_lengths, [33, 66])

# =========================================================
# PER-CELL FRACTIONS
# =========================================================
def per_cell_fractions(df, cell_id_col, length_col, q1, q2):
    rows = []
    for cid, sub in df.groupby(cell_id_col):
        vals = sub[length_col].values
        total = len(vals)
        if total == 0:
            continue

        rows.append({
            "CellID": cid,
            "Short": np.sum(vals <= q1) / total * 100,
            "Intermediate": np.sum((vals > q1) & (vals <= q2)) / total * 100,
            "Long": np.sum(vals > q2) / total * 100
        })
    return pd.DataFrame(rows)

left_cell = per_cell_fractions(left_df, CELL_ID_COL, LENGTH_COL, q1, q2)
right_cell = per_cell_fractions(right_df, CELL_ID_COL, LENGTH_COL, q1, q2)

def mean_sem(df):
    mean = df.mean().values
    sem = df.std(ddof=1).values / np.sqrt(len(df))
    return mean, sem

cats = ["Short", "Intermediate", "Long"]
left_mean, left_sem = mean_sem(left_cell[cats])
right_mean, right_sem = mean_sem(right_cell[cats])

# =========================================================
# STATS FOR LONG FRACTION
# =========================================================
left_long = left_cell["Long"].values
right_long = right_cell["Long"].values

stat, pval = mannwhitneyu(left_long, right_long, alternative="two-sided")

# =========================================================
# HELPER: style axes
# =========================================================
def style_axes(ax):
    ax.tick_params(
        axis="x",
        labelsize=FONT_SIZE_XTICK,
        width=TICK_WIDTH,
        length=TICK_LENGTH_X,
        pad=2
    )
    ax.tick_params(
        axis="y",
        labelsize=FONT_SIZE_YTICK,
        width=TICK_WIDTH,
        length=TICK_LENGTH_Y
    )

    for label in ax.get_xticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_rotation(X_LABEL_ROTATION)
        label.set_ha(X_LABEL_HA)
        label.set_rotation_mode("anchor")

    for label in ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(LINE_WIDTH_AXIS)
    ax.spines["bottom"].set_linewidth(LINE_WIDTH_AXIS)

# =========================================================
# FIGURE 1: STACKED BAR
# =========================================================
fig1, ax1 = plt.subplots(figsize=(STACKED_FIG_W, STACKED_FIG_H))

x = np.array([0, 1])
bottoms = np.zeros(2)

group_means = np.vstack([left_mean, right_mean]).T
group_sems = np.vstack([left_sem, right_sem]).T

for i, cat in enumerate(cats):
    colors = [LEFT_COLORS[i], RIGHT_COLORS[i]]

    ax1.bar(
        x,
        group_means[i],
        bottom=bottoms,
        color=colors,
        edgecolor="none",
        width=BAR_WIDTH
    )

    centers = bottoms + group_means[i] / 2
    ax1.errorbar(
        x,
        centers,
        yerr=group_sems[i],
        fmt="none",
        ecolor="black",
        capsize=ERROR_CAPSIZE,
        lw=LINE_WIDTH_BAR_ERROR
    )

    bottoms += group_means[i]

ax1.set_xticks(x)
ax1.set_xticklabels([LEFT_LABEL, RIGHT_LABEL], fontsize=FONT_SIZE_XTICK, fontname=FONT_FAMILY)
ax1.set_ylabel("Per-cell fraction (%)", fontsize=FONT_SIZE_LABEL, fontname=FONT_FAMILY)
ax1.set_title("Per-cell mitochondria length distribution", fontsize=FONT_SIZE_TITLE, fontname=FONT_FAMILY)

style_axes(ax1)

ax1.text(
    0.0, 1.02,
    f"Short ≤ {q1:.2f}, Intermediate ≤ {q2:.2f}, Long > {q2:.2f}",
    transform=ax1.transAxes,
    fontsize=FONT_SIZE_NOTE,
    fontname=FONT_FAMILY
)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color="#d9d9d9"),
    plt.Rectangle((0, 0), 1, 1, color="#9e9e9e"),
    plt.Rectangle((0, 0), 1, 1, color="#4d4d4d"),
]
ax1.legend(
    legend_handles,
    ["Short", "Intermediate", "Long"],
    frameon=False,
    loc="upper right",
    prop={"family": FONT_FAMILY, "size": FONT_SIZE_LEGEND}
)

fig1.tight_layout()
fig1.savefig(OUT_SVG_STACKED, format="svg", bbox_inches="tight")
plt.close(fig1)

# =========================================================
# FIGURE 2: BOXPLOT + DOTS
# =========================================================
fig2, ax2 = plt.subplots(figsize=(BOX_FIG_W, BOX_FIG_H))

bp = ax2.boxplot(
    [left_long, right_long],
    positions=[1, 2],
    widths=BOX_WIDTH,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="white", linewidth=LINE_WIDTH_MEDIAN),
    whiskerprops=dict(color="black", linewidth=LINE_WIDTH_WHISKER),
    capprops=dict(color="black", linewidth=LINE_WIDTH_CAP),
    boxprops=dict(linewidth=LINE_WIDTH_BOX_EDGE),
)

bp["boxes"][0].set_facecolor(LEFT_MAIN)
bp["boxes"][0].set_edgecolor(LEFT_EDGE)
bp["boxes"][0].set_alpha(BOX_ALPHA)

bp["boxes"][1].set_facecolor(RIGHT_MAIN)
bp["boxes"][1].set_edgecolor(RIGHT_EDGE)
bp["boxes"][1].set_alpha(BOX_ALPHA)

rng = np.random.default_rng(RNG_SEED)

def add_box_dots(ax, x_center, values, width, color):
    jitter = rng.uniform(-width * 0.22, width * 0.22, size=len(values))
    ax.scatter(
        np.full(len(values), x_center) + jitter,
        values,
        s=DOT_SIZE,
        color=color,
        edgecolors="white",
        linewidths=LINE_WIDTH_DOT_EDGE,
        alpha=DOT_ALPHA,
        zorder=3
    )

add_box_dots(ax2, 1, left_long, BOX_WIDTH, LEFT_EDGE)
add_box_dots(ax2, 2, right_long, BOX_WIDTH, RIGHT_EDGE)

ax2.set_xticks([1, 2])
ax2.set_xticklabels([LEFT_LABEL, RIGHT_LABEL], fontsize=FONT_SIZE_XTICK, fontname=FONT_FAMILY)
ax2.set_ylabel("Long mitochondria (%) per cell", fontsize=FONT_SIZE_LABEL, fontname=FONT_FAMILY)

style_axes(ax2)

# =========================================================
# SIGNIFICANCE
# =========================================================
ymax = max(np.max(left_long), np.max(right_long))
ymin = min(np.min(left_long), np.min(right_long))

if ymax == ymin:
    ymin = 0
    ymax = ymax + 1

yrange = ymax - ymin
y = ymax + yrange * SIG_Y_OFFSET
h = yrange * SIG_H

ax2.plot([1, 1, 2, 2], [y, y + h, y + h, y], color="black", linewidth=LINE_WIDTH_SIG)

sig = "ns"
if pval < 1e-4:
    sig = "****"
elif pval < 1e-3:
    sig = "***"
elif pval < 1e-2:
    sig = "**"
elif pval < 5e-2:
    sig = "*"

ax2.text(
    1.5,
    y + h * 1.1,
    sig,
    ha="center",
    va="bottom",
    fontsize=FONT_SIZE_SIG,
    fontweight="bold",
    fontname=FONT_FAMILY
)

padding_bottom = yrange * YLIM_BOTTOM_PAD
padding_top = yrange * YLIM_TOP_PAD
ax2.set_ylim(max(0, ymin - padding_bottom), y + h + padding_top)

fig2.tight_layout()
fig2.savefig(OUT_SVG_BOX, format="svg", bbox_inches="tight")
plt.close(fig2)

print("Done.")
print("Comparison mode:", COMPARISON_MODE)
print("Saved stacked plot to:", OUT_SVG_STACKED)
print("Saved boxplot to:", OUT_SVG_BOX)
print(f"{LEFT_LABEL}: n_cells = {len(left_long)}, mean long% = {np.mean(left_long):.2f}, median long% = {np.median(left_long):.2f}")
print(f"{RIGHT_LABEL}: n_cells = {len(right_long)}, mean long% = {np.mean(right_long):.2f}, median long% = {np.median(right_long):.2f}")
print(f"Mann-Whitney U p-value = {pval:.6g}")
