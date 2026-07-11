#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# File paths
# ==============================
normal_path = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_curvature/curvature_results_normal.xlsx"
nras_path   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_curvature/curvature_results_nras.xlsx"
beta_path   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_curvature/curvature_results_beta.xlsx"

output_dir = os.path.dirname(os.path.abspath(normal_path))

# ==============================
# Settings
# ==============================
SHEET_NAME = "sampled_vertices"

# Only show thresholds > 0.3
THRESHOLDS = np.array([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=float)

# Sort within each group by enrichment at this threshold
SORT_BY_THRESHOLD = 0.40

# Heatmap color scaling
VMIN = 0.0
VMAX = 0.10

# Figure size (inches). Change if you want exact mm sizing.
FIGSIZE = (5, 6)
DPI = 600

# Font sizes
FONT_BASE = 12
XTICK_SIZE = 12
YTICK_SIZE = 12
AXIS_LABEL_SIZE = 15
GROUP_LABEL_SIZE = 15
CBAR_LABEL_SIZE = 15


# ==============================
# Load sampled_vertices
# ==============================
def load_sampled_vertices(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_NAME)

    # Ensure required column exists
    if "cell_id" not in df.columns:
        raise ValueError(f"'cell_id' column not found in sheet '{SHEET_NAME}' of: {path}")

    # Choose curvature column:
    # - prefer numeric columns containing "curv"
    # - else fall back to first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric columns found in sheet '{SHEET_NAME}' of: {path}")

    curv_candidates = [c for c in numeric_cols if "curv" in c.lower()]
    curv_col = curv_candidates[0] if len(curv_candidates) > 0 else numeric_cols[0]

    out = df[["cell_id", curv_col]].rename(columns={curv_col: "curv"}).copy()
    return out


# ==============================
# Compute per-cell enrichment
# ==============================
def per_cell_fraction_over_thresholds(df: pd.DataFrame, thresholds: np.ndarray) -> np.ndarray:
    fracs = []
    for _, g in df.groupby("cell_id", sort=False):
        curv = g["curv"].to_numpy()
        curv = curv[np.isfinite(curv)]

        if curv.size == 0:
            frac = np.zeros(len(thresholds), dtype=float)
        else:
            abs_curv = np.abs(curv)
            frac = np.array([(abs_curv > t).mean() for t in thresholds], dtype=float)

        fracs.append(frac)

    if len(fracs) == 0:
        return np.zeros((0, len(thresholds)), dtype=float)

    return np.vstack(fracs)


def sort_by_threshold(M: np.ndarray, thresholds: np.ndarray, sort_t: float) -> np.ndarray:
    # Find index (robust to float formatting)
    idx_arr = np.where(np.isclose(thresholds, sort_t, rtol=0, atol=1e-12))[0]
    if idx_arr.size == 0:
        raise ValueError(
            f"SORT_BY_THRESHOLD={sort_t} not found in THRESHOLDS={thresholds.tolist()}. "
            "Add it to THRESHOLDS or change SORT_BY_THRESHOLD."
        )
    idx = int(idx_arr[0])
    return M[np.argsort(M[:, idx])[::-1]]


# ==============================
# Load data
# ==============================
dfN = load_sampled_vertices(normal_path)
dfR = load_sampled_vertices(nras_path)
dfB = load_sampled_vertices(beta_path)

MN = per_cell_fraction_over_thresholds(dfN, THRESHOLDS)
MR = per_cell_fraction_over_thresholds(dfR, THRESHOLDS)
MB = per_cell_fraction_over_thresholds(dfB, THRESHOLDS)

# Sort within each group by enrichment at SORT_BY_THRESHOLD
MN = sort_by_threshold(MN, THRESHOLDS, SORT_BY_THRESHOLD)
MR = sort_by_threshold(MR, THRESHOLDS, SORT_BY_THRESHOLD)
MB = sort_by_threshold(MB, THRESHOLDS, SORT_BY_THRESHOLD)

# Stack all groups
M = np.vstack([MN, MR, MB])

# ==============================
# Plot heatmap
# ==============================
plt.rcParams.update({
    "font.size": FONT_BASE,
    "axes.labelsize": FONT_BASE,     # default axes label size (we override with explicit fontsize below)
    "axes.titlesize": FONT_BASE,
    "xtick.labelsize": XTICK_SIZE,
    "ytick.labelsize": YTICK_SIZE,
    "legend.fontsize": FONT_BASE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

im = ax.imshow(M, aspect="auto", cmap="cividis", vmin=VMIN, vmax=VMAX)

cbar = fig.colorbar(im, ax=ax)
cbar.set_ticks([0, 0.05, 0.10])
cbar.set_label("Fraction of vertices with |curv| > threshold", fontsize=CBAR_LABEL_SIZE)

ax.set_xticks(np.arange(len(THRESHOLDS)))
ax.set_xticklabels(
    [f"{t:.2f}" for t in THRESHOLDS],
    rotation=45,        # angle
    ha="left",         # align so labels don't collide
    fontsize=13         # larger font
)
ax.set_yticks([])                # remove tick labels
ax.tick_params(left=False)       # remove tick marks

ax.set_xlabel("High-curvature threshold |curv| (1/µm)", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("Cells (sorted within group)", fontsize=AXIS_LABEL_SIZE, labelpad=40)

# Group separators
yN = MN.shape[0]
yR = yN + MR.shape[0]

ax.axhline(yN - 0.5, color="white", linewidth=2)
ax.axhline(yR - 0.5, color="white", linewidth=2)

# Group labels (rotate)
ax.text(-0.98, yN / 2, "Normal", rotation=90, va="center", fontsize=GROUP_LABEL_SIZE)
ax.text(-0.98, yN + MR.shape[0] / 2, "NRAS", rotation=90, va="center", fontsize=GROUP_LABEL_SIZE)
ax.text(-0.98, yR + MB.shape[0] / 2, "CTNNB1", rotation=90, va="center", fontsize=GROUP_LABEL_SIZE)

fig.tight_layout()

# ==============================
# Save as SVG
# ==============================
save_path = os.path.join(output_dir, "High_curvature_enrichment_heatmap.svg")
fig.savefig(save_path, format="svg")
plt.close(fig)

print("Saved to:", save_path)
