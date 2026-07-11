#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# USER SETTINGS
# ============================================================
FEATURE = "TotalSkeletonLength_um"
CELL_ID_COL = "ImageName"

LEFT_GROUP = "Normal"
RIGHT_GROUP = "CTNNB1"

GROUPS = {
    "Normal": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/mutation/normal_cleaned.xlsx",
        "color": "#7f7f7f",
        "dark": "#4d4d4d"
    },
    "CTNNB1": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/mutation/beta_cleaned.xlsx",
        "color": "#e377c2",
        "dark": "#c44e9b"
    },
    #"NRAS": {
       # "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/mitochondria_NRAS_cleaned.xlsx",
        #"color": "#1f77b4",
       # "dark": "#0f4c81"
   # },
    "PCC": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/mitochondria_PCC_cleaned.xlsx",
        "color": "#ADD8E6",
        "dark": "#1f77b4"
    },
    "PPC": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/mitocondira _PPC_cleaned.xlsx",
        "color": "#D8B4F8",
        "dark": "#7b3fcf"
    }
}

OUTPUT_DIR = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation"
OUTPUT = f"{FEATURE}_{LEFT_GROUP}_vs_{RIGHT_GROUP}_per_cell.svg"


# ============================================================
# HELPERS
# ============================================================
def p_to_stars(p):
    if p < 1e-4: return "****"
    elif p < 1e-3: return "***"
    elif p < 1e-2: return "**"
    elif p < 0.05: return "*"
    return "ns"


def clean_df(df, feature):
    df = df.copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[feature, CELL_ID_COL])
    df = df[df[feature] > 0]
    return df


# ⭐ CORE FIX: per-cell normalization
def get_per_cell_values(df, feature, method="median"):
    grouped = df.groupby(CELL_ID_COL)[feature]

    if method == "median":
        return grouped.median().values
    elif method == "mean":
        return grouped.mean().values
    else:
        raise ValueError("method must be 'median' or 'mean'")


# ============================================================
# LOAD DATA
# ============================================================
left_info = GROUPS[LEFT_GROUP]
right_info = GROUPS[RIGHT_GROUP]

df_left = pd.read_excel(left_info["file"])
df_right = pd.read_excel(right_info["file"])

df_left = clean_df(df_left, FEATURE)
df_right = clean_df(df_right, FEATURE)

# ⭐ PER-CELL VALUES
vals_left = get_per_cell_values(df_left, FEATURE, method="median")
vals_right = get_per_cell_values(df_right, FEATURE, method="median")

# ============================================================
# STATS
# ============================================================
pval = mannwhitneyu(vals_left, vals_right).pvalue
stars = p_to_stars(pval)

print(f"{LEFT_GROUP} n_cells = {len(vals_left)}")
print(f"{RIGHT_GROUP} n_cells = {len(vals_right)}")
print(f"p = {pval:.3e} ({stars})")

# ============================================================
# SUMMARY STATS
# ============================================================
q1_l, med_l, q3_l = np.percentile(vals_left, [25, 50, 75])
q1_r, med_r, q3_r = np.percentile(vals_right, [25, 50, 75])

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(2.5, 3.8))

parts = ax.violinplot(
    [vals_left, vals_right],
    positions=[1, 2],
    widths=0.7,
    showextrema=False
)

colors = [left_info["color"], right_info["color"]]

for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(colors[i])
    body.set_alpha(1)

# median + quartiles
for x, q1, med, q3 in [(1, q1_l, med_l, q3_l), (2, q1_r, med_r, q3_r)]:
    ax.hlines([q1, q3], x-0.2, x+0.2, colors="black", linestyles=":")
    ax.hlines(med, x-0.2, x+0.2, colors="black", linewidth=1.5)

# ⭐ plot individual cells (VERY IMPORTANT)
np.random.seed(0)
for i, vals in enumerate([vals_left, vals_right]):
    x = np.random.normal(i+1, 0.04, size=len(vals))
    ax.scatter(x, vals, color="black", s=8, alpha=0.6, zorder=3)

# significance
ymax = max(vals_left.max(), vals_right.max())
yrange = ymax - min(vals_left.min(), vals_right.min())

ax.plot([1,1,2,2],
        [ymax+0.05*yrange, ymax+0.1*yrange, ymax+0.1*yrange, ymax+0.05*yrange],
        color="black")

ax.text(1.5, ymax+0.12*yrange, stars, ha="center", fontsize=14)

# style
ax.set_xticks([1,2])
ax.set_xticklabels([LEFT_GROUP, RIGHT_GROUP], fontsize=12)
ax.set_ylabel(FEATURE, fontsize=12)

for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()

# ============================================================
# SAVE
# ============================================================
out_path = os.path.join(OUTPUT_DIR, OUTPUT)
fig.savefig(out_path, bbox_inches="tight", transparent=True)

plt.show()
