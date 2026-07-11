#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ============================================================
# User settings
# ============================================================
PCC_FILE = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/PCC_ExM.xlsx"
PPC_FILE = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/PPC_ExM.xlsx"
OUTPUT_PNG = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/selected_feature_correlation_clustered.png"
OUTPUT_SVG = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/selected_feature_correlation_clustered.svg"
OUTPUT_XLSX = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/selected_feature_correlation_matrix.xlsx"

FEATURES = [
    "Volume_um3",
    "MajorAxisLength_um",
    "MinorAxisLength_um",
    "TotalSkeletonLength_um",
    "AvgDiameterFromVolume_um",
    "CrossSectionArea_um2",
]

# Optional shorter display labels
DISPLAY_LABELS = {
    "Volume_um3": "Volume",
    "MajorAxisLength_um": "MajorAxisLength",
    "MinorAxisLength_um": "MinorAxisLength",
    "TotalSkeletonLength_um": "TotalSkeletonLength",
    "AvgDiameterFromVolume_um": "AvgDiameter",
    "CrossSectionArea_um2": "CrossSectionArea",
}

FIG_W = 6
FIG_H = 6
TITLE = "Feature–Feature Correlation"

FONT_SIZE = 16
TITLE_SIZE = 22
CBAR_FONT_SIZE = 16


# ============================================================
# Load data
# ============================================================
pcc = pd.read_excel(PCC_FILE)
ppc = pd.read_excel(PPC_FILE)

df = pd.concat([pcc, ppc], ignore_index=True)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[FEATURES].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

if len(df) == 0:
    raise ValueError("No valid rows remain after cleaning.")

# Pearson correlation
corr = df.corr(method="pearson")
corr.to_excel(OUTPUT_XLSX)

# Rename labels for display
corr_plot = corr.rename(index=DISPLAY_LABELS, columns=DISPLAY_LABELS)


# ============================================================
# Hierarchical clustering
# ============================================================
# Distance based on correlation similarity
distance = 1 - corr_plot.abs()
np.fill_diagonal(distance.values, 0)

linkage_matrix = linkage(
    squareform(distance.values),
    method="average"
)


# ============================================================
# Plot clustered heatmap
# ============================================================
sns.set_context("paper")

g = sns.clustermap(
    corr_plot,
    row_linkage=linkage_matrix,
    col_linkage=linkage_matrix,
    cmap="vlag",
    vmin=-1,
    vmax=1,
    center=0,
    figsize=(FIG_W, FIG_H),
    linewidths=0.3,
    linecolor="lightgray",
    annot=False,
    dendrogram_ratio=(0.14, 0.14),
    colors_ratio=0.03,
    cbar_pos=(0.90, 0.35, 0.035, 0.22),
    cbar_kws={"label": "Pearson r", "ticks": [-1, 0, 1]},
)

# Title
g.fig.suptitle(TITLE, fontsize=TITLE_SIZE, y=1.02)

# Axis label styling
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(),
    rotation=90,
    ha="center",
    va="top",
    fontsize=FONT_SIZE
)

g.ax_heatmap.set_yticklabels(
    g.ax_heatmap.get_yticklabels(),
    rotation=0,
    fontsize=FONT_SIZE
)

# Remove axis labels
g.ax_heatmap.set_xlabel("")
g.ax_heatmap.set_ylabel("")

# Colorbar styling
g.cax.set_ylabel("Pearson r", fontsize=CBAR_FONT_SIZE)
g.cax.tick_params(labelsize=CBAR_FONT_SIZE)

# Make dendrogram lines black and thicker
for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
    for line in ax.collections:
        line.set_color("black")
        line.set_linewidth(1.3)

# Remove dendrogram ticks
g.ax_row_dendrogram.set_xticks([])
g.ax_row_dendrogram.set_yticks([])
g.ax_col_dendrogram.set_xticks([])
g.ax_col_dendrogram.set_yticks([])

# Save
g.fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
g.fig.savefig(OUTPUT_SVG, bbox_inches="tight")
plt.close(g.fig)


# ============================================================
# Print summary
# ============================================================
print("Correlation matrix:")
print(corr.round(3).to_string())
print(f"\nSaved PNG: {OUTPUT_PNG}")
print(f"Saved SVG: {OUTPUT_SVG}")
print(f"Saved Excel: {OUTPUT_XLSX}")
