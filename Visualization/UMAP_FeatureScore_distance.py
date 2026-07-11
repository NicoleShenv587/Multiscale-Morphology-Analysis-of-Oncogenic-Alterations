#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:54:00 2026

@author: s227698
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# ==============================
# SVG / FONT SETTINGS
# ==============================
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==============================
# INPUT FILES
# ==============================
normal_path = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Mito_segment/QC/normal_Exm.xlsx"
beta_path   = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Mito_segment/QC/beta_Exm.xlsx"
pcc_path    = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Mito_segment/QC/PCC_ExM.xlsx"
ppc_path    = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Mito_segment/QC/PPC_ExM.xlsx"

# ==============================
# SAVE FOLDER
# ==============================
SAVE_DIR = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Mito_segment/QC/z-score_Distance"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# OUTPUT FILES
# ==============================
umap_svg_out = os.path.join(SAVE_DIR, "UMAP_4groups_with_median.svg")
umap_png_out = os.path.join(SAVE_DIR, "UMAP_4groups_with_median.png")

umap_dist_csv_out = os.path.join(SAVE_DIR, "UMAP_group_median_distances_2D.csv")
umap_median_csv_out = os.path.join(SAVE_DIR, "UMAP_group_median_coordinates_2D.csv")

umap_heatmap_svg_out = os.path.join(SAVE_DIR, "UMAP_group_median_distance_heatmap_2D.svg")
umap_heatmap_png_out = os.path.join(SAVE_DIR, "UMAP_group_median_distance_heatmap_2D.png")

umap_pair_csv_out = os.path.join(SAVE_DIR, "UMAP_group_pairwise_median_distances_ranked_2D.csv")

feature_median_csv_out = os.path.join(SAVE_DIR, "Original_feature_space_group_median_coordinates_zscore.csv")
feature_dist_csv_out = os.path.join(SAVE_DIR, "Original_feature_space_group_median_distances_zscore.csv")
feature_pair_csv_out = os.path.join(SAVE_DIR, "Original_feature_space_pairwise_median_distances_ranked_zscore.csv")

feature_heatmap_svg_out = os.path.join(SAVE_DIR, "Original_feature_space_group_median_distance_heatmap_zscore.svg")
feature_heatmap_png_out = os.path.join(SAVE_DIR, "Original_feature_space_group_median_distance_heatmap_zscore.png")

# ==============================
# FEATURES TO USE
# ==============================
features = [
    "Volume_um3",
    "MinorAxisLength_um",
    "AvgDiameterFromVolume_um",
    "ElongationRatio",
    "MajorAxisLength_um",
]

# ==============================
# USER-EDITABLE PLOT SETTINGS
# ==============================
FIGSIZE_UMAP = (2.1, 2.1)
FIGSIZE_HEATMAP = (2.1, 2.3)

POINT_SIZE = 0.5
POINT_ALPHA = 0.8

MEDIAN_SIZE = 15
MEDIAN_EDGEWIDTH = 0.2

X_LABEL_SIZE = 8
Y_LABEL_SIZE = 8
TITLE_SIZE = 9
TICK_SIZE = 6
LEGEND_SIZE = 6
HEATMAP_NUMBER_SIZE = 6
COLORBAR_TICK_SIZE = 6

X_LABEL_WEIGHT = "normal"
Y_LABEL_WEIGHT = "normal"
TITLE_WEIGHT = "normal"

X_LABEL_ROTATION = 0
Y_LABEL_ROTATION = 90
HEATMAP_XTICK_ROTATION = 45
HEATMAP_YTICK_ROTATION = 0

AXIS_SPINE_WIDTH = 1.2
TICK_WIDTH = 1.0
TICK_LENGTH = 4

LEGEND_MARKERSCALE = 2.0

SHOW_UMAP_TITLE = True
SHOW_HEATMAP_TITLE = True

# ==============================
# LOAD DATA
# ==============================
normal = pd.read_excel(normal_path)
beta   = pd.read_excel(beta_path)
pcc    = pd.read_excel(pcc_path)
ppc    = pd.read_excel(ppc_path)

normal["Group"] = "Normal"
beta["Group"]   = "CTNNB1"
pcc["Group"]    = "PCC"
ppc["Group"]    = "PPC"

df = pd.concat([normal, beta, pcc, ppc], ignore_index=True)

# ==============================
# CLEAN DATA
# ==============================
df = df.dropna(subset=features).copy()

X = df[features].values
labels = df["Group"].values

plot_order = ["Normal", "CTNNB1", "PCC", "PPC"]

colors = {
    "Normal": "#7f7f7f",
    "CTNNB1": "#e377c2",
    "PCC": "#1f77b4",
    "PPC": "#C78BE3"
}

# ==============================
# STANDARDIZE FEATURES
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# ORIGINAL STANDARDIZED FEATURE-SPACE DISTANCE
# This is the rigorous distance using the 5 original features
# after z-score normalization.
# ============================================================
scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df["Group"] = labels

feature_median_rows = []

for group in plot_order:
    sub = scaled_df[scaled_df["Group"] == group]

    row = {
        "Group": group,
        "N_points": len(sub)
    }

    for f in features:
        row[f"Median_z_{f}"] = sub[f].median()

    feature_median_rows.append(row)

feature_median_df = pd.DataFrame(feature_median_rows)
feature_median_df.to_csv(feature_median_csv_out, index=False)

feature_dist_matrix = np.zeros((len(plot_order), len(plot_order)), dtype=float)

for i, g1 in enumerate(plot_order):
    v1 = feature_median_df.loc[
        feature_median_df["Group"] == g1,
        [f"Median_z_{f}" for f in features]
    ].values[0]

    for j, g2 in enumerate(plot_order):
        v2 = feature_median_df.loc[
            feature_median_df["Group"] == g2,
            [f"Median_z_{f}" for f in features]
        ].values[0]

        dist = np.sqrt(np.sum((v1 - v2) ** 2))
        feature_dist_matrix[i, j] = dist

feature_dist_df = pd.DataFrame(
    feature_dist_matrix,
    index=plot_order,
    columns=plot_order
)

feature_dist_df.to_csv(feature_dist_csv_out)

feature_pair_rows = []

for i in range(len(plot_order)):
    for j in range(i + 1, len(plot_order)):
        feature_pair_rows.append({
            "Group1": plot_order[i],
            "Group2": plot_order[j],
            "Distance_original_zscore_feature_space": feature_dist_matrix[i, j]
        })

feature_pair_df = pd.DataFrame(feature_pair_rows).sort_values(
    "Distance_original_zscore_feature_space",
    ascending=True
)

feature_pair_df.to_csv(feature_pair_csv_out, index=False)

print("\nGroup median coordinates in original standardized feature space:")
print(feature_median_df)

print("\nPairwise Euclidean distances between group medians in original standardized feature space:")
print(feature_dist_df)

print("\nRanked pairwise distances in original standardized feature space:")
print(feature_pair_df)

# ==============================
# UMAP 2D
# ==============================
reducer = UMAP(
    n_neighbors=50,
    min_dist=0.05,
    n_components=2,
    spread=1,
    random_state=42,
    low_memory=True
)

X_umap = reducer.fit_transform(X_scaled)

df["UMAP1"] = X_umap[:, 0]
df["UMAP2"] = X_umap[:, 1]

# ==============================
# CALCULATE 2D UMAP MEDIAN COORDINATES
# ==============================
umap_median_rows = []

for group in plot_order:
    sub = df[df["Group"] == group]

    median_x = sub["UMAP1"].median()
    median_y = sub["UMAP2"].median()

    umap_median_rows.append({
        "Group": group,
        "Median_UMAP1": median_x,
        "Median_UMAP2": median_y,
        "N_points": len(sub)
    })

umap_median_df = pd.DataFrame(umap_median_rows)
umap_median_df.to_csv(umap_median_csv_out, index=False)

# ==============================
# 2D UMAP DISTANCE MATRIX
# ==============================
umap_dist_matrix = np.zeros((len(plot_order), len(plot_order)), dtype=float)

for i, g1 in enumerate(plot_order):
    x1 = umap_median_df.loc[umap_median_df["Group"] == g1, "Median_UMAP1"].values[0]
    y1 = umap_median_df.loc[umap_median_df["Group"] == g1, "Median_UMAP2"].values[0]

    for j, g2 in enumerate(plot_order):
        x2 = umap_median_df.loc[umap_median_df["Group"] == g2, "Median_UMAP1"].values[0]
        y2 = umap_median_df.loc[umap_median_df["Group"] == g2, "Median_UMAP2"].values[0]

        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        umap_dist_matrix[i, j] = dist

umap_dist_df = pd.DataFrame(
    umap_dist_matrix,
    index=plot_order,
    columns=plot_order
)

umap_dist_df.to_csv(umap_dist_csv_out)

umap_pair_rows = []

for i in range(len(plot_order)):
    for j in range(i + 1, len(plot_order)):
        umap_pair_rows.append({
            "Group1": plot_order[i],
            "Group2": plot_order[j],
            "Distance_2D_UMAP": umap_dist_matrix[i, j]
        })

umap_pair_df = pd.DataFrame(umap_pair_rows).sort_values(
    "Distance_2D_UMAP",
    ascending=True
)

umap_pair_df.to_csv(umap_pair_csv_out, index=False)

print("\n2D UMAP median coordinates:")
print(umap_median_df)

print("\nPairwise Euclidean distances between group medians in 2D UMAP space:")
print(umap_dist_df)

print("\nRanked pairwise distances in 2D UMAP space:")
print(umap_pair_df)

# ==============================
# HELPER: STYLE AXES
# ==============================
def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_SPINE_WIDTH)

    ax.tick_params(
        axis="both",
        which="both",
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        labelsize=TICK_SIZE
    )

# ==============================
# UMAP PLOT
# ==============================
fig, ax = plt.subplots(figsize=FIGSIZE_UMAP, dpi=300)

for group in plot_order:
    sub = df[df["Group"] == group]

    ax.scatter(
        sub["UMAP1"],
        sub["UMAP2"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        color=colors[group],
        label=group,
        linewidths=0.3
    )

for group in plot_order:
    sub_med = umap_median_df[umap_median_df["Group"] == group]

    median_x = sub_med["Median_UMAP1"].values[0]
    median_y = sub_med["Median_UMAP2"].values[0]

    ax.scatter(
        median_x,
        median_y,
        s=MEDIAN_SIZE,
        color=colors[group],
        edgecolor="black",
        linewidth=MEDIAN_EDGEWIDTH,
        marker="X",
        zorder=10
    )

ax.set_xlabel(
    "UMAP1",
    fontsize=X_LABEL_SIZE,
    fontweight=X_LABEL_WEIGHT,
    rotation=X_LABEL_ROTATION
)

ax.set_ylabel(
    "UMAP2",
    fontsize=Y_LABEL_SIZE,
    fontweight=Y_LABEL_WEIGHT,
    rotation=Y_LABEL_ROTATION
)

if SHOW_UMAP_TITLE:
    ax.set_title(
        "UMAP of Mitochondrial Features",
        fontsize=TITLE_SIZE,
        fontweight=TITLE_WEIGHT
    )

style_axes(ax)

ax.legend(
    frameon=False,
    fontsize=LEGEND_SIZE,
    markerscale=LEGEND_MARKERSCALE
)

plt.tight_layout()
plt.savefig(umap_svg_out, format="svg", bbox_inches="tight")
plt.savefig(umap_png_out, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==============================
# HEATMAP: 2D UMAP MEDIAN DISTANCES
# ==============================
fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP, dpi=300)

im = ax.imshow(
    umap_dist_matrix,
    cmap="coolwarm",
    alpha=0.5
)

ax.set_xticks(np.arange(len(plot_order)))
ax.set_yticks(np.arange(len(plot_order)))

ax.set_xticklabels(
    plot_order,
    fontsize=TICK_SIZE,
    rotation=HEATMAP_XTICK_ROTATION,
    ha="right"
)

ax.set_yticklabels(
    plot_order,
    fontsize=TICK_SIZE,
    rotation=HEATMAP_YTICK_ROTATION
)

for i in range(len(plot_order)):
    for j in range(len(plot_order)):
        ax.text(
            j,
            i,
            f"{umap_dist_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=HEATMAP_NUMBER_SIZE,
            color="black"
        )

if SHOW_HEATMAP_TITLE:
    ax.set_title(
        "Distance Between Group Median UMAP Positions",
        fontsize=TITLE_SIZE,
        fontweight=TITLE_WEIGHT
    )

style_axes(ax)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(
    labelsize=COLORBAR_TICK_SIZE,
    width=TICK_WIDTH,
    length=TICK_LENGTH
)

plt.tight_layout()
plt.savefig(umap_heatmap_svg_out, format="svg", bbox_inches="tight")
plt.savefig(umap_heatmap_png_out, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==============================
# HEATMAP: ORIGINAL FEATURE-SPACE MEDIAN DISTANCES
# ==============================
fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP, dpi=300)

im = ax.imshow(
    feature_dist_matrix,
    cmap="coolwarm",
    alpha=0.5
)

ax.set_xticks(np.arange(len(plot_order)))
ax.set_yticks(np.arange(len(plot_order)))

ax.set_xticklabels(
    plot_order,
    fontsize=TICK_SIZE,
    rotation=HEATMAP_XTICK_ROTATION,
    ha="right"
)

ax.set_yticklabels(
    plot_order,
    fontsize=TICK_SIZE,
    rotation=HEATMAP_YTICK_ROTATION
)

for i in range(len(plot_order)):
    for j in range(len(plot_order)):
        ax.text(
            j,
            i,
            f"{feature_dist_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=HEATMAP_NUMBER_SIZE,
            color="black"
        )

if SHOW_HEATMAP_TITLE:
    ax.set_title(
        "Distance Between Group Medians\nOriginal Z-scored Feature Space",
        fontsize=TITLE_SIZE,
        fontweight=TITLE_WEIGHT
    )

style_axes(ax)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(
    labelsize=COLORBAR_TICK_SIZE,
    width=TICK_WIDTH,
    length=TICK_LENGTH
)

plt.tight_layout()
plt.savefig(feature_heatmap_svg_out, format="svg", bbox_inches="tight")
plt.savefig(feature_heatmap_png_out, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==============================
# FINAL PRINT
# ==============================
print("\nSaved files:")
print("UMAP SVG:", umap_svg_out)
print("UMAP PNG:", umap_png_out)

print("2D UMAP median coordinates CSV:", umap_median_csv_out)
print("2D UMAP distance matrix CSV:", umap_dist_csv_out)
print("2D UMAP ranked pairwise distance CSV:", umap_pair_csv_out)
print("2D UMAP distance heatmap SVG:", umap_heatmap_svg_out)
print("2D UMAP distance heatmap PNG:", umap_heatmap_png_out)

print("Original feature-space median coordinates CSV:", feature_median_csv_out)
print("Original feature-space distance matrix CSV:", feature_dist_csv_out)
print("Original feature-space ranked pairwise distance CSV:", feature_pair_csv_out)
print("Original feature-space distance heatmap SVG:", feature_heatmap_svg_out)
print("Original feature-space distance heatmap PNG:", feature_heatmap_png_out)
