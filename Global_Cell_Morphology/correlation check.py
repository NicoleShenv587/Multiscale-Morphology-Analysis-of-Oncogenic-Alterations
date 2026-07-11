#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
workbook format:
- One Excel file
- Each FEATURE is a separate sheet
- Each feature sheet has 3 columns: Normal, CTNNB1, NRAS (one value per row/cell)
- There is a 'SampleID' sheet with columns Normal/CTNNB1/NRAS (optional, used as metadata)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Arial"
# ==============================
# 1) User settings
# ==============================
XLSX_PATH = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/GraphPad_Feature_Comparison.xlsx"  # <-- change this
OUTDIR = os.path.dirname(os.path.abspath(XLSX_PATH))

CELLTYPES = ["Normal", "CTNNB1", "NRAS"]

# sheets you do NOT want to treat as features
EXCLUDE_SHEETS = {"cellID", "NShortLength", "SampleID"}  # add more if needed, not measurement data

# min #cells required to compute a correlation pair (after dropping NaNs)
MIN_PAIRS = 8

# ==============================
# 2) Load workbook + decide feature sheets
# ==============================
xl = pd.ExcelFile(XLSX_PATH)
feature_sheets = [s for s in xl.sheet_names if s not in EXCLUDE_SHEETS]

# Optional SampleID sheet (used only as metadata, not as a feature)
has_sampleid = "SampleID" in xl.sheet_names
if has_sampleid:
    df_sample = pd.read_excel(XLSX_PATH, sheet_name="SampleID")
else:
    df_sample = None


# ==============================
# 3) Build combined per-cell table
# ==============================
combined_blocks = []

for ct in CELLTYPES:
    # Determine how many rows/cells exist for this celltype:
    # Prefer SampleID sheet if present; otherwise use the first feature sheet
    if df_sample is not None and ct in df_sample.columns:
        n = int(df_sample[ct].notna().sum())
    else:
        first_feat = pd.read_excel(XLSX_PATH, sheet_name=feature_sheets[0])
        n = int(first_feat[ct].notna().sum())

    block = pd.DataFrame({
        "CellType": [ct] * n,
        "RowIndex": np.arange(1, n + 1)
    })

    if df_sample is not None and ct in df_sample.columns:
        block["SampleID"] = df_sample.loc[:n - 1, ct].astype("Int64").to_numpy()

    # add each feature column
    for feat in feature_sheets:
        df_feat = pd.read_excel(XLSX_PATH, sheet_name=feat)
        if ct not in df_feat.columns:
            raise ValueError(f"Sheet '{feat}' is missing column '{ct}'")
        block[feat] = df_feat.loc[:n - 1, ct].to_numpy()

    combined_blocks.append(block)

all_df = pd.concat(combined_blocks, ignore_index=True)


# ==============================
# 4) Pearson correlation + p-values (pooled across all 3 groups)
# ==============================
feat_cols = feature_sheets

# Use pairwise-complete observations (drop NaNs per pair)
corr = pd.DataFrame(index=feat_cols, columns=feat_cols, dtype=float)
pvals = pd.DataFrame(index=feat_cols, columns=feat_cols, dtype=float)

for f1 in feat_cols:
    for f2 in feat_cols:
        x = all_df[f1].to_numpy(dtype=float)
        y = all_df[f2].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < MIN_PAIRS:
            corr.loc[f1, f2] = np.nan
            pvals.loc[f1, f2] = np.nan
        else:
            r, p = pearsonr(x[m], y[m])
            corr.loc[f1, f2] = r
            pvals.loc[f1, f2] = p

# Ranked pairs list
pairs = []
for i in range(len(feat_cols)):
    for j in range(i + 1, len(feat_cols)):
        f1, f2 = feat_cols[i], feat_cols[j]
        r = corr.loc[f1, f2]
        p = pvals.loc[f1, f2]
        if np.isfinite(r):
            pairs.append([f1, f2, float(r), float(p), float(abs(r))])

pairs_df = pd.DataFrame(
    pairs, columns=["Feature1", "Feature2", "Pearson_r", "p_value", "abs_r"]
).sort_values("abs_r", ascending=False)


# ==============================
# 5) Save to Excel
# ==============================
out_xlsx = os.path.join(OUTDIR, "Combined_Features_for_Correlation.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    all_df.to_excel(w, index=False, sheet_name="Combined")
    corr.to_excel(w, sheet_name="Pearson_r")
    pvals.to_excel(w, sheet_name="p_values")
    pairs_df.to_excel(w, index=False, sheet_name="Pairs_ranked")

print("Saved:", out_xlsx)


# ==============================
# 6) Heatmap (matplotlib only)
# ==============================
vals = corr.to_numpy(dtype=float)

fig = plt.figure(figsize=(10, 9))
ax = plt.gca()

im = ax.imshow(vals, vmin=-1, vmax=1, cmap="vlag")

ax.set_xticks(range(len(feat_cols)))
ax.set_yticks(range(len(feat_cols)))
ax.set_xticklabels(feat_cols, rotation=90, fontsize=20)   # bigger font
ax.set_yticklabels(feat_cols, fontsize=20)                # bigger font

cbar = plt.colorbar(im)
cbar.set_label("Pearson r", fontsize=20)
cbar.ax.tick_params(labelsize=18)

ax.set_title("Feature–Feature Correlation (All groups combined)",
             fontsize=20, pad=15)

plt.tight_layout()

out_png = os.path.join(OUTDIR, "Feature_Correlation_Heatmap.png")
out_svg = os.path.join(OUTDIR, "Feature_Correlation_Heatmap.svg")
plt.savefig(out_png, dpi=400)
plt.savefig(out_svg)
plt.close(fig)
# ==============================
# 7) Clustered correlation plot (seaborn clustermap)
# ==============================
corr_df = corr.astype(float)

# (Important) if there are any NaNs (rare), fill diagonal and remaining NaNs
np.fill_diagonal(corr_df.values, 1.0)
corr_df = corr_df.fillna(0.0)

# # ---- Option A (simple): just clustermap, no color bars ----
# g = sns.clustermap(
#     corr_df,
#     center=0,
#     cmap="vlag",
#     linewidths=.8,
#     figsize=(12, 13),
#     dendrogram_ratio=(.1, .2),
#     cbar_pos=(.02, .32, .03, .2),
#     method="average",          # clustering method
#     metric="euclidean"         # distance metric on rows/cols of corr matrix
# )
# out_cluster_png = os.path.join(OUTDIR, "Feature_Correlation_Clustermap.png")
# out_cluster_svg = os.path.join(OUTDIR, "Feature_Correlation_Clustermap.svg")
# g.savefig(out_cluster_png, dpi=400)
# g.savefig(out_cluster_svg)
# print("Saved:", out_cluster_png)
# print("Saved:", out_cluster_svg)

# ---- Option B (better): add row/col color bars from feature clusters ----

# Convert correlation to a distance matrix suitable for clustering:
# distance = 1 - corr (so highly correlated features are "close")
dist = 1 - corr_df
np.fill_diagonal(dist.values, 0.0)

# squareform expects a condensed distance vector
dist_condensed = squareform(dist.values, checks=False)

# Hierarchical clustering
Z = linkage(dist_condensed, method="average")

# Choose number of feature clusters (tune this: 4–8 often looks good)
N_CLUSTERS = 5
cluster_id = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

# Make a color for each cluster and map features -> colors
palette = sns.color_palette("tab10", n_colors=N_CLUSTERS)
feature_to_color = {
    feat: palette[cid - 1] for feat, cid in zip(corr_df.index, cluster_id)
}

# seaborn accepts a list/Series of colors aligned to rows/cols
network_colors = pd.Series(corr_df.index, index=corr_df.index).map(feature_to_color)

g = sns.clustermap(
    corr_df,
    center=0,
    cmap="vlag",
    vmin=-1, vmax=1,
    row_colors=network_colors,
    col_colors=network_colors,
    linewidths=.75,
    figsize=(12, 13),
    dendrogram_ratio=(.1, .2),
    cbar_pos=(.02, .32, .03, .2),
    method="average",
    metric="euclidean"
)

# Increase X/Y tick label size
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=50)
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=50)

# Make colorbar text bigger
g.cax.tick_params(labelsize=50)
g.cax.set_ylabel("Pearson r", fontsize=50)

# Make title bigger (if you use one)
g.ax_heatmap.set_title(
    "Feature–Feature Correlation (All groups combined)",
    fontsize=55,
    pad=25
)

# Optional: make dendrogram tick labels bigger (rarely needed but safe)
plt.setp(g.ax_row_dendrogram.get_xticklabels(), fontsize=20)
plt.setp(g.ax_col_dendrogram.get_yticklabels(), fontsize=20)
out_cluster_png = os.path.join(OUTDIR, "Feature_Correlation_Clustermap_withClusterColors.png")
out_cluster_svg = os.path.join(OUTDIR, "Feature_Correlation_Clustermap_withClusterColors.svg")
g.savefig(out_cluster_png, dpi=400)
g.savefig(out_cluster_svg)
print("Saved:", out_cluster_png)
print("Saved:", out_cluster_svg)


# Print top correlations
print("\nTop 10 strongest |r| feature pairs:")
print(pairs_df.head(10).to_string(index=False))
