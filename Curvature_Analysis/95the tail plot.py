#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# NEW: Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ==============================
# File paths
# ==============================
normal_path = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/curvature_1/curvature_results_normal.xlsx"
beta_path   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/curvature_1/curvature_results_beta.xlsx"
nras_path   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/curvature_1/curvature_results_nras.xlsx"

output_dir = os.path.dirname(os.path.abspath(normal_path))
save_path_png  = os.path.join(output_dir, "P95_violin_38x50mm.png")
save_path_svg  = os.path.join(output_dir, "P95_violin_38x50mm.svg")

SHEET = "sampled_vertices"

# ==============================
# Exact figure size in mm
# ==============================
MM_TO_INCH = 1 / 25.4
FIG_W = 80 * MM_TO_INCH
FIG_H = 100 * MM_TO_INCH


# ==============================
# Compute per-cell 95th percentile
# ==============================
def per_cell_p95(path):
    df = pd.read_excel(path, sheet_name=SHEET)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric columns found in sheet '{SHEET}' for file:\n  {path}")

    curv_col = next((c for c in numeric_cols if "curv" in str(c).lower()), numeric_cols[0])

    if "cell_id" in df.columns:
        cell_col = "cell_id"
    elif "cell" in df.columns:
        cell_col = "cell"
    else:
        non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if len(non_num) == 0:
            raise ValueError(f"Could not infer cell id column in file:\n  {path}")
        cell_col = non_num[0]

    df = df[[cell_col, curv_col]].rename(columns={cell_col: "cell", curv_col: "curv"}).dropna()
    df = df[np.isfinite(df["curv"].to_numpy())]

    return df.groupby("cell")["curv"].apply(
        lambda x: np.percentile(np.abs(x.to_numpy()), 95)
    ).to_numpy()


# ==============================
# Helpers for significance
# ==============================
def p_to_stars(p):
    # Tukey-adjusted p-values
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    else:
        return "ns"


def add_sig_bar(ax, x1, x2, y, h, text, lw=1.8, fs=12):
    """
    Draws a bracket from x1 to x2 at height y, with bracket height h,
    and puts `text` (stars) centered above.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c="black", lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fs)


# ==============================
# Load data
# ==============================
p95_normal = per_cell_p95(normal_path)
p95_beta   = per_cell_p95(beta_path)
p95_nras   = per_cell_p95(nras_path)

groups = [p95_normal, p95_beta, p95_nras]
labels = ["Normal", "CTNNB1", "NRAS"]

data_clean = [np.asarray(g)[np.isfinite(g)] for g in groups]


# ==============================
# ANOVA + Tukey HSD
# ==============================
F, p_anova = stats.f_oneway(*data_clean)
print("\nOmnibus: one-way ANOVA (classic)")
print(f"F = {F:.6g}")
print(f"p = {p_anova:.16e}")  # full precision

# Prepare long-form for Tukey
vals = np.concatenate(data_clean)
grps = np.concatenate([[labels[i]] * len(data_clean[i]) for i in range(len(labels))])

tukey = pairwise_tukeyhsd(endog=vals, groups=grps, alpha=0.05)
tukey_table = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])

print("\nPosthoc: Tukey HSD (adjusted p-values)")
for _, r in tukey_table.iterrows():
    g1, g2, p_adj = r["group1"], r["group2"], float(r["p-adj"])
    print(f"  {g1} vs {g2} : {p_adj:.16e}")


# ==============================
# Plot
# ==============================
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=600)

x = np.arange(len(labels))
colors = ["#7f7f7f", "#e377c2", "#1f77b4"]

# Violin
vp = ax.violinplot(
    data_clean,
    positions=x,
    widths=0.75,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for body, c in zip(vp["bodies"], colors):
    body.set_facecolor(c)
    body.set_edgecolor("black")
    body.set_alpha(0.35)
    body.set_linewidth(1.5)

# Points
rng = np.random.default_rng(0)
for i, (g, c) in enumerate(zip(data_clean, colors)):
    jitter = rng.uniform(-0.09, 0.09, size=len(g))
    ax.scatter(
        np.full(len(g), x[i]) + jitter,
        g,
        s=24,
        facecolor=c,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        zorder=3
    )

# Dashed mean ± SD inside violin
dash = (0, (4, 3))
half_width = 0.25

for i, g in enumerate(data_clean):
    mean = np.mean(g)
    sd = np.std(g, ddof=1)

    ax.plot([x[i]-half_width, x[i]+half_width], [mean, mean],
            linestyle=dash, linewidth=2, color="black", zorder=5)

    ax.plot([x[i]-half_width, x[i]+half_width], [mean - sd, mean - sd],
            linestyle=dash, linewidth=1.5, color="black", alpha=0.9, zorder=5)

    ax.plot([x[i]-half_width, x[i]+half_width], [mean + sd, mean + sd],
            linestyle=dash, linewidth=1.5, color="black", alpha=0.9, zorder=5)


# ==============================
# Add significance stars (Tukey)
# ==============================
# Map group name -> x position
pos = {lab: xi for lab, xi in zip(labels, x)}

# Start above the max point
y0 = max(np.max(g) for g in data_clean)
yr = (y0 - min(np.min(g) for g in data_clean)) if len(data_clean) else 1.0
y = y0 + 0.05 * yr          # baseline for first bar
h = 0.015 * yr              # bracket height
step = 0.06 * yr            # vertical spacing between bars

# Add only the pairwise comparisons (typical for your plot)
# (Tukey table already has all 3 pairs)
for _, r in tukey_table.iterrows():
    g1, g2, p_adj = r["group1"], r["group2"], float(r["p-adj"])
    stars = p_to_stars(p_adj)
    add_sig_bar(ax, pos[g1], pos[g2], y, h, stars, lw=1.8, fs=14)
    y += step

# Make sure there's headroom
ax.set_ylim(top=y + 0.05 * yr)


# ==============================
# Styling
# ==============================
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)

ax.set_ylabel("95th Percentile |Curvature| per Cell (1/µm)", fontsize=10)

ax.tick_params(axis="y", labelsize=10, width=1.5, length=4)
ax.tick_params(axis="x", labelsize=10, width=1.5, length=0)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(1.5)

fig.tight_layout()
fig.savefig(save_path_png, transparent=True)
fig.savefig(save_path_svg, transparent=True)
plt.close(fig)

print("\nSaved:")
print("PNG:", save_path_png)
print("SVG:", save_path_svg)
print("Figure size (inch):", FIG_W, "x", FIG_H)
