#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inputs:
  NORMAL_PATH / BETA_PATH / NRAS_PATH
  Each Excel must include columns:
    - SampleID
    - CellID
    - numeric feature columns

Outputs (saved into the folder that contains Normal.xlsx):
  <SAVE_PREFIX>.png
  <SAVE_PREFIX>.svg
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict

# Optional: convex hull via scipy
try:
    from scipy.spatial import ConvexHull
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def hull_polygon(points_2d: np.ndarray) -> Optional[np.ndarray]:
    """Return closed polygon vertices for convex hull, or None if not available."""
    pts = np.asarray(points_2d)
    if pts.shape[0] < 3 or not HAVE_SCIPY:
        return None
    h = ConvexHull(pts)
    poly = pts[h.vertices]
    return np.vstack([poly, poly[0]])


def load_numeric_with_group(path: str, group_name: str) -> pd.DataFrame:
    """
    Read Excel file and return:
      - numeric feature columns
      - Group (cell type)
      - SampleID (for marker shapes + shaded colors)
      - CellID (for labeling points)
    """
    df = pd.read_excel(path)

    for col in ["SampleID", "CellID"]:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in file:\n  {path}")

    sample_col = df["SampleID"].astype(str).values
    cell_col   = df["CellID"].astype(str).values

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df["Group"] = group_name
    numeric_df["SampleID"] = sample_col
    numeric_df["CellID"] = cell_col
    return numeric_df


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    """'#RRGGBB' -> np.array([r,g,b]) in [0,1]."""
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=float)


def shade_toward_white(base_rgb01: np.ndarray, t: float) -> tuple:
    """
    Blend base color toward white by fraction t in [0,1].
      t=0 -> base, t=1 -> white
    Returns RGBA tuple usable by matplotlib.
    """
    t = float(np.clip(t, 0.0, 1.0))
    rgb = (1 - t) * base_rgb01 + t * np.array([1.0, 1.0, 1.0])
    return (rgb[0], rgb[1], rgb[2], 1.0)


# ==============================
# User options
# ==============================
REMOVE_EXTREME_OUTLIER = True

# Set to None to use all shared numeric columns.
FEATURES_TO_USE: Optional[List[str]] = None

SAVE_PREFIX = "PCA_groupColor_sampleShape_gradient_CellID"

# Base colors for cell type groups (paper style)
COL: Dict[str, str] = {"Normal": "#7f7f7f", "CTNNB1": "#e377c2", "Nras": "#1f77b4"}

# Marker list for SampleIDs (extend if you have more)
MARKER_LIST = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H", "p", "8"]

# Gradient strength (how much to lighten samples within each group)
SHADE_MIN = 0.05
SHADE_MAX = 0.55

# CellID label appearance
LABEL_EACH_POINT = True
LABEL_FONTSIZE = 3
LABEL_ALPHA = 0.75

# ==============================
# Paths (EDIT THESE)
# ==============================
NORMAL_PATH = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/global/WithsampleID/Normal_withSampleID.xlsx"
BETA_PATH   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/global/WithsampleID/Beta_withSampleID.xlsx"
NRAS_PATH   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/global/WithsampleID/Nras_withSampleID.xlsx"

# ==============================
# Load data
# ==============================
normal = load_numeric_with_group(NORMAL_PATH, "Normal")
beta   = load_numeric_with_group(BETA_PATH,   "CTNNB1")
nras   = load_numeric_with_group(NRAS_PATH,   "Nras")

# ==============================
# Choose / align feature columns across the 3 files
# ==============================
n_feat = set([c for c in normal.columns if c not in ["Group", "SampleID", "CellID"]])
b_feat = set([c for c in beta.columns   if c not in ["Group", "SampleID", "CellID"]])
r_feat = set([c for c in nras.columns   if c not in ["Group", "SampleID", "CellID"]])

common_features = sorted(list(n_feat & b_feat & r_feat))
if len(common_features) < 2:
    raise ValueError(f"Not enough shared numeric features across files. Shared: {common_features}")

if FEATURES_TO_USE is None or len(FEATURES_TO_USE) == 0:
    features = common_features
else:
    missing = [c for c in FEATURES_TO_USE if c not in common_features]
    if missing:
        raise ValueError(
            "These selected FEATURES_TO_USE are missing from at least one file.\n"
            f"Missing: {missing}\n"
            f"Available shared features: {common_features}"
        )
    features = FEATURES_TO_USE

# Keep only selected features + Group + SampleID + CellID
normal = normal[features + ["Group", "SampleID", "CellID"]].copy()
beta   = beta[features + ["Group", "SampleID", "CellID"]].copy()
nras   = nras[features + ["Group", "SampleID", "CellID"]].copy()

df = pd.concat([normal, beta, nras], ignore_index=True)

# Drop rows with NaNs
df = df.dropna(subset=features + ["Group", "SampleID", "CellID"]).reset_index(drop=True)

X = df[features].values.astype(float)
groups = df["Group"].values.astype(str)
samples = df["SampleID"].values.astype(str)
cell_ids = df["CellID"].values.astype(str)

# ==============================
# PCA (standardized)
# ==============================
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)
expl = pca.explained_variance_ratio_ * 100

# Optional: remove one extreme outlier (largest |PC2|)
if REMOVE_EXTREME_OUTLIER and scores.shape[0] >= 3:
    outlier_idx = int(np.argmax(np.abs(scores[:, 1])))
    scores = np.delete(scores, outlier_idx, axis=0)
    groups = np.delete(groups, outlier_idx, axis=0)
    samples = np.delete(samples, outlier_idx, axis=0)
    cell_ids = np.delete(cell_ids, outlier_idx, axis=0)

# ==============================
# Map SampleID -> marker shape (global)
# ==============================
unique_samples = sorted(np.unique(samples))
if len(unique_samples) > len(MARKER_LIST):
    raise ValueError(
        f"Too many SampleIDs ({len(unique_samples)}) for MARKER_LIST ({len(MARKER_LIST)}). "
        "Please extend MARKER_LIST."
    )
sample_marker = {s: MARKER_LIST[i] for i, s in enumerate(unique_samples)}

# ==============================
# Map (Group, SampleID) -> shaded color (gradient within original group color)
# ==============================
group_base_rgb = {g: hex_to_rgb01(COL[g]) for g in ["Normal", "CTNNB1", "Nras"]}
sample_shade_color = {}  # key: (group, sampleID) -> RGBA

for g in ["Normal", "CTNNB1", "Nras"]:
    s_in_group = sorted(np.unique(samples[groups == g]))
    if len(s_in_group) == 0:
        continue

    if len(s_in_group) == 1:
        t_list = [0.20]
    else:
        t_list = np.linspace(SHADE_MIN, SHADE_MAX, len(s_in_group))

    for s, t in zip(s_in_group, t_list):
        sample_shade_color[(g, s)] = shade_toward_white(group_base_rgb[g], t)

# ==============================
# Plot
# ==============================
plt.figure(figsize=(6.2, 4.7))
ax = plt.gca()

# Light crosshair at origin
ax.axhline(0, color="0.7", linestyle=":", linewidth=1.0, zorder=0)
ax.axvline(0, color="0.7", linestyle=":", linewidth=1.0, zorder=0)

# Scatter: marker = SampleID, color = shaded (still original group hue)
for g in ["Normal", "CTNNB1", "Nras"]:
    group_idx = (groups == g)
    s_in_group = sorted(np.unique(samples[group_idx]))

    for s in s_in_group:
        idx = group_idx & (samples == s)
        if np.sum(idx) == 0:
            continue

        ax.scatter(
            scores[idx, 0],
            scores[idx, 1],
            s=34,
            marker=sample_marker[s],
            c=[sample_shade_color[(g, s)]],
            alpha=0.80,
            edgecolors="black",
            linewidths=0.35,
            zorder=2,
        )

# Axis limits (nice padding) -- do this BEFORE labeling to compute offsets correctly
score_xmax = float(np.max(np.abs(scores[:, 0]))) if scores.size else 1.0
score_ymax = float(np.max(np.abs(scores[:, 1]))) if scores.size else 1.0
xmax = max(score_xmax, 1e-9) * 1.25
ymax = max(score_ymax, 1e-9) * 1.25
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)

# ---------------------------------
# Label each dot with CellID
# ---------------------------------
if LABEL_EACH_POINT:
    dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for i in range(scores.shape[0]):
        ax.text(
            scores[i, 0] + dx,
            scores[i, 1] + dy,
            cell_ids[i],
            fontsize=LABEL_FONTSIZE,
            alpha=LABEL_ALPHA,
            color="black",
            zorder=10,
        )

# Group hulls + centroid bubbles (use ORIGINAL base color)
for g in ["Normal", "CTNNB1", "Nras"]:
    pts = scores[groups == g]
    if pts.shape[0] == 0:
        continue

    poly = hull_polygon(pts)
    if poly is not None:
        ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=1.8,
                color=COL[g], alpha=0.9, zorder=3)
        ax.fill(poly[:, 0], poly[:, 1], color=COL[g], alpha=0.08, zorder=1)

    # Centroid bubble + dot
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ax.scatter([cx], [cy], s=850, color=COL[g], alpha=0.18, edgecolors="none", zorder=2)
    ax.scatter([cx], [cy], s=70,  color=COL[g], alpha=0.95, edgecolors="white",
               linewidths=0.8, zorder=4)

# Labels with variance %
ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)", fontsize=13)
ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)", fontsize=13)

# Grid + frame
ax.grid(True, linestyle=":", linewidth=0.8, color="0.85")
ax.set_axisbelow(True)
ax.tick_params(axis="both", labelsize=10, length=5, width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(1.1)
    spine.set_color("0.35")

# ------------------------------
# Two legends:
#   1) Cell type (base color)
#   2) SampleID (shape + representative shade)
# ------------------------------
from matplotlib.lines import Line2D

group_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=COL[g], markeredgecolor="black",
           markersize=8, label=g)
    for g in ["Normal", "CTNNB1", "Nras"]
]
legend1 = ax.legend(handles=group_handles, title="Cell type",
                    frameon=False, fontsize=10, title_fontsize=10,
                    loc="upper left")
ax.add_artist(legend1)

sample_handles = []
for s in unique_samples:
    # pick representative color for this sample from the first group where it exists
    rep_color = None
    for g in ["Normal", "CTNNB1", "Nras"]:
        if (g, s) in sample_shade_color:
            rep_color = sample_shade_color[(g, s)]
            break
    if rep_color is None:
        rep_color = (0.2, 0.2, 0.2, 1.0)

    sample_handles.append(
        Line2D([0], [0], marker=sample_marker[s], color='w',
               markerfacecolor=rep_color, markeredgecolor="black",
               linestyle='None', markersize=8, label=s)
    )

ax.legend(handles=sample_handles, title="SampleID (shape + shade)",
          frameon=False, fontsize=9, title_fontsize=10,
          loc="lower right", ncol=1)

plt.tight_layout()

# ==============================
# Save
# ==============================
save_dir = os.path.dirname(NORMAL_PATH)
png_path = os.path.join(save_dir, f"{SAVE_PREFIX}.png")
svg_path = os.path.join(save_dir, f"{SAVE_PREFIX}.svg")

plt.savefig(png_path, dpi=400, transparent=True)
plt.savefig(svg_path, transparent=True)
plt.show()

print("Saved:")
print("  ", png_path)
print("  ", svg_path)
print("Features used:", features)
print("SampleIDs:", unique_samples)
