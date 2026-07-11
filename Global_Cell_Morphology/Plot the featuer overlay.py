#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.spatial import ConvexHull
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def hull_polygon(points_2d: np.ndarray) -> Optional[np.ndarray]:
    pts = np.asarray(points_2d)
    if pts.shape[0] < 3 or not HAVE_SCIPY:
        return None
    h = ConvexHull(pts)
    poly = pts[h.vertices]
    return np.vstack([poly, poly[0]])


def load_numeric_with_group(path: str, group_name: str) -> pd.DataFrame:
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
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=float)


def shade_toward_white(base_rgb01: np.ndarray, t: float) -> tuple:
    t = float(np.clip(t, 0.0, 1.0))
    rgb = (1 - t) * base_rgb01 + t * np.array([1.0, 1.0, 1.0])
    return (rgb[0], rgb[1], rgb[2], 1.0)


def _scale_feature(vals: np.ndarray, zscore=True, clip_pct=(2, 98)):
    v = vals.astype(float).copy()
    if zscore:
        mu = np.nanmean(v)
        sd = np.nanstd(v) + 1e-12
        v = (v - mu) / sd
    lo, hi = np.nanpercentile(v, clip_pct)
    v = np.clip(v, lo, hi)
    return v, lo, hi


def plot_overlay(ax, scores, values_scaled, lo, hi, expl, title, cmap="coolwarm",
                 point_size=38, alpha=0.95, compression=1.0):
    # ONLY visual compression on the plot; PCA coords unchanged
    scores_plot = scores * float(compression)

    ax.scatter(scores_plot[:, 0], scores_plot[:, 1],
               s=12, c="lightgray", alpha=0.18,
               edgecolors="none", zorder=1)

    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    sc = ax.scatter(scores_plot[:, 0], scores_plot[:, 1],
                    s=point_size, c=values_scaled,
                    cmap=cmap, norm=norm,
                    alpha=alpha, edgecolors="none", zorder=2)

    ax.axhline(0, color="0.85", linestyle=":", linewidth=1.0, zorder=0)
    ax.axvline(0, color="0.85", linestyle=":", linewidth=1.0, zorder=0)
    ax.grid(True, linestyle=":", linewidth=0.8, color="0.90")
    ax.set_axisbelow(True)

    ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)", fontsize=15)
    ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)", fontsize=15)
    ax.set_title(title, fontsize=18)

    return sc


# ==============================
# User options (KEEP SAME AS SCRIPT 1 FOR IDENTICAL PCA)
# ==============================
REMOVE_EXTREME_OUTLIER = True
FEATURES_TO_USE: Optional[List[str]] = None

# Base colors
COL: Dict[str, str] = {"Normal": "#7f7f7f", "CTNNB1": "#e377c2", "Nras": "#1f77b4"}

MARKER_LIST = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H", "p", "8"]
SHADE_MIN = 0.05
SHADE_MAX = 0.55

SAVE_PREFIX = "PCA_referenceDots_with_overlays"

# Overlays
MAKE_FEATURE_OVERLAYS = True
FEATURES_TO_OVERLAY = ["CompactNess", "VolumeSphericity"]  # must exist in df numeric columns
OVERLAY_ZSCORE = True
OVERLAY_CLIP_PCT = (2, 98)
OVERLAY_CMAP = "coolwarm"
OVERLAY_ALPHA = 0.95
OVERLAY_POINT_SIZE = 38
OVERLAY_COMPRESSION = 1.0   # set 0.8 if you want only overlay panels tighter (visual only)

SAVE_OVERLAY_INDIVIDUAL = True
SAVE_OVERLAY_STACKED = True
OVERLAY_PREFIX = "PCA_feature_overlay"

# ==============================
# Paths
# ==============================
NORMAL_PATH = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/Normal_withSampleID.xlsx"
BETA_PATH   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/Beta_withSampleID.xlsx"
NRAS_PATH   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/Nras_withSampleID.xlsx"
# ==============================
# Load + PCA (EXACT SAME AS SCRIPT 1)
# ==============================
normal = load_numeric_with_group(NORMAL_PATH, "Normal")
beta   = load_numeric_with_group(BETA_PATH,   "CTNNB1")
nras   = load_numeric_with_group(NRAS_PATH,   "Nras")

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
        raise ValueError(f"FEATURES_TO_USE missing: {missing}")
    features = FEATURES_TO_USE

normal = normal[features + ["Group", "SampleID", "CellID"]].copy()
beta   = beta[features + ["Group", "SampleID", "CellID"]].copy()
nras   = nras[features + ["Group", "SampleID", "CellID"]].copy()

df = pd.concat([normal, beta, nras], ignore_index=True)
df = df.dropna(subset=features + ["Group", "SampleID", "CellID"]).reset_index(drop=True)

X = df[features].values.astype(float)
groups   = df["Group"].values.astype(str)
samples  = df["SampleID"].values.astype(str)
cell_ids = df["CellID"].values.astype(str)

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)
expl = pca.explained_variance_ratio_ * 100

if REMOVE_EXTREME_OUTLIER and scores.shape[0] >= 3:
    outlier_idx = int(np.argmax(np.abs(scores[:, 1])))
    scores   = np.delete(scores,   outlier_idx, axis=0)
    groups   = np.delete(groups,   outlier_idx, axis=0)
    samples  = np.delete(samples,  outlier_idx, axis=0)
    cell_ids = np.delete(cell_ids, outlier_idx, axis=0)
    df = df.drop(df.index[outlier_idx]).reset_index(drop=True)

# Sample -> marker
unique_samples = sorted(np.unique(samples))
if len(unique_samples) > len(MARKER_LIST):
    raise ValueError("Too many SampleIDs for MARKER_LIST; extend it.")
sample_marker = {s: MARKER_LIST[i] for i, s in enumerate(unique_samples)}

# Shade map
group_base_rgb = {g: hex_to_rgb01(COL[g]) for g in ["Normal", "CTNNB1", "Nras"]}
sample_shade_color = {}
for g in ["Normal", "CTNNB1", "Nras"]:
    s_in_group = sorted(np.unique(samples[groups == g]))
    if len(s_in_group) == 0:
        continue
    t_list = [0.20] if len(s_in_group) == 1 else np.linspace(SHADE_MIN, SHADE_MAX, len(s_in_group))
    for s, t in zip(s_in_group, t_list):
        sample_shade_color[(g, s)] = shade_toward_white(group_base_rgb[g], t)

# ==============================
# Save base PCA figure (same dot arrangement)
# ==============================
plt.figure(figsize=(6.2, 4.7))
ax = plt.gca()

ax.axhline(0, color="0.7", linestyle=":", linewidth=1.0, zorder=0)
ax.axvline(0, color="0.7", linestyle=":", linewidth=1.0, zorder=0)

for g in ["Normal", "CTNNB1", "Nras"]:
    group_idx = (groups == g)
    s_in_group = sorted(np.unique(samples[group_idx]))
    for s in s_in_group:
        idx = group_idx & (samples == s)
        if np.sum(idx) == 0:
            continue
        ax.scatter(scores[idx, 0], scores[idx, 1],
                   s=34, marker=sample_marker[s],
                   c=[sample_shade_color[(g, s)]],
                   alpha=0.80, edgecolors="black", linewidths=0.35, zorder=2)

# hulls + centroids
for g in ["Normal", "CTNNB1", "Nras"]:
    pts = scores[groups == g]
    if pts.shape[0] == 0:
        continue
    poly = hull_polygon(pts)
    if poly is not None:
        ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=1.8,
                color=COL[g], alpha=0.9, zorder=3)
        ax.fill(poly[:, 0], poly[:, 1], color=COL[g], alpha=0.08, zorder=1)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ax.scatter([cx], [cy], s=850, color=COL[g], alpha=0.18, edgecolors="none", zorder=2)
    ax.scatter([cx], [cy], s=70,  color=COL[g], alpha=0.95, edgecolors="white", linewidths=0.8, zorder=4)

ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)", fontsize=13)
ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)", fontsize=13)
ax.grid(True, linestyle=":", linewidth=0.8, color="0.85")
plt.tight_layout()

save_dir = os.path.dirname(NORMAL_PATH)
base_png = os.path.join(save_dir, f"{SAVE_PREFIX}_BASE.png")
base_svg = os.path.join(save_dir, f"{SAVE_PREFIX}_BASE.svg")
plt.savefig(base_png, dpi=400, transparent=True)
plt.savefig(base_svg, transparent=True)
plt.show()

print("Saved base PCA:", base_png)

# ==============================
# Feature overlays (same PCA coords)
# ==============================
if MAKE_FEATURE_OVERLAYS:
    missing_overlay = [f for f in FEATURES_TO_OVERLAY if f not in df.columns]
    if missing_overlay:
        raise ValueError(f"Overlay features not found: {missing_overlay}")

    if SAVE_OVERLAY_INDIVIDUAL:
        for feat in FEATURES_TO_OVERLAY:
            vals = df[feat].values
            vals_s, lo, hi = _scale_feature(vals, zscore=OVERLAY_ZSCORE, clip_pct=OVERLAY_CLIP_PCT)

            fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)
            sc = plot_overlay(ax, scores, vals_s, lo, hi, expl, title=feat,
                              cmap=OVERLAY_CMAP, point_size=OVERLAY_POINT_SIZE,
                              alpha=OVERLAY_ALPHA, compression=OVERLAY_COMPRESSION)

            cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cbar.set_label("z-score" if OVERLAY_ZSCORE else "value", fontsize=16)
            cbar.ax.tick_params(labelsize=14)

            plt.tight_layout()
            out_png = os.path.join(save_dir, f"{OVERLAY_PREFIX}_{feat}.png")
            out_svg = os.path.join(save_dir, f"{OVERLAY_PREFIX}_{feat}.svg")
            plt.savefig(out_png, dpi=400, transparent=True, bbox_inches="tight")
            plt.savefig(out_svg, transparent=True, bbox_inches="tight")
            plt.close(fig)
            print("Saved overlay:", out_png)

    if SAVE_OVERLAY_STACKED:
        n = len(FEATURES_TO_OVERLAY)
        ncols = 1
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4.6 * ncols, 4.0 * nrows),
                                 dpi=300, sharex=True, sharey=True)
        axes_flat = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]

        last_sc = None
        for i, feat in enumerate(FEATURES_TO_OVERLAY):
            ax = axes_flat[i]
            vals = df[feat].values
            vals_s, lo, hi = _scale_feature(vals, zscore=OVERLAY_ZSCORE, clip_pct=OVERLAY_CLIP_PCT)
            last_sc = plot_overlay(ax, scores, vals_s, lo, hi, expl, title=feat,
                                   cmap=OVERLAY_CMAP, point_size=OVERLAY_POINT_SIZE,
                                   alpha=OVERLAY_ALPHA, compression=OVERLAY_COMPRESSION)
            ax.label_outer()

        for j in range(n, nrows * ncols):
            axes_flat[j].axis("off")

        cbar = fig.colorbar(last_sc, ax=axes_flat[:n], fraction=0.03, pad=0.02)
        cbar.set_label("z-score" if OVERLAY_ZSCORE else "value", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        fig.tight_layout(rect=[0, 0, 0.95, 1])
        out_png = os.path.join(save_dir, f"{OVERLAY_PREFIX}_STACKED.png")
        out_svg = os.path.join(save_dir, f"{OVERLAY_PREFIX}_STACKED.svg")
        plt.savefig(out_png, dpi=400, transparent=True, bbox_inches="tight")
        plt.savefig(out_svg, transparent=True, bbox_inches="tight")
        plt.close(fig)

        print("Saved stacked overlays:", out_png)
