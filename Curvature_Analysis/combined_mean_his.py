#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================
SHEET_NAME = "sampled_vertices"
XRANGE = (-0.2, 1)
N_BINS = 40

# Add your 3 excel files here
DATASETS = [
    ("Normal", r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_curvature/curvature_results_normal.xlsx"),
    ("NRAS",   r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_curvature/curvature_results_nras.xlsx"),
    ("CTNNB1",   r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_curvature/curvature_results_beta.xlsx"),
]

# Colors (change if you want)
COLORS = {
    "Normal": "#7f7f7f",
    "NRAS":   "#1f77b4",
    "CTNNB1":   "#e377c2",
}

# If you know exact column names, set them here.
# Otherwise leave as None and the script will auto-detect PER FILE.
CELL_ID_COL = None   # e.g., "cell_id"
CURV_COL    = None   # e.g., "mean_curvature" or "H_um_inv"

# Save format
DPI = 300

# Output
# (Saves in the folder of the FIRST excel file)
OUT_NAME = "AllTypes_combined_mean_SEM.png"
OUT_CSV  = "AllTypes_binned_stats_mean_SEM.csv"

# =========================
# Helpers
# =========================
def autodetect_cell_id_col(df: pd.DataFrame) -> str:
    candidates = ["cell_id", "cell", "cellid", "cellID", "label", "object", "obj", "id"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    for c in df.columns:
        if "cell" in c.lower():
            return c
    raise ValueError("Could not auto-detect a cell id column. Please set CELL_ID_COL manually.")

def autodetect_curv_col(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found for curvature.")
    patterns = [
        ("mean_curv", 5),
        ("mean curvature", 5),
        ("curvature", 4),
        ("curv", 3),
        ("h_um", 2),
        ("mean", 1),
        ("h", 1),
    ]
    best, best_score = None, -1
    for c in numeric_cols:
        name = c.lower()
        score = 0
        for p, w in patterns:
            if p in name:
                score += w
        if score > best_score:
            best_score, best = score, c
    return best if best is not None else numeric_cols[0]

def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-_\. ]", "_", s)
    s = s.strip().replace(" ", "_")
    return s[:180] if len(s) > 180 else s

def compute_cell_histograms(excel_path: str, sheet: str, x_range, n_bins,
                            cell_id_col=None, curv_col=None):
    """Returns: bin_centers, bins, cell_histograms (n_cells x n_bins), used cols, n_cells"""
    df = pd.read_excel(excel_path, sheet_name=sheet)

    # Detect columns per file if not provided
    cid = cell_id_col if cell_id_col is not None else autodetect_cell_id_col(df)
    ccol = curv_col if curv_col is not None else autodetect_curv_col(df)

    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    cell_histograms = []
    for _, g in df.groupby(cid):
        vals = g[ccol].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        hist, _ = np.histogram(vals, bins=bins)
        if hist.sum() == 0:
            continue
        cell_histograms.append(hist / hist.sum())  # fraction per bin (per-cell normalized)

    cell_histograms = np.array(cell_histograms)
    if cell_histograms.size == 0:
        raise RuntimeError(f"No valid cells/histograms in {excel_path}. Check columns/data.")

    return bin_centers, bins, cell_histograms, cid, ccol, cell_histograms.shape[0]

# =========================
# Main: compute mean+SEM per type and plot together
# =========================
# Global bins (must be identical for all types)
bins = np.linspace(XRANGE[0], XRANGE[1], N_BINS + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Store stats for CSV
all_stats_rows = []

plt.figure(figsize=(12, 6))

for cell_type, excel_path in DATASETS:
    bc, b, cell_hists, used_cid, used_ccol, n_cells = compute_cell_histograms(
        excel_path, SHEET_NAME, XRANGE, N_BINS, CELL_ID_COL, CURV_COL
    )

    # mean + SEM across cells
    mean_hist = cell_hists.mean(axis=0)
    std_hist = cell_hists.std(axis=0, ddof=1) if n_cells > 1 else np.zeros_like(mean_hist)
    sem_hist = std_hist / np.sqrt(n_cells) if n_cells > 0 else np.zeros_like(mean_hist)

    color = COLORS.get(cell_type, None)

    # Mean line
    plt.plot(bin_centers, mean_hist, linewidth=3, label=f"{cell_type}", color=color)

    # SEM band
    plt.fill_between(
        bin_centers,
        mean_hist - sem_hist,
        mean_hist + sem_hist,
        alpha=0.20,
        color=color
    )

    # Save rows for CSV
    for i in range(len(bin_centers)):
        all_stats_rows.append({
            "cell_type": cell_type,
            "bin_left": bins[i],
            "bin_right": bins[i + 1],
            "bin_center": bin_centers[i],
            "mean_fraction": mean_hist[i],
            "sem_fraction": sem_hist[i],
            "n_cells": n_cells
        })

plt.xlim(*XRANGE)
plt.xlabel("Mean Curvature (μm⁻¹)")
plt.ylabel("Fraction of Surface Vertices (per cell normalized)")
plt.title("Curvature Distribution (cell-as-unit) | mean ± SEM")
plt.legend(frameon=False)
plt.tight_layout()

# Output dir = folder of the first dataset
out_root = os.path.dirname(os.path.abspath(DATASETS[0][1]))
out_png = os.path.join(out_root, OUT_NAME)
plt.savefig(out_png, dpi=DPI)
plt.close()

# Save CSV
stats_df = pd.DataFrame(all_stats_rows)
out_csv = os.path.join(out_root, OUT_CSV)
stats_df.to_csv(out_csv, index=False)

print(f"Saved overlay plot to: {out_png}")
print(f"Saved binned stats to: {out_csv}")
