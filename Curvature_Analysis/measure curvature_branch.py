#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 15:09:01 2026

@author: s227698
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
u-Unwrap3D batch workflow:
  - batch over all 3D binary masks in a folder
  - extract surface mesh
  - compute mean curvature
  - export colored mesh (.obj) + save curvature arrays (.npz)
  - combine ALL cells into ONE Excel file:
      * summary_per_cell
      * sampled_vertices

Reference notebook:
https://fyz11.github.io/u-Unwrap3D_notebooks/03_basic_workflow/Step0_find_cell_surface_from_segmentation.html
"""

import os
import glob
import numpy as np
import tifffile as tif
from matplotlib import cm

import pandas as pd
import unwrap3D.Mesh.meshtools as meshtools
import unwrap3D.Visualisation.colors as vol_colors


# ============================================================
# ======================= USER INPUT =========================
# ============================================================

# 1) Where are your masks?
MASK_DIR   = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/cells/visualization_Nras"
PATTERN    = "cell_0008.tif"   # <-- CHANGED: batch all cells, e.g. "cell_*.tif" or "*.tif"

# 2) Where do you want outputs?
OUT_DIR    = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/cells/visualization_Nras/unwrap_1_5/show"

# 3) Voxel size (um)
VOXEL_SIZE_UM = 0.17

# 4) Curvature colormap scaling (units: um^-1)
VMIN_UM_INV = -1
VMAX_UM_INV =  1

# 5) Curvature calculation parameters
SMOOTH_GRADIENT = 5.0
INVERT_H        = False

# 6) Mesh extraction parameters
PRESMOOTH       = 1.0
CONTOURLEVEL    = 0.5
REMESH          = True
REMESH_METHOD   = "CGAL"   # requires CGAL; if fails, set REMESH=False
REMESH_SAMPLES  = 0.5
MIN_MESH_SIZE   = 10000
UPSAMPLEMETHOD  = "inplane"

# If already exists:
# - if True: do NOT recompute OBJ/NPZ, but DO still add this cell to Excel by reading NPZ
# - if False: recompute everything
SKIP_IF_EXISTS = True

# ---------- Excel export options ----------
SAVE_EXCEL = True
EXCEL_NAME = "curvature_results.xlsx"

# Excel cannot store millions of vertices per cell; sample for a distribution sheet.
SAVE_VERTEX_SAMPLES = True
SAMPLE_N_PER_CELL   = 5000   # set 0 to disable sampled_vertices sheet

# Optional: reproducible sampling
RANDOM_SEED = 0

# ============================================================
# ============================================================


def load_binary_mask(path: str) -> np.ndarray:
    """Load a 3D mask and convert to boolean (nonzero = foreground)."""
    m = tif.imread(path)
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {m.shape} from {path}")
    return (m > 0)


def curvature_summary_stats(x: np.ndarray) -> dict:
    """Return compact stats for one cell."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return dict(
            n_vertices=0, mean=np.nan, median=np.nan, std=np.nan,
            min=np.nan, max=np.nan, p05=np.nan, p25=np.nan, p75=np.nan, p95=np.nan
        )
    return dict(
        n_vertices=int(x.size),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        min=float(np.min(x)),
        max=float(np.max(x)),
        p05=float(np.percentile(x, 5)),
        p25=float(np.percentile(x, 25)),
        p75=float(np.percentile(x, 75)),
        p95=float(np.percentile(x, 95)),
    )


def make_summary_row(base: str, mask_path: str, out_obj: str, out_npz: str,
                     surf_H_um_inv: np.ndarray) -> dict:
    stats = curvature_summary_stats(surf_H_um_inv)
    return dict(
        cell=base,
        mask_path=mask_path,
        obj_path=out_obj,
        npz_path=out_npz,
        voxel_size_um=VOXEL_SIZE_UM,
        vmin_um_inv=VMIN_UM_INV,
        vmax_um_inv=VMAX_UM_INV,
        smooth_gradient=SMOOTH_GRADIENT,
        invert_H=INVERT_H,
        **stats
    )


from typing import Optional

def sample_vertices_df(base: str, surf_H_um_inv: np.ndarray) -> Optional[pd.DataFrame]:
    if not (SAVE_VERTEX_SAMPLES and SAMPLE_N_PER_CELL and surf_H_um_inv.size):
        return None
    n = min(int(SAMPLE_N_PER_CELL), int(surf_H_um_inv.size))
    idx = np.random.choice(surf_H_um_inv.size, size=n, replace=False)
    return pd.DataFrame({
        "cell": base,
        "vertex_idx": idx.astype(np.int64),
        "curv_um_inv": surf_H_um_inv[idx].astype(np.float64),
    })


def process_one(mask_path: str):
    """
    Returns:
      - summary_row: dict for summary_per_cell sheet
      - sampled_df:  DataFrame with sampled per-vertex curvature (or None)
    """
    base = os.path.splitext(os.path.basename(mask_path))[0]
    os.makedirs(OUT_DIR, exist_ok=True)

    out_obj = os.path.join(OUT_DIR, f"{base}_curvature_colored.obj")
    out_npz = os.path.join(OUT_DIR, f"{base}_curvature_values.npz")

    # ---------- NEW: If already computed, read NPZ and still contribute to Excel ----------
    if SKIP_IF_EXISTS and os.path.exists(out_npz):
        try:
            d = np.load(out_npz, allow_pickle=True)
            if "surf_H_um_inv" in d:
                surf_H_um_inv = d["surf_H_um_inv"]
            elif "surf_H" in d:
                # fallback if older file stored only surf_H
                surf_H_um_inv = d["surf_H"] / float(VOXEL_SIZE_UM)
            else:
                raise KeyError("NPZ missing surf_H_um_inv (and surf_H).")

            summary_row = make_summary_row(base, mask_path, out_obj, out_npz, surf_H_um_inv)
            sampled_df = sample_vertices_df(base, surf_H_um_inv)

            print(f"[EXISTING->EXCEL] {base} (read NPZ, no recompute)")
            return summary_row, sampled_df
        except Exception as e:
            print(f"[WARN] {base}: NPZ exists but could not be read ({e}). Will recompute.")

    # ---------- Full compute path ----------
    # 1) Load binary
    img_binary = load_binary_mask(mask_path)

    # 2) Mesh extraction (Step0 notebook uses transpose(2,1,0))
    mesh = meshtools.marching_cubes_mesh_binary(
        img_binary.transpose(2, 1, 0),
        presmooth=PRESMOOTH,
        contourlevel=CONTOURLEVEL,
        remesh=REMESH,
        remesh_method=REMESH_METHOD,
        remesh_samples=REMESH_SAMPLES,
        predecimate=False,
        min_mesh_size=MIN_MESH_SIZE,
        upsamplemethod=UPSAMPLEMETHOD,
    )

    # 3) Curvature from binary
    surf_H, (H_binary, H_sdf_vol_normal, H_sdf_vol) = meshtools.compute_mean_curvature_from_binary(
        mesh,
        img_binary.transpose(2, 1, 0),
        smooth_gradient=SMOOTH_GRADIENT,
        eps=1e-12,
        invert_H=INVERT_H,
        return_H_img=True,
    )

    # Convert to um^-1
    surf_H_um_inv = surf_H / float(VOXEL_SIZE_UM)

    # 4) Color mapping
    colors = vol_colors.get_colors(
        surf_H_um_inv,
        colormap=cm.coolwarm, # original is Spectral_r
        vmin=VMIN_UM_INV,
        vmax=VMAX_UM_INV,
    )
    mesh.visual.vertex_colors = np.uint8(255 * colors[..., :3])

    # 5) Export colored mesh
    mesh.export(out_obj)

    # 6) Save arrays
    np.savez_compressed(
        out_npz,
        surf_H=surf_H,
        surf_H_um_inv=surf_H_um_inv,
        vmin_um_inv=VMIN_UM_INV,
        vmax_um_inv=VMAX_UM_INV,
        voxel_size_um=VOXEL_SIZE_UM,
        mask_path=mask_path,
        obj_path=out_obj,
    )

    # 7) Excel rows
    summary_row = make_summary_row(base, mask_path, out_obj, out_npz, surf_H_um_inv)
    sampled_df = sample_vertices_df(base, surf_H_um_inv)

    print(f"[OK] {base}")
    print(f"     OBJ: {out_obj}")
    print(f"     NPZ: {out_npz}")

    return summary_row, sampled_df


def main():
    if RANDOM_SEED is not None:
        np.random.seed(int(RANDOM_SEED))

    mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, PATTERN)))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {MASK_DIR} with pattern {PATTERN}")

    print(f"Found {len(mask_paths)} masks.")
    print(f"Saving outputs to: {OUT_DIR}")

    summary_rows = []
    sampled_tables = []

    for mp in mask_paths:
        try:
            summary_row, sampled_df = process_one(mp)
            if summary_row is not None:
                summary_rows.append(summary_row)
            if sampled_df is not None:
                sampled_tables.append(sampled_df)
        except Exception as e:
            print(f"[FAIL] {os.path.basename(mp)} -> {e}")

    # Write ONE combined Excel
    if SAVE_EXCEL and len(summary_rows) > 0:
        excel_path = os.path.join(OUT_DIR, EXCEL_NAME)

        summary_df = pd.DataFrame(summary_rows).sort_values("cell")
        sampled_df_all = pd.concat(sampled_tables, ignore_index=True) if len(sampled_tables) else None

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary_per_cell", index=False)
            if sampled_df_all is not None and len(sampled_df_all) > 0:
                sampled_df_all.to_excel(writer, sheet_name="sampled_vertices", index=False)

        print(f"\n[EXCEL SAVED] {excel_path}")
        if sampled_df_all is None or len(sampled_df_all) == 0:
            print("  Note: sampled_vertices sheet not written (disabled or empty).")
    elif SAVE_EXCEL:
        print("\n[EXCEL] Nothing written (no cells processed).")


if __name__ == "__main__":
    main()
