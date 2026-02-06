#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch preprocess the SAME-NAMED file in each folder listed in an Excel sheet.

- Excel contains a column with folder paths (each folder contains the same file name).
- For each folder:
    input  = <folder>/<INPUT_FILENAME>
    output = <folder>/<OUTPUT_SUBDIR>/{segprep_bgsub_float01_uint16.tif, segprep_bgsub_float01_uint8.tif}

Notes:
- Requires: pandas + openpyxl (for .xlsx)
    conda install -c conda-forge pandas openpyxl
or  pip install pandas openpyxl
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio
import skimage.exposure as skexposure
import skimage.filters as skfilters
import skimage.morphology as skmorph

import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D
import segment3D.filters as uSegment3D_filters


# =============================================================================
# Segmentation-oriented preprocessing (BG subtract + robust rescale + optional CLAHE + smoothing)
# =============================================================================
def preprocess_for_segmentation(
    vol_zyx: np.ndarray,
    # ---- background removal ----
    bg_method="gaussian",          # "gaussian" or "morph"
    bg_sigma=25.0,                 # gaussian sigma for background (increase if bg varies slowly)
    morph_radius=40,               # disk radius for morph opening (must be larger than objects!)
    # ---- robust contrast ----
    clip_p=(1.0, 99.8),            # percentile stretch AFTER bg subtraction
    # ---- optional CLAHE ----
    use_clahe=True,
    clahe_clip_limit=0.01,
    clahe_kernel_size=64,
    # ---- smoothing ----
    gaussian_sigma=1.0,
):
    """
    Input:
        vol_zyx: float32, shape (Z,Y,X)
    Output:
        float32, shape (Z,Y,X), normalized to [0,1]
    """
    vol = vol_zyx.astype(np.float32, copy=False)

    # Robust normalize to [0,1] first (stabilizes bg estimation)
    lo0, hi0 = np.percentile(vol, (0.5, 99.9))
    hi0 = max(float(hi0), float(lo0) + 1e-6)
    vol0 = np.clip(vol, lo0, hi0)
    vol0 = (vol0 - lo0) / (hi0 - lo0)

    out = np.zeros_like(vol0, dtype=np.float32)

    # Pre-build structuring element for morph method
    selem = skmorph.disk(int(morph_radius)) if bg_method == "morph" else None

    for z in range(vol0.shape[0]):
        sl = vol0[z]

        # 1) Estimate and subtract background
        if bg_method == "gaussian":
            bg = skfilters.gaussian(sl, sigma=bg_sigma, preserve_range=True).astype(np.float32)
        elif bg_method == "morph":
            bg = skmorph.opening(sl, selem).astype(np.float32)
        else:
            raise ValueError("bg_method must be 'gaussian' or 'morph'")

        sl = sl - bg
        sl = np.clip(sl, 0, None)

        # 2) Robust contrast stretch
        lo, hi = np.percentile(sl, clip_p)
        hi = max(float(hi), float(lo) + 1e-6)
        sl = skexposure.rescale_intensity(sl, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32)

        # 3) Optional CLAHE
        if use_clahe:
            sl = skexposure.equalize_adapthist(
                sl,
                kernel_size=clahe_kernel_size,
                clip_limit=clahe_clip_limit
            ).astype(np.float32)

        # 4) Mild smoothing
        sl = skfilters.gaussian(sl, sigma=gaussian_sigma, preserve_range=True).astype(np.float32)

        out[z] = np.clip(sl, 0, 1)

    return out


# =============================================================================
# Excel helpers
# =============================================================================
def read_folder_list_from_excel(excel_path: str | Path, sheet_name=0, folder_col="folder") -> list[Path]:
    """
    Read a list of folder paths from an Excel sheet.
    - folder_col: column name containing folder paths. (Change if your column name differs.)
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    if folder_col not in df.columns:
        raise ValueError(
            f"Column '{folder_col}' not found in Excel. Columns are: {list(df.columns)}"
        )

    folders = []
    for v in df[folder_col].dropna().astype(str).tolist():
        v = v.strip()
        if not v:
            continue
        folders.append(Path(v))

    if not folders:
        raise ValueError("No folder paths found in the Excel column.")

    return folders


# =============================================================================
# Per-folder processing
# =============================================================================
def process_one_folder(
    folder: Path,
    input_filename: str,
    output_subdir: str,
    # u-Segment3D preprocess params
    usegment_factor: float = 0.2,
    usegment_do_bg: bool = True,
    voxel_res=(1, 1, 1),
    # segprep params
    segprep_kwargs: dict | None = None,
) -> tuple[bool, str]:
    """
    Returns (success, message).
    """
    folder = Path(folder)
    imgfile = folder / input_filename
    if not imgfile.exists():
        return False, f"Missing input: {imgfile}"

    outdir = folder / output_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load image -> enforce (Z,Y,X)
    img = skio.imread(str(imgfile))

    if img.ndim == 4:
        img = img[..., 0]
    elif img.ndim != 3:
        return False, f"Unexpected image shape {img.shape} for {imgfile}"

    img = img.astype(np.float32, copy=False)

    # u-Segment3D preprocessing (resize + background correction)
    preprocess_params = uSegment3D_params.get_preprocess_params()
    preprocess_params["do_bg_correction"] = bool(usegment_do_bg)
    preprocess_params["factor"] = float(usegment_factor)
    preprocess_params["voxel_res"] = list(voxel_res)

    # uSegment3D expects (C,Z,Y,X)
    img_pre = uSegment3D.preprocess_imgs(img[None, ...], params=preprocess_params)[0]
    img_pre = uSegment3D_filters.normalize(img_pre, clip=True).astype(np.float32)

    # Step 4: segmentation-ready preprocessing
    if segprep_kwargs is None:
        segprep_kwargs = {}
    img_segprep = preprocess_for_segmentation(img_pre, **segprep_kwargs)

    # Save outputs
    out_u16 = outdir / "segprep_bgsub_float01_uint16.tif"
    out_u8  = outdir / "segprep_bgsub_float01_uint8.tif"

    skio.imsave(str(out_u16), (65535 * img_segprep).astype(np.uint16), check_contrast=False)
    skio.imsave(str(out_u8),  (255   * img_segprep).astype(np.uint8),  check_contrast=False)

    return True, f"OK: {imgfile} -> {outdir}"


# =============================================================================
# Main
# =============================================================================
def main():
    # --------------------------
    # USER SETTINGS
    # --------------------------
    EXCEL_PATH = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/Beta.xlsx"   # <<< CHANGE ME
    SHEET_NAME = 0                             # or "Sheet1"
    FOLDER_COLUMN = "folder"                   # <<< CHANGE if your column name differs

    # The SAME-NAMED file that exists inside each folder listed in Excel:
    INPUT_FILENAME = "downsample4_fused_tp_0_ch_1.tif"  # <<< CHANGE ME

    # Output subfolder created inside each folder
    OUTPUT_SUBDIR = "preprocess_output"               # <<< CHANGE ME

    # u-Segment3D preprocessing params
    USEGMENT_FACTOR = 0.2
    USEGMENT_DO_BG = True
    VOXEL_RES = (1, 1, 1)

    # segprep parameters (same as your script defaults)
    SEG_PREP_KWARGS = dict(
        bg_method="gaussian",   # try "gaussian" first; if uneven, try "morph"
        bg_sigma=25.0,
        morph_radius=40,
        clip_p=(1.0, 99.8),
        use_clahe=True,
        clahe_clip_limit=0.01,
        clahe_kernel_size=64,
        gaussian_sigma=1.0,
    )

    # --------------------------
    # RUN
    # --------------------------
    folders = read_folder_list_from_excel(EXCEL_PATH, sheet_name=SHEET_NAME, folder_col=FOLDER_COLUMN)

    print(f"Found {len(folders)} folders in Excel.")
    print(f"Input filename: {INPUT_FILENAME}")
    print(f"Output subdir : {OUTPUT_SUBDIR}\n")

    ok_count = 0
    fail_count = 0

    for i, folder in enumerate(folders, 1):
        folder = Path(folder)
        print(f"[{i}/{len(folders)}] Processing: {folder}")

        try:
            ok, msg = process_one_folder(
                folder=folder,
                input_filename=INPUT_FILENAME,
                output_subdir=OUTPUT_SUBDIR,
                usegment_factor=USEGMENT_FACTOR,
                usegment_do_bg=USEGMENT_DO_BG,
                voxel_res=VOXEL_RES,
                segprep_kwargs=SEG_PREP_KWARGS,
            )
            if ok:
                ok_count += 1
                print("   ?", msg)
            else:
                fail_count += 1
                print("   ?", msg)

        except Exception as e:
            fail_count += 1
            print("   ? ERROR:", repr(e))
            traceback.print_exc()

        print("")

    print("========== DONE ==========")
    print(f"Success: {ok_count}")
    print(f"Failed : {fail_count}")


if __name__ == "__main__":
    main()
