#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: rescale u-Segment3D labels to RAW image shape, then guided-filter at RAW resolution.

For each folder listed in an Excel sheet:
  - load labels: uSegment3D_labels_postprocess-diffuse_labels_finetuning.tif
  - load raw:    downsample4_fused_tp_0_ch_1.tif
  - (optional) normalize raw
  - rescale labels -> raw shape (nearest-neighbor)
  - guided filter using raw as guide (uSegment3D guided_filter_3D_cell_segmentation_MP)
  - save: uSegment3D_labels_guided_filter_RAW.tif

Requirements:
  - pandas + openpyxl installed in your env
  - segment3D (u-Segment3D) installed and importable
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import skimage.io as skio

import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D
import segment3D.filters as uSegment3D_filters
import segment3D.file_io as uSegment3D_fio


# =============================================================================
# USER SETTINGS
# =============================================================================
EXCEL_PATH = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/Nras.xlsx"   # <-- change to your excel path
SHEET_NAME = "Sheet5"   # <-- change to your actual sheet name


# Column name that contains folder paths. If not found, the script uses the first column.
FOLDER_COLNAME = "folder"
LABEL_FILENAME = "uSegment3D_labels_postprocess-diffuse_labels_finetuning.tif"
RAW_FILENAME   = "downsample4_fused_tp_0_ch_0.tif"

# Where to save output:
#   - "same": save next to input label/raw in each folder
#   - "subfolder": save into <folder>/<OUTPUT_SUBFOLDER>/
OUTPUT_MODE = "same"
OUTPUT_SUBFOLDER = "guided_filter"

OUTPUT_FILENAME = "uSegment3D_labels_guided_filter_RAW.tif"


# =============================================================================
# Helpers
# =============================================================================
def crop_or_pad_to_shape(arr_zyx: np.ndarray, target_shape_zyx, pad_value=0) -> np.ndarray:
    """Crop then zero-pad to exactly match target shape (Z,Y,X)."""
    tz, ty, tx = map(int, target_shape_zyx)
    out = arr_zyx[:min(arr_zyx.shape[0], tz),
                  :min(arr_zyx.shape[1], ty),
                  :min(arr_zyx.shape[2], tx)]
    pad_z = tz - out.shape[0]
    pad_y = ty - out.shape[1]
    pad_x = tx - out.shape[2]
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        out = np.pad(out,
                     ((0, max(pad_z, 0)), (0, max(pad_y, 0)), (0, max(pad_x, 0))),
                     mode="constant",
                     constant_values=pad_value)
    return out


def load_folder_list(excel_path: str, colname: str = "folder"):
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    if colname in df.columns:
        folders = df[colname].dropna().astype(str).tolist()
    else:
        # fallback: use first column
        folders = df.iloc[:, 0].dropna().astype(str).tolist()

    # normalize whitespace
    folders = [f.strip() for f in folders if f.strip()]
    return folders


# =============================================================================
# Guided filter parameters (tune here)
# =============================================================================
guided_filter_params = uSegment3D_params.get_guided_filter_params()

guided_filter_params["ridge_filter"]["do_ridge_enhance"] = True
guided_filter_params["ridge_filter"]["mix_ratio"] = 0.5
guided_filter_params["ridge_filter"]["sigmas"] = [3.0]   # tune based on noise/feature size

guided_filter_params["guide_filter"]["radius"] = 35
guided_filter_params["guide_filter"]["eps"] = 1e-4
guided_filter_params["guide_filter"]["mode"] = "additive"
guided_filter_params["guide_filter"]["base_erode"] = 2
guided_filter_params["guide_filter"]["collision_erode"] = 0
guided_filter_params["guide_filter"]["collision_close"] = 0
guided_filter_params["guide_filter"]["collision_fill_holes"] = True


# =============================================================================
# Main
# =============================================================================
def main():
    try:
        folders = load_folder_list(EXCEL_PATH, FOLDER_COLNAME)
    except Exception as e:
        print("? Failed to read Excel:", EXCEL_PATH)
        print("   Error:", repr(e))
        print("\nIf you see 'Missing optional dependency openpyxl', install it in your env:")
        print("  conda install -c conda-forge openpyxl")
        return 1

    if len(folders) == 0:
        print("? No folders found in Excel.")
        return 1

    print(f"Found {len(folders)} folders in Excel.")
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, folder in enumerate(folders, start=1):
        print("\n===================================================")
        print(f"[{i}/{len(folders)}] Folder: {folder}")

        label_path = os.path.join(folder, LABEL_FILENAME)
        raw_path   = os.path.join(folder, RAW_FILENAME)

        if not os.path.exists(folder):
            print("??  Skip: folder does not exist")
            n_skip += 1
            continue
        if not os.path.exists(label_path):
            print("??  Skip: label not found:", label_path)
            n_skip += 1
            continue
        if not os.path.exists(raw_path):
            print("??  Skip: raw not found:", raw_path)
            n_skip += 1
            continue

        try:
            # -------------------------
            # Load label + raw
            # -------------------------
            labels = skio.imread(label_path).astype(np.int32)

            raw = skio.imread(raw_path)
            # Handle (Z,Y,X,C) or (Z,Y,X)
            if raw.ndim == 4:
                raw = raw[..., 0]
            raw = raw.astype(np.float32)

            # Normalize raw to [0,1] (recommended)
            guide_image_raw = uSegment3D_filters.normalize(raw, clip=True)

            print("  labels shape:", labels.shape, labels.dtype)
            print("  raw shape   :", guide_image_raw.shape, guide_image_raw.dtype)

            # -------------------------
            # Rescale labels -> raw shape
            # -------------------------
            target_shape = guide_image_raw.shape  # (Z,Y,X)
            src_shape = labels.shape

            zoom_factors = np.array(target_shape, dtype=np.float32) / np.array(src_shape, dtype=np.float32)
            print("  zoom_factors:", zoom_factors)

            labels_rs = ndimage.zoom(
                labels,
                zoom=zoom_factors,
                order=0,          # nearest neighbor for labels
                mode="nearest"
            ).astype(np.int32)

            labels_rs = crop_or_pad_to_shape(labels_rs, target_shape, pad_value=0)
            print("  labels_rs shape:", labels_rs.shape, labels_rs.dtype)

            # -------------------------
            # Guided filter at raw resolution
            # -------------------------
            labels_guided, _ = uSegment3D.guided_filter_3D_cell_segmentation_MP(
                labels_rs,
                guide_image=guide_image_raw,
                params=guided_filter_params
            )
            labels_guided = labels_guided.astype(np.int32)

            # -------------------------
            # Save output
            # -------------------------
            if OUTPUT_MODE == "subfolder":
                out_dir = os.path.join(folder, OUTPUT_SUBFOLDER)
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = folder

            out_path = os.path.join(out_dir, OUTPUT_FILENAME)
            uSegment3D_fio.save_segmentation(out_path, labels_guided)

            print("? Saved:", out_path)
            n_ok += 1

        except Exception as e:
            print("? Failed in folder:", folder)
            print("   Error:", repr(e))
            n_fail += 1
            continue

    print("\n==================== SUMMARY ====================")
    print("  OK   :", n_ok)
    print("  SKIP :", n_skip)
    print("  FAIL :", n_fail)
    print("=================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
