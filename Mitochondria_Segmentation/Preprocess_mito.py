#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import skimage.io as skio
import skimage.exposure as skexposure
import skimage.filters as skfilters
from scipy.ndimage import gaussian_filter

import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D
import segment3D.filters as uSegment3D_filters


# =============================================================================
# Segmentation-oriented preprocessing
# =============================================================================
def preprocess_for_segmentation(
    vol_zyx: np.ndarray,
    clip_p=(0.5, 99.9),
    bg_sigma=18.0,
    gaussian_sigma=1.4,
    clahe_kernel_size=(96, 96),
    clahe_clip_limit=0.01,
    gamma=0.90,
):
    """
    Input:
        vol_zyx: float32, shape (Z,Y,X)
    Output:
        float32, shape (Z,Y,X), normalized to [0,1]
    """
    vol = vol_zyx.astype(np.float32)

    # robust global normalization
    lo, hi = np.percentile(vol, clip_p)
    hi = max(hi, lo + 1e-6)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)

    # background subtraction
    if bg_sigma and bg_sigma > 0:
        bg = gaussian_filter(vol, sigma=(0, bg_sigma, bg_sigma))
        vol = np.clip(vol - bg, 0, 1)

    out = np.zeros_like(vol, dtype=np.float32)

    for z in range(vol.shape[0]):
        sl = vol[z]

        if gamma is not None:
            sl = skexposure.adjust_gamma(sl, gamma)

        sl = skexposure.equalize_adapthist(
            sl,
            kernel_size=clahe_kernel_size,
            clip_limit=clahe_clip_limit
        ).astype(np.float32)

        if gaussian_sigma and gaussian_sigma > 0:
            sl = skfilters.gaussian(
                sl,
                sigma=gaussian_sigma,
                preserve_range=True
            ).astype(np.float32)

        out[z] = np.clip(sl, 0, 1)

    return out


# =============================================================================
# Process one image
# =============================================================================
def process_one_image(
    imgfile,
    preprocess_outdir,
    clip_p=(0.5, 99.9),
    bg_sigma=18.0,
    gaussian_sigma=1.4,
    clahe_kernel_size=(96, 96),
    clahe_clip_limit=0.01,
    gamma=0.90
):
    print(f"\nProcessing: {imgfile}")

    if not os.path.isfile(imgfile):
        raise FileNotFoundError(f"Image not found: {imgfile}")

    os.makedirs(preprocess_outdir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load image -> enforce (Z,Y,X)
    # -------------------------------------------------------------------------
    img = skio.imread(imgfile)

    if img.ndim == 4:
        img = img[..., 0]
    elif img.ndim != 3:
        raise ValueError(f"Unexpected image shape for {imgfile}: {img.shape}")

    img = img.astype(np.float32)

    # -------------------------------------------------------------------------
    # u-Segment3D preprocessing
    # -------------------------------------------------------------------------
    preprocess_params = uSegment3D_params.get_preprocess_params()
    preprocess_params["do_bg_correction"] = True
    preprocess_params["factor"] = 1
    preprocess_params["voxel_res"] = [1, 1, 1]

    # uSegment3D expects (C,Z,Y,X)
    img_pre = uSegment3D.preprocess_imgs(
        img[None, ...],
        params=preprocess_params
    )[0]

    img_pre = uSegment3D_filters.normalize(img_pre, clip=True)

    # -------------------------------------------------------------------------
    # segmentation-ready preprocessing
    # -------------------------------------------------------------------------
    img_segprep = preprocess_for_segmentation(
        img_pre,
        clip_p=clip_p,
        bg_sigma=bg_sigma,
        gaussian_sigma=gaussian_sigma,
        clahe_kernel_size=clahe_kernel_size,
        clahe_clip_limit=clahe_clip_limit,
        gamma=gamma
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    uint16_path = os.path.join(preprocess_outdir, "segprep_preprocess_uint16.tif")
    uint8_path  = os.path.join(preprocess_outdir, "segprep_preprocess_uint8.tif")

    skio.imsave(
        uint16_path,
        (65535 * img_segprep).astype(np.uint16),
        check_contrast=False
    )

    skio.imsave(
        uint8_path,
        (255 * img_segprep).astype(np.uint8),
        check_contrast=False
    )

    print("  Saved:")
    print(f"   - {uint16_path}")
    print(f"   - {uint8_path}")


# =============================================================================
# Get all tif files inside one folder
# =============================================================================
def get_tif_files_from_folder(folder_path):
    tif_files = []
    tif_files.extend(glob.glob(os.path.join(folder_path, "*.tif")))
    tif_files.extend(glob.glob(os.path.join(folder_path, "*.tiff")))
    tif_files = sorted(tif_files)
    return tif_files


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Excel file containing folder paths
    # Required column: folder
    # -------------------------------------------------------------------------
    excel_file = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Mito_segment/mito_segmentation.xlsx"
    sheet_name = 'Sheet7'   # or "Sheet1"

    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    if "folder" not in df.columns:
        raise ValueError("Excel must contain a column named 'folder'.")

    total_folders = 0
    total_files = 0
    success_files = 0
    failed_files = 0

    for idx, row in df.iterrows():
        folder = str(row["folder"]).strip()

        if folder == "" or folder.lower() == "nan":
            print(f"\nSkipping row {idx + 2}: empty folder path")
            continue

        if not os.path.isdir(folder):
            print(f"\nSkipping row {idx + 2}: folder does not exist -> {folder}")
            continue

        total_folders += 1
        print(f"\n{'=' * 80}")
        print(f"Folder {total_folders}: {folder}")

        tif_files = get_tif_files_from_folder(folder)

        if len(tif_files) == 0:
            print("  No .tif or .tiff files found in this folder.")
            continue

        print(f"  Found {len(tif_files)} tif file(s).")

        for imgfile in tif_files:
            total_files += 1
            try:
                tif_name = os.path.splitext(os.path.basename(imgfile))[0]

                # save each tif result into:
                # folder/Pre_downsampe1_new/<tif_name>/
                preprocess_outdir = os.path.join(
                    folder,
                    "Pre_downsampe1_new",
                    tif_name
                )

                process_one_image(
                    imgfile=imgfile,
                    preprocess_outdir=preprocess_outdir,
                    clip_p=(0.5, 99.9),
                    bg_sigma=18.0,
                    gaussian_sigma=1.4,
                    clahe_kernel_size=(96, 96),
                    clahe_clip_limit=0.01,
                    gamma=0.90
                )
                success_files += 1

            except Exception as e:
                failed_files += 1
                print(f"  Failed on file: {imgfile}")
                print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("Batch preprocessing complete.")
    print(f"Folders processed : {total_folders}")
    print(f"Total tif files   : {total_files}")
    print(f"Success files     : {success_files}")
    print(f"Failed files      : {failed_files}")
    print("=" * 80)