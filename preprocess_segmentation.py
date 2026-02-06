#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import skimage.io as skio
import skimage.exposure as skexposure
import skimage.filters as skfilters

import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D
import segment3D.filters as uSegment3D_filters


# =============================================================================
# Segmentation-oriented preprocessing (CLAHE + smoothing)
# =============================================================================
def preprocess_for_segmentation(
    vol_zyx: np.ndarray,
    clip_p=(1.0, 99.8),
    clahe_clip_limit=0.01,
    gaussian_sigma=1.0,
):
    """
    Input:
        vol_zyx: float32, shape (Z,Y,X)
    Output:
        float32, shape (Z,Y,X), normalized to [0,1]
    """

    vol = vol_zyx.astype(np.float32)

    # --- robust global normalization ---
    lo, hi = np.percentile(vol, clip_p)
    hi = max(hi, lo + 1e-6)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)

    out = np.zeros_like(vol, dtype=np.float32)

    for z in range(vol.shape[0]):
        sl = vol[z]

        # local contrast enhancement
        sl = skexposure.equalize_adapthist(
            sl,
            clip_limit=clahe_clip_limit
        ).astype(np.float32)

        # noise suppression (important for SAM)
        sl = skfilters.gaussian(
            sl,
            sigma=gaussian_sigma,
            preserve_range=True
        ).astype(np.float32)

        out[z] = np.clip(sl, 0, 1)

    return out


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    imgfile = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/NRAS_1/488/2025-12-24/38X_005/downsample4_fused_tp_0_ch_0.tif"

    preprocess_outdir = "/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/NRAS_1/488/2025-12-24/38X_005/preprocess_output"
    os.makedirs(preprocess_outdir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load image ? enforce (Z,Y,X)
    # -------------------------------------------------------------------------
    img = skio.imread(imgfile)

    if img.ndim == 4:
        img = img[..., 0]
    elif img.ndim != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    img = img.astype(np.float32)

    # -------------------------------------------------------------------------
    # u-Segment3D preprocessing (resize + background correction)
    # -------------------------------------------------------------------------
    preprocess_params = uSegment3D_params.get_preprocess_params()
    preprocess_params["do_bg_correction"] = True
    preprocess_params["factor"] = 0.2
    preprocess_params["voxel_res"] = [1, 1, 1]

    # uSegment3D expects (C,Z,Y,X)
    img_pre = uSegment3D.preprocess_imgs(
        img[None, ...],
        params=preprocess_params
    )[0]

    img_pre = uSegment3D_filters.normalize(img_pre, clip=True)

    # -------------------------------------------------------------------------
    # Step 4: segmentation-ready preprocessing
    # -------------------------------------------------------------------------
    img_segprep = preprocess_for_segmentation(
        img_pre,
        clip_p=(1.0, 99.8),
        clahe_clip_limit=0.01,
        gaussian_sigma=1.0
    )

    # -------------------------------------------------------------------------
    # SAVE ONLY final processed images
    # -------------------------------------------------------------------------
    skio.imsave(
        os.path.join(preprocess_outdir, "segprep_float01_uint16.tif"),
        (65535 * img_segprep).astype(np.uint16)
    )

    skio.imsave(
        os.path.join(preprocess_outdir, "segprep_float01_uint8.tif"),
        (255 * img_segprep).astype(np.uint8)
    )

    print("? Preprocessing complete.")
    print("Saved files:")
    print(" - segprep_float01_uint16.tif")
    print(" - segprep_float01_uint8.tif")
