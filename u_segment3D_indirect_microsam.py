#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
u-Segment3D indirect method (XY/XZ/YZ -> 3D) with:
- Robust axis fixing (auto transpose) + exact size enforcement (crop/pad)
- Optional downsampling (nearest-neighbor)
- XY-weighted foreground prior (XY influences more than XZ/YZ)
- Mask cleaning (2D slice clean + 3D small-object removal)
- Postprocess (size + flow consistency filtering)
- Label diffusion
- Guided filter

NOTE about "rescale back":
- This script DOES NOT rescale the final labels back to the original image size,
  per your last instruction ("but don't need to rescale back").
- Instead, it resamples the raw image to the SEGMENTATION GRID for guided filtering,
  so everything stays in one consistent grid.

Inputs needed:
- PATH_XY, PATH_XZ, PATH_YZ : label stacks from your 2D model
- PATH_IMG (optional but recommended): raw image stack for guided filtering + overlays
  (expects shape (Z,Y,X) or (Z,Y,X,C); uses channel 0 if multi-channel)

Outputs (in SAVE_DIR):
- labels_*_downsampled.tif
- precomputed_binary_xy_weighted.tif
- uSegment3D_3D_labels_downsampled_raw.tif
- uSegment3D_3D_labels_postprocess.tif
- uSegment3D_labels_postprocess-diffuse.tif
- uSegment3D_labels_postprocess-diffuse-guided_filter.tif
- overlay_postprocess.png (if PATH_IMG provided)
"""

if __name__ == "__main__":
    import os
    import itertools
    import numpy as np
    import skimage.io as skio
    import skimage.segmentation as sksegmentation
    import scipy.ndimage as ndimage
    import pylab as plt

    import segment3D.plotting as uSegment3D_plotting
    import segment3D.parameters as uSegment3D_params
    import segment3D.usegment3d as uSegment3D
    import segment3D.filters as uSegment3D_filters
    import segment3D.file_io as uSegment3D_fio

    # =============================================================================
    # 0) USER I/O
    # =============================================================================
    PATH_XY = r'/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/NRAS_1/488/2025-12-24/38X_003/preprocess_output/segment/03_segprep_float01_uint8_seg_xy_linked_ZYX.tif'
    PATH_XZ = r'/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/NRAS_1/488/2025-12-24/38X_003/preprocess_output/segment/03_segprep_float01_uint8_seg_xz_linked_ZYX.tif'
    PATH_YZ = r'/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/NRAS_1/488/2025-12-24/38X_003/preprocess_output/segment/03_segprep_float01_uint8_seg_yz_linked_ZYX.tif'

    # Raw image stack for guided filter + overlays (set to None to skip those parts)
    # Example: PATH_IMG = r'/path/to/raw_image.tif'
    PATH_IMG = None

    SAVE_DIR = r'/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/NRAS_1/488/2025-12-24/38X_003/preprocess_output/segment/usegment3D_indirect_xy_weighted_postprocess'
    uSegment3D_fio.mkdir(SAVE_DIR)

    # --- Downsample factors for XY grid (Z,Y,X). Use (1,1,1) to disable.
    DS_Z, DS_Y, DS_X = 1, 1, 1

    # --- XY weighting (bigger => trust XY more)
    W_XY, W_XZ, W_YZ = 3.0, 1.0, 1.0
    # With (3,1,1) vote range is 0..5; 2.5 = "mostly XY"
    VOTE_THRESHOLD = 2.5#2.5

    # --- Mask cleaning
    SLICE_MINSIZE = 8        # 2D slice cleanup (set None to disable)
    FINAL_MIN_SIZE_3D = 250  # remove tiny 3D objects after aggregation (set None to disable)

    DEBUG_VIZ = True

    # =============================================================================
    # 1) Utilities
    # =============================================================================
    def _find_perm_to_target(shape, target):
        """Return permutation p such that transpose(arr,p).shape == target, else None."""
        for p in itertools.permutations([0, 1, 2]):
            if tuple(shape[i] for i in p) == tuple(target):
                return p
        return None

    def _center_crop_or_pad(arr, target_shape, pad_value=0):
        """Force arr to exactly target_shape by center-cropping or padding."""
        out = arr

        # crop
        slices = []
        for d, td in enumerate(target_shape):
            sd = out.shape[d]
            if sd > td:
                start = (sd - td) // 2
                slices.append(slice(start, start + td))
            else:
                slices.append(slice(0, sd))
        out = out[tuple(slices)]

        # pad
        pad_width = []
        for d, td in enumerate(target_shape):
            sd = out.shape[d]
            if sd < td:
                total = td - sd
                pre = total // 2
                post = total - pre
                pad_width.append((pre, post))
            else:
                pad_width.append((0, 0))
        if any(p != (0, 0) for p in pad_width):
            out = np.pad(out, pad_width, mode="constant", constant_values=pad_value)

        return out

    def downsample_labels(vol_labels, factors):
        """Nearest-neighbor downsample for label volumes."""
        if factors == (1, 1, 1):
            return vol_labels
        return ndimage.zoom(vol_labels, zoom=factors, order=0)

    def imsave_uint32(path, arr):
        skio.imsave(path, arr.astype(np.uint32), check_contrast=False)

    def _load_raw_image(path_img):
        """Load raw image; return float32 array (Z,Y,X) as single-channel."""
        img = skio.imread(path_img)
        img = np.asarray(img)
        if img.ndim == 4:
            # assume (Z,Y,X,C) or (Z,C,Y,X) etc is possible in some pipelines
            # Most common for microscopy tiffs via skimage is (Z,Y,X,C).
            if img.shape[-1] <= 4:
                img1c = img[..., 0]
            else:
                # fallback: take first plane along axis 1 if looks like (Z,C,Y,X)
                img1c = img[:, 0, ...]
        elif img.ndim == 3:
            img1c = img
        else:
            raise ValueError(f"Unsupported raw image ndim={img.ndim}, shape={img.shape}")

        img1c = img1c.astype(np.float32)
        return img1c

    # =============================================================================
    # 2) Load stacks
    # =============================================================================
    labels_xy = skio.imread(PATH_XY).astype(np.int32)
    labels_xz = skio.imread(PATH_XZ).astype(np.int32)
    labels_yz = skio.imread(PATH_YZ).astype(np.int32)

    Z, Y, X = labels_xy.shape
    target_xz = (Y, Z, X)  # (Y,Z,X)
    target_yz = (X, Z, Y)  # (X,Z,Y)

    print("Loaded label shapes:")
    print("  XY:", labels_xy.shape, " expected (Z,Y,X) =", (Z, Y, X))
    print("  XZ:", labels_xz.shape, " expected (Y,Z,X) =", target_xz)
    print("  YZ:", labels_yz.shape, " expected (X,Z,Y) =", target_yz)

    # =============================================================================
    # 3) Fix orientations + enforce exact sizes (prevents var_combine mismatch)
    # =============================================================================
    # XZ -> (Y,Z,X)
    if labels_xz.shape != target_xz:
        p = _find_perm_to_target(labels_xz.shape, target_xz)
        if p is None:
            raise ValueError(f"XZ shape {labels_xz.shape} cannot be permuted to (Y,Z,X)={target_xz}")
        labels_xz = np.transpose(labels_xz, p)
        print(f"Permuted XZ axes {p} -> {labels_xz.shape}")
    labels_xz = _center_crop_or_pad(labels_xz, target_xz, pad_value=0)

    # YZ -> (X,Z,Y)
    if labels_yz.shape != target_yz:
        p = _find_perm_to_target(labels_yz.shape, target_yz)
        if p is None:
            raise ValueError(f"YZ shape {labels_yz.shape} cannot be permuted to (X,Z,Y)={target_yz}")
        labels_yz = np.transpose(labels_yz, p)
        print(f"Permuted YZ axes {p} -> {labels_yz.shape}")
    labels_yz = _center_crop_or_pad(labels_yz, target_yz, pad_value=0)

    print("After orientation/size fix:")
    print("  XY:", labels_xy.shape)
    print("  XZ:", labels_xz.shape)
    print("  YZ:", labels_yz.shape)

    # =============================================================================
    # 4) Clean masks (2D slice cleanup)
    # =============================================================================
    if SLICE_MINSIZE is not None and SLICE_MINSIZE > 0:
        labels_xy = uSegment3D_filters.filter_2d_label_slices(labels_xy, bg_label=0, minsize=int(SLICE_MINSIZE))
        labels_xz = uSegment3D_filters.filter_2d_label_slices(labels_xz, bg_label=0, minsize=int(SLICE_MINSIZE))
        labels_yz = uSegment3D_filters.filter_2d_label_slices(labels_yz, bg_label=0, minsize=int(SLICE_MINSIZE))

    # =============================================================================
    # 5) Downsample labels (optional)
    # =============================================================================
    labels_xy_ds = downsample_labels(labels_xy, (DS_Z, DS_Y, DS_X))  # (Z,Y,X)
    labels_xz_ds = downsample_labels(labels_xz, (DS_Y, DS_Z, DS_X))  # (Y,Z,X)
    labels_yz_ds = downsample_labels(labels_yz, (DS_X, DS_Z, DS_Y))  # (X,Z,Y)

    # Enforce exact expected DS sizes (zoom rounding protection)
    Zds, Yds, Xds = labels_xy_ds.shape
    labels_xz_ds = _center_crop_or_pad(labels_xz_ds, (Yds, Zds, Xds), pad_value=0)
    labels_yz_ds = _center_crop_or_pad(labels_yz_ds, (Xds, Zds, Yds), pad_value=0)

    print("Final (to aggregator) shapes:")
    print("  XY:", labels_xy_ds.shape, "should be (Z,Y,X)")
    print("  XZ:", labels_xz_ds.shape, "should be (Y,Z,X)")
    print("  YZ:", labels_yz_ds.shape, "should be (X,Z,Y)")

    # Save DS inputs (QA)
    imsave_uint32(os.path.join(SAVE_DIR, "labels_xy_downsampled.tif"), labels_xy_ds)
    imsave_uint32(os.path.join(SAVE_DIR, "labels_xz_downsampled.tif"), labels_xz_ds)
    imsave_uint32(os.path.join(SAVE_DIR, "labels_yz_downsampled.tif"), labels_yz_ds)

    # =============================================================================
    # 6) XY-weighted foreground prior
    # =============================================================================
    xy_fg = (labels_xy_ds > 0)  # (Z,Y,X)
    xz_fg = (labels_xz_ds > 0).transpose(1, 0, 2)  # (Y,Z,X)->(Z,Y,X)
    yz_fg = (labels_yz_ds > 0).transpose(1, 2, 0)  # (X,Z,Y)->(Z,Y,X)

    vote = (W_XY * xy_fg.astype(np.float32) +
            W_XZ * xz_fg.astype(np.float32) +
            W_YZ * yz_fg.astype(np.float32))

    precomputed_binary = (vote >= float(VOTE_THRESHOLD))
    imsave_uint32(os.path.join(SAVE_DIR, "precomputed_binary_xy_weighted.tif"),
                 precomputed_binary.astype(np.uint32))

    # =============================================================================
    # 7) Params for indirect aggregation
    # =============================================================================
    aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()

    aggregation_params["indirect_method"]["dtform_method"] = "cellpose_skel"
    aggregation_params["indirect_method"]["smooth_skel_sigma"] = 3

    aggregation_params["gradient_descent"]["gradient_decay"] = 0.1
    aggregation_params["gradient_descent"]["n_iter"] = 120
    aggregation_params["gradient_descent"]["momenta"] = 0.98
    aggregation_params["gradient_descent"]["debug_viz"] = bool(DEBUG_VIZ)

    # =============================================================================
    # 8) Aggregate (indirect method)
    # =============================================================================
    segmentation3D, (prob3d, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(
        segmentations=[labels_xy_ds, labels_xz_ds, labels_yz_ds],
        img_xy_shape=labels_xy_ds.shape,          # (Z,Y,X) grid
        precomputed_binary=precomputed_binary,    # XY-weighted prior
        params=aggregation_params,
        savefolder=None,
        basename=None
    )

    # Remove tiny 3D objects right after aggregation (optional but recommended)
    if FINAL_MIN_SIZE_3D is not None and FINAL_MIN_SIZE_3D > 0:
        segmentation3D = uSegment3D_filters.remove_small_labels(segmentation3D, min_size=int(FINAL_MIN_SIZE_3D))

    uSegment3D_fio.save_segmentation(
        os.path.join(SAVE_DIR, "uSegment3D_3D_labels_downsampled_raw.tif"),
        segmentation3D.astype(np.uint32)
    )

    # =============================================================================
    # 9) Postprocess: size + flow consistency filtering (your snippet)
    # =============================================================================
    postprocess_segment_params = uSegment3D_params.get_postprocess_segmentation_params()
    print("========== Default postprocess parameters ========")
    print(postprocess_segment_params)
    print("=================================================")

    segmentation3D_filt, flow_consistency_intermediates = uSegment3D.postprocess_3D_cell_segmentation(
        segmentation3D,
        aggregation_params=aggregation_params,
        postprocess_params=postprocess_segment_params,
        cell_gradients=gradients3D,
        savefolder=None,
        basename=None
    )

    uSegment3D_fio.save_segmentation(
        os.path.join(SAVE_DIR, "uSegment3D_3D_labels_postprocess.tif"),
        segmentation3D_filt.astype(np.uint32)
    )

    # =============================================================================
    # 10) Overlay plots (requires raw image)
    # =============================================================================
    if PATH_IMG is not None:
        img_raw_1c = _load_raw_image(PATH_IMG)  # (Z0,Y0,X0)

        # Resample raw image onto segmentation grid (Zds,Yds,Xds) for clean overlay
        zoom_to_seg = np.array(segmentation3D_filt.shape, dtype=np.float32) / np.array(img_raw_1c.shape, dtype=np.float32)
        img_preprocess_1c = ndimage.zoom(img_raw_1c, zoom=zoom_to_seg, order=1, mode="reflect")
        img_preprocess_1c = uSegment3D_filters.normalize(img_preprocess_1c, clip=True)

        midz = img_preprocess_1c.shape[0] // 2
        midy = img_preprocess_1c.shape[1] // 2
        midx = img_preprocess_1c.shape[2] // 2

        plt.figure(figsize=(12, 10))

        plt.subplot(131)
        plt.title("Overlay mid XY")
        plt.imshow(sksegmentation.mark_boundaries(
            np.dstack([img_preprocess_1c[midz]] * 3),
            segmentation3D_filt[midz],
            color=(0, 1, 0), mode="thick"
        ))

        plt.subplot(132)
        plt.title("Overlay mid XZ")
        plt.imshow(sksegmentation.mark_boundaries(
            np.dstack([img_preprocess_1c[:, midy, :]] * 3),
            segmentation3D_filt[:, midy, :],
            color=(0, 1, 0), mode="thick"
        ))

        plt.subplot(133)
        plt.title("Overlay mid YZ")
        plt.imshow(sksegmentation.mark_boundaries(
            np.dstack([img_preprocess_1c[:, :, midx]] * 3),
            segmentation3D_filt[:, :, midx],
            color=(0, 1, 0), mode="thick"
        ))

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "overlay_postprocess.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        # still define img_preprocess_1c for diffusion/guided filter guide if possible
        img_preprocess_1c = None

    # =============================================================================
    # 11) Label diffusion (your snippet)
    # =============================================================================
    label_diffusion_params = uSegment3D_params.get_label_diffusion_params()
    label_diffusion_params["diffusion"]["refine_iters"] = 15
    label_diffusion_params["diffusion"]["refine_alpha"] = 0.5
    label_diffusion_params["diffusion"]["refine_clamp"] = 0.75

    # Guide image for diffusion:
    # - If PATH_IMG provided: we already computed img_preprocess_1c on segmentation grid.
    # - Else: fall back to a simple normalized mask-derived "guide" (less ideal).
    if img_preprocess_1c is None:
        guide_for_diffusion = uSegment3D_filters.normalize((segmentation3D_filt > 0).astype(np.float32), clip=True)
    else:
        guide_for_diffusion = img_preprocess_1c

    segmentation3D_filt_diffuse = uSegment3D.label_diffuse_3D_cell_segmentation_MP(
        segmentation3D_filt,
        guide_image=guide_for_diffusion,
        params=label_diffusion_params
    )

    uSegment3D_fio.save_segmentation(
        os.path.join(SAVE_DIR, "uSegment3D_labels_postprocess-diffuse.tif"),
        segmentation3D_filt_diffuse.astype(np.uint32)
    )

    # =============================================================================
    # 12) Guided filter (your snippet, adapted to NO "rescale back")
    # =============================================================================
    guided_filter_params = uSegment3D_params.get_guided_filter_params()
    guided_filter_params["ridge_filter"]["do_ridge_enhance"] = True
    guided_filter_params["ridge_filter"]["mix_ratio"] = 0.5
    guided_filter_params["ridge_filter"]["sigmas"] = [3.0]

    guided_filter_params["guide_filter"]["radius"] = 35
    guided_filter_params["guide_filter"]["eps"] = 1e-4
    guided_filter_params["guide_filter"]["mode"] = "additive"
    guided_filter_params["guide_filter"]["base_erode"] = 2
    guided_filter_params["guide_filter"]["collision_erode"] = 0
    guided_filter_params["guide_filter"]["collision_close"] = 0
    guided_filter_params["guide_filter"]["collision_fill_holes"] = True

    # Build guide image ON THE SAME GRID as segmentation (no rescale needed)
    if PATH_IMG is not None:
        # img_preprocess_1c already exists and matches segmentation grid
        guide_image = uSegment3D_filters.normalize(img_preprocess_1c.astype(np.float32), clip=True)
    else:
        # fallback guide: smoothed foreground (less ideal)
        guide_image = ndimage.gaussian_filter((segmentation3D_filt_diffuse > 0).astype(np.float32), sigma=1.0)
        guide_image = uSegment3D_filters.normalize(guide_image, clip=True)

    print("guide_image.shape =", guide_image.shape, "segmentation.shape =", segmentation3D_filt_diffuse.shape)

    # Because guide_image and segmentation are same shape now, no resampling step required
    segmentation3D_filt_guide, guide_image_used = uSegment3D.guided_filter_3D_cell_segmentation_MP(
        segmentation3D_filt_diffuse.astype(np.int32),
        guide_image=guide_image,
        params=guided_filter_params
    )

    uSegment3D_fio.save_segmentation(
        os.path.join(SAVE_DIR, "uSegment3D_labels_postprocess-diffuse-guided_filter.tif"),
        segmentation3D_filt_guide.astype(np.uint32)
    )

    # =============================================================================
    # Done
    # =============================================================================
    n_inst = int(np.unique(segmentation3D_filt_guide).size - 1)
    print("Done.")
    print("Instances (final, guided filter, segmentation grid):", n_inst)
    print(f"XY weighting: W_XY={W_XY}, W_XZ={W_XZ}, W_YZ={W_YZ}, threshold={VOTE_THRESHOLD}")
    print("NOTE: Final labels are NOT rescaled back to original image size (per request).")
