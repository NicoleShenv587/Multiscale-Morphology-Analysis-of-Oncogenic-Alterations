#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path

import numpy as np
import imageio.v3 as imageio
import tifffile
from skimage.segmentation import mark_boundaries

from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)

# =========================
# USER SETTINGS
# =========================
INPUT_STACK = r"/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/mutation/NRAS_1/488/2025-12-24/38X_003/preprocess_output/03_segprep_float01_uint8.tif"  # <-- change me

MODEL_TYPE = "vit_b_lm"
CHECKPOINT = None

# Per-view modes (recommended to avoid APG crash in XZ/YZ)
MODE_XY = "apg"   # "amg" / "ais" / "apg"
MODE_XZ = "apg"   # use "ais" here for stability
MODE_YZ = "apg"   # use "ais" here for stability

# 2D tiling (for large 2D slices)
IS_TILED = False
TILE_SHAPE = (1024, 1024) if IS_TILED else None
HALO = (256, 256) if IS_TILED else None

NORMALIZE_TO_UINT8 = True

# If your stack loads as (Y,X,Z), set to "yxz"
STACK_LAYOUT = "zyx"  # "zyx" or "yxz"

# Linking thresholds (consistent IDs along slicing axis)
LINK_IOU_THRESH = 0.10
LINK_FRAC_CURR_THRESH = 0.20

# Overlay style (uSegment3D-like)
BOUNDARY_COLOR = (0, 1, 0)  # green
BOUNDARY_MODE = "thick"

# Save resliced raw stacks?
SAVE_RESLICE_RAW_STACKS = True


# =========================
# Helpers
# =========================

# for remove the lines

from skimage.morphology import opening, rectangle
from skimage.measure import label
from skimage.morphology import remove_small_objects


def remove_streaks_directional_from_labelmask(
    label2d: np.ndarray,
    direction: str,
    line_len: int = 60,
    line_thickness: int = 2,
    min_line_area: int = 80,
) -> np.ndarray:
    """
    Remove thin line artifacts from a 2D label mask by:
      1) extracting line-like foreground with morphological opening
      2) filtering extracted line components by area
      3) subtracting them from the label mask

    direction: "vertical" or "horizontal"
    """
    out = label2d.copy()
    fg = out > 0
    if not np.any(fg):
        return out

    direction = direction.lower()
    if direction == "vertical":
        se = rectangle(line_len, line_thickness)       # tall + thin
    elif direction == "horizontal":
        se = rectangle(line_thickness, line_len)       # wide + thin
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    # Extract long thin streaks
    streaks = opening(fg, se)

    # Keep only sufficiently large streak components (avoid removing real boundaries)
    cc = label(streaks)
    cc = remove_small_objects(cc, min_size=min_line_area) > 0

    # Subtract streak pixels
    out[cc] = 0
    return out


def clean_streaks_in_stack(
    lbl_stack: np.ndarray,
    direction: str,
    line_len: int = 60,
    line_thickness: int = 2,
    min_line_area: int = 80,
) -> np.ndarray:
    """
    Apply directional streak removal to each slice in a 3D stack (N,H,W).
    """
    out = lbl_stack.copy()
    for i in range(out.shape[0]):
        out[i] = remove_streaks_directional_from_labelmask(
            out[i],
            direction=direction,
            line_len=line_len,
            line_thickness=line_thickness,
            min_line_area=min_line_area,
        )
    return out


#for clean the mask

import scipy.ndimage as ndi


def relabel_consecutive(lbl: np.ndarray) -> np.ndarray:
    """Relabel non-zero IDs to 1..N (consecutive), keep 0 as background."""
    lbl = lbl.astype(np.int32, copy=False)
    ids = np.unique(lbl)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros_like(lbl, dtype=np.int32)

    new = np.zeros_like(lbl, dtype=np.int32)
    for new_id, old_id in enumerate(ids, start=1):
        new[lbl == old_id] = new_id
    return new


def clean_instance_labels_3d(
    lbl_zyx: np.ndarray,
    min_size: int = 200,
    max_size: int | None = None,
    connectivity: int = 1,
    relabel: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Remove objects that are too small / too large based on voxel count.

    Parameters
    ----------
    lbl_zyx : (Z,Y,X) int labels (instances)
    min_size : remove instances with voxel count < min_size
    max_size : remove instances with voxel count > max_size (None = no upper filter)
    connectivity : for connected-component splitting within each instance if needed
                   (1 = 6-connectivity in 3D). Usually keep 1.
    relabel : make output IDs consecutive 1..N

    Returns
    -------
    cleaned_lbl : cleaned labels
    stats : dict with counts
    """
    lbl = lbl_zyx.astype(np.int32, copy=False)

    # Count voxels per instance id
    max_id = int(lbl.max())
    if max_id == 0:
        return np.zeros_like(lbl, dtype=np.int32), {
            "kept": 0, "removed_small": 0, "removed_large": 0, "original": 0
        }

    counts = np.bincount(lbl.ravel(), minlength=max_id + 1)
    counts[0] = 0  # ignore background

    too_small = counts < int(min_size)
    too_small[0] = False

    if max_size is None:
        too_large = np.zeros_like(too_small, dtype=bool)
    else:
        too_large = counts > int(max_size)
        too_large[0] = False

    remove = too_small | too_large
    keep = ~remove
    keep[0] = True  # keep background

    # Fast remap table: removed ids -> 0, kept ids -> same (for now)
    remap = np.arange(max_id + 1, dtype=np.int32)
    remap[remove] = 0
    cleaned = remap[lbl]

    # Optional: enforce connectivity per kept instance (splits accidental disjoint parts)
    # Comment out if you want to preserve original IDs exactly.
    if connectivity is not None and connectivity > 0:
        structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
        # For each remaining id, ensure it's one connected component; split into new ids if not.
        # This is safer for removing tiny disconnected fragments attached to big objects.
        out = np.zeros_like(cleaned, dtype=np.int32)
        next_id = 1
        ids = np.unique(cleaned)
        ids = ids[ids != 0]
        for oid in ids:
            mask = (cleaned == oid)
            cc, ncc = ndi.label(mask, structure=structure)
            if ncc == 0:
                continue
            # keep all CCs, but they become separate instance IDs
            for k in range(1, ncc + 1):
                out[cc == k] = next_id
                next_id += 1
        cleaned = out

    if relabel:
        cleaned = relabel_consecutive(cleaned)

    stats = {
        "original": int((counts > 0).sum()),
        "removed_small": int(too_small.sum()),
        "removed_large": int(too_large.sum()),
        "kept": int(np.unique(cleaned).size - 1),
    }
    return cleaned.astype(np.int32, copy=False), stats


# for xz yz use apg
def patch_sam_maskdata_filter_cast_keep():
    """
    Fix APG NMS crash:
    Ensure 'keep' used for indexing is bool/int (not float).
    """
    import torch
    from segment_anything.utils.amg import MaskData

    orig_filter = MaskData.filter

    def filter_cast(self, keep):
        # keep can be torch tensor, numpy array, list...
        if isinstance(keep, torch.Tensor):
            if keep.dtype not in (
                torch.bool,
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            ):
                # Most often keep is float 0/1 -> convert to bool
                keep = keep.bool()
        return orig_filter(self, keep)

    MaskData.filter = filter_cast


# Call immediately
patch_sam_maskdata_filter_cast_keep()


def normalize_to_uint8(img2d: np.ndarray) -> np.ndarray:
    if img2d.dtype == np.uint8:
        return img2d
    x = img2d.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99.8))
    if hi <= lo:
        return np.zeros_like(img2d, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (255.0 * x).astype(np.uint8)


def normalize01(img2d: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99.8))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay_boundaries_useg_style(raw2d: np.ndarray, labels2d: np.ndarray) -> np.ndarray:
    raw01 = normalize01(raw2d)
    raw_rgb = np.dstack([raw01, raw01, raw01])
    over = mark_boundaries(raw_rgb, labels2d, color=BOUNDARY_COLOR, mode=BOUNDARY_MODE)
    return (over * 255).astype(np.uint8)


def get_generate_kwargs(mode: str) -> dict:
    if mode == "amg":
        return {
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "box_nms_thresh": 0.7,
            "crop_nms_thresh": 0.7,
            "min_mask_region_area": 0,
        }
    if mode == "ais":
        return {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.5,
            "foreground_threshold": 0.5,
            "foreground_smoothing": 1.0,
            "distance_smoothing": 1.6,
            "min_size": 0,
        }
    if mode == "apg":
        # keep APG default-ish; XZ/YZ will use AIS to avoid NMS dtype crash
        return {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.5,
            "foreground_threshold": 0.5,
            "nms_threshold": 0.7,#0.9,
            "intersection_over_min": 0.0,
            # If you still see APG issues even in XY, try lowering nms_threshold to 0.7
            # or add: "intersection_over_min": 0.0
        }
    raise ValueError("Mode must be 'amg', 'ais', or 'apg'.")


def safe_cast_labels(lbl: np.ndarray) -> np.ndarray:
    mx = int(lbl.max()) if lbl.size else 0
    if mx <= 65535:
        return lbl.astype(np.uint16, copy=False)
    return lbl.astype(np.uint32, copy=False)


def load_stack(path: Path) -> np.ndarray:
    vol = imageio.imread(path)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape={vol.shape}")

    if STACK_LAYOUT.lower() == "zyx":
        return vol
    if STACK_LAYOUT.lower() == "yxz":
        return np.moveaxis(vol, -1, 0)
    raise ValueError("STACK_LAYOUT must be 'zyx' or 'yxz'.")


def segment_view_stack(predictor, segmenter, view_stack: np.ndarray, gen_kwargs: dict, view_name: str) -> np.ndarray:
    """
    view_stack: (N, H, W)
    return: (N, H, W) int32 label images (may be 0 if no objects)
    """
    n = view_stack.shape[0]
    out = np.zeros_like(view_stack, dtype=np.int32)

    for i in range(n):
        img2d = view_stack[i]
        if NORMALIZE_TO_UINT8:
            img2d = normalize_to_uint8(img2d)

        seg_res = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=img2d,
            ndim=2,
            tile_shape=TILE_SHAPE,
            halo=HALO,
            **gen_kwargs,
        )

        # micro-sam can return ndarray or list; convert to label image
        seg2d = to_label_image(seg_res, img2d.shape)
        out[i] = seg2d

        if (i % max(1, n // 10)) == 0 or i == n - 1:
            print(f"  [{view_name}] slice {i+1}/{n} seg.max()={int(seg2d.max()) if seg2d.size else 0}")

    return out


def to_label_image(seg_result, shape_hw: tuple[int, int]) -> np.ndarray:
    H, W = shape_hw

    if isinstance(seg_result, np.ndarray):
        if seg_result.ndim != 2:
            raise ValueError(f"Expected 2D label array, got shape={seg_result.shape}")
        return seg_result.astype(np.int32, copy=False)

    if isinstance(seg_result, list):
        labels = np.zeros((H, W), dtype=np.int32)
        next_id = 1
        for inst in seg_result:
            if isinstance(inst, dict) and "segmentation" in inst:
                m = inst["segmentation"]
            elif hasattr(inst, "segmentation"):
                m = inst.segmentation
            else:
                m = inst
            m = np.asarray(m).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"Mask shape {m.shape} != {(H, W)}")
            # overwrite allowed (often better for cytoplasm)
            labels[m] = next_id
            next_id += 1
        return labels

    raise TypeError(f"Unsupported segmentation output type: {type(seg_result)}")


def compute_intersections(prev: np.ndarray, curr: np.ndarray, prev_max: int, curr_max: int) -> np.ndarray:
    mask = (prev > 0) & (curr > 0)
    if not np.any(mask):
        return np.zeros((prev_max + 1, curr_max + 1), dtype=np.int64)
    prev_ids = prev[mask].ravel()
    curr_ids = curr[mask].ravel()
    flat = prev_ids * (curr_max + 1) + curr_ids
    counts = np.bincount(flat, minlength=(prev_max + 1) * (curr_max + 1))
    return counts.reshape((prev_max + 1, curr_max + 1))


def link_labels_along_first_axis(lbl_nhw: np.ndarray) -> np.ndarray:
    n = lbl_nhw.shape[0]
    linked = np.zeros_like(lbl_nhw, dtype=np.int32)

    linked[0] = lbl_nhw[0]
    global_max = int(linked[0].max()) if linked[0].size else 0

    for i in range(1, n):
        prev = linked[i - 1]
        curr = lbl_nhw[i]

        prev_max = int(prev.max()) if prev.size else 0
        curr_max = int(curr.max()) if curr.size else 0

        if curr_max == 0:
            linked[i] = 0
            continue

        prev_area = np.bincount(prev.ravel(), minlength=prev_max + 1).astype(np.float32)
        curr_area = np.bincount(curr.ravel(), minlength=curr_max + 1).astype(np.float32)
        inter = compute_intersections(prev, curr, prev_max, curr_max).astype(np.float32)

        mapping = np.zeros(curr_max + 1, dtype=np.int32)
        used_prev = set()

        for cid in range(1, curr_max + 1):
            if curr_area[cid] <= 0:
                continue
            inter_col = inter[:, cid]
            if inter_col.max() <= 0:
                continue

            pid = int(np.argmax(inter_col))
            if pid == 0:
                continue

            i_ = float(inter_col[pid])
            union = float(prev_area[pid] + curr_area[cid] - i_)
            iou = (i_ / union) if union > 0 else 0.0
            frac_curr = (i_ / float(curr_area[cid])) if curr_area[cid] > 0 else 0.0

            if (iou >= LINK_IOU_THRESH or frac_curr >= LINK_FRAC_CURR_THRESH) and (pid not in used_prev):
                mapping[cid] = pid
                used_prev.add(pid)

        out = np.zeros_like(curr, dtype=np.int32)
        for cid in range(1, curr_max + 1):
            m = (curr == cid)
            if not np.any(m):
                continue
            gid = mapping[cid]
            if gid == 0:
                global_max += 1
                gid = global_max
            out[m] = gid

        linked[i] = out

    return linked


# =========================
# Reslicing (raw) and mapping back to ZYX
# =========================
def make_xy_stack(vol_zyx: np.ndarray) -> np.ndarray:
    return vol_zyx  # (Z,Y,X)


def make_xz_stack(vol_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(vol_zyx, (1, 0, 2))  # (Y,Z,X)


def make_yz_stack(vol_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(vol_zyx, (2, 0, 1))  # (X,Z,Y)


def map_xz_to_zyx(lbl_xz: np.ndarray) -> np.ndarray:
    return np.transpose(lbl_xz, (1, 0, 2))  # (Y,Z,X) -> (Z,Y,X)


def map_yz_to_zyx(lbl_yz: np.ndarray) -> np.ndarray:
    return np.transpose(lbl_yz, (1, 2, 0))  # (X,Z,Y) -> (Z,Y,X)


# =========================
# Main
# =========================
def main():
    in_path = Path(INPUT_STACK)
    vol_zyx = load_stack(in_path)
    z, y, x = vol_zyx.shape
    print(f"Loaded: {in_path}")
    print(f"  shape (Z,Y,X) = {vol_zyx.shape}, dtype={vol_zyx.dtype}")
    print(f"Modes: XY={MODE_XY}, XZ={MODE_XZ}, YZ={MODE_YZ}\n")

    out_dir = in_path.parent / "segment"
    out_dir.mkdir(exist_ok=True)

    # Build raw view stacks
    xy_raw = make_xy_stack(vol_zyx)  # (Z,Y,X)
    xz_raw = make_xz_stack(vol_zyx)  # (Y,Z,X)
    yz_raw = make_yz_stack(vol_zyx)  # (X,Z,Y)

    if SAVE_RESLICE_RAW_STACKS:
        out_xz_raw = out_dir / f"{in_path.stem}_raw_xz_YZX.tif"
        out_yz_raw = out_dir / f"{in_path.stem}_raw_yz_XZY.tif"
        tifffile.imwrite(out_xz_raw, xz_raw)
        tifffile.imwrite(out_yz_raw, yz_raw)
        print(f"Saved raw XZ stack: {out_xz_raw}")
        print(f"Saved raw YZ stack: {out_yz_raw}\n")

    # Create predictor+segmenter per view mode (safe + explicit)
    predictor_xy, segmenter_xy = get_predictor_and_segmenter(
        model_type=MODEL_TYPE, checkpoint=CHECKPOINT, segmentation_mode=MODE_XY, is_tiled=IS_TILED
    )
    predictor_xz, segmenter_xz = get_predictor_and_segmenter(
        model_type=MODEL_TYPE, checkpoint=CHECKPOINT, segmentation_mode=MODE_XZ, is_tiled=IS_TILED
    )
    predictor_yz, segmenter_yz = get_predictor_and_segmenter(
        model_type=MODEL_TYPE, checkpoint=CHECKPOINT, segmentation_mode=MODE_YZ, is_tiled=IS_TILED
    )

    kwargs_xy = get_generate_kwargs(MODE_XY)
    kwargs_xz = get_generate_kwargs(MODE_XZ)
    kwargs_yz = get_generate_kwargs(MODE_YZ)

    # Segment each view
    print("Segmenting XY...")
    lbl_xy = segment_view_stack(predictor_xy, segmenter_xy, xy_raw, kwargs_xy, "xy")  # (Z,Y,X)

    print("\nSegmenting XZ...")
    lbl_xz = segment_view_stack(predictor_xz, segmenter_xz, xz_raw, kwargs_xz, "xz")  # (Y,Z,X)

    print("\nSegmenting YZ...")
    lbl_yz = segment_view_stack(predictor_yz, segmenter_yz, yz_raw, kwargs_yz, "yz")  # (X,Z,Y)

    #remove streaks 

    # Remove horizontal streaks in XZ slices
    lbl_xz = clean_streaks_in_stack(
        lbl_xz,
        direction="horizontal",
        line_len=80,          # tune
        line_thickness=2,     # tune
        min_line_area=120,    # tune
    )
    
    # Remove vertical streaks in YZ slices
    lbl_yz = clean_streaks_in_stack(
        lbl_yz,
        direction="vertical",
        line_len=60,          # tune
        line_thickness=2,     # tune
        min_line_area=100,    # tune
    )
    
    # Now do linking as you already do:
    lbl_xz_link = link_labels_along_first_axis(lbl_xz)
    lbl_yz_link = link_labels_along_first_axis(lbl_yz)


    # Link IDs along slicing axis
    print("\nLinking IDs: XY along Z...")
    lbl_xy_link = link_labels_along_first_axis(lbl_xy)

    print("Linking IDs: XZ along Y...")
    lbl_xz_link = link_labels_along_first_axis(lbl_xz)

    print("Linking IDs: YZ along X...")
    lbl_yz_link = link_labels_along_first_axis(lbl_yz)

    # Map to ZYX
    xy_zyx = lbl_xy_link
    xz_zyx = map_xz_to_zyx(lbl_xz_link)
    yz_zyx = map_yz_to_zyx(lbl_yz_link)
    
    #clean mask
    # =========================
    # FINAL CLEANING (remove tiny + huge instances)
    # =========================
    MIN_SIZE = 200        # <-- adjust (voxels)
    MAX_SIZE = 100000    # <-- adjust (voxels), or set None to disable
    
    xy_zyx, st_xy = clean_instance_labels_3d(xy_zyx, min_size=MIN_SIZE, max_size=MAX_SIZE, connectivity=1, relabel=True)
    xz_zyx, st_xz = clean_instance_labels_3d(xz_zyx, min_size=MIN_SIZE, max_size=MAX_SIZE, connectivity=1, relabel=True)
    yz_zyx, st_yz = clean_instance_labels_3d(yz_zyx, min_size=MIN_SIZE, max_size=MAX_SIZE, connectivity=1, relabel=True)
    
    print("\nCleaning stats:")
    print("  XY:", st_xy)
    print("  XZ:", st_xz)
    print("  YZ:", st_yz)


    # Save 3 final masks
    out_xy = out_dir / f"{in_path.stem}_seg_xy_linked_ZYX.tif"
    out_xz = out_dir / f"{in_path.stem}_seg_xz_linked_ZYX.tif"
    out_yz = out_dir / f"{in_path.stem}_seg_yz_linked_ZYX.tif"
    tifffile.imwrite(out_xy, safe_cast_labels(xy_zyx))
    tifffile.imwrite(out_xz, safe_cast_labels(xz_zyx))
    tifffile.imwrite(out_yz, safe_cast_labels(yz_zyx))

    print("\nSaved final 3 masks:")
    print(f"  XY: {out_xy}")
    print(f"  XZ: {out_xz}")
    print(f"  YZ: {out_yz}")

    # Save mid-slice overlays (TIFF RGB)
    z0 = z // 2
    y0 = y // 2
    x0 = x // 2

    ov_xy = overlay_boundaries_useg_style(xy_raw[z0], lbl_xy_link[z0])
    ov_xz = overlay_boundaries_useg_style(xz_raw[y0], lbl_xz_link[y0])  # slice is (Z,X)
    ov_yz = overlay_boundaries_useg_style(yz_raw[x0], lbl_yz_link[x0])  # slice is (Z,Y)

    out_ov_xy = out_dir / f"{in_path.stem}_mid_xy_overlay_uSeg.tif"
    out_ov_xz = out_dir / f"{in_path.stem}_mid_xz_overlay_uSeg.tif"
    out_ov_yz = out_dir / f"{in_path.stem}_mid_yz_overlay_uSeg.tif"

    tifffile.imwrite(out_ov_xy, ov_xy, photometric="rgb")
    tifffile.imwrite(out_ov_xz, ov_xz, photometric="rgb")
    tifffile.imwrite(out_ov_yz, ov_yz, photometric="rgb")

    print("\nSaved middle overlays:")
    print(f"  XY (z={z0}): {out_ov_xy}")
    print(f"  XZ (y={y0}): {out_ov_xz}")
    print(f"  YZ (x={x0}): {out_ov_yz}")

    print("\nMax label IDs (sanity):")
    print("  XY:", int(xy_zyx.max()))
    print("  XZ:", int(xz_zyx.max()))
    print("  YZ:", int(yz_zyx.max()))


if __name__ == "__main__":
    main()
