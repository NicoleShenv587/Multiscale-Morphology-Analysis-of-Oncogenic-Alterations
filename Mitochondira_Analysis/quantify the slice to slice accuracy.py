

# ============================================================
# USER SETTINGS
# ============================================================


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tifffile as tiff

import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk

# ============================================================
# USER SETTINGS
# ============================================================

RAW_IMAGE_PATH = r"Z:/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Control_tub_mito/Segmented data/Cell2/Pre_downsampe1_new/Cell/segprep_preprocess_uint8.tif"
MASK1_PATH = r"Z:/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Control_tub_mito/Segmented data/Cell2/Pre_downsampe1_new/usegment3D_indirect_xy_weighted_postprocess_original/labels_xy_downsampled.tif"
MASK2_PATH = r"Z:/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Control_tub_mito/Segmented data/Cell2/Pre_downsampe1_new/usegment3D_indirect_xy_weighted_postprocess_original/uSegment3D_labels_postprocess-diffuse_labels.tif"

OUT_DIR = r"Z:/Liver/1stHCC_Kelly_betacatin_NRas/EXASLM/Control_tub_mito/Segmented data/Cell2/accuracy"


DPI = 600

RAW_P_LOW = 1
RAW_P_HIGH = 99.8

MASK1_NAME = "Mask1"
MASK2_NAME = "Mask2"

RAW_THRESHOLD_METHOD = "otsu"      # "otsu" or "percentile"
RAW_FOREGROUND_PERCENTILE = 80     # used only if RAW_THRESHOLD_METHOD="percentile"


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_tif(path):
    return tiff.imread(path)


def match_volume_shape(reference, moving, is_mask=True):
    """
    Resize moving volume to match reference shape.
    Masks use nearest-neighbor interpolation to preserve labels.
    """
    if reference.shape == moving.shape:
        return moving

    print(f"Resizing {moving.shape} -> {reference.shape}")

    resized = resize(
        moving,
        reference.shape,
        order=0 if is_mask else 1,
        preserve_range=True,
        anti_aliasing=False if is_mask else True
    )

    if is_mask:
        resized = np.rint(resized).astype(moving.dtype)
    else:
        resized = resized.astype(moving.dtype)

    return resized


def normalize_raw_slice(raw_2d):
    lo, hi = np.percentile(raw_2d, [RAW_P_LOW, RAW_P_HIGH])
    if hi <= lo:
        return np.zeros_like(raw_2d, dtype=np.float32)
    return rescale_intensity(raw_2d, in_range=(lo, hi), out_range=(0, 1))


def extract_slice(volume, orientation, index):
    """
    Input volume shape must be (Z, Y, X).
    """
    orientation = orientation.lower()

    if orientation == "xy":
        return volume[index, :, :]
    elif orientation == "xz":
        return volume[:, index, :]
    elif orientation == "yz":
        return volume[:, :, index]
    else:
        raise ValueError("orientation must be xy, xz, or yz")


def get_n_slices(shape, orientation):
    z, y, x = shape

    if orientation == "xy":
        return z
    elif orientation == "xz":
        return y
    elif orientation == "yz":
        return x
    else:
        raise ValueError("orientation must be xy, xz, or yz")


def get_middle_index(shape, orientation):
    z, y, x = shape

    if orientation == "xy":
        return z // 2
    elif orientation == "xz":
        return y // 2
    elif orientation == "yz":
        return x // 2
    else:
        raise ValueError("orientation must be xy, xz, or yz")


# ============================================================
# METRICS
# ============================================================

def binary_iou(a, b):
    a = a > 0
    b = b > 0

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0

    return inter / union


def binary_dice(a, b):
    a = a > 0
    b = b > 0

    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()

    if total == 0:
        return 1.0

    return 2 * inter / total


def raw_to_foreground(raw_2d):
    """
    Convert raw image slice to binary foreground.
    This is approximate, not true ground truth.
    """
    raw_float = raw_2d.astype(np.float32)

    if np.max(raw_float) == np.min(raw_float):
        return np.zeros_like(raw_float, dtype=bool)

    if RAW_THRESHOLD_METHOD.lower() == "otsu":
        thresh = threshold_otsu(raw_float)
    elif RAW_THRESHOLD_METHOD.lower() == "percentile":
        thresh = np.percentile(raw_float, RAW_FOREGROUND_PERCENTILE)
    else:
        raise ValueError("RAW_THRESHOLD_METHOD must be 'otsu' or 'percentile'")

    return raw_float > thresh


# ============================================================
# OVERLAY PLOTS
# ============================================================

def make_single_mask_overlay(raw_2d, mask_2d, color="yellow", thickness=2):
    """
    Overlay one mask boundary on raw image with adjustable thickness.
    """
    raw_norm = normalize_raw_slice(raw_2d)
    rgb = np.dstack([raw_norm, raw_norm, raw_norm])

    boundary = find_boundaries(mask_2d, mode="outer")

    # --- make thicker WITHOUT extra libraries ---
    for _ in range(thickness):
        boundary = np.logical_or(boundary, np.roll(boundary, 1, axis=0))
        boundary = np.logical_or(boundary, np.roll(boundary, -1, axis=0))
        boundary = np.logical_or(boundary, np.roll(boundary, 1, axis=1))
        boundary = np.logical_or(boundary, np.roll(boundary, -1, axis=1))

    if color == "yellow":
        rgb[boundary, 0] = 1.0
        rgb[boundary, 1] = 1.0
        rgb[boundary, 2] = 0.0

    elif color == "green":
        rgb[boundary, 0] = 0.0
        rgb[boundary, 1] = 1.0
        rgb[boundary, 2] = 0.0

    return rgb

def save_middle_slice_side_by_side_raw_mask1_mask2(raw, mask1, mask2, out_dir):
    """
    Save middle slice for XY/XZ/YZ:
    Raw | Raw + Mask1 | Raw + Mask2
    """
    for orientation in ["xy", "xz", "yz"]:
        idx = get_middle_index(raw.shape, orientation)

        raw_2d = extract_slice(raw, orientation, idx)
        mask1_2d = extract_slice(mask1, orientation, idx)
        mask2_2d = extract_slice(mask2, orientation, idx)

        raw_norm = normalize_raw_slice(raw_2d)
        overlay_mask1 = make_single_mask_overlay(raw_2d, mask1_2d, color="yellow", thickness = 2)
        overlay_mask2 = make_single_mask_overlay(raw_2d, mask2_2d, color="green", thickness =2)

        iou_mask2_mask1 = binary_iou(mask2_2d, mask1_2d)
        raw_fg = raw_to_foreground(raw_2d)
        iou_mask2_raw = binary_iou(mask2_2d, raw_fg)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(raw_norm, cmap="gray")
        axes[0].set_title("Raw")
        axes[0].axis("off")

        axes[1].imshow(overlay_mask1)
        axes[1].set_title(f"Raw + {MASK1_NAME}\nyellow boundary")
        axes[1].axis("off")

        axes[2].imshow(overlay_mask2)
        axes[2].set_title(f"Raw + {MASK2_NAME}\ngreen boundary")
        axes[2].axis("off")

        fig.suptitle(
            f"{orientation.upper()} middle slice {idx} | "
            f"Mask2 vs Mask1 IoU={iou_mask2_mask1:.3f}, "
            f"Mask2 vs raw IoU={iou_mask2_raw:.3f}",
            fontsize=11
        )

        plt.tight_layout()

        out_base = os.path.join(
            out_dir,
            f"{orientation}_middle_raw_mask1_mask2_side_by_side"
        )

        plt.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
        plt.savefig(out_base + ".svg", bbox_inches="tight")
        plt.close()


# ============================================================
# ACCURACY CALCULATION
# ============================================================

def evaluate_accuracy(mask1, mask2, raw):
    rows = []

    for orientation in ["xy", "xz", "yz"]:
        n = get_n_slices(mask1.shape, orientation)

        for i in range(n):
            mask1_2d = extract_slice(mask1, orientation, i)
            mask2_2d = extract_slice(mask2, orientation, i)
            raw_2d = extract_slice(raw, orientation, i)

            raw_fg = raw_to_foreground(raw_2d)

            rows.append({
                "orientation": orientation,
                "slice_index": i,

                "iou_mask2_vs_mask1": binary_iou(mask2_2d, mask1_2d),
                "dice_mask2_vs_mask1": binary_dice(mask2_2d, mask1_2d),

                "iou_mask2_vs_raw": binary_iou(mask2_2d, raw_fg),
                "dice_mask2_vs_raw": binary_dice(mask2_2d, raw_fg),

                "mask1_foreground_pixels": int((mask1_2d > 0).sum()),
                "mask2_foreground_pixels": int((mask2_2d > 0).sum()),
                "raw_foreground_pixels": int(raw_fg.sum()),
            })

    return pd.DataFrame(rows)


# ============================================================
# ACCURACY PLOTS
# ============================================================

def auto_ylim(y_values, padding=0.15):
    """
    Automatically zoom y-axis around data so dots are clearly visible.
    """
    y_values = np.asarray(y_values, dtype=float)
    y_values = y_values[np.isfinite(y_values)]

    if len(y_values) == 0:
        return 0, 1

    y_min = np.min(y_values)
    y_max = np.max(y_values)

    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05

    y_range = y_max - y_min

    y_low = max(0, y_min - padding * y_range)
    y_high = min(1, y_max + padding * y_range)

    if y_high - y_low < 0.1:
        center = (y_high + y_low) / 2
        y_low = max(0, center - 0.05)
        y_high = min(1, center + 0.05)

    return y_low, y_high


def plot_accuracy_curves(df, out_dir, metric_col, title, ylabel, out_name):
    plt.figure(figsize=(10, 3.8))

    colors = {
        "xy": "#1f77b4",
        "xz": "#ff7f0e",
        "yz": "#2ca02c"
    }

    all_y = []

    for orientation in ["xy", "xz", "yz"]:
        sub = df[df["orientation"] == orientation]

        x = sub["slice_index"].to_numpy()
        y = sub[metric_col].to_numpy()

        all_y.extend(y)

        plt.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=orientation.upper(),
            color=colors[orientation]
        )

    y_low, y_high = auto_ylim(all_y, padding=0.15)

    plt.xlabel("Slice index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(y_low, y_high)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_base = os.path.join(out_dir, out_name)
    plt.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
    plt.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close()


def plot_all_accuracy(df, out_dir):
    plot_accuracy_curves(
        df,
        out_dir,
        metric_col="iou_mask2_vs_mask1",
        title="Per-slice segmentation accuracy: Mask2 vs Mask1",
        ylabel="IoU per slice",
        out_name="accuracy_mask2_vs_mask1_meanIoU_zoomed"
    )

    plot_accuracy_curves(
        df,
        out_dir,
        metric_col="iou_mask2_vs_raw",
        title="Per-slice segmentation accuracy: Mask2 vs raw foreground",
        ylabel="IoU per slice",
        out_name="accuracy_mask2_vs_raw_meanIoU_zoomed"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    print("Loading data...")
    raw = load_tif(RAW_IMAGE_PATH)
    mask1 = load_tif(MASK1_PATH)
    mask2 = load_tif(MASK2_PATH)

    print("Original shapes:")
    print("Raw:  ", raw.shape)
    print("Mask1:", mask1.shape)
    print("Mask2:", mask2.shape)

    mask1 = match_volume_shape(raw, mask1, is_mask=True)
    mask2 = match_volume_shape(raw, mask2, is_mask=True)

    print("Aligned shapes:")
    print("Raw:  ", raw.shape)
    print("Mask1:", mask1.shape)
    print("Mask2:", mask2.shape)

    save_middle_slice_side_by_side_raw_mask1_mask2(raw, mask1, mask2, OUT_DIR)

    df = evaluate_accuracy(mask1, mask2, raw)

    csv_path = os.path.join(OUT_DIR, "slice_accuracy_mask2_vs_mask1_and_raw.csv")
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    plot_all_accuracy(df, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
