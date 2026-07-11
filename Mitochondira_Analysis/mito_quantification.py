#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile as tiff

from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from skimage.measure import regionprops
from skimage import morphology

try:
    from skimage.morphology import skeletonize_3d
except Exception:
    skeletonize_3d = None

from skan import Skeleton, summarize


# ============================================================
# USER SETTINGS
# ============================================================
INPUT_FOLDER = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Vessel/QC/PCC"
OUTPUT_DIR   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Vessel/QC/PCC/results"

VOXEL_SIZE_UM = (0.17, 0.17, 0.17)   # (z, y, x)
MIN_SIZE_VOXELS = 20
FILE_PATTERN = None   # e.g. "*.tif" or "*label*.tif"

# ------------------------------------------------------------
# Multiprocessing controls
# ------------------------------------------------------------
# Number of images processed in parallel
N_IMAGE_WORKERS = 2

# Number of object workers used inside each image process
N_OBJECT_WORKERS = 8

# Number of labeled objects per chunk for image-level object multiprocessing
OBJECT_CHUNK_SIZE = 30

# Automatically limit object workers when image parallelism is used,
# to avoid CPU oversubscription.
AUTO_LIMIT_OBJECT_WORKERS = True

# ------------------------------------------------------------
# Object cleanup before skeletonization
# ------------------------------------------------------------
APPLY_BINARY_OPENING = False
OPENING_RADIUS = 1

APPLY_BINARY_CLOSING = True
CLOSING_RADIUS = 1

FILL_HOLES = True

# ------------------------------------------------------------
# Skeleton cleanup / branch settings
# ------------------------------------------------------------
PRUNE_SHORT_TERMINAL_BRANCHES = True
MIN_TERMINAL_BRANCH_LENGTH_UM = 1.0   # prune tiny ghost tips before classification
MIN_SIDE_BRANCH_LEN_UM = 2.0          # keep side branches only if >= this
KEEP_LARGEST_SKELETON_COMPONENT = False

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
PRINT_EVERY_N_OBJECTS = 10


# ============================================================
# HELPERS
# ============================================================
def log(msg):
    print(msg, flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_input_tiffs(folder, file_pattern=None):
    if file_pattern is not None:
        paths = sorted(glob.glob(os.path.join(folder, file_pattern)))
    else:
        paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
        paths += sorted(glob.glob(os.path.join(folder, "*.tiff")))
        paths = sorted(list(set(paths)))
    return [p for p in paths if os.path.isfile(p)]


def safe_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        stem = stem.replace(ch, "_")
    return stem


def choose_object_workers(n_image_workers, requested_object_workers):
    cpu_total = os.cpu_count() or 1
    requested_object_workers = max(1, int(requested_object_workers))
    n_image_workers = max(1, int(n_image_workers))

    if not AUTO_LIMIT_OBJECT_WORKERS:
        return requested_object_workers

    max_per_image = max(1, cpu_total // n_image_workers)
    return min(requested_object_workers, max_per_image)


def load_labeled_mask(path):
    arr = tiff.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF, got shape {arr.shape} for file: {path}")
    return arr


def filter_small_labels(labeled, min_size=20):
    """
    Relabel surviving objects consecutively: 1, 2, 3, ...
    """
    out = np.zeros_like(labeled, dtype=np.uint16)
    new_id = 1

    ids = np.unique(labeled)
    ids = ids[ids > 0]

    for old_id in ids:
        obj = (labeled == old_id)
        if int(obj.sum()) >= int(min_size):
            out[obj] = new_id
            new_id += 1

    return out


def safe_regionprops_single(obj_mask, spacing):
    label_img = obj_mask.astype(np.uint8)
    try:
        props = regionprops(label_img, spacing=spacing)
    except Exception:
        return None
    if len(props) == 0:
        return None
    return props[0]


def safe_get_axis_lengths(rp):
    major_axis_length_um = np.nan
    minor_axis_length_um = np.nan

    if rp is None:
        return major_axis_length_um, minor_axis_length_um

    try:
        major_axis_length_um = float(rp.axis_major_length)
        if not np.isfinite(major_axis_length_um):
            major_axis_length_um = np.nan
    except Exception:
        major_axis_length_um = np.nan

    try:
        minor_axis_length_um = float(rp.axis_minor_length)
        if not np.isfinite(minor_axis_length_um):
            minor_axis_length_um = np.nan
    except Exception:
        minor_axis_length_um = np.nan

    return major_axis_length_um, minor_axis_length_um


def normalize_skan_columns(tbl: pd.DataFrame) -> pd.DataFrame:
    tbl = tbl.copy()

    rename_map = {}
    if "branch_type" not in tbl.columns:
        if "type" in tbl.columns:
            rename_map["type"] = "branch_type"
        elif "branch-type" in tbl.columns:
            rename_map["branch-type"] = "branch_type"

    if "branch_distance" not in tbl.columns:
        if "distance" in tbl.columns:
            rename_map["distance"] = "branch_distance"
        elif "branch-distance" in tbl.columns:
            rename_map["branch-distance"] = "branch_distance"

    if rename_map:
        tbl = tbl.rename(columns=rename_map)

    return tbl


def get_skan_length_column(branch_table):
    for candidate in ["branch_distance", "branch-distance", "distance"]:
        if candidate in branch_table.columns:
            return candidate
    return None


def get_skan_type_column(branch_table):
    for candidate in ["branch_type", "branch-type", "type"]:
        if candidate in branch_table.columns:
            return candidate
    return None


def compute_skeleton_degree(skel):
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    neighbor_count = ndi.convolve(skel_u8, kernel, mode="constant", cval=0)
    neighbor_count = neighbor_count - skel_u8
    return neighbor_count


def count_endpoints_and_junctions(skel):
    degree = compute_skeleton_degree(skel)
    endpoints = np.logical_and(skel, degree == 1)
    junctions = np.logical_and(skel, degree >= 3)
    return int(endpoints.sum()), int(junctions.sum())


def has_at_least_one_skeleton_edge(skel):
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = ndi.convolve(skel_u8, kernel, mode="constant", cval=0)
    return np.any((skel_u8 > 0) & (neighbor_count > 0))


def keep_largest_component(binary):
    binary = binary.astype(bool)
    if binary.sum() == 0:
        return binary

    lab, n = ndi.label(binary)
    if n <= 1:
        return binary

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep_id = np.argmax(sizes)
    return lab == keep_id


def clean_object(obj):
    out = obj.astype(bool).copy()

    if APPLY_BINARY_OPENING:
        out = morphology.binary_opening(out, morphology.ball(OPENING_RADIUS))

    if APPLY_BINARY_CLOSING:
        out = morphology.binary_closing(out, morphology.ball(CLOSING_RADIUS))

    if FILL_HOLES:
        out = ndi.binary_fill_holes(out)

    return out.astype(bool)


def coords_to_mask(shape, coords):
    out = np.zeros(shape, dtype=bool)
    coords = np.asarray(coords)

    if coords.size == 0:
        return out

    coords = np.round(coords).astype(int)

    valid = np.ones(len(coords), dtype=bool)
    for d in range(coords.shape[1]):
        valid &= (coords[:, d] >= 0) & (coords[:, d] < shape[d])

    coords = coords[valid]
    if len(coords) > 0:
        out[tuple(coords.T)] = True

    return out


def prune_short_terminal_branches(skel, spacing=(1, 1, 1), min_length_um=1.0, max_iter=20):
    """
    Remove short terminal tips before backbone/side-branch classification.
    """
    skel = skel.astype(bool).copy()

    for _ in range(max_iter):
        if skel.sum() == 0:
            break

        try:
            sk = Skeleton(skel, spacing=spacing)
            bt = summarize(sk, separator="_")
            bt = normalize_skan_columns(bt)
        except Exception:
            break

        if len(bt) == 0:
            break

        length_col = get_skan_length_column(bt)
        type_col = get_skan_type_column(bt)

        if length_col is None or type_col is None:
            break

        degree = compute_skeleton_degree(skel)
        removed_any = False
        skel_new = skel.copy()

        for i, row in bt.iterrows():
            try:
                branch_len = float(row[length_col])
                branch_type = int(row[type_col])
            except Exception:
                continue

            if branch_type == 1 and branch_len < float(min_length_um):
                try:
                    coords = np.round(sk.path_coordinates(i)).astype(int)
                except Exception:
                    continue

                for z, y, x in coords:
                    if (
                        0 <= z < skel.shape[0]
                        and 0 <= y < skel.shape[1]
                        and 0 <= x < skel.shape[2]
                    ):
                        if degree[z, y, x] < 3:
                            skel_new[z, y, x] = False
                            removed_any = True

        skel = skel_new

        if KEEP_LARGEST_SKELETON_COMPONENT and skel.sum() > 0:
            skel = keep_largest_component(skel)

        if not removed_any:
            break

    return skel


def longest_geodesic_path_mask(skel, spacing=(1, 1, 1)):
    """
    Longest endpoint-to-endpoint geodesic path = backbone mask.
    """
    skel = skel.astype(bool)
    if skel.sum() == 0:
        return skel.copy()

    coords = np.argwhere(skel)
    n_nodes = len(coords)

    if n_nodes <= 2:
        return skel.copy()

    index_map = -np.ones(skel.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords):
        index_map[z, y, x] = i

    degree = compute_skeleton_degree(skel)
    endpoints_mask = np.logical_and(skel, degree == 1)
    endpoint_coords = np.argwhere(endpoints_mask)

    if len(endpoint_coords) < 2:
        return keep_largest_component(skel)

    rows = []
    cols = []
    vals = []

    sz, sy, sx = spacing
    neighbor_offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                dist = np.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2)
                neighbor_offsets.append((dz, dy, dx, dist))

    shape = skel.shape
    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx, dist in neighbor_offsets:
            zz, yy, xx = z + dz, y + dy, x + dx
            if 0 <= zz < shape[0] and 0 <= yy < shape[1] and 0 <= xx < shape[2]:
                j = index_map[zz, yy, xx]
                if j >= 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(dist)

    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    endpoint_indices = np.array([index_map[z, y, x] for z, y, x in endpoint_coords], dtype=int)

    dist_matrix, predecessors = csgraph.dijkstra(
        graph,
        directed=False,
        indices=endpoint_indices,
        return_predecessors=True,
    )

    dist_matrix = np.asarray(dist_matrix)
    finite = np.isfinite(dist_matrix)
    if not finite.any():
        return keep_largest_component(skel)

    dist_copy = dist_matrix.copy()
    dist_copy[~finite] = -np.inf

    ep_row, node_j = np.unravel_index(np.argmax(dist_copy), dist_copy.shape)
    start_node = endpoint_indices[ep_row]
    end_node = node_j

    if not np.isfinite(dist_matrix[ep_row, end_node]):
        return keep_largest_component(skel)

    path_nodes = []
    cur = end_node
    while cur != start_node and cur != -9999:
        path_nodes.append(cur)
        cur = predecessors[ep_row, cur]
    path_nodes.append(start_node)

    out = np.zeros_like(skel, dtype=bool)
    for node in path_nodes:
        z, y, x = coords[node]
        out[z, y, x] = True

    return out


def branch_attaches_to_backbone(path_mask, backbone_mask):
    """
    A side branch must touch backbone in a 3x3x3 neighborhood.
    """
    if not np.any(path_mask):
        return False
    if not np.any(backbone_mask):
        return False

    dilated_backbone = ndi.binary_dilation(
        backbone_mask,
        structure=np.ones((3, 3, 3), dtype=bool)
    )
    return np.any(path_mask & dilated_backbone)


def classify_branches_for_mito(skel, spacing=(1.0, 1.0, 1.0), min_side_len_um=2.0):
    """
    Returns
    -------
    tbl : DataFrame
        Per-branch skan table with:
        - branch_type
        - branch_distance
        - path_id
        - is_backbone_branch
        - is_side_branch
    backbone_mask : bool ndarray
        Longest geodesic backbone
    kept_mask : bool ndarray
        backbone + accepted side branches
    """
    if skel.sum() == 0:
        return pd.DataFrame(), np.zeros_like(skel, dtype=bool), np.zeros_like(skel, dtype=bool)

    sk = Skeleton(skel, spacing=spacing)
    tbl = summarize(sk, separator="_").copy()
    tbl = normalize_skan_columns(tbl)

    if tbl.empty:
        return pd.DataFrame(), np.zeros_like(skel, dtype=bool), np.zeros_like(skel, dtype=bool)

    if "branch_type" not in tbl.columns or "branch_distance" not in tbl.columns:
        raise KeyError(
            "Missing required skan columns. Available columns: "
            f"{list(tbl.columns)}"
        )

    tbl = tbl.reset_index(drop=True)
    tbl["path_id"] = np.arange(len(tbl), dtype=int)
    tbl["is_backbone_branch"] = False
    tbl["is_side_branch"] = False

    backbone_mask = longest_geodesic_path_mask(skel, spacing=spacing)
    kept_mask = np.zeros_like(skel, dtype=bool)

    # mark backbone branches
    for i, row in tbl.iterrows():
        pid = int(row["path_id"])
        coords = sk.path_coordinates(pid)
        path_mask = coords_to_mask(skel.shape, coords)

        if np.any(path_mask & backbone_mask):
            tbl.loc[i, "is_backbone_branch"] = True
            kept_mask |= path_mask

    # mark accepted side branches
    for i, row in tbl.iterrows():
        if bool(tbl.loc[i, "is_backbone_branch"]):
            continue

        if int(row["branch_type"]) != 1:
            continue

        if float(row["branch_distance"]) < float(min_side_len_um):
            continue

        pid = int(row["path_id"])
        coords = sk.path_coordinates(pid)
        path_mask = coords_to_mask(skel.shape, coords)

        if branch_attaches_to_backbone(path_mask, backbone_mask):
            tbl.loc[i, "is_side_branch"] = True
            kept_mask |= path_mask

    kept_mask |= backbone_mask

    return tbl, backbone_mask, kept_mask


def skeletonize_mask(binary_mask, voxel_size_um):
    if skeletonize_3d is None:
        raise ImportError(
            "skeletonize_3d is not available in your scikit-image version. "
            "Please install a compatible scikit-image version."
        )

    obj_clean = clean_object(binary_mask > 0)

    raw_skel = skeletonize_3d(obj_clean) > 0
    pruned_skel = raw_skel.copy()

    if KEEP_LARGEST_SKELETON_COMPONENT and pruned_skel.sum() > 0:
        pruned_skel = keep_largest_component(pruned_skel)

    if PRUNE_SHORT_TERMINAL_BRANCHES and pruned_skel.sum() > 0:
        pruned_skel = prune_short_terminal_branches(
            pruned_skel,
            spacing=voxel_size_um,
            min_length_um=MIN_TERMINAL_BRANCH_LENGTH_UM
        )

    if KEEP_LARGEST_SKELETON_COMPONENT and pruned_skel.sum() > 0:
        pruned_skel = keep_largest_component(pruned_skel)

    return raw_skel, pruned_skel


def analyze_branch_table_from_skeleton(skel, voxel_size_um, obj_id):
    if skel.sum() == 0:
        return {
            "branch_number": 0,
            "total_skeleton_length_um": 0.0,
            "mean_branch_length_um": np.nan,
            "median_branch_length_um": np.nan,
            "max_branch_length_um": np.nan,
            "min_branch_length_um": np.nan,
            "branch_df": pd.DataFrame()
        }

    try:
        sk = Skeleton(skel, spacing=voxel_size_um)
        branch_table = summarize(sk, separator="_")
        branch_table = normalize_skan_columns(branch_table)
    except Exception as e:
        return {
            "branch_number": np.nan,
            "total_skeleton_length_um": np.nan,
            "mean_branch_length_um": np.nan,
            "median_branch_length_um": np.nan,
            "max_branch_length_um": np.nan,
            "min_branch_length_um": np.nan,
            "branch_df": pd.DataFrame(),
            "skan_error": f"Object {obj_id}: {e}"
        }

    length_col = get_skan_length_column(branch_table)

    if len(branch_table) == 0 or length_col is None:
        return {
            "branch_number": 0,
            "total_skeleton_length_um": 0.0,
            "mean_branch_length_um": np.nan,
            "median_branch_length_um": np.nan,
            "max_branch_length_um": np.nan,
            "min_branch_length_um": np.nan,
            "branch_df": pd.DataFrame()
        }

    try:
        branch_lengths = branch_table[length_col].astype(float).to_numpy()
    except Exception:
        branch_lengths = np.array([], dtype=float)

    branch_df = branch_table.copy()
    branch_df.insert(0, "ImageObjectID", obj_id)

    if len(branch_lengths) == 0:
        return {
            "branch_number": 0,
            "total_skeleton_length_um": 0.0,
            "mean_branch_length_um": np.nan,
            "median_branch_length_um": np.nan,
            "max_branch_length_um": np.nan,
            "min_branch_length_um": np.nan,
            "branch_df": branch_df
        }

    return {
        "branch_number": int(len(branch_lengths)),
        "total_skeleton_length_um": float(np.nansum(branch_lengths)),
        "mean_branch_length_um": float(np.nanmean(branch_lengths)),
        "median_branch_length_um": float(np.nanmedian(branch_lengths)),
        "max_branch_length_um": float(np.nanmax(branch_lengths)),
        "min_branch_length_um": float(np.nanmin(branch_lengths)),
        "branch_df": branch_df
    }


# ============================================================
# PER-MITO MEASUREMENT
# ============================================================
def measure_one_mito(obj_mask, obj_id, voxel_size_um):
    """
    Final rule requested:

    - backbone_length_um = OLD backbone measurement
      (longest geodesic path only)

    - n_side_branches and side_branch_length_um
      from corrected side-branch classification

    - if n_side_branches > 0:
          TotalSkeletonLength_um = backbone_length_um + side_branch_length_um
      else:
          TotalSkeletonLength_um = backbone_length_um
    """
    voxel_size_um = tuple(float(v) for v in voxel_size_um)
    voxel_volume_um3 = voxel_size_um[0] * voxel_size_um[1] * voxel_size_um[2]

    if obj_mask.sum() == 0:
        return None, None, None, None

    voxel_count = int(obj_mask.sum())
    volume_um3 = voxel_count * voxel_volume_um3

    rp = safe_regionprops_single(obj_mask, spacing=voxel_size_um)
    major_axis_length_um, minor_axis_length_um = safe_get_axis_lengths(rp)

    raw_skel, pruned_skel = skeletonize_mask(obj_mask, voxel_size_um)

    if int(pruned_skel.sum()) < 2:
        return None, None, None, None

    if not has_at_least_one_skeleton_edge(pruned_skel):
        return None, None, None, None

    # ------------------------------------------------------------
    # 1) OLD backbone measurement
    # ------------------------------------------------------------
    backbone_mask = longest_geodesic_path_mask(pruned_skel, spacing=voxel_size_um)

    old_backbone_metrics = analyze_branch_table_from_skeleton(
        backbone_mask,
        voxel_size_um,
        obj_id
    )
    backbone_length_um = float(old_backbone_metrics["total_skeleton_length_um"])

    # ------------------------------------------------------------
    # 2) NEW corrected side-branch classification
    # ------------------------------------------------------------
    try:
        branch_table, backbone_mask_for_classify, kept_mask_classify = classify_branches_for_mito(
            pruned_skel,
            spacing=voxel_size_um,
            min_side_len_um=MIN_SIDE_BRANCH_LEN_UM
        )
    except Exception as e:
        raise RuntimeError(f"classify_branches_for_mito failed for object {obj_id}: {e}")

    if branch_table.empty:
        return None, None, None, None

    length_col = "branch_distance"
    side_tbl = branch_table.loc[branch_table["is_side_branch"]].copy()

    n_side_branches = int(len(side_tbl))
    side_branch_length_um = float(side_tbl[length_col].sum()) if len(side_tbl) > 0 else 0.0

    # ------------------------------------------------------------
    # 3) FINAL total skeleton length rule
    # ------------------------------------------------------------
    if n_side_branches > 0:
        total_skeleton_length_um = float(backbone_length_um + side_branch_length_um)
    else:
        total_skeleton_length_um = float(backbone_length_um)

    # ------------------------------------------------------------
    # 4) final kept mask for output:
    #    exact old backbone + accepted side branches
    # ------------------------------------------------------------
    final_kept_mask = backbone_mask.copy()

    if n_side_branches > 0:
        sk_full = Skeleton(pruned_skel, spacing=voxel_size_um)
        for _, row in side_tbl.iterrows():
            pid = int(row["path_id"])
            coords = sk_full.path_coordinates(pid)
            path_mask = coords_to_mask(pruned_skel.shape, coords)
            final_kept_mask |= path_mask

    endpoints, junctions = count_endpoints_and_junctions(final_kept_mask)

    cross_section_area_um2 = np.nan
    avg_diameter_from_volume_um = np.nan

    # keep width-related calculations based on backbone length
    if np.isfinite(backbone_length_um) and backbone_length_um > 0:
        try:
            cross_section_area_um2 = volume_um3 / backbone_length_um
        except Exception:
            cross_section_area_um2 = np.nan

        try:
            inside = volume_um3 / (np.pi * backbone_length_um)
            if np.isfinite(inside) and inside > 0:
                avg_diameter_from_volume_um = 2.0 * np.sqrt(inside)
        except Exception:
            avg_diameter_from_volume_um = np.nan

    if (
        np.isfinite(avg_diameter_from_volume_um)
        and avg_diameter_from_volume_um > 0
        and np.isfinite(major_axis_length_um)
    ):
        try:
            elongation_ratio = major_axis_length_um / avg_diameter_from_volume_um
        except Exception:
            elongation_ratio = np.nan
    else:
        elongation_ratio = np.nan

    summary_row = {
        "ImageObjectID": obj_id,
        "VoxelCount": voxel_count,
        "Volume_um3": volume_um3,
        "MajorAxisLength_um": major_axis_length_um,
        "MinorAxisLength_um": minor_axis_length_um,
        "CrossSectionArea_um2": cross_section_area_um2,
        "AvgDiameterFromVolume_um": avg_diameter_from_volume_um,
        "ElongationRatio": elongation_ratio,

        # requested skeleton outputs
        "TotalSkeletonLength_um": total_skeleton_length_um,
        "n_side_branches": n_side_branches,
        "side_branch_length_um": side_branch_length_um,
        "backbone_length_um": backbone_length_um,

        # extra outputs
        "AcceptedSkeletonVoxelCount": int(final_kept_mask.sum()),
        "BackboneVoxelCount": int(backbone_mask.sum()),
        "EndpointCount": endpoints,
        "JunctionVoxelCount": junctions,
    }

    branch_df = branch_table.copy()
    branch_df.insert(0, "ImageObjectID", obj_id)

    return summary_row, branch_df, final_kept_mask.astype(np.uint8), backbone_mask.astype(np.uint8)


# ============================================================
# CROPPED OBJECT EXTRACTION
# ============================================================
def extract_cropped_objects(labeled):
    ids = np.unique(labeled)
    ids = ids[ids > 0]

    records = []
    for obj_id in ids:
        region_mask = (labeled == obj_id)
        if not np.any(region_mask):
            continue

        found = ndi.find_objects(region_mask.astype(np.uint8))
        if not found or found[0] is None:
            continue

        slc = found[0]
        slice_tuple = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, dim))
            for s, dim in zip(slc, labeled.shape)
        )

        sub = labeled[slice_tuple]
        obj_mask = (sub == obj_id)

        records.append({
            "obj_id": int(obj_id),
            "slice_tuple": slice_tuple,
            "obj_mask": obj_mask
        })
    return records


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def measure_objects_chunk(task):
    object_chunk, voxel_size_um, image_name, image_path = task

    chunk_summary_rows = []
    chunk_branch_tables = []
    chunk_error_rows = []
    chunk_kept_updates = []
    chunk_backbone_updates = []

    for rec in object_chunk:
        obj_id = rec["obj_id"]
        slc = rec["slice_tuple"]
        obj_mask = rec["obj_mask"]

        try:
            out = measure_one_mito(
                obj_mask=obj_mask,
                obj_id=obj_id,
                voxel_size_um=voxel_size_um
            )

            if out[0] is None:
                continue

            summary_row, branch_df_one, kept_mask, backbone_mask = out

            summary_row["ImageName"] = image_name
            summary_row["ImagePath"] = image_path
            chunk_summary_rows.append(summary_row)

            if branch_df_one is not None and len(branch_df_one) > 0:
                branch_df_one = branch_df_one.copy()
                branch_df_one.insert(0, "ImageName", image_name)
                branch_df_one.insert(1, "ImagePath", image_path)
                chunk_branch_tables.append(branch_df_one)

            if kept_mask is not None and np.any(kept_mask):
                chunk_kept_updates.append((slc, kept_mask, obj_id))

            if backbone_mask is not None and np.any(backbone_mask):
                chunk_backbone_updates.append((slc, backbone_mask, obj_id))

        except Exception as e:
            chunk_error_rows.append({
                "ImageName": image_name,
                "ImagePath": image_path,
                "ImageObjectID": obj_id,
                "Error": str(e),
                "Traceback": traceback.format_exc()
            })

    return (
        chunk_summary_rows,
        chunk_branch_tables,
        chunk_error_rows,
        chunk_kept_updates,
        chunk_backbone_updates
    )


# ============================================================
# IMAGE-LEVEL SUMMARY
# ============================================================
def summarize_one_image(summary_df_one, branch_df_one, image_name, image_path):
    if summary_df_one is None or len(summary_df_one) == 0:
        return pd.DataFrame([{
            "ImageName": image_name,
            "ImagePath": image_path,
            "MitoCount": 0,
            "TotalVolume_um3": 0.0,
            "MeanVolume_um3": np.nan,
            "MeanMajorAxisLength_um": np.nan,
            "MeanMinorAxisLength_um": np.nan,
            "MeanCrossSectionArea_um2": np.nan,
            "MeanAvgDiameterFromVolume_um": np.nan,
            "MeanElongationRatio": np.nan,
            "TotalSkeletonLength_um": 0.0,
            "Total_n_side_branches": 0,
            "Total_side_branch_length_um": 0.0,
            "Total_backbone_length_um": 0.0,
            "TotalEndpointCount": 0,
            "TotalJunctionVoxelCount": 0,
        }])

    row = {
        "ImageName": image_name,
        "ImagePath": image_path,
        "MitoCount": int(len(summary_df_one)),
        "TotalVolume_um3": float(summary_df_one["Volume_um3"].sum()),
        "MeanVolume_um3": float(summary_df_one["Volume_um3"].mean()),
        "MeanMajorAxisLength_um": float(summary_df_one["MajorAxisLength_um"].mean()),
        "MeanMinorAxisLength_um": float(summary_df_one["MinorAxisLength_um"].mean()),
        "MeanCrossSectionArea_um2": float(summary_df_one["CrossSectionArea_um2"].mean()),
        "MeanAvgDiameterFromVolume_um": float(summary_df_one["AvgDiameterFromVolume_um"].mean()),
        "MeanElongationRatio": float(summary_df_one["ElongationRatio"].mean()),

        "TotalSkeletonLength_um": float(np.nansum(summary_df_one["TotalSkeletonLength_um"])),
        "Total_n_side_branches": int(np.nansum(summary_df_one["n_side_branches"])),
        "Total_side_branch_length_um": float(np.nansum(summary_df_one["side_branch_length_um"])),
        "Total_backbone_length_um": float(np.nansum(summary_df_one["backbone_length_um"])),

        "TotalEndpointCount": int(np.nansum(summary_df_one["EndpointCount"])),
        "TotalJunctionVoxelCount": int(np.nansum(summary_df_one["JunctionVoxelCount"])),
    }
    return pd.DataFrame([row])


def make_overall_summary(all_summary_df):
    if all_summary_df is None or len(all_summary_df) == 0:
        return pd.DataFrame([{
            "Scope": "ALL_IMAGES",
            "ImageCount": 0,
            "MitoCount": 0,
            "Volume_um3": 0.0,
            "MajorAxisLength_um_mean": np.nan,
            "MinorAxisLength_um_mean": np.nan,
            "CrossSectionArea_um2_mean": np.nan,
            "AvgDiameterFromVolume_um_mean": np.nan,
            "ElongationRatio_mean": np.nan,
            "TotalSkeletonLength_um": 0.0,
            "Total_n_side_branches": 0,
            "Total_side_branch_length_um": 0.0,
            "Total_backbone_length_um": 0.0,
            "EndpointCount": 0,
            "JunctionVoxelCount": 0,
        }])

    overall_row = {
        "Scope": "ALL_IMAGES",
        "ImageCount": int(all_summary_df["ImageName"].nunique()),
        "MitoCount": int(len(all_summary_df)),
        "Volume_um3": float(np.nansum(all_summary_df["Volume_um3"])),
        "MajorAxisLength_um_mean": float(np.nanmean(all_summary_df["MajorAxisLength_um"])),
        "MinorAxisLength_um_mean": float(np.nanmean(all_summary_df["MinorAxisLength_um"])),
        "CrossSectionArea_um2_mean": float(np.nanmean(all_summary_df["CrossSectionArea_um2"])),
        "AvgDiameterFromVolume_um_mean": float(np.nanmean(all_summary_df["AvgDiameterFromVolume_um"])),
        "ElongationRatio_mean": float(np.nanmean(all_summary_df["ElongationRatio"])),

        "TotalSkeletonLength_um": float(np.nansum(all_summary_df["TotalSkeletonLength_um"])),
        "Total_n_side_branches": int(np.nansum(all_summary_df["n_side_branches"])),
        "Total_side_branch_length_um": float(np.nansum(all_summary_df["side_branch_length_um"])),
        "Total_backbone_length_um": float(np.nansum(all_summary_df["backbone_length_um"])),

        "EndpointCount": int(np.nansum(all_summary_df["EndpointCount"])),
        "JunctionVoxelCount": int(np.nansum(all_summary_df["JunctionVoxelCount"])),
    }
    return pd.DataFrame([overall_row])


# ============================================================
# PROCESS ONE IMAGE
# ============================================================
def process_one_image(image_path, output_dir, voxel_size_um, n_object_workers):
    t0 = time.time()
    image_name = os.path.basename(image_path)
    image_stem = safe_stem(image_path)
    prefix = f"[{image_name}] "

    log(f"{prefix}Loading image...")
    labeled = load_labeled_mask(image_path)
    log(f"{prefix}Shape: {labeled.shape}, dtype: {labeled.dtype}")

    labeled = filter_small_labels(labeled, min_size=MIN_SIZE_VOXELS)

    ids = np.unique(labeled)
    ids = ids[ids > 0]
    n_ids = len(ids)

    log(f"{prefix}Objects after filtering: {n_ids}")
    log(f"{prefix}Nonzero voxels after filtering: {(labeled > 0).sum()}")
    log(f"{prefix}Using object workers: {n_object_workers}")

    summary_rows = []
    branch_tables = []
    error_rows = []

    kept_skeleton_label_volume = np.zeros_like(labeled, dtype=np.uint16)
    backbone_label_volume = np.zeros_like(labeled, dtype=np.uint16)

    if n_ids == 0:
        summary_df = pd.DataFrame()
        branch_df = pd.DataFrame()
        error_df = pd.DataFrame()

    else:
        object_records = extract_cropped_objects(labeled)
        log(f"{prefix}Prepared {len(object_records)} cropped objects")

        if n_object_workers <= 1:
            for obj_idx, rec in enumerate(object_records, start=1):
                obj_id = rec["obj_id"]
                slc = rec["slice_tuple"]
                obj_mask = rec["obj_mask"]

                if obj_idx == 1 or obj_idx % PRINT_EVERY_N_OBJECTS == 0 or obj_idx == len(object_records):
                    log(f"{prefix}Processing object {obj_idx}/{len(object_records)} (label={obj_id})")

                try:
                    out = measure_one_mito(
                        obj_mask=obj_mask,
                        obj_id=obj_id,
                        voxel_size_um=voxel_size_um
                    )

                    if out[0] is None:
                        continue

                    summary_row, branch_df_one, kept_mask, backbone_mask = out

                    summary_row["ImageName"] = image_name
                    summary_row["ImagePath"] = image_path
                    summary_rows.append(summary_row)

                    if branch_df_one is not None and len(branch_df_one) > 0:
                        branch_df_one = branch_df_one.copy()
                        branch_df_one.insert(0, "ImageName", image_name)
                        branch_df_one.insert(1, "ImagePath", image_path)
                        branch_tables.append(branch_df_one)

                    if kept_mask is not None and np.any(kept_mask):
                        subvol = kept_skeleton_label_volume[slc]
                        subvol[kept_mask > 0] = obj_id

                    if backbone_mask is not None and np.any(backbone_mask):
                        subvol = backbone_label_volume[slc]
                        subvol[backbone_mask > 0] = obj_id

                except Exception as e:
                    error_rows.append({
                        "ImageName": image_name,
                        "ImagePath": image_path,
                        "ImageObjectID": obj_id,
                        "Error": str(e),
                        "Traceback": traceback.format_exc()
                    })
                    log(f"{prefix}FAILED object {obj_idx}/{len(object_records)} (label={obj_id}): {e}")

        else:
            object_chunks = list(chunk_list(object_records, OBJECT_CHUNK_SIZE))
            log(
                f"{prefix}Running {len(object_chunks)} chunk tasks with "
                f"N_OBJECT_WORKERS={n_object_workers}, OBJECT_CHUNK_SIZE={OBJECT_CHUNK_SIZE}"
            )

            tasks = [
                (chunk, voxel_size_um, image_name, image_path)
                for chunk in object_chunks
            ]

            with ProcessPoolExecutor(max_workers=n_object_workers) as executor:
                futures = [executor.submit(measure_objects_chunk, task) for task in tasks]

                finished = 0
                total_tasks = len(futures)

                for future in as_completed(futures):
                    finished += 1
                    log(f"{prefix}Finished chunk {finished}/{total_tasks}")

                    (
                        chunk_summary_rows,
                        chunk_branch_tables,
                        chunk_error_rows,
                        chunk_kept_updates,
                        chunk_backbone_updates
                    ) = future.result()

                    summary_rows.extend(chunk_summary_rows)
                    branch_tables.extend(chunk_branch_tables)
                    error_rows.extend(chunk_error_rows)

                    for slc, kept_mask, obj_id in chunk_kept_updates:
                        subvol = kept_skeleton_label_volume[slc]
                        subvol[kept_mask > 0] = obj_id

                    for slc, backbone_mask, obj_id in chunk_backbone_updates:
                        subvol = backbone_label_volume[slc]
                        subvol[backbone_mask > 0] = obj_id

        summary_df = pd.DataFrame(summary_rows)
        if len(summary_df) > 0:
            summary_df = summary_df.sort_values("ImageObjectID").reset_index(drop=True)

        branch_df = pd.concat(branch_tables, ignore_index=True) if len(branch_tables) > 0 else pd.DataFrame()
        error_df = pd.DataFrame(error_rows)

    image_summary_df = summarize_one_image(summary_df, branch_df, image_name, image_path)

    # Save into one subfolder per image
    image_out_dir = os.path.join(output_dir, image_stem)
    ensure_dir(image_out_dir)

    summary_xlsx_path = os.path.join(image_out_dir, f"{image_stem}_summary.xlsx")
    branch_csv_path = os.path.join(image_out_dir, f"{image_stem}_branches.csv")
    kept_skel_tif_path = os.path.join(image_out_dir, f"{image_stem}_kept_skeleton_labels.tif")
    backbone_tif_path = os.path.join(image_out_dir, f"{image_stem}_backbone_labels.tif")
    error_csv_path = os.path.join(image_out_dir, f"{image_stem}_errors.csv")

    with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="per_mito_summary")
        image_summary_df.to_excel(writer, index=False, sheet_name="image_summary")

    branch_df.to_csv(branch_csv_path, index=False)
    tiff.imwrite(kept_skel_tif_path, kept_skeleton_label_volume)
    tiff.imwrite(backbone_tif_path, backbone_label_volume)

    if len(error_df) > 0:
        error_df.to_csv(error_csv_path, index=False)

    dt = time.time() - t0
    log(f"{prefix}Done in {dt:.2f} s")

    return {
        "image_name": image_name,
        "image_path": image_path,
        "summary_df": summary_df,
        "branch_df": branch_df,
        "image_summary_df": image_summary_df,
        "error_df": error_df,
    }


def process_one_image_wrapper(args):
    return process_one_image(*args)


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUTPUT_DIR)

    image_paths = list_input_tiffs(INPUT_FOLDER, FILE_PATTERN)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No TIFF files found in: {INPUT_FOLDER}")

    log(f"Found {len(image_paths)} TIFF files")

    n_image_workers = max(1, int(N_IMAGE_WORKERS))
    n_object_workers = choose_object_workers(
        n_image_workers=n_image_workers,
        requested_object_workers=N_OBJECT_WORKERS
    )

    log(f"CPU count detected: {os.cpu_count()}")
    log(f"N_IMAGE_WORKERS = {n_image_workers}")
    log(f"N_OBJECT_WORKERS requested = {N_OBJECT_WORKERS}")
    log(f"N_OBJECT_WORKERS used per image = {n_object_workers}")

    results = []

    if n_image_workers <= 1:
        for idx, image_path in enumerate(image_paths, start=1):
            log(f"\n=== Image {idx}/{len(image_paths)} ===")
            results.append(
                process_one_image(
                    image_path=image_path,
                    output_dir=OUTPUT_DIR,
                    voxel_size_um=VOXEL_SIZE_UM,
                    n_object_workers=n_object_workers
                )
            )
    else:
        tasks = [
            (image_path, OUTPUT_DIR, VOXEL_SIZE_UM, n_object_workers)
            for image_path in image_paths
        ]

        with ProcessPoolExecutor(max_workers=n_image_workers) as executor:
            futures = [executor.submit(process_one_image_wrapper, task) for task in tasks]

            for i, future in enumerate(as_completed(futures), start=1):
                log(f"\n=== Finished image task {i}/{len(futures)} ===")
                results.append(future.result())

    all_summary = []
    all_branches = []
    all_image_summaries = []
    all_errors = []

    for res in results:
        if res["summary_df"] is not None and len(res["summary_df"]) > 0:
            all_summary.append(res["summary_df"])
        if res["branch_df"] is not None and len(res["branch_df"]) > 0:
            all_branches.append(res["branch_df"])
        if res["image_summary_df"] is not None and len(res["image_summary_df"]) > 0:
            all_image_summaries.append(res["image_summary_df"])
        if res["error_df"] is not None and len(res["error_df"]) > 0:
            all_errors.append(res["error_df"])

    all_summary_df = pd.concat(all_summary, ignore_index=True) if len(all_summary) > 0 else pd.DataFrame()
    all_branch_df = pd.concat(all_branches, ignore_index=True) if len(all_branches) > 0 else pd.DataFrame()
    all_image_summary_df = pd.concat(all_image_summaries, ignore_index=True) if len(all_image_summaries) > 0 else pd.DataFrame()
    all_error_df = pd.concat(all_errors, ignore_index=True) if len(all_errors) > 0 else pd.DataFrame()

    overall_summary_df = make_overall_summary(all_summary_df)

    overall_xlsx_path = os.path.join(OUTPUT_DIR, "ALL_IMAGES_summary.xlsx")
    all_branch_csv_path = os.path.join(OUTPUT_DIR, "ALL_IMAGES_branches.csv")
    all_error_csv_path = os.path.join(OUTPUT_DIR, "ALL_IMAGES_errors.csv")

    with pd.ExcelWriter(overall_xlsx_path, engine="openpyxl") as writer:
        all_summary_df.to_excel(writer, index=False, sheet_name="all_per_mito_summary")
        all_image_summary_df.to_excel(writer, index=False, sheet_name="all_image_summary")
        overall_summary_df.to_excel(writer, index=False, sheet_name="overall_summary")

    all_branch_df.to_csv(all_branch_csv_path, index=False)

    if len(all_error_df) > 0:
        all_error_df.to_csv(all_error_csv_path, index=False)

    log("\nFinished all images.")
    log(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
