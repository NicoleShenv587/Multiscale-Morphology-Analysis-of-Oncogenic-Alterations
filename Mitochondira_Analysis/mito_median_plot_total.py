#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"   # keep text editable in Illustrator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# USER SETTINGS: edit only this section
# ============================================================

# ---- choose feature ----
FEATURE =  "Volume_um3"#"AvgDiameterFromVolume_um"   # e.g. "Volume_um3", "SkeletonLength_um", etc.
Y_LABEL =  "Volume_um3"#"AvgDiameter (µm)"
TITLE = "Median-based comparison"

# ---- choose which two groups to compare ----
LEFT_GROUP = "Normal"
RIGHT_GROUP = "CTNNB1"

# ---- output ----
OUTPUT_DIR = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/Mito_segmentation/mutation"
OUTPUT_NAME = f"{FEATURE}_{LEFT_GROUP}_vs_{RIGHT_GROUP}_violin.svg"
OUTPUT_SVG = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

# ---- figure settings ----
FIG_WIDTH = 1.5
FIG_HEIGHT = 1.8
POSITIVE_ONLY = True
TRANSPARENT_BG = True

# ---- font settings ----
FONT_FAMILY = "Arial"
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 10
FONT_SIZE_XTICK = 9
FONT_SIZE_YTICK = 8
FONT_SIZE_STATS = 14

# ---- x tick label angle ----
X_LABEL_ROTATION = 0     # set angle here
X_LABEL_HA = "right"      # "right", "center", or "left"

# ---- line thickness settings ----
LINE_WIDTH_AXIS = 1.2
LINE_WIDTH_MEDIAN = 1.2
LINE_WIDTH_QUARTILE = 1.2
LINE_WIDTH_BRACKET = 1.2
TICK_WIDTH = 1.2
TICK_LENGTH_Y = 5
TICK_LENGTH_X = 0

# ---- violin / summary settings ----
VIOLIN_WIDTH = 0.5
LINE_HALF_WIDTH = 0.23
BW_METHOD = 0.3

# ---- optional font settings for Illustrator / publication consistency ----
mpl.rcParams["font.family"] = FONT_FAMILY
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# ============================================================
# DEFINE ALL AVAILABLE GROUPS HERE
# Add/edit groups once, then reuse
# ============================================================
GROUPS = {
    "Normal": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/mutation/New/normal_corrected_Exm.xlsx",
        "color": "#7f7f7f",
        "dark": "#4d4d4d"
    },
    "CTNNB1": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/mutation/New/beta_corrected_Exm.xlsx",
        "color": "#e377c2",
        "dark": "#c44e9b"
    },
     "NRAS": {
        "path": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/Mito_segmentation/mutation/Nras_corrected_original_scale.xlsx",
        "color": "#ff7f0e",   # orange (matplotlib default)
        "dark": "#cc6600"
    },
    "N_Normal": {
        "path": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/Mito_segmentation/mutation/Nras_normal_corrected_original_scale.xlsx",
        "color": "#2ca02c",   # green (matplotlib default)
        "dark": "#1e7a1e"
    },
    "PCC": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/New/PCC_corrected_ExM.xlsx",
        "color": "#C6DBEF",
        "dark": "#2171B5"
    },
    "PPC": {
        "file": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/New/PPC_corrected_ExM.xlsx",
        "color": "#E6C9F2",
        "dark": "#6A2FB8"
    }
}


# ============================================================
# Helper functions
# ============================================================
def p_to_stars(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    return "ns"


def clean_feature_values(df, feature, positive_only=True):
    if feature not in df.columns:
        raise ValueError(
            f"Feature '{feature}' not found in file. Available columns:\n{list(df.columns)}"
        )

    vals = pd.to_numeric(df[feature], errors="coerce")
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if positive_only:
        vals = vals[vals > 0]

    return vals


def get_group_info(group_name):
    if group_name not in GROUPS:
        raise ValueError(
            f"Group '{group_name}' not found.\n"
            f"Available groups: {list(GROUPS.keys())}"
        )
    return GROUPS[group_name]


def plot_two_group_violin_svg(
    left_group,
    right_group,
    feature,
    y_label=None,
    title="",
    output_svg="violin_plot.svg",
    positive_only=True,
    transparent=True,
    fig_width=2.6,
    fig_height=2.6,
):
    left_info = get_group_info(left_group)
    right_info = get_group_info(right_group)

    left_file = left_info["file"]
    right_file = right_info["file"]
    left_color = left_info["color"]
    right_color = right_info["color"]

    # Read data
    df_left = pd.read_excel(left_file)
    df_right = pd.read_excel(right_file)

    # Clean values
    vals_left = clean_feature_values(df_left, feature, positive_only=positive_only)
    vals_right = clean_feature_values(df_right, feature, positive_only=positive_only)

    if len(vals_left) == 0 or len(vals_right) == 0:
        raise ValueError("One of the groups has no valid values after cleaning.")

    # Stats
    pval = mannwhitneyu(vals_left, vals_right, alternative="two-sided").pvalue
    stars = p_to_stars(pval)

    # Summary stats
    q1_left, med_left, q3_left = np.percentile(vals_left, [25, 50, 75])
    q1_right, med_right, q3_right = np.percentile(vals_right, [25, 50, 75])

    if y_label is None:
        y_label = feature

    # Figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1)

    # Violin
    parts = ax.violinplot(
        [vals_left, vals_right],
        positions=[1, 2],
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=BW_METHOD,
    )

    violin_colors = [left_color, right_color]
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(violin_colors[i])
        body.set_edgecolor("none")
        body.set_alpha(1.0)

    # Quartile + median lines
    summary_info = [
        (1, q1_left, med_left, q3_left),
        (2, q1_right, med_right, q3_right),
    ]

    for x, q1, med, q3 in summary_info:
        ax.hlines(
            [q1, q3],
            x - LINE_HALF_WIDTH,
            x + LINE_HALF_WIDTH,
            colors="black",
            linestyles=":",
            linewidth=LINE_WIDTH_QUARTILE,
            zorder=3,
        )
        ax.hlines(
            med,
            x - LINE_HALF_WIDTH,
            x + LINE_HALF_WIDTH,
            colors="black",
            linestyles="-",
            linewidth=LINE_WIDTH_MEDIAN,
            zorder=4,
        )

    # Significance bracket
    y_min_data = min(vals_left.min(), vals_right.min())
    y_max_data = max(vals_left.max(), vals_right.max())
    y_range = y_max_data - y_min_data if y_max_data > y_min_data else 1.0

    bracket_y = y_max_data + 0.06 * y_range
    bracket_h = 0.08 * y_range

    ax.plot(
        [1, 1, 2, 2],
        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
        color="black",
        linewidth=LINE_WIDTH_BRACKET,
        clip_on=False,
    )
    ax.text(
        1.5,
        bracket_y + bracket_h + 0.015 * y_range,
        stars,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE_STATS,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )

    # Axes styling
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [left_group, right_group],
        fontsize=FONT_SIZE_XTICK,
        rotation=X_LABEL_ROTATION,
        ha=X_LABEL_HA,
        rotation_mode="anchor",
        fontname=FONT_FAMILY,
    )
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL, fontname=FONT_FAMILY)

    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontname=FONT_FAMILY)

    ax.tick_params(
        axis="y",
        labelsize=FONT_SIZE_YTICK,
        width=TICK_WIDTH,
        length=TICK_LENGTH_Y
    )
    ax.tick_params(
        axis="x",
        labelsize=FONT_SIZE_XTICK,
        width=TICK_WIDTH,
        length=TICK_LENGTH_X,
        pad=2
    )

    for label in ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(LINE_WIDTH_AXIS)

    ax.set_ylim(y_min_data, y_max_data + 0.22 * y_range)

    fig.tight_layout()
    fig.savefig(output_svg, format="svg", bbox_inches="tight", transparent=transparent)
    plt.close(fig)

    print(f"Feature: {feature}")
    print(f"{left_group}: median={med_left:.4f}, q1={q1_left:.4f}, q3={q3_left:.4f}, n={len(vals_left)}")
    print(f"{right_group}: median={med_right:.4f}, q1={q1_right:.4f}, q3={q3_right:.4f}, n={len(vals_right)}")
    print(f"Mann-Whitney p={pval:.3e} ({stars})")
    print(f"Saved SVG: {output_svg}")


# ============================================================
# Run
# ============================================================
plot_two_group_violin_svg(
    left_group=LEFT_GROUP,
    right_group=RIGHT_GROUP,
    feature=FEATURE,
    y_label=Y_LABEL,
    title=TITLE,
    output_svg=OUTPUT_SVG,
    positive_only=POSITIVE_ONLY,
    transparent=TRANSPARENT_BG,
    fig_width=FIG_WIDTH,
    fig_height=FIG_HEIGHT,
)
