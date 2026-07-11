import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
import os

# =========================================================
# GLOBAL STYLE SETTINGS (EDIT HERE ONLY)
# =========================================================

# ---- font ----
FONT_FAMILY = "Arial"

FONT_SIZE_GLOBAL = 10
FONT_SIZE_LABEL = 10
FONT_SIZE_TITLE = 11
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9

# ---- x label rotation (optional) ----
X_LABEL_ROTATION = 0
X_LABEL_HA = "center"

# ---- figure size ----
FIG_WIDTH = 4
FIG_HEIGHT = 1.8
SHOW_TITLE = False
# ---- line thickness ----
LINE_WIDTH_CURVE = 1.8
LINE_WIDTH_MEDIAN = 1.8
LINE_WIDTH_AXIS = 1.2
LINE_WIDTH_TICK = 1.2

# ---- SEM band ----
SEM_ALPHA = 0.25

# ---- histogram ----
MIN_MITO_PER_CELL = 5
N_BINS = 46

# ---- output ----
OUTPUT_DIR = "C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/Mito_segmentation/mutation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COL = "Volume_um3"#"TotalSkeletonLength_um" #"Volume_um3"
CELL_ID_COL = "ImageName"

# =========================================================
# FORCE FONT
# =========================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY],
    "font.size": FONT_SIZE_GLOBAL,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

# =========================================================
# DATASETS
# =========================================================
DATASETS = {
    "Normal": {
        "path": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/mutation/normal_Exm.xlsx",
        "color": "#7f7f7f",
        "dark": "#4d4d4d"
    },
    "CTNNB1": {
        "path": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/mutation/beta_Exm.xlsx",
        "color": "#e377c2",
        "dark": "#c44e9b"
    },
    "PCC": {
        "path": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/PCC_ExM.xlsx",
        "color": "#7FB9DD",
        "dark": "#2171B5"
    },
    "PPC": {
        "path": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Segmentation/Mito_segmentation/PP_PC/PPC_ExM.xlsx",
        "color": "#C78BE3",
        "dark": "#6A2FB8"
    },
     "NRAS": {
        "path": "C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/mitochondira quantification/Mito_segmentation/mutation/Nras_corrected_preExpansion.xlsx",
        "color": "#ff7f0e",   # orange (matplotlib default)
        "dark": "#cc6600"
    },
}

# =========================================================
# SELECT GROUPS
# =========================================================
GROUPS_TO_PLOT = ["NRAS", "Normal"] #["Normal" , "CTNNB1"] #["PPC", "PCC"]

PLOT_TITLE = "Per-cell normalized distribution"
X_LABEL =  "Volume (µm³)"#TotalSkeletonLength (µm)" # "Volume (µm³)"
Y_LABEL = "Relative frequency per cell (%)"

# =========================================================
# FUNCTIONS
# =========================================================
def clean_df(df):
    df = df.copy()
    df[FEATURE_COL] = pd.to_numeric(df[FEATURE_COL], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[FEATURE_COL, CELL_ID_COL])
    df = df[df[FEATURE_COL] > 0]
    return df


def per_cell_hist(df, bins):
    hist_list = []

    for cid, sub in df.groupby(CELL_ID_COL):
        vals = sub[FEATURE_COL].values

        if len(vals) < MIN_MITO_PER_CELL:
            continue

        h, _ = np.histogram(vals, bins=bins)

        if h.sum() == 0:
            continue

        h = h / h.sum() * 100.0
        hist_list.append(h)

    hist_array = np.array(hist_list)

    mean_curve = hist_array.mean(axis=0)
    sem_curve = hist_array.std(axis=0, ddof=1) / np.sqrt(hist_array.shape[0])

    return mean_curve, sem_curve


def load_selected_groups(group_names):
    loaded = {}
    for group in group_names:
        info = DATASETS[group]
        df = pd.read_excel(info["path"])
        df = clean_df(df)
        loaded[group] = {
            "df": df,
            "color": info["color"],
            "dark": info["dark"]
        }
    return loaded


def make_shared_bins(loaded_groups):
    all_vals = np.concatenate([
        g["df"][FEATURE_COL].values for g in loaded_groups.values()
    ])

    bins = np.logspace(np.log10(all_vals.min()), np.log10(all_vals.max()), N_BINS)
    centers = np.sqrt(bins[:-1] * bins[1:])
    return bins, centers


def style_axes(ax):
    ax.tick_params(
        axis="x",
        labelsize=FONT_SIZE_TICK,
        width=LINE_WIDTH_TICK,
        length=0
    )
    ax.tick_params(
        axis="y",
        labelsize=FONT_SIZE_TICK,
        width=LINE_WIDTH_TICK,
        length=4
    )

    for label in ax.get_xticklabels():
        label.set_rotation(X_LABEL_ROTATION)
        label.set_ha(X_LABEL_HA)
        label.set_fontname(FONT_FAMILY)

    for label in ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(LINE_WIDTH_AXIS)
    ax.spines["bottom"].set_linewidth(LINE_WIDTH_AXIS)


def plot_groups(group_names, save_prefix=None):
    loaded = load_selected_groups(group_names)
    bins, centers = make_shared_bins(loaded)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for group_name in group_names:
        g = loaded[group_name]

        mean_curve, sem_curve = per_cell_hist(g["df"], bins)
        median_val = np.median(g["df"][FEATURE_COL].values)

        ax.plot(
            centers, mean_curve,
            color=g["color"],
            linewidth=LINE_WIDTH_CURVE,
            label=group_name
        )

        ax.fill_between(
            centers,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=g["color"],
            alpha=SEM_ALPHA
        )

        ax.axvline(
            median_val,
            color=g["dark"],
            linestyle="--",
            linewidth=LINE_WIDTH_MEDIAN
        )

    ax.set_xscale("log")

    ax.set_xlabel(X_LABEL, fontsize=FONT_SIZE_LABEL, fontname=FONT_FAMILY)
    ax.set_ylabel(Y_LABEL, fontsize=FONT_SIZE_LABEL, fontname=FONT_FAMILY)
    if SHOW_TITLE:
      ax.set_title(PLOT_TITLE, fontsize=FONT_SIZE_TITLE, fontname=FONT_FAMILY)

    style_axes(ax)

    legend = ax.legend(frameon=False, fontsize=FONT_SIZE_LEGEND)
    for text in legend.get_texts():
        text.set_fontname(FONT_FAMILY)

    fig.tight_layout()

    if save_prefix is None:
        save_prefix = "_vs_".join(group_names)

    fig.savefig(os.path.join(OUTPUT_DIR, save_prefix + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, save_prefix + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, save_prefix + ".svg"), bbox_inches="tight")

    plt.show()
    plt.close()


# =========================================================
# RUN
# =========================================================
plot_groups(GROUPS_TO_PLOT, save_prefix="volume_per_cell_" + "_vs_".join(GROUPS_TO_PLOT))
