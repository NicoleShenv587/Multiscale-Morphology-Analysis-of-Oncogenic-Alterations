"""
Multiscale Structural Phenotyping and Classification of Oncogenic States

This script performs:
1. Multiscale feature extraction summary (Panel A)
2. Integrated PCA visualization (Panel B)
3. Pseudo-bulk classification using GroupKFold cross-validation (Panel C)
4. Feature importance analysis (Panel D)

"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif

from scipy.spatial import ConvexHull
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =========================================================
# Global plotting style
# =========================================================
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0


# =========================================================
# Fixed paths
# =========================================================
BASE = Path(r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure4")
OUTDIR = BASE / "figure4_update_2"
OUTDIR.mkdir(exist_ok=True)

FILES = {
    "Normal": {
        "morph": BASE / "Normal_withSampleID.xlsx",
        "curv": BASE / "curvature_results_normal.xlsx",
        "mito": BASE / "Normal_Exm.xlsx",
    },
    "CTNNB1": {
        "morph": BASE / "Beta_withSampleID.xlsx",
        "curv": BASE / "curvature_results_beta.xlsx",
        "mito": BASE / "beta_Exm.xlsx",
    },
    "NRAS": {
        "morph": BASE / "Nras_withSampleID.xlsx",
        "curv": BASE / "curvature_results_nras.xlsx",
        "mito": BASE / "Nras_Exm.xlsx",
    },
}


# =========================================================
# Exact feature set
# =========================================================
GLOBAL_FEATURES = [
    "Sphericity",
    "SurfaceArea",
    "Solidity",
    "AspectRatio",
    "Roughness",
    "Volume",
    "CompactNess",
    "LongLength",
]

CURV_FEATURE = ["p95", "median"]

MITO_FEATURES = [
    "AvgDiameterFromVolume_um",
    "Volume_um3",
    "TotalSkeletonLength_um",
]

LOG_FEATURES = {
    "Volume",
    "CompactNess",
    "AvgDiameterFromVolume_um",
    "Volume_um3",
    "TotalSkeletonLength_um",
    "SurfaceArea",
    "LongLength",
}

GROUP_ORDER = ["Normal", "CTNNB1", "NRAS"]
GROUP_COLORS = {
    "Normal": "#7f7f7f",
    "CTNNB1": "#e377c2",
    "NRAS": "#1f77b4",
}


# =========================================================
# Parameters
# =========================================================
N_PSEUDOBULKS_PER_GROUP = 40
N_MORPH_PER_BULK = 4
N_CURV_PER_BULK = 4
N_MITO_PER_BULK = 120

N_REPEATS = 15
N_SPLITS = 3

PCA_SEED = 42
PCA_PSEUDOBULKS_PER_GROUP = 50
PCA_OUTLIERS_PER_GROUP = 3
PCA_KEEP_QUANTILE = 0.90

COMBINED_MODEL = "rf"
COMBINED_SELECT_K = 14
W_GLOBAL = 1.0
W_CURV = 1.7
W_MITO = 1.7
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1

PANEL_D_TOP_N = 10


# =========================================================
# Small helpers
# =========================================================
def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, width=1.0)


def add_panel_label(ax, label, fontsize=16):
    ax.text(-0.12, 1.03, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="left")


def require_columns(df: pd.DataFrame, cols, name: str):
    if isinstance(cols, str):
        cols = [cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def clean_numeric(df: pd.DataFrame, cols):
    if isinstance(cols, str):
        cols = [cols]
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    for c in cols:
        if c in LOG_FEATURES:
            out[c] = np.log1p(out[c].clip(lower=0))
    return out


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.values.astype(float)
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return pd.DataFrame((arr - mu) / sd, index=df.index, columns=df.columns)


# =========================================================
# Source-group helpers
# =========================================================
def detect_source_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def attach_source_group(df: pd.DataFrame, modality: str, group_name: str) -> pd.DataFrame:
    if modality == "morph":
        candidates = ["SampleID", "sample", "ImageName", "ImagePath", "Sample", "sample_id"]
    elif modality == "curv":
        candidates = ["SampleID", "sample", "ImageName", "ImagePath", "cell", "cell_id"]
    elif modality == "mito":
        candidates = ["SampleID", "sample", "ImageName", "ImagePath", "image", "image_id"]
    else:
        candidates = []

    src_col = detect_source_column(df, candidates)
    out = df.copy()

    if src_col is not None:
        out["__source_group__"] = out[src_col].astype(str)
    else:
        out["__source_group__"] = [f"{modality}_row_{i}" for i in range(len(out))]

    out["__source_group__"] = [
        f"{group_name}|{modality}|{x}" for x in out["__source_group__"]
    ]
    return out


# =========================================================
# Load tables
# =========================================================
def load_group_tables(group: str, paths):
    morph_raw = pd.read_excel(paths["morph"])
    curv_raw = pd.read_excel(paths["curv"], sheet_name="summary_per_cell")
    mito_raw = pd.read_excel(paths["mito"])

    require_columns(morph_raw, GLOBAL_FEATURES, f"{group} morphology")
    require_columns(curv_raw, CURV_FEATURE, f"{group} curvature")
    require_columns(mito_raw, MITO_FEATURES, f"{group} mitochondria")

    morph_raw = attach_source_group(morph_raw, "morph", group)
    curv_raw = attach_source_group(curv_raw, "curv", group)
    mito_raw = attach_source_group(mito_raw, "mito", group)

    morph = clean_numeric(morph_raw, GLOBAL_FEATURES)
    curv = clean_numeric(curv_raw, CURV_FEATURE)
    mito = clean_numeric(mito_raw, MITO_FEATURES)

    morph["__source_group__"] = morph_raw.loc[morph.index, "__source_group__"].values
    curv["__source_group__"] = curv_raw.loc[curv.index, "__source_group__"].values
    mito["__source_group__"] = mito_raw.loc[mito.index, "__source_group__"].values

    return morph, curv, mito


# =========================================================
# Feature and model helpers
# =========================================================
def split_feature_blocks(feature_names):
    global_idx = [i for i, f in enumerate(feature_names) if f.startswith("global_")]
    curv_idx = [i for i, f in enumerate(feature_names) if f.startswith("curv_")]
    mito_idx = [i for i, f in enumerate(feature_names) if f.startswith("mito_")]
    return global_idx, curv_idx, mito_idx


def transform_blocks_train_test(X_train, X_test, feature_names,
                                w_global=1.0, w_curv=1.0, w_mito=1.0):
    global_idx, curv_idx, mito_idx = split_feature_blocks(feature_names)
    Xtr = X_train.copy()
    Xte = X_test.copy()

    if global_idx:
        sg = StandardScaler()
        Xtr[:, global_idx] = sg.fit_transform(Xtr[:, global_idx])
        Xte[:, global_idx] = sg.transform(Xte[:, global_idx])
        Xtr[:, global_idx] *= w_global
        Xte[:, global_idx] *= w_global

    if curv_idx:
        sc = StandardScaler()
        Xtr[:, curv_idx] = sc.fit_transform(Xtr[:, curv_idx])
        Xte[:, curv_idx] = sc.transform(Xte[:, curv_idx])
        Xtr[:, curv_idx] *= w_curv
        Xte[:, curv_idx] *= w_curv

    if mito_idx:
        sm = StandardScaler()
        Xtr[:, mito_idx] = sm.fit_transform(Xtr[:, mito_idx])
        Xte[:, mito_idx] = sm.transform(Xte[:, mito_idx])
        Xtr[:, mito_idx] *= w_mito
        Xte[:, mito_idx] *= w_mito

    return Xtr, Xte


def select_features_train_test(X_train, y_train, X_test, k):
    k_use = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_use)
    Xtr = selector.fit_transform(X_train, y_train)
    Xte = selector.transform(X_test)
    return Xtr, Xte, selector


def get_model(model_name: str, seed: int):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=seed
        )
    return LogisticRegression(max_iter=4000)


# =========================================================
# GroupKFold helpers
# =========================================================
def get_effective_n_splits(group_tables, requested_splits):
    counts = []
    for g in GROUP_ORDER:
        morph, curv, mito = group_tables[g]
        counts.extend([
            morph["__source_group__"].nunique(),
            curv["__source_group__"].nunique(),
            mito["__source_group__"].nunique(),
        ])
    return max(2, min(requested_splits, min(counts)))


def make_group_folds_for_table(df: pd.DataFrame, n_splits=3):
    uniq = pd.DataFrame({"__source_group__": pd.unique(df["__source_group__"])})
    X_dummy = np.zeros((len(uniq), 1))
    y_dummy = np.zeros(len(uniq))
    groups = uniq["__source_group__"].values

    n_use = min(n_splits, len(uniq))
    if n_use < 2:
        raise ValueError("Not enough unique source groups for GroupKFold.")

    gkf = GroupKFold(n_splits=n_use)
    folds = []
    for tr_idx, te_idx in gkf.split(X_dummy, y_dummy, groups):
        train_groups = set(uniq.iloc[tr_idx]["__source_group__"])
        test_groups = set(uniq.iloc[te_idx]["__source_group__"])
        folds.append((train_groups, test_groups))
    return folds


def build_modality_group_splits(group_tables, fold_idx: int, n_splits=3):
    split_maps = {}

    for g in GROUP_ORDER:
        morph, curv, mito = group_tables[g]

        morph_folds = make_group_folds_for_table(morph, n_splits=n_splits)
        curv_folds = make_group_folds_for_table(curv, n_splits=n_splits)
        mito_folds = make_group_folds_for_table(mito, n_splits=n_splits)

        split_maps[g] = {
            "morph": morph_folds[fold_idx % len(morph_folds)],
            "curv": curv_folds[fold_idx % len(curv_folds)],
            "mito": mito_folds[fold_idx % len(mito_folds)],
        }

    return split_maps


# =========================================================
# Pseudo-bulk builders
# =========================================================
def pseudobulk_signature_from_pool(df: pd.DataFrame, features, n_items: int, rng, prefix: str):
    idx = rng.integers(0, len(df), size=n_items)
    s = df.iloc[idx]

    vals = []
    names = []
    for c in features:
        vals.extend([s[c].mean(), s[c].std(ddof=1)])
        names.extend([f"{prefix}{c}_mean", f"{prefix}{c}_std"])
    return np.array(vals), names


def build_pseudobulks_from_split(group_tables, split_maps, block="all", seed=0, n_per_group=40):
    rng = np.random.default_rng(seed)

    X_train, y_train = [], []
    X_test, y_test = [], []
    feature_names = None

    for g in GROUP_ORDER:
        morph, curv, mito = group_tables[g]

        train_morph_groups, test_morph_groups = split_maps[g]["morph"]
        train_curv_groups, test_curv_groups = split_maps[g]["curv"]
        train_mito_groups, test_mito_groups = split_maps[g]["mito"]

        morph_tr = morph[morph["__source_group__"].isin(train_morph_groups)]
        morph_te = morph[morph["__source_group__"].isin(test_morph_groups)]

        curv_tr = curv[curv["__source_group__"].isin(train_curv_groups)]
        curv_te = curv[curv["__source_group__"].isin(test_curv_groups)]

        mito_tr = mito[mito["__source_group__"].isin(train_mito_groups)]
        mito_te = mito[mito["__source_group__"].isin(test_mito_groups)]

        morph_tr_f = morph_tr[GLOBAL_FEATURES]
        morph_te_f = morph_te[GLOBAL_FEATURES]
        curv_tr_f = curv_tr[CURV_FEATURE]
        curv_te_f = curv_te[CURV_FEATURE]
        mito_tr_f = mito_tr[MITO_FEATURES]
        mito_te_f = mito_te[MITO_FEATURES]

        if (
            len(morph_tr_f) == 0 or len(morph_te_f) == 0 or
            len(curv_tr_f) == 0 or len(curv_te_f) == 0 or
            len(mito_tr_f) == 0 or len(mito_te_f) == 0
        ):
            continue

        for _ in range(n_per_group):
            parts = []
            names = []

            if block in ("global", "global+curv", "all"):
                vec, nm = pseudobulk_signature_from_pool(
                    morph_tr_f, GLOBAL_FEATURES, N_MORPH_PER_BULK, rng, "global_"
                )
                parts.append(vec)
                names += nm

            if block in ("curv", "global+curv", "all"):
                vec, nm = pseudobulk_signature_from_pool(
                    curv_tr_f, CURV_FEATURE, N_CURV_PER_BULK, rng, "curv_"
                )
                parts.append(vec)
                names += nm

            if block in ("mito", "all"):
                vec, nm = pseudobulk_signature_from_pool(
                    mito_tr_f, MITO_FEATURES, N_MITO_PER_BULK, rng, "mito_"
                )
                parts.append(vec)
                names += nm

            X_train.append(np.concatenate(parts))
            y_train.append(g)
            feature_names = names

            parts = []
            names = []

            if block in ("global", "global+curv", "all"):
                vec, nm = pseudobulk_signature_from_pool(
                    morph_te_f, GLOBAL_FEATURES, N_MORPH_PER_BULK, rng, "global_"
                )
                parts.append(vec)
                names += nm

            if block in ("curv", "global+curv", "all"):
                vec, nm = pseudobulk_signature_from_pool(
                    curv_te_f, CURV_FEATURE, N_CURV_PER_BULK, rng, "curv_"
                )
                parts.append(vec)
                names += nm

            if block in ("mito", "all"):
                vec, nm = pseudobulk_signature_from_pool(
                    mito_te_f, MITO_FEATURES, N_MITO_PER_BULK, rng, "mito_"
                )
                parts.append(vec)
                names += nm

            X_test.append(np.concatenate(parts))
            y_test.append(g)

    return (
        np.vstack(X_train), np.array(y_train),
        np.vstack(X_test), np.array(y_test),
        feature_names
    )


def build_visual_dataset(group_tables, seed=42, n_per_group=50):
    rng = np.random.default_rng(seed)
    X, y, feature_names = [], [], None

    for g in GROUP_ORDER:
        morph, curv, mito = group_tables[g]
        morph_f = morph[GLOBAL_FEATURES]
        curv_f = curv[CURV_FEATURE]
        mito_f = mito[MITO_FEATURES]

        for _ in range(n_per_group):
            parts = []
            names = []

            vec, nm = pseudobulk_signature_from_pool(
                morph_f, GLOBAL_FEATURES, N_MORPH_PER_BULK, rng, "global_"
            )
            parts.append(vec)
            names += nm

            vec, nm = pseudobulk_signature_from_pool(
                curv_f, CURV_FEATURE, N_CURV_PER_BULK, rng, "curv_"
            )
            parts.append(vec)
            names += nm

            vec, nm = pseudobulk_signature_from_pool(
                mito_f, MITO_FEATURES, N_MITO_PER_BULK, rng, "mito_"
            )
            parts.append(vec)
            names += nm

            X.append(np.concatenate(parts))
            y.append(g)
            feature_names = names

    return np.vstack(X), np.array(y), feature_names


# =========================================================
# Panel A helpers
# =========================================================
def panelA_feature_labels():
    global_labels = {
        "Sphericity": "Sphericity",
        "SurfaceArea": "Surface\nArea",
        "Solidity": "Solidity",
        "AspectRatio": "Aspect\nratio",
        "Roughness": "Roughness",
        "Volume": "Volume",
        "VolumeSphericity": "Volume\nsphericity",
        "CompactNess": "Compactness",
        "LongLength": "Long\nlength",
    }
    curv_labels = {
        "p95": "Curvature\np95",
        "mean": "Curvature\nmean",
        "median": "Curvature\nmedian",
    }
    mito_labels = {
        "AvgDiameterFromVolume_um": "AvgDiameter\nFromVolume (µm)",
        "Volume_um3": "Volume\n(µm³)",
        "TotalSkeletonLength_um": "TotalSkeleton\nLength (µm)",
    }

    labels = []
    for f in GLOBAL_FEATURES:
        labels.append(global_labels.get(f, f))
    for f in CURV_FEATURE:
        labels.append(curv_labels.get(f, f))
    for f in MITO_FEATURES:
        labels.append(mito_labels.get(f, f))
    return labels


def build_panelA_tables(group_tables):
    heatmap_rows = []
    for g in GROUP_ORDER:
        morph, curv, mito = group_tables[g]
        row = {"Group": g}
        for f in GLOBAL_FEATURES:
            row[f] = morph[f].mean()
        for f in CURV_FEATURE:
            row[f] = curv[f].mean()
        for f in MITO_FEATURES:
            row[f] = mito[f].mean()
        heatmap_rows.append(row)

    heatmap_df = pd.DataFrame(heatmap_rows).set_index("Group")
    heatmap_z = zscore_rows(heatmap_df)
    return heatmap_df, heatmap_z


def draw_panel_A(ax, heatmap_z, group_tables,
                 title_left="Multiscale feature signature ",
                 title_right="(z-score of group means)",
                 label_fontsize=10,
                 group_fontsize=13,
                 header_fontsize=10,
                 title_fontsize=14):
    n_global = len(GLOBAL_FEATURES)
    n_curv = len(CURV_FEATURE)
    n_mito = len(MITO_FEATURES)
    total_cols = n_global + n_curv + n_mito
    labels = panelA_feature_labels()

    ax.set_xlim(-1.6, total_cols + 0.2)
    ax.set_ylim(-0.65, 4.30)
    ax.axis("off")

    ax.text(-0.25, 4.16, title_left, fontsize=title_fontsize, fontweight="bold",
            ha="left", va="bottom")
    ax.text(3.95, 4.16, title_right, fontsize=title_fontsize,
            ha="left", va="bottom")

    y_header = 3.60
    h_header = 0.52

    ax.add_patch(Rectangle((0, y_header), n_global, h_header,
                           facecolor="#ead7c8", edgecolor="0.7", lw=0.8))
    ax.text(n_global / 2, y_header + h_header / 2,
            "Global cell morphology", ha="center", va="center",
            fontsize=header_fontsize, fontweight="bold")

    ax.add_patch(Rectangle((n_global, y_header), n_curv, h_header,
                           facecolor="#eadfeb", edgecolor="0.7", lw=0.8))
    ax.text(n_global + n_curv / 2, y_header + h_header / 2,
            "Membrane curvature", ha="center", va="center",
            fontsize=header_fontsize, fontweight="bold")

    ax.add_patch(Rectangle((n_global + n_curv, y_header), n_mito, h_header,
                           facecolor="#dfe9d8", edgecolor="0.7", lw=0.8))
    ax.text(n_global + n_curv + n_mito / 2, y_header + h_header / 2,
            "Mitochondria morphology", ha="center", va="center",
            fontsize=header_fontsize, fontweight="bold")

    for j, lab in enumerate(labels):
        ax.add_patch(Rectangle((j, 0), 1, 3, facecolor="none", edgecolor="0.82", lw=0.8))
        ax.text(j + 0.5, 2.94, lab, rotation=90,
                ha="center", va="top", fontsize=label_fontsize)

    ax.text(-1.05, 2.0, "Group", fontsize=11, fontweight="bold", ha="center")
    for i, g in enumerate(GROUP_ORDER):
        y = 2 - i
        ax.text(-0.18, y + 0.5, g, color=GROUP_COLORS[g],
                fontsize=group_fontsize, fontweight="bold",
                ha="right", va="center")
        ax.text(-0.22, y + 0.12, f"(n={len(group_tables[g][0])})",
                fontsize=9.5, ha="right", va="center")

    cmap = plt.cm.coolwarm
    vmin, vmax = -2.2, 2.2
    for i, g in enumerate(GROUP_ORDER):
        y = 2 - i
        for j, f in enumerate(heatmap_z.columns):
            val = heatmap_z.loc[g, f]
            norm_val = np.clip((val - vmin) / (vmax - vmin), 0, 1)
            ax.add_patch(Rectangle((j, y), 1, 1,
                                   facecolor=cmap(norm_val),
                                   edgecolor="white", lw=1.5))

    ax.add_patch(Rectangle((0, 0), total_cols, 3, fill=False, edgecolor="0.55", lw=1.0))
    ax.plot([n_global, n_global], [0, 3], color="white", lw=2.2)
    ax.plot([n_global + n_curv, n_global + n_curv], [0, 3], color="white", lw=2.2)


def add_panelA_colorbar(fig, left=0.18, bottom=0.11, width=0.32, height=0.022):
    vmin, vmax = -2.2, 2.2
    cb_ax = fig.add_axes([left, bottom, width, height])
    grad = np.linspace(vmin, vmax, 256).reshape(1, -1)
    cb_ax.imshow(grad, aspect="auto", cmap=plt.cm.coolwarm, extent=[vmin, vmax, 0, 1])
    cb_ax.set_yticks([])
    cb_ax.set_xticks([-2, -1, 0, 1, 2])
    cb_ax.tick_params(labelsize=10)
    cb_ax.set_xlabel("z-score", fontsize=11, labelpad=-2)
    for s in cb_ax.spines.values():
        s.set_visible(False)


# =========================================================
# Panel B helpers
# =========================================================
def build_panelB_pca(group_tables):
    X, y, feature_names = build_visual_dataset(
        group_tables, seed=PCA_SEED, n_per_group=PCA_PSEUDOBULKS_PER_GROUP
    )

    # Raw pseudo-bulk feature table before scaling
    raw_df = pd.DataFrame(X, columns=feature_names)
    raw_df.insert(0, "Group", y)
    raw_df.insert(0, "PseudoBulkID", [f"PB_{i+1:04d}" for i in range(len(raw_df))])

    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(Xs, columns=feature_names)
    scaled_df.insert(0, "Group", y)
    scaled_df.insert(0, "PseudoBulkID", [f"PB_{i+1:04d}" for i in range(len(scaled_df))])

    # PCA
    pca = PCA(n_components=2, random_state=PCA_SEED)
    Z = pca.fit_transform(Xs)

    pca_df = pd.DataFrame({
        "PseudoBulkID": [f"PB_{i+1:04d}" for i in range(len(Z))],
        "Group": y,
        "PC1": Z[:, 0],
        "PC2": Z[:, 1],
    })

    # PCA loadings
    loading_df = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=["PC1_loading", "PC2_loading"]
    ).reset_index().rename(columns={"index": "Feature"})

    variance_df = pd.DataFrame({
        "PC": ["PC1", "PC2"],
        "Explained_variance_ratio": pca.explained_variance_ratio_,
        "Explained_variance_percent": pca.explained_variance_ratio_ * 100
    })

    # Remove extreme outliers for cleaner plot, same as your original code
    keep_mask = np.ones(len(pca_df), dtype=bool)
    for g in GROUP_ORDER:
        idx = np.where(pca_df["Group"].values == g)[0]
        pts = Z[idx]
        center = pts.mean(axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        cutoff = np.quantile(dist, PCA_KEEP_QUANTILE)
        keep_local = dist <= cutoff
        keep_mask[idx] = keep_local

    pca_df_clean = pca_df.loc[keep_mask].reset_index(drop=True)

    # Save all PCA-related data
    excel_path = OUTDIR / "panel_B_PCA_all_plotting_data.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="pseudo_bulk_raw_features", index=False)
        scaled_df.to_excel(writer, sheet_name="pseudo_bulk_scaled_features", index=False)
        pca_df.to_excel(writer, sheet_name="PCA_points_all", index=False)
        pca_df_clean.to_excel(writer, sheet_name="PCA_points_plotted", index=False)
        loading_df.to_excel(writer, sheet_name="PCA_loadings", index=False)
        variance_df.to_excel(writer, sheet_name="PCA_variance", index=False)

    print("Saved PCA plotting data to:", excel_path)

    return pca, pca_df_clean


def add_group_scatter_with_polygon(ax, pts, color, label=None, alpha=0.14, lw=1.2):
    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=28, color=color, edgecolor="white", linewidth=0.4,
        label=label, alpha=0.9, zorder=3
    )

    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            poly = Polygon(
                hull_pts,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                lw=lw,
                zorder=1
            )
            ax.add_patch(poly)
        except Exception:
            pass


def draw_panel_B(ax, pca, pca_df, title, label_fontsize=12, tick_fontsize=10, legend_fontsize=9):
    style_axes(ax)
    for g in GROUP_ORDER:
        sub = pca_df[pca_df["Group"] == g]
        pts = sub[["PC1", "PC2"]].values
        add_group_scatter_with_polygon(ax, pts, GROUP_COLORS[g], label=g)

    ax.axhline(0, color="0.88", lw=0.8, zorder=0)
    ax.axvline(0, color="0.88", lw=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=label_fontsize)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=label_fontsize)
    ax.set_title(title, loc="left", fontsize=13, pad=8)
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(frameon=False, fontsize=legend_fontsize, loc="best")


# =========================================================
# Panel C helpers
# =========================================================
def build_panelC_repeat_values(group_tables, effective_splits):
    blocks = [
        ("global", "Cell only"),
        ("curv", "Curvature only"),
        ("mito", "Mito only"),
        ("global+curv", "Cell + Curvature"),
        ("all", "All combined"),
    ]

    records = []

    for block_key, block_label in blocks:
        for seed in range(N_REPEATS):
            fold_acc = []

            for fold_idx in range(effective_splits):
                split_maps = build_modality_group_splits(
                    group_tables, fold_idx, n_splits=effective_splits
                )

                Xtr, ytr, Xte, yte, feature_names = build_pseudobulks_from_split(
                    group_tables,
                    split_maps,
                    block=block_key,
                    seed=seed * 100 + fold_idx,
                    n_per_group=N_PSEUDOBULKS_PER_GROUP
                )

                # Use Random Forest for ALL feature sets for a fair comparison
                Xtr, Xte = transform_blocks_train_test(
                    Xtr, Xte, feature_names,
                    w_global=W_GLOBAL,
                    w_curv=W_CURV,
                    w_mito=W_MITO
                 )

# Feature selection only when the number of features is larger than k
                if Xtr.shape[1] > COMBINED_SELECT_K:
                   Xtr, Xte, _ = select_features_train_test(
                        Xtr, ytr, Xte, k=COMBINED_SELECT_K
                   )

                clf = RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    max_depth=RF_MAX_DEPTH,
                    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                    random_state=seed
                )

                clf.fit(Xtr, ytr)
                pred = clf.predict(Xte)
                fold_acc.append(f1_score(yte, pred, average="macro"))

            records.append({
                "Block": block_label,
                "Seed": seed,
                "Accuracy": np.mean(fold_acc) * 100
            })

    return pd.DataFrame(records)


def panelC_summary_and_stats(panelC_df):
    order = [
        "Cell only",
        "Curvature only",
        "Mito only",
        "Cell + Curvature",
        "All combined",
    ]

    df = panelC_df.copy()
    df["Block"] = pd.Categorical(df["Block"], categories=order, ordered=True)
    df = df.sort_values(["Block", "Seed"]).reset_index(drop=True)

    summary_df = (
        df.groupby("Block", observed=False)["Accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Accuracy_mean", "std": "Accuracy_std"})
    )
    summary_df["Block"] = pd.Categorical(summary_df["Block"], categories=order, ordered=True)
    summary_df = summary_df.sort_values("Block").reset_index(drop=True)

    groups = [df.loc[df["Block"] == b, "Accuracy"].values for b in order]
    anova_stat, anova_p = f_oneway(*groups)
    print("Panel C ANOVA p =", anova_p)

    tukey = pairwise_tukeyhsd(
        endog=df["Accuracy"],
        groups=df["Block"],
        alpha=0.05
    )
    print(tukey)

    return df, summary_df, order, anova_p, tukey


def sig_label(p):
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def draw_panel_C(ax, repeat_df, summary_df, order, tukey,
                 title="C. Classification accuracy",
                 label_fontsize=13, tick_fontsize=11, bracket_fontsize=11):
    style_axes(ax)

    x = np.arange(len(order))
    y = summary_df["Accuracy_mean"].values
    yerr = summary_df["Accuracy_std"].fillna(0).values

    bar_colors = ["#d9d9d9", "#d9d9d9", "#d9d9d9", "#9ecae1", "#1f4e79"]

    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=5,
        error_kw=dict(lw=1.2, capthick=1.2, ecolor="black"),
        width=0.68,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.9,
        zorder=2
    )

    for i, val in enumerate(y):
        ax.text(
            i, val + 0.45,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold"
        )

    rng = np.random.default_rng(1234)
    for i, block in enumerate(order):
        vals = repeat_df.loc[repeat_df["Block"] == block, "Accuracy"].values
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        dot_color = "#08306b" if block == "All combined" else "#4d4d4d"

        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=26,
            color=dot_color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=3
        )

    combined = "All combined"
    combined_idx = order.index(combined)

    sig_rows = []
    for res in tukey.summary().data[1:]:
        g1 = res[0]
        g2 = res[1]
        p_adj = float(res[3])
        reject = bool(res[-1])

        if combined not in (g1, g2):
            continue

        other = g2 if g1 == combined else g1
        other_idx = order.index(other)
        sig_rows.append((other_idx, p_adj, reject))

    y_top = max(np.max(y + yerr), repeat_df["Accuracy"].max())
    step = 1.2
    current = y_top + 1.2
    sig_rows = sorted(sig_rows, key=lambda t: t[0])

    for other_idx, p_adj, reject in sig_rows:
        label = sig_label(p_adj)
        if label == "ns":
            continue

        x1, x2 = other_idx, combined_idx
        y_bracket = current

        ax.plot(
            [x1, x1, x2, x2],
            [y_bracket - 0.18, y_bracket, y_bracket, y_bracket - 0.18],
            color="black", lw=1.1, zorder=5
        )
        ax.text(
            (x1 + x2) / 2,
            y_bracket + 0.12,
            label,
            ha="center", va="bottom",
            fontsize=bracket_fontsize,
            fontweight="bold",
            zorder=6
        )
        current += step

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Cell", "Curvature", "Mito", "Cell +\nCurvature", "All\ncombined"],
        fontsize=tick_fontsize
    )
    ax.set_ylabel("Macro F1 (%)", fontsize=label_fontsize)
    ax.set_xlabel("Feature set", fontsize=label_fontsize)
    ax.set_title(title, loc="left", fontsize=13, pad=10)

    ax.grid(axis="y", color="0.90", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ymin = max(0, np.floor((repeat_df["Accuracy"].min() - 2) / 2) * 2)
    ymax = min(100, np.ceil((current + 1.0) / 2) * 2)
    ax.set_ylim(ymin, ymax)


# =========================================================
# Panel D helpers
# =========================================================
def clean_feature_label(name: str) -> str:
    x = name
    x = x.replace("global_", "")
    x = x.replace("curv_", "")
    x = x.replace("mito_", "")
    x = x.replace("_mean", "")
    x = x.replace("_median", "")
    x = x.replace("_std", "")

    x = x.replace("AvgDiameterFromVolume_um", "AvgDiameter")
    x = x.replace("Volume_um3", "MitoVolume")
    x = x.replace("TotalSkeletonLength_um", "SkeletonLength")
    x = x.replace("CompactNess", "Compactness")
    x = x.replace("AspectRatio", "Aspect ratio")
    x = x.replace("SurfaceArea", "Surface area")
    x = x.replace("LongLength", "Long length")
    x = x.replace("p95", "Curvature p95")
    x = x.replace("median", "Curvature median")
    return x


def modality_from_feature(name: str) -> str:
    if name.startswith("global_"):
        return "Global"
    if name.startswith("curv_"):
        return "Curvature"
    if name.startswith("mito_"):
        return "Mitochondria"
    return "Other"


def build_panelD_importance(group_tables, effective_splits):
    importance_records = []

    for seed in range(N_REPEATS):
        for fold_idx in range(effective_splits):
            split_maps = build_modality_group_splits(
                group_tables, fold_idx, n_splits=effective_splits
            )

            Xtr, ytr, Xte, yte, feature_names = build_pseudobulks_from_split(
                group_tables,
                split_maps,
                block="all",
                seed=seed * 100 + fold_idx,
                n_per_group=N_PSEUDOBULKS_PER_GROUP
            )

            Xtr, Xte = transform_blocks_train_test(
                Xtr, Xte, feature_names,
                w_global=W_GLOBAL, w_curv=W_CURV, w_mito=W_MITO
            )

            Xtr_sel, Xte_sel, selector = select_features_train_test(
                Xtr, ytr, Xte, k=COMBINED_SELECT_K
            )

            selected_idx = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_idx]

            clf = get_model(COMBINED_MODEL, seed)
            clf.fit(Xtr_sel, ytr)

            if hasattr(clf, "feature_importances_"):
                imps = clf.feature_importances_
            else:
                imps = np.abs(clf.coef_).mean(axis=0)

            for feat, imp in zip(selected_features, imps):
                if feat.endswith("_std"):
                    continue
                importance_records.append({
                    "Feature": feat,
                    "Importance": float(imp),
                    "Modality": modality_from_feature(feat),
                    "Label": clean_feature_label(feat)
                })

    imp_df = pd.DataFrame(importance_records)

    imp_summary = (
        imp_df.groupby(["Feature", "Label", "Modality"], as_index=False)["Importance"]
        .mean()
        .sort_values("Importance", ascending=False)
    )

    top_df = imp_summary.head(PANEL_D_TOP_N).copy()
    top_df = top_df.sort_values("Importance", ascending=True)

    return imp_df, imp_summary, top_df


def draw_panel_D(ax, top_df, title="D. Top features driving the combined model",
                 label_fontsize=12, tick_fontsize=10):
    style_axes(ax)

    color_map = {
        "Global": "#c7b299",
        "Curvature": "#b39ddb",
        "Mitochondria": "#9dc3a6",
        "Other": "#d9d9d9",
    }
    bar_colors = [color_map.get(m, "#d9d9d9") for m in top_df["Modality"]]

    ax.barh(
        top_df["Label"],
        top_df["Importance"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_xlabel("Mean model importance", fontsize=label_fontsize)
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=13, pad=10)

    ax.grid(axis="x", color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)

    ax.text(
        1.02, 0.95, "Global",
        transform=ax.transAxes, color=color_map["Global"],
        fontsize=10, fontweight="bold", ha="left", va="top"
    )
    ax.text(
        1.02, 0.87, "Curvature",
        transform=ax.transAxes, color=color_map["Curvature"],
        fontsize=10, fontweight="bold", ha="left", va="top"
    )
    ax.text(
        1.02, 0.79, "Mitochondria",
        transform=ax.transAxes, color=color_map["Mitochondria"],
        fontsize=10, fontweight="bold", ha="left", va="top"
    )


# =========================================================
# Separate panel writers
# =========================================================
def plot_panel_A(heatmap_df, heatmap_z, group_tables):
    total_cols = len(GLOBAL_FEATURES) + len(CURV_FEATURE) + len(MITO_FEATURES)
    fig_w = max(11.5, 0.70 * total_cols + 3.8)
    fig_h = 5.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_axes([0.05, 0.22, 0.83, 0.66])

    draw_panel_A(
        ax, heatmap_z, group_tables,
        label_fontsize=10,
        group_fontsize=13,
        header_fontsize=10,
        title_fontsize=14
    )
    add_panelA_colorbar(fig, left=0.18, bottom=0.12, width=0.28, height=0.022)

    heatmap_df.to_excel(OUTDIR / "panel_A_heatmap_values.xlsx")
    fig.savefig(OUTDIR / "panel_A_heatmap_publication.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTDIR / "panel_A_heatmap_publication.svg", bbox_inches="tight")
    plt.close(fig)


def plot_panel_B(pca, pca_df):
    fig, ax = plt.subplots(figsize=(mm_to_inch(72), mm_to_inch(64)), dpi=300)
    draw_panel_B(ax, pca, pca_df, "B. Integrated feature space (PCA)")
    pca_df.to_excel(OUTDIR / "panel_B_pca_points.xlsx", index=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "panel_B_pca_publication.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTDIR / "panel_B_pca_publication.svg", bbox_inches="tight")
    plt.close(fig)


def plot_panel_C(repeat_df, summary_df, order, anova_p, tukey):
    fig, ax = plt.subplots(figsize=(mm_to_inch(100), mm_to_inch(72)), dpi=300)
    draw_panel_C(ax, repeat_df, summary_df, order, tukey)

    excel_path = OUTDIR / "panel_C_repeat_values.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        repeat_df.to_excel(writer, sheet_name="repeat_values", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    fig.tight_layout()
    fig.savefig(OUTDIR / "panel_C_classifier_publication.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTDIR / "panel_C_classifier_publication.svg", bbox_inches="tight")
    plt.close(fig)

    print("Panel C ANOVA p =", anova_p)
    print("Saved Panel C repeat values to:", excel_path)


def plot_panel_D(imp_df, imp_summary, top_df):
    fig, ax = plt.subplots(figsize=(mm_to_inch(86), mm_to_inch(80)), dpi=300)
    draw_panel_D(ax, top_df)

    excel_path = OUTDIR / "panel_D_feature_importance.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        imp_df.to_excel(writer, sheet_name="all_fold_importance", index=False)
        imp_summary.to_excel(writer, sheet_name="mean_importance", index=False)
        top_df.to_excel(writer, sheet_name="top_features", index=False)

    fig.tight_layout()
    fig.savefig(OUTDIR / "panel_D_features_publication.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTDIR / "panel_D_features_publication.svg", bbox_inches="tight")
    plt.close(fig)

    print("Saved Panel D feature importance to:", excel_path)


# =========================================================
# Combined figure A-D
# =========================================================
def plot_combined_figure(heatmap_z, group_tables, pca, pca_df, panelC_repeat_df, panelC_summary_df, panelC_order, panelC_tukey, top_df):
    fig = plt.figure(figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Figure 3. Multiscale integration of structural features improves discrimination of oncogenic states",
        fontsize=17, fontweight="bold", y=0.98
    )

    # A
    axA = fig.add_axes([0.05, 0.58, 0.58, 0.32])
    add_panel_label(axA, "A")
    draw_panel_A(
        axA, heatmap_z, group_tables,
        label_fontsize=9.0,
        group_fontsize=12.0,
        header_fontsize=9.0,
        title_fontsize=13.0
    )
    add_panelA_colorbar(fig, left=0.15, bottom=0.555, width=0.22, height=0.014)

    # B
    axB = fig.add_axes([0.68, 0.60, 0.26, 0.28])
    add_panel_label(axB, "B")
    draw_panel_B(axB, pca, pca_df, "Integrated feature space (PCA)",
                 label_fontsize=11, tick_fontsize=9, legend_fontsize=8)

    # C
    axC = fig.add_axes([0.07, 0.13, 0.40, 0.32])
    add_panel_label(axC, "C")
    draw_panel_C(axC, panelC_repeat_df, panelC_summary_df, panelC_order, panelC_tukey,
                 title="Classification accuracy",
                 label_fontsize=12, tick_fontsize=10, bracket_fontsize=10)

    # D
    axD = fig.add_axes([0.58, 0.13, 0.32, 0.32])
    add_panel_label(axD, "D")
    draw_panel_D(axD, top_df,
                 title="Top features driving the combined model",
                 label_fontsize=12, tick_fontsize=9)

    fig.savefig(OUTDIR / "figure3_combined_AD.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTDIR / "figure3_combined_AD.svg", bbox_inches="tight")
    plt.close(fig)

def plot_raw_vs_scaled_distributions(group_tables):
    import matplotlib.pyplot as plt
    import numpy as np

    FEATURES_TO_PLOT = [
        "Volume_um3",
        "AvgDiameterFromVolume_um",
        "TotalSkeletonLength_um",
        "Volume",
        "Sphericity"
    ]

    # Collect raw values
    data_raw = {f: [] for f in FEATURES_TO_PLOT}
    labels = []

    for g in GROUP_ORDER:
        morph, curv, mito = group_tables[g]

        for f in FEATURES_TO_PLOT:
            if f in morph.columns:
                vals = morph[f].values
            elif f in mito.columns:
                vals = mito[f].values
            else:
                continue

            data_raw[f].append(vals)
        labels.append(g)

    # Flatten for scaling
    all_features = []
    feature_names = []

    for f in FEATURES_TO_PLOT:
        for g_idx, g in enumerate(GROUP_ORDER):
            vals = data_raw[f][g_idx]
            all_features.append(vals)
            feature_names.append(f"{g}_{f}")

    all_concat = np.concatenate(all_features).reshape(-1, 1)

    # Z-score
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_concat).flatten()

    # Split back
    split_scaled = []
    idx = 0
    for vals in all_features:
        split_scaled.append(all_scaled[idx:idx+len(vals)])
        idx += len(vals)

    # =========================================================
    # Plot
    # =========================================================
    n_features = len(FEATURES_TO_PLOT)
    fig, axes = plt.subplots(n_features, 2, figsize=(8, 2.2*n_features))

    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i, f in enumerate(FEATURES_TO_PLOT):

        # RAW
        ax = axes[i, 0]
        for j, g in enumerate(GROUP_ORDER):
            ax.hist(
                data_raw[f][j],
                bins=40,
                alpha=0.5,
                label=g,
                density=True
            )
        ax.set_title(f"{f} (raw/log)")
        ax.set_ylabel("Density")
        if i == n_features - 1:
            ax.set_xlabel("Value")

        # SCALED
        ax = axes[i, 1]
        for j, g in enumerate(GROUP_ORDER):
            vals = split_scaled[i*len(GROUP_ORDER) + j]
            ax.hist(
                vals,
                bins=40,
                alpha=0.5,
                label=g,
                density=True
            )
        ax.set_title(f"{f} (z-score)")
        if i == n_features - 1:
            ax.set_xlabel("Z-score")

    axes[0, 0].legend(frameon=False)

    plt.tight_layout()

    out_png = OUTDIR / "raw_vs_scaled_distributions.png"
    out_svg = OUTDIR / "raw_vs_scaled_distributions.svg"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()

    print("Saved:", out_png)
    print("Saved:", out_svg)
# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    print("Loading data...")
    group_tables = {}
    for g, paths in FILES.items():
        group_tables[g] = load_group_tables(g, paths)
        morph, curv, mito = group_tables[g]
        print(
            f"{g}: morph={morph.shape}, curv={curv.shape}, mito={mito.shape}, "
            f"morph_groups={morph['__source_group__'].nunique()}, "
            f"curv_groups={curv['__source_group__'].nunique()}, "
            f"mito_groups={mito['__source_group__'].nunique()}"
        )

    effective_splits = get_effective_n_splits(group_tables, N_SPLITS)
    print(f"Using effective GroupKFold splits = {effective_splits}")

    # Panel A data
    heatmap_df, heatmap_z = build_panelA_tables(group_tables)

    # Panel B data
    pca, pca_df = build_panelB_pca(group_tables)

    # Panel C data
    panelC_repeat_df = build_panelC_repeat_values(group_tables, effective_splits)
    panelC_repeat_df, panelC_summary_df, panelC_order, panelC_anova_p, panelC_tukey = panelC_summary_and_stats(panelC_repeat_df)

    # Panel D data
    imp_df, imp_summary, top_df = build_panelD_importance(group_tables, effective_splits)

    # Save separate panels
    print("Saving separate panels...")
    plot_panel_A(heatmap_df, heatmap_z, group_tables)
    plot_panel_B(pca, pca_df)
    plot_panel_C(panelC_repeat_df, panelC_summary_df, panelC_order, panelC_anova_p, panelC_tukey)
    plot_panel_D(imp_df, imp_summary, top_df)

    # Save combined figure
    print("Saving combined figure...")
    plot_combined_figure(
        heatmap_z, group_tables,
        pca, pca_df,
        panelC_repeat_df, panelC_summary_df, panelC_order, panelC_tukey,
        top_df
    )
    plot_raw_vs_scaled_distributions(group_tables)
    print("Done.")
    print("Saved files:")
    print(OUTDIR / "panel_A_heatmap.svg")
    print(OUTDIR / "panel_B_pca.svg")
    print(OUTDIR / "panel_C_classifier.svg")
    print(OUTDIR / "panel_D_features.svg")
    #print(OUTDIR / "figure3_combined_AD.svg")
