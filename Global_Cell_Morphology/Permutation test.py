#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outputs (in OUT_DIR):
1) PCA_true_labels.png / .svg
2) PCA_random_labels.png / .svg
3) Permutation_accuracy_hist.png / .svg

Notes:
- Automatically excludes non-feature columns:
  CellID, ExpCondition, cellID, SampleID, NShortLength, CellType
- Uses z-score normalization (StandardScaler)
- Uses multinomial Logistic Regression + 5-fold Stratified CV
- Permutation test by shuffling labels N times
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Arial"
# =========================
# User settings
# =========================
NORMAL_XLSX = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/Normal_withSampleID.xlsx"
BETA_XLSX   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/Beta_withSampleID.xlsx"
NRAS_XLSX   = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/Nras_withSampleID.xlsx"

OUT_DIR = r"C:/Users/shenq/OneDrive - University of Texas Southwestern/Manuscripts/Nanoscale image/paper figure/3D cells/Figure2_Global/celltype_label_shuffle_test"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 0
N_PERM = 500              # increase to 1000+ for publication-grade p-values
N_SPLITS = 5              # CV folds

# =========================
# Load + combine
# =========================
def load_with_label(path: str, celltype: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["CellType"] = celltype
    return df

df_all = pd.concat(
    [
        load_with_label(NORMAL_XLSX, "Normal"),
        load_with_label(BETA_XLSX,   "CTNNB1"),
        load_with_label(NRAS_XLSX,   "Nras"),
    ],
    ignore_index=True,
)

# =========================
# Select feature columns
# =========================
exclude_cols = {
    "CellID", "ExpCondition", "cellID", "SampleID", "NShortLength", "CellType"
}

feature_cols = [
    c for c in df_all.columns
    if (c not in exclude_cols) and pd.api.types.is_numeric_dtype(df_all[c])
]

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found after exclusions. Check your sheet columns.")

X = df_all[feature_cols].to_numpy()
y_true = df_all["CellType"].to_numpy()

# Z-score
Xz = StandardScaler().fit_transform(X)

# =========================
# PCA (2D) for visualization
# =========================
pca = PCA(n_components=2, random_state=RANDOM_SEED)
scores = pca.fit_transform(Xz)
expl = pca.explained_variance_ratio_ * 100

# =========================
# Helper plotting
# =========================
def save_scatter(scores_2d, labels, out_png, out_svg, title):
    plt.figure(figsize=(4.2, 3.8))
    for lab in np.unique(labels):
        m = (labels == lab)
        plt.scatter(scores_2d[m, 0], scores_2d[m, 1], s=30, alpha=0.85, label=str(lab))
    plt.xlabel(f"PC1 ({expl[0]:.1f}%)", fontsize = 10)
    plt.ylabel(f"PC2 ({expl[1]:.1f}%)", fontsize = 10)
    plt.title(title)
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, transparent=True)
    plt.savefig(out_svg, transparent=True)
    plt.close()

# True labels PCA
save_scatter(
    scores,
    y_true,
    os.path.join(OUT_DIR, "PCA_true_labels.png"),
    os.path.join(OUT_DIR, "PCA_true_labels.svg"),
    "PCA colored by TRUE labels",
)

# Random labels PCA (one example shuffle)
rng = np.random.default_rng(RANDOM_SEED)
y_rand_example = rng.permutation(y_true)
save_scatter(
    scores,
    y_rand_example,
    os.path.join(OUT_DIR, "PCA_random_labels.png"),
    os.path.join(OUT_DIR, "PCA_random_labels.svg"),
    "PCA colored by RANDOM labels (example shuffle)",
)

# =========================
# Quantitative proof: permutation test (classification)
# =========================
clf = LogisticRegression(max_iter=5000, multi_class="auto")
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

acc_true = cross_val_score(clf, Xz, y_true, cv=cv, scoring="accuracy")
acc_true_mean = float(np.mean(acc_true))
acc_true_std  = float(np.std(acc_true))

acc_perm = []
for _ in range(N_PERM):
    y_shuf = rng.permutation(y_true)
    acc_perm.append(float(np.mean(cross_val_score(clf, Xz, y_shuf, cv=cv, scoring="accuracy"))))
acc_perm = np.array(acc_perm)

# Permutation p-value (one-sided: accuracy >= observed)
p_value = (np.sum(acc_perm >= acc_true_mean) + 1) / (N_PERM + 1)

# Plot permutation histogram
plt.figure(figsize=(4.2, 3.8))
plt.hist(acc_perm, bins=25, alpha=0.9)
plt.axvline(acc_true_mean, linewidth=2)
plt.xlabel("5-fold CV accuracy", fontsize = 10 )
plt.ylabel("Count (permutations)", fontsize = 10)
plt.title(f"Permutation test (N={N_PERM})\nTrue acc={acc_true_mean:.3f} ± {acc_true_std:.3f}, p={p_value:.4g}", fontsize = 12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Permutation_accuracy_hist.png"), dpi=600, transparent=True)
plt.savefig(os.path.join(OUT_DIR, "Permutation_accuracy_hist.svg"), transparent=True)
plt.close()

# =========================
# Console summary
# =========================
print("=================================================")
print("Cell-type separability test (true vs random labels)")
print("-------------------------------------------------")
print(f"N cells total: {len(df_all)}")
print("Counts by label:")
print(pd.Series(y_true).value_counts())
print("-------------------------------------------------")
print(f"Features used ({len(feature_cols)}): {feature_cols}")
print("-------------------------------------------------")
print(f"TRUE labels:  mean CV accuracy = {acc_true_mean:.3f}  (std={acc_true_std:.3f})")
print(f"SHUFFLED labels: mean = {acc_perm.mean():.3f}  (std={acc_perm.std():.3f}), max={acc_perm.max():.3f}")
print(f"Permutation p-value (>= true): {p_value:.4g}")
print("-------------------------------------------------")
print(f"Saved figures to: {OUT_DIR}")
print("=================================================")
