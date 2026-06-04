"""
predict_negaffect.py
SURGE Project — Ridge Regression: FC → Negative Affect

Usage:
    python predict_negaffect.py --fc_dir ~/surge/fc_vectors

What it does:
    1. Loads all {subject_id}_fc.npy files from fc_dir
    2. Matches subjects to behavioral.csv on Subject ID
    3. Computes NIH Toolbox Negative Affect composite:
           NegAffect = mean(AngAffect_Unadj, FearAffect_Unadj, Sadness_Unadj)
    4. Runs Ridge Regression with 5-fold cross-validation (RidgeCV, log-spaced alphas)
    5. Outputs:
          - Console: r, r², MAE, best alpha
          - predicted_vs_observed.png: scatter plot with regression line
          - predictions.csv: subject-level predicted vs. observed scores

Theory:
    Following Finn et al. (2015, Nat Neurosci) and connectome-based predictive
    modeling (CPM; Shen et al. 2017), we treat each subject's FC vector as
    a high-dimensional feature vector and use regularized regression to predict
    a continuous behavioral outcome. Ridge regression (L2 penalty) is appropriate
    here because:
      - n_features (71,631) >> n_subjects (~888), requiring regularization
      - FC edges are highly correlated, so ridge is preferred over lasso
      - We want to retain all edges (distributed prediction), not sparse selection
    5-fold CV is used to select alpha and evaluate generalization.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy import stats

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

BEHAVIORAL_CSV = Path(__file__).parent / "behavioral.csv"
TARGET_COLS = ["AngAffect_Unadj", "FearAffect_Unadj", "Sadness_Unadj"]
ALPHAS = np.logspace(1, 6, 20)   # Ridge regularization strengths to search
N_FOLDS = 5
RANDOM_STATE = 42

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def load_fc_matrix(fc_dir):
    """
    Load all FC vectors from fc_dir.
    Returns X (n_subjects, n_edges), subject_ids list.
    """
    fc_files = sorted([
        f for f in os.listdir(fc_dir)
        if f.endswith("_fc.npy")
    ])
    if not fc_files:
        raise FileNotFoundError(f"No _fc.npy files found in {fc_dir}")

    vectors = []
    subject_ids = []
    for fname in fc_files:
        sid = fname.replace("_fc.npy", "")
        vec = np.load(os.path.join(fc_dir, fname))
        vectors.append(vec)
        subject_ids.append(sid)

    X = np.vstack(vectors)  # (n_subjects, n_edges)
    print(f"Loaded {X.shape[0]} subjects × {X.shape[1]} edges")
    return X, subject_ids


def load_behavioral(subject_ids):
    """
    Load behavioral data and compute Negative Affect composite.
    Returns DataFrame indexed to match subject_ids order, dropping NaN rows.
    """
    df = pd.read_csv(BEHAVIORAL_CSV)
    df["Subject"] = df["Subject"].astype(str)
    df["NegAffect"] = df[TARGET_COLS].mean(axis=1)
    df = df[["Subject", "NegAffect"] + TARGET_COLS].dropna()
    df = df.set_index("Subject")

    # Align to loaded FC subjects
    df_aligned = df.loc[df.index.isin(subject_ids)]
    return df_aligned


def plot_results(y_true, y_pred, r, r2, mae, out_path):
    """Scatter plot: observed vs. predicted negative affect with regression line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#534AB7", edgecolors="none")

    # Regression line
    m, b, *_ = stats.linregress(y_true, y_pred)
    x_range = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_range, m * x_range + b, color="#E24B4A", linewidth=2, label=f"r = {r:.3f}")

    # Identity line for reference
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.4, label="identity")

    ax.set_xlabel("Observed Negative Affect (NIH Toolbox T-score)", fontsize=12)
    ax.set_ylabel("Predicted Negative Affect", fontsize=12)
    ax.set_title(
        f"Ridge Regression: FC → Negative Affect\n"
        f"r = {r:.3f}   r² = {r2:.3f}   MAE = {mae:.2f}",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.text(
        0.05, 0.92,
        f"n = {len(y_true)} subjects\n{N_FOLDS}-fold CV",
        transform=ax.transAxes,
        fontsize=10,
        color="#555"
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_subscale_breakdown(y_pred, df_aligned, out_dir):
    """
    Three scatter subplots showing how the composite prediction correlates
    with each constituent subscale (anger, fear, sadness).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    subscale_labels = {
        "AngAffect_Unadj": "Anger Affect",
        "FearAffect_Unadj": "Fear Affect",
        "Sadness_Unadj": "Sadness",
    }

    for ax, col in zip(axes, TARGET_COLS):
        y_sub = df_aligned[col].values
        r_sub, p_sub = stats.pearsonr(y_sub, y_pred)
        ax.scatter(y_sub, y_pred, alpha=0.5, s=18, color="#1D9E75", edgecolors="none")
        m, b, *_ = stats.linregress(y_sub, y_pred)
        x_range = np.linspace(y_sub.min(), y_sub.max(), 100)
        ax.plot(x_range, m * x_range + b, color="#E24B4A", linewidth=2)
        ax.set_xlabel(f"Observed {subscale_labels[col]}", fontsize=11)
        ax.set_ylabel("Predicted NegAffect Composite", fontsize=11)
        ax.set_title(f"r = {r_sub:.3f}  (p = {p_sub:.3e})", fontsize=11)

    fig.suptitle("Predicted Composite vs. Individual Subscales", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "subscale_breakdown.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Subscale plot saved: {out_path}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ridge Regression: FC vectors → Negative Affect"
    )
    parser.add_argument(
        "--fc_dir",
        type=str,
        default=str(Path.home() / "surge" / "fc_vectors"),
        help="Directory containing {subject_id}_fc.npy files"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.home() / "surge" / "results"),
        help="Directory for output figures and CSVs"
    )
    args = parser.parse_args()

    fc_dir = os.path.expanduser(args.fc_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    print("\n=== Loading FC vectors ===")
    X, subject_ids = load_fc_matrix(fc_dir)

    print("\n=== Loading behavioral data ===")
    df_aligned = load_behavioral(subject_ids)

    # Keep only subjects present in both FC and behavioral
    common_ids = [sid for sid in subject_ids if sid in df_aligned.index]
    id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    keep_idx = [id_to_idx[sid] for sid in common_ids]

    X_matched = X[keep_idx]
    y = df_aligned.loc[common_ids, "NegAffect"].values
    df_matched = df_aligned.loc[common_ids]

    print(f"\nMatched: {len(common_ids)} subjects")
    print(f"  NegAffect — mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.2f}, max={y.max():.2f}")

    if len(common_ids) < N_FOLDS:
        print(
            f"\n⚠  Only {len(common_ids)} matched subject(s). Need at least "
            f"{N_FOLDS} for {N_FOLDS}-fold CV.\n"
            "Run run_batch_fc.py on more subjects first (requires CHPC cluster "
            "access for bulk HCP downloads). Exiting."
        )
        return

    if len(common_ids) < 30:
        print(
            f"\n⚠  Only {len(common_ids)} matched subjects — results will be "
            "unreliable. Continuing for debugging purposes..."
        )

    # --------------------------------------------------
    # 2. Standardize features
    # --------------------------------------------------
    print("\n=== Standardizing features ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_matched)
    print(f"  X shape: {X_scaled.shape}")

    # --------------------------------------------------
    # 3. Ridge Regression — 5-fold CV
    # --------------------------------------------------
    print(f"\n=== Ridge Regression ({N_FOLDS}-fold CV) ===")
    print(f"  Alpha search range: {ALPHAS[0]:.0f} – {ALPHAS[-1]:.0f} ({len(ALPHAS)} values)")

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # cross_val_predict gives out-of-fold predictions for every subject
    ridge = RidgeCV(alphas=ALPHAS, cv=cv)
    ridge.fit(X_scaled, y)
    best_alpha = ridge.alpha_
    print(f"  Best alpha (from RidgeCV): {best_alpha:.1f}")

    # Re-run cross_val_predict with the chosen alpha for honest OOF predictions
    from sklearn.linear_model import Ridge
    ridge_fixed = Ridge(alpha=best_alpha)
    y_pred = cross_val_predict(ridge_fixed, X_scaled, y, cv=cv)

    # --------------------------------------------------
    # 4. Evaluate
    # --------------------------------------------------
    r, p_val = stats.pearsonr(y, y_pred)
    r2 = r ** 2
    mae = np.mean(np.abs(y - y_pred))

    print(f"\n=== Results ===")
    print(f"  Pearson r  : {r:.4f}  (p = {p_val:.3e})")
    print(f"  r²         : {r2:.4f}")
    print(f"  MAE        : {mae:.4f} T-score units")
    print(f"  Best alpha : {best_alpha:.1f}")

    # --------------------------------------------------
    # 5. Save outputs
    # --------------------------------------------------
    # Main scatter plot
    plot_path = os.path.join(out_dir, "predicted_vs_observed.png")
    plot_results(y, y_pred, r, r2, mae, plot_path)

    # Subscale breakdown (only if all 3 subscales are available)
    if all(col in df_matched.columns for col in TARGET_COLS):
        plot_subscale_breakdown(y_pred, df_matched, out_dir)

    # Predictions CSV
    preds_df = pd.DataFrame({
        "subject_id": common_ids,
        "observed_NegAffect": y,
        "predicted_NegAffect": y_pred,
        "residual": y - y_pred,
    })
    for col in TARGET_COLS:
        if col in df_matched.columns:
            preds_df[col] = df_matched[col].values
    csv_path = os.path.join(out_dir, "predictions.csv")
    preds_df.to_csv(csv_path, index=False)
    print(f"Predictions CSV saved: {csv_path}")

    # Summary stats
    summary = {
        "n_subjects": len(common_ids),
        "n_edges": X_matched.shape[1],
        "pearson_r": round(r, 4),
        "r_squared": round(r2, 4),
        "mae": round(mae, 4),
        "p_value": float(f"{p_val:.3e}"),
        "best_alpha": best_alpha,
        "n_folds": N_FOLDS,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(out_dir, "model_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Model summary saved: {summary_path}")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
