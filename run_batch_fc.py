"""
run_batch_fc.py
SURGE Project — Batch FC computation for all HCP subjects

Usage:
    python run_batch_fc.py --data_root ~/Downloads --out_dir ~/surge/fc_vectors

What it does:
    For each subject in behavioral.csv with 3T_Full_MR_Compl == True,
    concatenates all 4 resting-state runs, parcellates with MMP (379 parcels),
    computes the upper-triangle Fisher-z FC vector (71,631 edges), and saves
    it as {subject_id}_fc.npy in out_dir. Skipped subjects (missing data) are
    logged to skip_log.txt.

    This produces the feature matrix X (n_subjects x 71631) used by
    predict_negaffect.py for Ridge Regression.
"""

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import hcp_utils as hcp
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

RUNS = [
    "rfMRI_REST1_LR",
    "rfMRI_REST1_RL",
    "rfMRI_REST2_LR",
    "rfMRI_REST2_RL",
]

# Try the more aggressively cleaned version first (from BALSA zip),
# fall back to the S3 version (downloaded via hcp_s3_download.py)
DTSERIES_SUFFIXES = [
    "_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii",  # BALSA
    "_Atlas_MSMAll_hp2000_clean.dtseries.nii",                 # S3
]
BEHAVIORAL_CSV = Path(__file__).parent / "behavioral.csv"
PARCELLATION = hcp.mmp  # 379 non-trivial parcels → 71,631 edges

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def get_dtseries_path(data_root, subject_id, run):
    """Return the first dtseries path that exists, trying BALSA then S3 filename."""
    for suffix in DTSERIES_SUFFIXES:
        path = os.path.join(
            data_root, str(subject_id), "MNINonLinear", "Results", run,
            f"{run}{suffix}"
        )
        if os.path.exists(path):
            return path
    return None  # neither found


def load_parcel_timeseries(dtseries_path):
    """Load a dtseries.nii and parcellate with MMP. Returns (n_timepoints, n_parcels)."""
    img = nib.load(dtseries_path)
    data = img.get_fdata(dtype=np.float32)   # (n_timepoints, 91282)
    ts = hcp.parcellate(data, PARCELLATION)  # (n_timepoints, n_parcels)
    return ts


def compute_fc_vector(concat_ts):
    """
    Compute Fisher-z FC vector from concatenated parcel timeseries.
    concat_ts: (n_timepoints, n_parcels)
    Returns: 1D array of length n_parcels*(n_parcels-1)/2
    """
    r_matrix = np.corrcoef(concat_ts.T)           # (n_parcels, n_parcels)
    r_clipped = np.clip(r_matrix, -0.9999, 0.9999)
    z_matrix = np.arctanh(r_clipped)             # Fisher-z transform
    idx = np.triu_indices(z_matrix.shape[0], k=1)
    return z_matrix[idx]                          # (71631,)


def process_subject(data_root, subject_id):
    """
    Concatenate runs, compute FC vector.
    Returns (fc_vector, n_timepoints, runs_used) or raises RuntimeError.
    """
    all_ts = []
    runs_used = []

    for run in RUNS:
        path = get_dtseries_path(data_root, subject_id, run)
        if path is None:
            print(f"  [skip run] {run} not found for subject {subject_id}")
            continue
        ts = load_parcel_timeseries(path)
        all_ts.append(ts)
        runs_used.append(run)

    if len(all_ts) == 0:
        raise RuntimeError(f"No runs found for subject {subject_id}")

    concat_ts = np.concatenate(all_ts, axis=0)  # (total_timepoints, n_parcels)
    fc_vector = compute_fc_vector(concat_ts)
    return fc_vector, concat_ts.shape[0], runs_used


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch FC computation for HCP subjects")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(Path.home() / "Downloads"),
        help="Root directory containing subject folders (e.g. ~/Downloads/108020/)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.home() / "surge" / "fc_vectors"),
        help="Output directory for .npy FC vectors"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Run a single subject only (for testing). E.g. --subject 108020"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print subjects that would be processed without computing FC"
    )
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load behavioral data
    df = pd.read_csv(BEHAVIORAL_CSV)
    df_complete = df[df["3T_Full_MR_Compl"] == True].copy()
    subject_ids = df_complete["Subject"].astype(str).tolist()

    if args.subject:
        subject_ids = [args.subject]
        print(f"Single-subject mode: {args.subject}")
    else:
        print(f"Found {len(subject_ids)} subjects with complete 3T MRI")

    if args.dry_run:
        print("DRY RUN — subjects that would be processed:")
        for sid in subject_ids:
            print(f"  {sid}")
        return

    skipped = []
    completed = []

    for i, sid in enumerate(subject_ids):
        out_path = os.path.join(out_dir, f"{sid}_fc.npy")

        # Skip if already computed
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(subject_ids)}] {sid} — already exists, skipping")
            completed.append(sid)
            continue

        print(f"[{i+1}/{len(subject_ids)}] {sid} — computing FC...", end=" ", flush=True)
        try:
            fc_vector, n_tp, runs_used = process_subject(data_root, sid)
            np.save(out_path, fc_vector)
            completed.append(sid)
            print(f"✓  shape={fc_vector.shape}  timepoints={n_tp}  runs={len(runs_used)}")
        except Exception as e:
            skipped.append((sid, str(e)))
            print(f"✗  SKIPPED: {e}")

    # Write skip log
    skip_log_path = os.path.join(out_dir, "skip_log.txt")
    with open(skip_log_path, "w") as f:
        f.write(f"Skipped subjects ({len(skipped)} total)\n")
        f.write("=" * 50 + "\n")
        for sid, reason in skipped:
            f.write(f"{sid}: {reason}\n")

    print("\n" + "=" * 50)
    print(f"Done: {len(completed)} computed, {len(skipped)} skipped")
    print(f"FC vectors saved to: {out_dir}")
    print(f"Skip log: {skip_log_path}")


if __name__ == "__main__":
    main()
