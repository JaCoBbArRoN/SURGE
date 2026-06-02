"""
compute_fc.py
SURGE Project — Functional Connectivity Predicts Dimensional Negative Affect

Week 3 core task: for each subject, compute a Fisher-z transformed FC matrix
from their resting-state parcellated timeseries and vectorize the upper triangle.

Pipeline per subject:
  dtseries.nii → hcp.parcellate → (parcels x time) → Pearson r matrix
  → Fisher-z transform → upper triangle vector

Output:
  fc_matrix.npy  — (n_subjects, n_edges) array of upper-triangle FC vectors
  subject_ids.npy — (n_subjects,) array of subject ID strings
"""

import os
import numpy as np
import nibabel as nib
import hcp_utils as hcp
from scipy import stats

# -------------------------------------------------------
# CONFIG — update these paths before running
# -------------------------------------------------------

# Root directory where subject folders live after unzipping BALSA downloads
# Each subject folder should follow: {DATA_ROOT}/{subject_id}/MNINonLinear/Results/...
DATA_ROOT = "/Users/jacobbarron/Downloads"

# Which rs-fMRI runs to use per subject (can use one or concatenate all four)
# Options: REST1_LR, REST1_RL, REST2_LR, REST2_RL
RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]

# File suffix for the ICA-FIX denoised dense timeseries (HCP-YA 2025 naming)
DTSERIES_SUFFIX = "_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii"

# Parcellation — using HCP MMP (360 parcels) via hcp_utils
# hcp.mmp gives 360 cortical parcels derived from HCP subjects
PARCELLATION = hcp.mmp

# Output directory
OUTPUT_DIR = "/Users/jacobbarron/surge_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def get_dtseries_path(data_root, subject_id, run):
    """Build the path to a subject's dense timeseries for a given run."""
    return os.path.join(
        data_root, subject_id, "MNINonLinear", "Results", run,
        f"{run}{DTSERIES_SUFFIX}"
    )

def load_parcel_timeseries(dtseries_path, parcellation):
    """
    Load a dense timeseries and parcellate it.
    Returns array of shape (timepoints, n_parcels).
    """
    img = nib.load(dtseries_path)
    data = img.get_fdata(dtype=np.float32)  # shape: (timepoints, grayordinates)
    parcel_ts = hcp.parcellate(data, parcellation)  # shape: (timepoints, n_parcels)
    return parcel_ts

def compute_fc_vector(parcel_ts):
    """
    Compute Fisher-z transformed FC matrix from parcel timeseries.
    Returns the vectorized upper triangle (excluding diagonal).
    
    Steps:
      1. Pearson correlation across all parcel pairs -> r matrix
      2. Fisher-z transform: z = arctanh(r) -> stabilizes variance
      3. Vectorize upper triangle -> 1D feature vector
    """
    # 1. Pearson correlation matrix — shape (n_parcels, n_parcels)
    r_matrix = np.corrcoef(parcel_ts.T)

    # 2. Fisher-z transform
    # Clip r to avoid arctanh blowing up at exactly ±1
    r_clipped = np.clip(r_matrix, -0.9999, 0.9999)
    z_matrix = np.arctanh(r_clipped)

    # 3. Upper triangle indices (excluding diagonal, k=1)
    n_parcels = z_matrix.shape[0]
    upper_idx = np.triu_indices(n_parcels, k=1)
    fc_vector = z_matrix[upper_idx]

    return fc_vector

def concatenate_runs(data_root, subject_id, runs, parcellation):
    """
    Load and concatenate timeseries across all available runs for a subject.
    Concatenation increases FC reliability vs. using a single run.
    Returns concatenated parcel timeseries (timepoints_total, n_parcels).
    """
    all_ts = []
    for run in runs:
        path = get_dtseries_path(data_root, subject_id, run)
        if not os.path.exists(path):
            print(f"  Warning: {run} not found for subject {subject_id}, skipping")
            continue
        ts = load_parcel_timeseries(path, parcellation)
        all_ts.append(ts)

    if len(all_ts) == 0:
        return None

    return np.concatenate(all_ts, axis=0)  # stack along timepoints axis

# -------------------------------------------------------
# MAIN — process all subjects
# -------------------------------------------------------

def main():
    # Find all subject directories in DATA_ROOT
    subject_ids = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.isdigit()
    ])
    print(f"Found {len(subject_ids)} subject directories")

    # Compute n_edges from parcellation size
    n_parcels = PARCELLATION.map.max()  # number of parcels
    n_edges = int(n_parcels * (n_parcels - 1) / 2)
    print(f"Parcellation: {n_parcels} parcels → {n_edges} edges per subject")

    fc_matrix = []
    valid_subjects = []

    for i, subject_id in enumerate(subject_ids):
        print(f"[{i+1}/{len(subject_ids)}] Processing subject {subject_id}...")

        try:
            # Load and concatenate all runs
            parcel_ts = concatenate_runs(DATA_ROOT, subject_id, RUNS, PARCELLATION)

            if parcel_ts is None:
                print(f"  Skipping {subject_id}: no runs found")
                continue

            print(f"  Timeseries shape: {parcel_ts.shape}")

            # Compute FC vector
            fc_vector = compute_fc_vector(parcel_ts)
            print(f"  FC vector shape: {fc_vector.shape}")

            # Basic quality check — skip if NaNs or zero variance
            if np.isnan(fc_vector).any():
                print(f"  Skipping {subject_id}: NaNs in FC vector")
                continue
            if fc_vector.std() == 0:
                print(f"  Skipping {subject_id}: zero variance in FC vector")
                continue

            fc_matrix.append(fc_vector)
            valid_subjects.append(subject_id)

        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            continue

    # Stack into final array
    fc_matrix = np.array(fc_matrix)  # shape: (n_subjects, n_edges)
    subject_ids_out = np.array(valid_subjects)

    print(f"\nDone. FC matrix shape: {fc_matrix.shape}")
    print(f"Valid subjects: {len(valid_subjects)}")

    # Save outputs
    np.save(os.path.join(OUTPUT_DIR, "fc_matrix.npy"), fc_matrix)
    np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"), subject_ids_out)
    print(f"Saved to {OUTPUT_DIR}/fc_matrix.npy and subject_ids.npy")


# -------------------------------------------------------
# SINGLE SUBJECT SMOKE TEST
# Run this first to validate on your one downloaded subject
# before running the full pipeline on all subjects
# -------------------------------------------------------

def smoke_test():
    """Validate the full FC computation pipeline on subject 108020."""
    subject_id = "108020"
    print(f"Smoke test: subject {subject_id}")

    parcel_ts = concatenate_runs(DATA_ROOT, subject_id, RUNS, PARCELLATION)
    if parcel_ts is None:
        print("No runs found — check DATA_ROOT and subject directory")
        return

    print(f"Concatenated timeseries shape: {parcel_ts.shape}")

    fc_vector = compute_fc_vector(parcel_ts)
    print(f"FC vector shape: {fc_vector.shape}")
    print(f"FC vector stats: mean={fc_vector.mean():.4f}, std={fc_vector.std():.4f}")
    print(f"NaNs: {np.isnan(fc_vector).sum()}")
    print("Smoke test passed!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke_test()
    else:
        main()
