"""
hcp_s3_download.py
SURGE Project — Targeted S3 download of HCP resting-state dtseries files

Usage:
    python hcp_s3_download.py --n 30 --out_dir ~/Downloads

What it does:
    For each subject in behavioral.csv with 3T_Full_MR_Compl == True,
    downloads only the 4 dtseries files needed for FC computation directly
    from the HCP open-access S3 bucket — skipping the 10 GB BALSA zip.

    File downloaded per run (~418 MB each, ~1.7 GB per subject):
        rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
        rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii
        rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
        rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii

    Also downloads Movement_RelativeRMS.txt for motion QC.

    Requires AWS credentials configured via `aws configure` using
    your HCP ConnectomeDB S3 keys (db.humanconnectome.org → Profile
    → Amazon S3 Access Enabled).

Prerequisites:
    pip install boto3
    aws configure  (enter HCP S3 access key + secret key)
"""

import os
import argparse
import time
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.exceptions import ConnectTimeoutError
import pandas as pd
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

BUCKET = "hcp-openaccess"
HCP_PREFIX = "HCP_1200"
BEHAVIORAL_CSV = Path(__file__).parent / "behavioral.csv"

RUNS = [
    "rfMRI_REST1_LR",
    "rfMRI_REST1_RL",
    "rfMRI_REST2_LR",
    "rfMRI_REST2_RL",
]

# S3 has this version (no _rclean_tclean suffix)
DTSERIES_FILENAME = "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"

# Also grab motion file for QC
EXTRA_FILES = ["Movement_RelativeRMS.txt"]


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def s3_key(subject_id, run, filename):
    return f"{HCP_PREFIX}/{subject_id}/MNINonLinear/Results/{run}/{filename}"


def local_path(out_dir, subject_id, run, filename):
    """Mirror the HCP directory structure locally."""
    return os.path.join(
        out_dir, str(subject_id), "MNINonLinear", "Results", run, filename
    )


def download_file(s3_client, subject_id, run, filename, out_dir, dry_run=False):
    key = s3_key(subject_id, run, filename)
    dest = local_path(out_dir, subject_id, run, filename)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"    [exists] {filename} ({size_mb:.0f} MB)")
        return True

    if dry_run:
        print(f"    [dry-run] would download: {key}")
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            head = s3_client.head_object(Bucket=BUCKET, Key=key)
            size_mb = head["ContentLength"] / 1e6
            if attempt == 1:
                print(f"    ↓ {filename} ({size_mb:.0f} MB)...", end=" ", flush=True)
            else:
                print(f"    ↓ retry {attempt}/{max_retries}...", end=" ", flush=True)

            s3_client.download_file(BUCKET, key, dest)
            print("✓")
            return True

        except Exception as e:
            if os.path.exists(dest):
                os.remove(dest)
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"\n    [network error, waiting {wait}s before retry]", flush=True)
                time.sleep(wait)
            else:
                print(f"\n    ✗ FAILED after {max_retries} attempts: {filename}")
                return False


def download_subject(s3_client, subject_id, out_dir, dry_run=False):
    """Download all 4 runs + motion files for one subject."""
    success_count = 0
    fail_count = 0

    for run in RUNS:
        # Main dtseries file
        fname = DTSERIES_FILENAME.format(run=run)
        ok = download_file(s3_client, subject_id, run, fname, out_dir, dry_run)
        if ok:
            success_count += 1
        else:
            fail_count += 1

        # Motion file
        for extra in EXTRA_FILES:
            download_file(s3_client, subject_id, run, extra, out_dir, dry_run)

    return success_count, fail_count


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download HCP resting-state dtseries files from S3"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.home() / "Downloads"),
        help="Output directory (mirrors HCP folder structure)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of subjects to download (default: 30)"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of specific subject IDs to download"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded without actually downloading"
    )
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load subject list
    df = pd.read_csv(BEHAVIORAL_CSV)
    df_complete = df[df["3T_Full_MR_Compl"] == True].copy()
    all_subjects = df_complete["Subject"].astype(str).tolist()

    if args.subjects:
        subject_ids = [s.strip() for s in args.subjects.split(",")]
        print(f"Downloading {len(subject_ids)} specified subjects")
    else:
        subject_ids = all_subjects[: args.n]
        print(f"Downloading first {len(subject_ids)} of {len(all_subjects)} complete subjects")

    print(f"Output directory: {out_dir}")
    print(f"S3 bucket: s3://{BUCKET}/{HCP_PREFIX}/")
    print(f"File per run: {DTSERIES_FILENAME.format(run='<run>')}")
    print(f"~1.7 GB per subject (4 × ~418 MB)")
    total_gb = len(subject_ids) * 1.7
    print(f"Estimated total: ~{total_gb:.0f} GB for {len(subject_ids)} subjects")

    if args.dry_run:
        print("\n[DRY RUN — no files will be downloaded]\n")

    # Init S3 client
    s3 = boto3.client("s3")

    completed = []
    skipped = []

    for i, sid in enumerate(subject_ids):
        print(f"\n[{i+1}/{len(subject_ids)}] Subject {sid}")
        ok, fail = download_subject(s3, sid, out_dir, dry_run=args.dry_run)

        if fail == 0:
            completed.append(sid)
        elif ok > 0:
            completed.append(sid)  # partial but usable
            print(f"  ⚠ {fail} run(s) failed — subject may be partial")
        else:
            skipped.append(sid)
            print(f"  ✗ All runs failed — skipping subject")

    print("\n" + "=" * 50)
    print(f"Done: {len(completed)} subjects downloaded, {len(skipped)} failed")
    if skipped:
        print(f"Failed subjects: {', '.join(skipped)}")
    print(f"\nNext step: run run_batch_fc.py --data_root {out_dir}")


if __name__ == "__main__":
    main()
