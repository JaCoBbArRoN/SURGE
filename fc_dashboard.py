"""
fc_dashboard.py
SURGE Project — Local FC Pipeline Dashboard

Run with:
    pip install streamlit plotly
    streamlit run fc_dashboard.py

What it shows:
  - FC matrix heatmap (360x360 or whatever your parcellation gives)
  - Edge weight distribution (Fisher-z)
  - Network-level mean FC breakdown
  - Motion summary (mean FD per run)
  - Subject-level stats summary
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import nibabel as nib
import hcp_utils as hcp
import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting
from pathlib import Path

st.set_page_config(
    page_title="FC Pipeline Dashboard",
    layout="wide",
    page_icon="🧠"
)

st.markdown("""
<style>
    .metric-label { font-size: 12px; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 24px; font-weight: 600; }
    .metric-sub { font-size: 11px; color: #aaa; }
    .stPlotlyChart { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR — subject selection and config
# -------------------------------------------------------

with st.sidebar:
    st.title("🧠 FC Dashboard")
    st.caption("SURGE Project · HCP-YA 2025")

    data_root = st.text_input(
        "Data root directory",
        value=str(Path.home() / "Downloads"),
        help="Folder containing subject subdirectories e.g. ~/Downloads/108020/"
    )

    available_subjects = []
    if os.path.exists(data_root):
        available_subjects = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d.isdigit()
        ])

    if available_subjects:
        subject_id = st.selectbox("Subject", available_subjects)
    else:
        subject_id = st.text_input("Subject ID", value="108020")

    runs = st.multiselect(
        "Runs to include",
        ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"],
        default=["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
    )

    parcellation_choice = st.selectbox("Parcellation", ["MMP (360)", "Yeo 17", "Yeo 7"])
    parc_map = {"MMP (360)": hcp.mmp, "Yeo 17": hcp.yeo17, "Yeo 7": hcp.yeo7}
    parcellation = parc_map[parcellation_choice]

    run_button = st.button("▶ Compute FC", use_container_width=True)

# -------------------------------------------------------
# HELPER — load + compute FC
# -------------------------------------------------------

DTSERIES_SUFFIX = "_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii"

def load_motion(data_root, subject_id, run):
    path = os.path.join(data_root, subject_id, "MNINonLinear", "Results", run,
                        "Movement_RelativeRMS.txt")
    if os.path.exists(path):
        return np.loadtxt(path)
    return None

@st.cache_data(show_spinner=False)
def get_mmp_centroids():
    """Compute MMP parcel centroids in MNI space (cached — only runs once)."""
    coords_l = hcp.mesh.pial_left[0]
    coords_r = hcp.mesh.pial_right[0]
    grayl = np.array(hcp.vertex_info["grayl"])
    grayr = np.array(hcp.vertex_info["grayr"])
    parc_map = np.array(hcp.mmp.map_all)
    all_coords = np.zeros((91282, 3))
    all_coords[:len(grayl)] = coords_l[grayl]
    all_coords[len(grayl):len(grayl) + len(grayr)] = coords_r[grayr]
    parcel_ids = hcp.mmp.nontrivial_ids
    centroids = np.zeros((len(parcel_ids), 3))
    for i, pid in enumerate(parcel_ids):
        centroids[i] = all_coords[parc_map == pid].mean(axis=0)
    return centroids, parcel_ids


@st.cache_data(show_spinner=False)
def compute_subject_fc(data_root, subject_id, runs, parc_name):
    parcellation = parc_map[parc_name]
    all_ts = []
    motion_per_run = {}
    run_lengths = {}

    for run in runs:
        dtseries_path = os.path.join(
            data_root, subject_id, "MNINonLinear", "Results", run,
            f"{run}{DTSERIES_SUFFIX}"
        )
        if not os.path.exists(dtseries_path):
            continue

        img = nib.load(dtseries_path)
        data = img.get_fdata(dtype=np.float32)
        ts = hcp.parcellate(data, parcellation)
        all_ts.append(ts)
        run_lengths[run] = ts.shape[0]

        motion = load_motion(data_root, subject_id, run)
        if motion is not None:
            motion_per_run[run] = motion

    if not all_ts:
        return None

    concat_ts = np.concatenate(all_ts, axis=0)
    r_matrix = np.corrcoef(concat_ts.T)
    r_clipped = np.clip(r_matrix, -0.9999, 0.9999)
    z_matrix = np.arctanh(r_clipped)

    n = z_matrix.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    fc_vector = z_matrix[upper_idx]

    return {
        "z_matrix": z_matrix,
        "fc_vector": fc_vector,
        "concat_ts": concat_ts,
        "run_lengths": run_lengths,
        "motion_per_run": motion_per_run,
        "n_parcels": n,
        "n_timepoints": concat_ts.shape[0],
    }

# -------------------------------------------------------
# MAIN — render dashboard
# -------------------------------------------------------

st.title(f"Subject {subject_id}")
st.caption(f"{parcellation_choice} · {len(runs)} run(s) · Fisher-z transformed FC")

if run_button or True:
    with st.spinner(f"Computing FC for subject {subject_id}..."):
        result = compute_subject_fc(data_root, subject_id, runs, parcellation_choice)

    if result is None:
        st.error(f"No dtseries files found for subject {subject_id}. Check your data root and subject ID.")
        st.stop()

    z_matrix = result["z_matrix"]
    fc_vector = result["fc_vector"]
    n_parcels = result["n_parcels"]
    n_edges = len(fc_vector)

    # ---- STATS ROW ----
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Timepoints", f"{result['n_timepoints']:,}")
    with col2:
        st.metric("Parcels", str(n_parcels))
    with col3:
        st.metric("FC edges", f"{n_edges:,}")
    with col4:
        st.metric("Mean FC (z)", f"{fc_vector.mean():.3f}")
    with col5:
        st.metric("Std FC (z)", f"{fc_vector.std():.3f}")

    st.divider()

    # ---- ROW 1: FC HEATMAP + DISTRIBUTION ----
    col_left, col_right = st.columns([1.8, 1])

    with col_left:
        st.subheader("FC matrix")

        # Subsample for display if large
        display_n = min(n_parcels, 180)
        step = max(1, n_parcels // display_n)
        z_display = z_matrix[::step, ::step]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_display,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1.5,
            zmax=1.5,
            showscale=True,
            colorbar=dict(title="Fisher-z", thickness=12, len=0.8)
        ))
        fig_heat.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            xaxis=dict(showticklabels=False, title="parcel"),
            yaxis=dict(showticklabels=False, title="parcel", autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.subheader("Edge distribution")

        fig_dist = go.Figure(data=go.Histogram(
            x=fc_vector,
            nbinsx=60,
            marker_color="#534AB7",
            opacity=0.85
        ))
        fig_dist.add_vline(x=fc_vector.mean(), line_dash="dash",
                           line_color="#E24B4A", annotation_text=f"mean={fc_vector.mean():.2f}")
        fig_dist.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            xaxis_title="Fisher-z",
            yaxis_title="count",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Summary stats table
        st.caption("Edge weight stats")
        stats_df = pd.DataFrame({
            "stat": ["min", "p10", "p25", "median", "p75", "p90", "max"],
            "value": [
                f"{np.percentile(fc_vector, 0):.3f}",
                f"{np.percentile(fc_vector, 10):.3f}",
                f"{np.percentile(fc_vector, 25):.3f}",
                f"{np.median(fc_vector):.3f}",
                f"{np.percentile(fc_vector, 75):.3f}",
                f"{np.percentile(fc_vector, 90):.3f}",
                f"{np.percentile(fc_vector, 100):.3f}",
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # ---- ROW 2: MOTION ----
    if result["motion_per_run"]:
        st.subheader("Head motion (mean FD per run)")
        motion_cols = st.columns(len(result["motion_per_run"]))
        for i, (run_name, fd_trace) in enumerate(result["motion_per_run"].items()):
            with motion_cols[i]:
                mean_fd = fd_trace.mean()
                color = "normal" if mean_fd < 0.2 else ("off" if mean_fd < 0.5 else "inverse")
                st.metric(
                    run_name.replace("rfMRI_", ""),
                    f"{mean_fd:.3f} mm",
                    delta="below threshold" if mean_fd < 0.2 else "above threshold",
                    delta_color=color
                )

        fig_motion = go.Figure()
        colors = ["#534AB7", "#1D9E75", "#BA7517", "#D4537E"]
        for i, (run_name, fd_trace) in enumerate(result["motion_per_run"].items()):
            fig_motion.add_trace(go.Scatter(
                y=fd_trace,
                name=run_name.replace("rfMRI_REST", "REST"),
                line=dict(color=colors[i % len(colors)], width=1),
                opacity=0.8
            ))
        fig_motion.add_hline(y=0.2, line_dash="dot", line_color="#E24B4A",
                             annotation_text="0.2mm threshold")
        fig_motion.update_layout(
            height=180,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="timepoint",
            yaxis_title="FD (mm)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_motion, use_container_width=True)

    # ---- ROW 3: GLASS BRAIN (MMP only) ----
    if parcellation_choice == "MMP (360)":
        st.divider()
        st.subheader("Glass brain connectome")

        edge_pct = st.slider(
            "Show top X% strongest edges",
            min_value=1, max_value=10, value=1, step=1,
            help="Higher = more edges shown"
        )

        with st.spinner("Rendering glass brain..."):
            centroids, _ = get_mmp_centroids()

            n = n_parcels
            fc_matrix = np.zeros((n, n))
            idx = np.triu_indices(n, k=1)
            fc_matrix[idx] = fc_vector
            fc_matrix += fc_matrix.T

            threshold = np.percentile(np.abs(fc_vector), 100 - edge_pct)
            adj = np.where(np.abs(fc_matrix) >= threshold, fc_matrix, 0.0)

            fig_gb, axes_gb = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
            for ax in axes_gb:
                ax.set_facecolor("black")

            for i, view in enumerate(["x", "y", "z"]):
                plotting.plot_connectome(
                    adj, centroids,
                    node_size=6,
                    node_color="white",
                    edge_threshold=threshold * 0.99,
                    edge_vmin=-3,
                    edge_vmax=3,
                    edge_cmap="RdBu_r",
                    display_mode=view,
                    axes=axes_gb[i],
                    colorbar=(i == 2),
                    black_bg=True,
                    alpha=0.7,
                )

            plt.suptitle(
                f"Top {edge_pct}% FC edges — Subject {subject_id} (MMP 379 parcels)",
                color="white", fontsize=13, y=1.01
            )
            plt.tight_layout()

            buf = io.BytesIO()
            fig_gb.savefig(buf, format="png", dpi=130,
                           bbox_inches="tight", facecolor="black")
            plt.close(fig_gb)
            buf.seek(0)

        st.image(buf, use_container_width=True)

        st.download_button(
            "Download glass brain (.png)",
            data=buf.getvalue(),
            file_name=f"{subject_id}_glass_brain_top{edge_pct}pct.png",
            mime="image/png"
        )

    # ---- ROW 4: DOWNLOAD ----
    st.divider()
    col_dl1, col_dl2, _ = st.columns([1, 1, 2])
    with col_dl1:
        fc_bytes = fc_vector.astype(np.float32).tobytes()
        st.download_button(
            "Download FC vector (.npy)",
            data=fc_vector.astype(np.float32).tobytes(),
            file_name=f"{subject_id}_fc_vector.npy",
            mime="application/octet-stream"
        )
    with col_dl2:
        stats_csv = pd.DataFrame({
            "subject_id": [subject_id],
            "n_parcels": [n_parcels],
            "n_edges": [n_edges],
            "n_timepoints": [result["n_timepoints"]],
            "mean_fc_z": [round(float(fc_vector.mean()), 4)],
            "std_fc_z": [round(float(fc_vector.std()), 4)],
            "median_fc_z": [round(float(np.median(fc_vector)), 4)],
        })
        st.download_button(
            "Download stats (.csv)",
            data=stats_csv.to_csv(index=False),
            file_name=f"{subject_id}_fc_stats.csv",
            mime="text/csv"
        )