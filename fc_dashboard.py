"""
fc_dashboard.py
SURGE Project — Local FC Pipeline Dashboard

Run with:
    pip install streamlit plotly
    streamlit run fc_dashboard.py
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
    .context-box {
        background-color: #1a1a2e;
        border-left: 3px solid #534AB7;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #ccc;
        line-height: 1.6;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR — subject selection and config
# -------------------------------------------------------

with st.sidebar:
    st.title("🧠 FC Dashboard")
    st.caption("SURGE Project · HCP-YA 2025")

    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown(
        "This dashboard visualizes individual resting-state functional connectivity (FC) "
        "profiles computed from the NIH Human Connectome Project (HCP-YA). "
        "Each subject's fMRI timeseries is parcellated into brain regions, and pairwise "
        "correlations between regions form a FC matrix — the subject's unique "
        "\"connectome fingerprint\" (Finn et al., 2015). These fingerprints are then "
        "used to predict individual differences in negative affect (anxiety, fear, sadness) "
        "using Ridge Regression."
    )

    st.markdown("---")

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

    st.markdown("---")
    st.caption("Finn et al. (2015) *Nature Neuroscience* — FC fingerprinting using HCP data.")

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

st.title(f"Subject {subject_id} — Functional Connectivity Profile")
st.caption(f"{parcellation_choice} parcellation · {len(runs)} run(s) · Fisher-z transformed correlations")

# Project overview callout
st.markdown("""
<div class="context-box">
<b>Project overview</b> &mdash; Each row of the fMRI timeseries represents brain activity at one moment in time across 91,282 cortical surface vertices.
We average those vertices into <b>brain parcels</b> (regions of interest), then compute <b>pairwise Pearson correlations</b> between every pair of parcels' timeseries.
This produces a symmetric <b>FC matrix</b> whose off-diagonal elements encode the functional connectivity (coupling) between brain regions.
Correlations are Fisher-z transformed (arctanh) to normalize their distribution before statistical modeling.
The resulting FC vector &mdash; one value per edge &mdash; is this subject's connectome fingerprint, used downstream to predict negative affect scores.
</div>
""", unsafe_allow_html=True)

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
        st.metric("Timepoints", f"{result['n_timepoints']:,}",
                  help="Total fMRI volumes after concatenating all runs. Finn et al. recommend ≥500 timepoints for stable FC estimates.")
    with col2:
        st.metric("Parcels", str(n_parcels),
                  help="Number of brain regions in the selected parcellation. MMP gives 379 cortical parcels (180 per hemisphere + subcortex).")
    with col3:
        st.metric("FC edges", f"{n_edges:,}",
                  help="Number of unique pairwise connections = n*(n-1)/2. Each edge is one element of the FC vector used for prediction.")
    with col4:
        st.metric("Mean FC (z)", f"{fc_vector.mean():.3f}",
                  help="Mean Fisher-z across all edges. Positive values indicate overall positive coupling across the brain at rest.")
    with col5:
        st.metric("Std FC (z)", f"{fc_vector.std():.3f}",
                  help="Spread of edge weights. Higher std = more variability in coupling strength, which aids individual identification.")

    st.divider()

    # ---- ROW 1: FC HEATMAP + DISTRIBUTION ----
    col_left, col_right = st.columns([1.8, 1])

    with col_left:
        st.subheader("Functional connectivity matrix")
        st.markdown("""
<div class="context-box">
Each cell (i, j) shows the Fisher-z transformed Pearson correlation between the mean BOLD timeseries of parcel <i>i</i> and parcel <i>j</i>.
<b>Red</b> = strong positive coupling (regions activate together); <b>blue</b> = anticorrelation (regions suppress each other).
The block structure along the diagonal reflects known resting-state networks &mdash; regions within the same network
(e.g., default mode, frontoparietal) tend to show high within-network FC.
This matrix is the core output of the pipeline; its upper triangle is flattened into the FC vector used for Ridge Regression.
</div>
""", unsafe_allow_html=True)

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
        st.subheader("Edge weight distribution")
        st.markdown("""
<div class="context-box">
Distribution of all FC edge values (Fisher-z) for this subject.
A roughly Gaussian distribution centered near 0 is expected &mdash; most pairs of brain regions have near-zero coupling.
The positive tail reflects strongly connected pairs (e.g., homotopic regions, within-network edges).
The negative tail reflects anticorrelated pairs, often seen between the default mode and task-positive networks.
</div>
""", unsafe_allow_html=True)

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

        st.caption("Edge weight percentiles")
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

    # ---- ROW 2: NETWORK MATRIX PLOT ----
    st.divider()
    st.subheader("Network-level connectivity matrix (Yeo 7)")
    st.markdown("""
<div class="context-box">
This matrix summarizes FC at the <b>network level</b> by grouping parcels into Yeo 7 resting-state networks
(Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default Mode)
and computing the mean Fisher-z within and between each pair. The diagonal = within-network FC;
off-diagonal = between-network FC. Classic expected pattern: high DMN–DMN, high FPN–FPN,
and the DMN–DAN anticorrelation. Directly comparable to Shen et al. (2017) matrix plots.
</div>
""", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def build_network_matrix(z_key, parc_name):
        # z_key is a hashable tuple version of the matrix shape + mean for cache keying
        parc = parc_map[parc_name]
        yeo_map = np.array(hcp.yeo7.map_all)
        parcel_map_all = np.array(parc.map_all)
        parc_ids = parc.nontrivial_ids

        parcel_network = []
        for pid in parc_ids:
            mask = parcel_map_all == pid
            vals = yeo_map[mask]
            vals = vals[vals > 0]
            parcel_network.append(int(np.bincount(vals).argmax()) if len(vals) > 0 else 0)
        parcel_network = np.array(parcel_network)
        return parcel_network

    parcel_network = build_network_matrix(
        (z_matrix.shape, float(z_matrix.mean())), parcellation_choice
    )

    net_names = ["Visual", "SomMot", "DorsAttn", "VentAttn", "Limbic", "FrontPar", "Default"]
    net_ids = list(range(1, 8))
    n_nets = len(net_ids)
    net_matrix = np.zeros((n_nets, n_nets))
    for i, ni in enumerate(net_ids):
        for j, nj in enumerate(net_ids):
            mi = parcel_network == ni
            mj = parcel_network == nj
            if mi.sum() == 0 or mj.sum() == 0:
                continue
            sub = z_matrix[np.ix_(mi, mj)]
            if i == j:
                triu = sub[np.triu_indices(sub.shape[0], k=1)]
                net_matrix[i, j] = triu.mean() if len(triu) > 0 else 0
            else:
                net_matrix[i, j] = sub.mean()

    fig_net = go.Figure(data=go.Heatmap(
        z=net_matrix,
        x=net_names,
        y=net_names,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-0.5,
        zmax=0.8,
        showscale=True,
        colorbar=dict(title="Mean Fisher-z", thickness=12),
        text=np.round(net_matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=12)
    ))
    fig_net.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Network", tickangle=-30),
        yaxis=dict(title="Network", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_net, use_container_width=True)

    # ---- ROW 3: MOTION ----
    if result["motion_per_run"]:
        st.divider()
        st.subheader("Head motion")
        st.markdown("""
<div class="context-box">
Head motion is a major confound in FC analyses &mdash; even small movements correlate artifactually with BOLD signal,
inflating short-range connections and suppressing long-range ones (Van Dijk et al., 2012).
The metric shown is <b>framewise displacement (FD)</b>: the mean frame-to-frame RMS displacement across all translational and rotational axes.
Finn et al. excluded subjects with mean FD &gt; 0.14 mm for behavioral analyses.
The 0.2 mm threshold here flags runs that may require additional scrubbing before use in the main Ridge Regression pipeline.
</div>
""", unsafe_allow_html=True)

        motion_cols = st.columns(len(result["motion_per_run"]))
        for i, (run_name, fd_trace) in enumerate(result["motion_per_run"].items()):
            with motion_cols[i]:
                mean_fd = fd_trace.mean()
                color = "normal" if mean_fd < 0.2 else ("off" if mean_fd < 0.5 else "inverse")
                st.metric(
                    run_name.replace("rfMRI_", ""),
                    f"{mean_fd:.3f} mm",
                    delta="✓ below 0.2mm" if mean_fd < 0.2 else "⚠ above 0.2mm",
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

    # ---- ROW 3: GLASS BRAIN ----
    if parcellation_choice == "MMP (360)":
        st.divider()
        st.subheader("Glass brain connectome")
        st.markdown("""
<div class="context-box">
This view renders the subject's strongest FC edges projected onto a transparent brain surface,
viewed from three orthogonal angles (sagittal / coronal / axial).
Each node is the <b>centroid</b> of an MMP parcel; edge color encodes the sign and magnitude of the Fisher-z connection
(<b>red</b> = positive, <b>blue</b> = negative). Only the top X% of edges by absolute FC strength are shown.
Finn et al. found that the <b>frontoparietal</b> and <b>medial frontal</b> networks contribute most to individual
fingerprinting and behavioral prediction &mdash; look for dense clusters of edges in prefrontal and parietal regions.
</div>
""", unsafe_allow_html=True)

        gb_col1, gb_col2 = st.columns([1, 1])
        with gb_col1:
            edge_pct = st.slider(
                "Show top X% strongest edges",
                min_value=1, max_value=20, value=1, step=1,
                help="Top 1% = ~716 strongest connections. Higher % reveals more network structure."
            )
        with gb_col2:
            network_filter = st.selectbox(
                "Filter by network (show only edges involving this network)",
                options=["All networks"] + net_names,
                index=0,
                help="Restricts displayed edges to those where at least one endpoint parcel belongs to the selected Yeo 7 network."
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

            # Apply network filter if selected
            if network_filter != "All networks" and parcellation_choice == "MMP (360)":
                net_idx = net_names.index(network_filter) + 1  # 1-indexed Yeo IDs
                net_mask = parcel_network == net_idx  # shape (n_parcels,)
                # Zero out any edge where neither endpoint is in the selected network
                filter_matrix = np.zeros_like(adj)
                for pi in range(n):
                    for pj in range(pi + 1, n):
                        if net_mask[pi] or net_mask[pj]:
                            filter_matrix[pi, pj] = adj[pi, pj]
                            filter_matrix[pj, pi] = adj[pj, pi]
                adj = filter_matrix

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

            net_label = f" · {network_filter}" if network_filter != "All networks" else ""
            plt.suptitle(
                f"Top {edge_pct}% FC edges — Subject {subject_id} (MMP 379 parcels){net_label}",
                color="white", fontsize=13, y=1.01
            )
            plt.tight_layout()

            buf = io.BytesIO()
            fig_gb.savefig(buf, format="png", dpi=130,
                           bbox_inches="tight", facecolor="black")
            plt.close(fig_gb)
            buf.seek(0)

        st.image(buf, use_container_width=True)

        col_dl_gb, _ = st.columns([1, 3])
        with col_dl_gb:
            st.download_button(
                "Download glass brain (.png)",
                data=buf.getvalue(),
                file_name=f"{subject_id}_glass_brain_top{edge_pct}pct.png",
                mime="image/png"
            )

    # ---- ROW 4: DOWNLOAD ----
    st.divider()
    st.subheader("Export")
    st.markdown("""
<div class="context-box">
The <b>FC vector</b> (.npy) is the flattened upper triangle of the FC matrix &mdash; this is the feature vector
fed into Ridge Regression to predict negative affect scores. For n=379 MMP parcels, this yields 71,631 features per subject.
The batch pipeline (<code>run_batch_fc.py</code>) will save one such vector per subject to <code>~/surge/fc_vectors/</code>,
which <code>predict_negaffect.py</code> will then assemble into the full feature matrix for modeling.
</div>
""", unsafe_allow_html=True)

    col_dl1, col_dl2, _ = st.columns([1, 1, 2])
    with col_dl1:
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
