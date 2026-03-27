# ════════════════════════════════════════════════════════════════
# MgInsight — Module B: Corrosion Simulation Viewer
# File: pages/2_🔬_Module_B.py
# Source: ProjectB/results/ (9 files ✅)
# ════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Module B — Corrosion",
                   page_icon="🔬", layout="wide")

# ── Verified paths (audit-confirmed) ─────────────────────────
RESULTS_DIR = "ProjectB/results"     # 9 files ✅
MESH_DIR    = "ProjectB/mesh"        # 6 files ✅
CORR_CSV    = f"{RESULTS_DIR}/corrosion_profile.csv"
DAY28_PNG   = f"{RESULTS_DIR}/day28_concentration.png"
IMMUNE_CSV  = "ProjectC/results/immune_response.csv"   # ✅

# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_corrosion():
    return pd.read_csv(CORR_CSV) \
           if os.path.exists(CORR_CSV) else pd.DataFrame()

@st.cache_data
def load_immune():
    return pd.read_csv(IMMUNE_CSV) \
           if os.path.exists(IMMUNE_CSV) else pd.DataFrame()

corr_df   = load_corrosion()
immune_df = load_immune()

# ════════════════════════════════════════════════════════════════
# SIDEBAR — auto-populated from Module A via session state
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Module B Controls")
    st.markdown("---")
    st.markdown("### 🔗 Auto-populated from Module A")

    # Read from session_state if Module A pushed values
    default_D  = st.session_state.get("D_Mg",   7e-10)
    default_Rc = st.session_state.get("R_corr", 0.045)

    D_Mg   = st.number_input("D_Mg (m²/s)",   value=float(default_D),
                              format="%.2e",
                              help="Diffusivity from Module A prediction")
    R_corr = st.number_input("R_corr (mm/day)",value=float(default_Rc),
                              format="%.4f",
                              help="Corrosion rate proxy from Module A")
    st.markdown("---")
    st.markdown("### 🧬 Solver Info")
    st.markdown("""
    - **Solver:** DOLFINx 0.10.0
    - **Method:** Crank-Nicolson
    - **Steps:** 78,316
    - **dt:** 30.9 s
    - **Duration:** 28 days
    - **Nodes:** 708
    """)
    st.markdown("---")
    st.markdown("### 🔗 Modules")
    st.page_link("pages/1_🦾_Module_A.py", label="🦾 Module A — Alloy Design")
    st.page_link("pages/3_🧬_Module_C.py", label="🧬 Module C — Immune ODE")

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.title("🔬 Module B — Mg Corrosion Simulation")
st.markdown("""
FEniCS/DOLFINx finite-element solution of **Fick's 2nd Law** on a 3D
Gmsh-meshed Mg implant cylinder inside a 13×13×30 mm SBF tissue box.
Solver: Crank-Nicolson | 78,316 steps | 708 nodes.
""")
st.markdown("---")

# ════════════════════════════════════════════════════════════════
# MESH SUMMARY
# ════════════════════════════════════════════════════════════════
st.subheader("📐 Geometry & Mesh Summary")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Tissue box",      "13×13×30 mm")
col_m2.metric("Implant radius",  "1.5 – 7.5 mm")
col_m3.metric("Mesh nodes",      "708")
col_m4.metric("h_min",           "2.94×10⁻⁴ m")

mesh_files = os.listdir(MESH_DIR) if os.path.isdir(MESH_DIR) else []
st.success(f"✅ ProjectB/mesh/ — {len(mesh_files)} files: "
           f"{', '.join(mesh_files)}")

# ════════════════════════════════════════════════════════════════
# DAY SNAPSHOTS
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📸 Mg²⁺ Concentration Snapshots")

snap_days  = [1, 7, 14, 21, 28]
snap_files = [f"{RESULTS_DIR}/day{d}.png" for d in snap_days]
available  = [f for f in snap_files if os.path.exists(f)]

if available:
    tab_labels = [f"Day {d}" for d in snap_days
                  if os.path.exists(f"{RESULTS_DIR}/day{d}.png")]
    tabs = st.tabs(tab_labels)
    for tab, fpath in zip(tabs, available):
        with tab:
            st.image(fpath, use_container_width=True,
                     caption=f"Mg²⁺ concentration field — "
                             f"{os.path.basename(fpath).replace('.png','')}")
elif os.path.exists(DAY28_PNG):
    st.image(DAY28_PNG, caption="Day 28 — Mg²⁺ concentration field")
else:
    st.info("Snapshot PNGs not found. Run ProjectB FEniCS notebook to generate.")

# ════════════════════════════════════════════════════════════════
# CORROSION PROFILE TABLE + CHART
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 Corrosion Profile — 28-Day Mg²⁺ Trajectory")

if not corr_df.empty:
    # Smart column detection
    day_col  = next((c for c in corr_df.columns
                     if "day" in c.lower()), corr_df.columns[0])
    conc_col = next((c for c in corr_df.columns
                     if any(k in c.lower() for k in
                     ["mg","conc","concentration"])),
                    corr_df.columns[1] if len(corr_df.columns)>1 else None)

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        if conc_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=corr_df[day_col], y=corr_df[conc_col],
                mode="lines+markers", name="[Mg²⁺]",
                line=dict(width=3), fill="tozeroy"
            ))
            fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                          annotation_text="M2 EC50 = 1.5")
            fig.add_hline(y=2.5, line_dash="dash", line_color="red",
                          annotation_text="M1 IC50 = 2.5")
            fig.add_hline(y=0.8, line_dash="dot",  line_color="gray",
                          annotation_text="Baseline 0.8")
            fig.update_layout(
                title="Local Mg²⁺ Concentration (28 days)",
                height=380,
                xaxis_title="Time (days)",
                yaxis_title="[Mg²⁺] mol/m³",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    with col_t2:
        st.markdown("**corrosion_profile.csv**")
        st.dataframe(corr_df, use_container_width=True, height=380)
        st.download_button(
            "⬇️ Download corrosion_profile.csv",
            data=corr_df.to_csv(index=False),
            file_name="corrosion_profile.csv", mime="text/csv"
        )
else:
    st.warning("ProjectB/results/corrosion_profile.csv not found. "
               "Run the DOLFINx notebook first.")

# ════════════════════════════════════════════════════════════════
# SOLVER PARAMETERS CARD
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("⚙️ Active Solver Parameters")

col_p1, col_p2, col_p3 = st.columns(3)
col_p1.metric("D_Mg (from Module A)", f"{D_Mg:.2e} m²/s")
col_p2.metric("R_corr (from Module A)", f"{R_corr:.4f} mm/day")
col_p3.metric("Estimated Day 21 [Mg²⁺]",
    f"{3.115 * (D_Mg / 7e-10) ** 0.3:.3f} mol/m³",
    help="Scaled from baseline sim using diffusivity ratio")

st.info(
    f"💡 These values were auto-populated from Module A prediction. "
    f"Change D_Mg or R_corr in the sidebar to re-scale estimates.  "
    f"→ [🧬 Module C — Immune ODE](3_🧬_Module_C)"
)

# ════════════════════════════════════════════════════════════════
# IMMUNE ODE PREVIEW (from ProjectC — cross-module link)
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧬 Immune Response Preview (from ProjectC/results/)")

if not immune_df.empty:
    col_i1, col_i2 = st.columns([3, 1])
    with col_i1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=immune_df["day"],
            y=immune_df["M1_inflammatory"],
            name="M1 Pro-inflam.", mode="lines"))
        fig2.add_trace(go.Scatter(x=immune_df["day"],
            y=immune_df["M2_healing"],
            name="M2 Healing", mode="lines"))
        fig2.add_trace(go.Scatter(x=immune_df["day"],
            y=immune_df["M2_M1_ratio"],
            name="M2/M1 Ratio", mode="lines",
            line=dict(dash="dot", width=2)))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="red",
                       annotation_text="Threshold = 1.0")
        fig2.add_vline(x=21, line_dash="dash", line_color="purple",
                       annotation_text="Day 21")
        fig2.update_layout(
            title="M1 vs M2 + Polarisation Ratio (28 days)",
            height=350,
            xaxis_title="Time (days)",
            yaxis_title="Cell density / Ratio",
            legend=dict(orientation="h", y=1.12, x=0.5,
                        xanchor="center", yanchor="bottom")
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col_i2:
        day21 = immune_df[immune_df["day"]==21].iloc[0]
        st.metric("M2/M1 @ Day 21", f"{day21['M2_M1_ratio']:.4f}")
        st.metric("Phase",           day21["phase"].upper())
        st.metric("M1 peak day",     "Day 9")
        st.metric("Crossover",       "Day 20.5 ✅")
        st.markdown("→ [🧬 Full Module C](3_🧬_Module_C)")
else:
    st.info("Run ProjectC ODE notebook to generate immune_response.csv")

# ════════════════════════════════════════════════════════════════
# FILE AUDIT
# ════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📂 ProjectB File Audit", expanded=False):
    res_files  = os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []
    mesh_files = os.listdir(MESH_DIR)    if os.path.isdir(MESH_DIR)    else []
    st.markdown(f"**ProjectB/results/** ({len(res_files)} files)")
    st.code("\n".join(sorted(res_files)) or "empty")
    st.markdown(f"**ProjectB/mesh/** ({len(mesh_files)} files)")
    st.code("\n".join(sorted(mesh_files)) or "empty")

st.markdown("---")
st.caption("MgInsight · Module B · ProjectB/results/ (9 files ✅) · "
           "Solver: DOLFINx 0.10.0 | Crank-Nicolson | 78,316 steps")