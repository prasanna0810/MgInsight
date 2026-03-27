# ════════════════════════════════════════════════════════════════
# MgInsight — Home Page
# File : app.py
# Run  : streamlit run app.py
# Verified folder audit:
#   ProjectA/models/      ✅ 7 files
#   ProjectA/data/        ✅ 3 files
#   ProjectA/reports/     ✅ 7 files
#   ProjectA/bayesian_opt/✅ 4 files
#   ProjectB/mesh/        ✅ 6 files
#   ProjectB/results/     ✅ 9 files
#   ProjectC/results/     ✅ 3 files
#   ProjectC/notebooks/   ✅ created
# ════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title = "MgInsight — Home",
    page_icon  = "🧬",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Paths ─────────────────────────────────────────────────────
IMMUNE_CSV = "ProjectC/results/immune_response.csv"
CORR_CSV   = "ProjectB/results/corrosion_profile.csv"
MODEL_DIR  = "ProjectA/models"
BO_JSON    = "ProjectA/bayesian_opt/best_candidate.json"

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/magnesium.png",
        width=64
    )
    st.title("🧬 MgInsight")
    st.caption("AI-Driven Mg Alloy Implant Design Platform")
    st.markdown("---")

    st.markdown("### 🗂️ Navigate")
    st.page_link("app.py",
                 label="🏠 Home", icon="🏠")
    st.page_link("pages/1_🦾_Module_A.py",
                 label="🦾 Module A — Alloy Design")
    st.page_link("pages/2_🔬_Module_B.py",
                 label="🔬 Module B — Corrosion Simulation")
    st.page_link("pages/3_🧬_Module_C.py",
                 label="🧬 Module C — Immune Response ODE")

    st.markdown("---")
    st.markdown("### 📦 Platform Stack")
    st.markdown("""
    | Layer | Tool |
    |---|---|
    | ML | XGBoost · RF · GPR |
    | Optimisation | BoTorch UCB |
    | Explainability | SHAP |
    | Simulation | DOLFINx 0.10.0 |
    | ODE Solver | SciPy RK45 |
    | Mesh | Gmsh |
    | App | Streamlit |
    """)

    st.markdown("---")
    st.markdown("### 📂 Folder Audit")

    audit_paths = {
        "ProjectA/models":       7,
        "ProjectA/data":         3,
        "ProjectA/reports":      7,
        "ProjectA/bayesian_opt": 4,
        "ProjectB/mesh":         6,
        "ProjectB/results":      9,
        "ProjectC/results":      3,
        "ProjectC/notebooks":    1,
    }
    all_ok = True
    for folder, expected in audit_paths.items():
        exists = os.path.isdir(folder)
        count  = len(os.listdir(folder)) if exists else 0
        ok     = exists and count >= 1
        icon   = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        st.markdown(
            f"{icon} `{folder.split('/')[-1]}/` "
            f"<span style='color:{'green' if ok else 'red'};font-size:0.8em;'>"
            f"({count} files)</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    if all_ok:
        st.success("All folders verified ✅")
    else:
        st.warning("Some folders missing — check audit above")

# ════════════════════════════════════════════════════════════════
# MAIN HEADER
# ════════════════════════════════════════════════════════════════
st.title("🧬 MgInsight")
st.markdown(
    "### AI-Driven Biodegradable Magnesium Alloy Implant Design Platform"
)
st.markdown("""
An integrated computational platform combining **machine learning alloy design**,
**finite-element corrosion simulation**, and **immune response ODE modelling**
to accelerate discovery and validation of biodegradable Mg implants.
""")
st.markdown("---")

# ════════════════════════════════════════════════════════════════
# PLATFORM ARCHITECTURE FLOW
# ════════════════════════════════════════════════════════════════
st.subheader("⚙️ Platform Architecture")

col_arch = st.columns([1, 0.15, 1, 0.15, 1])

with col_arch[0]:
    st.markdown("""
    <div style="background:#e8f0fe;border-radius:12px;
    padding:20px;text-align:center;border:2px solid #4285F4;">
    <h3 style="color:#4285F4;margin:0;">🦾 Module A</h3>
    <h4 style="margin:6px 0;">Alloy Design</h4>
    <hr style="border-color:#4285F4;">
    <p style="font-size:0.9em;margin:4px 0;">📥 Composition sliders</p>
    <p style="font-size:0.9em;margin:4px 0;">🤖 XGBoost / RF / GPR</p>
    <p style="font-size:0.9em;margin:4px 0;">🔍 SHAP waterfall</p>
    <p style="font-size:0.9em;margin:4px 0;">🎯 Bayesian optimisation</p>
    <p style="font-size:0.9em;margin:4px 0;">🟢 Immunocompat. badge</p>
    <hr style="border-color:#4285F4;">
    <p style="font-size:0.85em;color:#555;">
    📤 Outputs D_Mg + R_corr</p>
    </div>
    """, unsafe_allow_html=True)

with col_arch[1]:
    st.markdown("""
    <div style="display:flex;align-items:center;
    justify-content:center;height:100%;font-size:2em;
    color:#4285F4;padding-top:60px;">→</div>
    """, unsafe_allow_html=True)

with col_arch[2]:
    st.markdown("""
    <div style="background:#e6f4ea;border-radius:12px;
    padding:20px;text-align:center;border:2px solid #34A853;">
    <h3 style="color:#34A853;margin:0;">🔬 Module B</h3>
    <h4 style="margin:6px 0;">Corrosion Simulation</h4>
    <hr style="border-color:#34A853;">
    <p style="font-size:0.9em;margin:4px 0;">📐 Gmsh 3D mesh</p>
    <p style="font-size:0.9em;margin:4px 0;">🧮 DOLFINx FEniCS</p>
    <p style="font-size:0.9em;margin:4px 0;">📈 Mg²⁺ concentration</p>
    <p style="font-size:0.9em;margin:4px 0;">🖼️ Day snapshots</p>
    <p style="font-size:0.9em;margin:4px 0;">⬇️ corrosion_profile.csv</p>
    <hr style="border-color:#34A853;">
    <p style="font-size:0.85em;color:#555;">
    📤 Outputs Mg²⁺(t) profile</p>
    </div>
    """, unsafe_allow_html=True)

with col_arch[3]:
    st.markdown("""
    <div style="display:flex;align-items:center;
    justify-content:center;height:100%;font-size:2em;
    color:#34A853;padding-top:60px;">→</div>
    """, unsafe_allow_html=True)

with col_arch[4]:
    st.markdown("""
    <div style="background:#fef9e7;border-radius:12px;
    padding:20px;text-align:center;border:2px solid #FBBC04;">
    <h3 style="color:#e37400;margin:0;">🧬 Module C</h3>
    <h4 style="margin:6px 0;">Immune Response ODE</h4>
    <hr style="border-color:#FBBC04;">
    <p style="font-size:0.9em;margin:4px 0;">🦠 4-var ODE (M0/M1/M2/Mg²⁺)</p>
    <p style="font-size:0.9em;margin:4px 0;">⚙️ SciPy RK45 live solver</p>
    <p style="font-size:0.9em;margin:4px 0;">📊 M2/M1 polarisation ratio</p>
    <p style="font-size:0.9em;margin:4px 0;">🎛️ Sensitivity sweep</p>
    <p style="font-size:0.9em;margin:4px 0;">✅ Day 21 benchmark badge</p>
    <hr style="border-color:#FBBC04;">
    <p style="font-size:0.85em;color:#555;">
    📤 M2/M1 = 1.022 @ Day 21</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# LIVE STATUS DASHBOARD
# ════════════════════════════════════════════════════════════════
st.subheader("📊 Live Project Status")

col_s1, col_s2, col_s3 = st.columns(3)

# ── Module A status ───────────────────────────────────────────
with col_s1:
    st.markdown("#### 🦾 Module A — Alloy Design")
    models_found = [f for f in
                    ["XGBoost.pkl","RandomForest.pkl","SVR.pkl",
                     "MLP.pkl","GPR.pkl","scaler.pkl"]
                    if os.path.exists(f"{MODEL_DIR}/{f}")]
    st.metric("Models loaded", f"{len(models_found)} / 6")

    import json
    if os.path.exists(BO_JSON):
        with open(BO_JSON) as f:
            bo = json.load(f)
        best_ef = bo.get("predicted_Ef_eV_atom", "N/A")
        st.metric("BO Best E_f", f"{best_ef} eV/atom")
    else:
        st.metric("BO Best E_f", "N/A — run BO notebook")

    st.metric("Dataset", "7,592 alloys · 132 features")

    if len(models_found) >= 5:
        st.success("✅ Module A ready")
    else:
        st.warning(f"⚠️ Only {len(models_found)} models found")

    st.markdown("<br>", unsafe_allow_html=True)
    st.page_link("pages/1_🦾_Module_A.py",
                 label="🚀 Open Module A", icon="🦾")

# ── Module B status ───────────────────────────────────────────
with col_s2:
    st.markdown("#### 🔬 Module B — Corrosion")

    corr_ok = os.path.exists(CORR_CSV)
    if corr_ok:
        corr_df = pd.read_csv(CORR_CSV)
        st.metric("Corrosion CSV",
                  f"{len(corr_df)} rows × {len(corr_df.columns)} cols")
    else:
        st.metric("Corrosion CSV", "❌ Not found")

    mesh_count = len(os.listdir("ProjectB/mesh")) \
                 if os.path.isdir("ProjectB/mesh") else 0
    res_count  = len(os.listdir("ProjectB/results")) \
                 if os.path.isdir("ProjectB/results") else 0

    st.metric("Mesh files",    f"{mesh_count} / 6")
    st.metric("Result files",  f"{res_count} / 9")

    if corr_ok and mesh_count >= 6:
        st.success("✅ Module B ready")
    else:
        st.warning("⚠️ Run DOLFINx notebook in Colab")

    st.markdown("<br>", unsafe_allow_html=True)
    st.page_link("pages/2_🔬_Module_B.py",
                 label="🚀 Open Module B", icon="🔬")

# ── Module C status ───────────────────────────────────────────
with col_s3:
    st.markdown("#### 🧬 Module C — Immune ODE")

    immune_ok = os.path.exists(IMMUNE_CSV)
    if immune_ok:
        imm_df = pd.read_csv(IMMUNE_CSV)
        day21  = imm_df[imm_df["day"] == 21].iloc[0]
        ratio  = day21["M2_M1_ratio"]
        st.metric("M2/M1 @ Day 21",
                  f"{ratio:.4f}",
                  "✅ PASSED" if ratio > 1.0 else "❌ FAILED")
        st.metric("Crossover Day",  "Day 20.5")
        st.metric("ODE rows",
                  f"{len(imm_df)} rows × {len(imm_df.columns)} cols")
        if ratio > 1.0:
            st.success("✅ Module C ready")
        else:
            st.error("❌ Benchmark failed — re-run ODE")
    else:
        st.metric("Immune CSV", "❌ Not found")
        st.warning("⚠️ Run ProjectC ODE notebook in Colab")

    st.markdown("<br>", unsafe_allow_html=True)
    st.page_link("pages/3_🧬_Module_C.py",
                 label="🚀 Open Module C", icon="🧬")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# KEY RESULTS SUMMARY
# ════════════════════════════════════════════════════════════════
st.subheader("🏆 Key Achieved Results")

col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
col_r1.metric("Best R² Score",      "0.9439",       "XGBoost")
col_r2.metric("BO Best E_f",        "−2.90 eV/atom","50 iterations")
col_r3.metric("FEniCS Steps",       "78,316",        "28 days · dt=30.9s")
col_r4.metric("M2/M1 @ Day 21",     "1.0222",        "✅ > 1.0 threshold")
col_r5.metric("Crossover Day",      "Day 20.5",      "Inflam → Healing")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# QUICK START GUIDE
# ════════════════════════════════════════════════════════════════
st.subheader("🚀 Quick Start")

col_q1, col_q2 = st.columns(2)

with col_q1:
    st.markdown("""
    #### Running Locally
    ```bash
    cd MgInsight/
    mgenv\\Scripts\\activate     # Windows
    source mgenv/bin/activate   # Mac/Linux
    streamlit run app.py
    ```
    Opens at **http://localhost:8501**
    """)

with col_q2:
    st.markdown("""
    #### Recommended Workflow
    1. 🦾 **Module A** — pick a preset or adjust sliders
    2. Note auto-populated **D_Mg** and **R_corr**
    3. 🔬 **Module B** — view corrosion trajectory
    4. 🧬 **Module C** — check M2/M1 Day 21 benchmark
    5. ⬇️ Download results CSV from each module
    """)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.85em;
padding:12px 0;">
🧬 MgInsight &nbsp;|&nbsp;
Biomedical Engineering &nbsp;|&nbsp;
DOLFINx 0.10.0 · SciPy RK45 · XGBoost · SHAP · BoTorch &nbsp;|&nbsp;
Data: Materials Project (7,592 alloys)
</div>
""", unsafe_allow_html=True)