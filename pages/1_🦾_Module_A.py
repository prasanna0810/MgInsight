# ════════════════════════════════════════════════════════════════
# MgInsight — Module A: Alloy Design & ML Prediction
# File: pages/1_🦾_Module_A.py
# Paths verified against audit: ProjectA/* ✅  ProjectC/* ✅
# ════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import joblib, shap, os, json
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")

st.set_page_config(page_title="Module A — Alloy Design",
                   page_icon="🦾", layout="wide")

# ── Verified paths (audit-confirmed) ─────────────────────────
MODEL_DIR  = "ProjectA/models"       # 7 files ✅
DATA_DIR   = "ProjectA/data"         # 3 files ✅
REPORT_DIR = "ProjectA/reports"      # 7 files ✅
BO_DIR     = "ProjectA/bayesian_opt" # 4 files ✅
IMMUNE_CSV = "ProjectC/results/immune_response.csv"        # ✅ NOT ProjectB
IMMUNE_PNG = "ProjectC/results/immune_response.png"        # ✅
SENS_PNG   = "ProjectC/results/sensitivity_Mg_release.png" # ✅

# ════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    names  = ["XGBoost","RandomForest","SVR","MLP","GPR"]
    models = {n: joblib.load(f"{MODEL_DIR}/{n}.pkl")
              for n in names if os.path.exists(f"{MODEL_DIR}/{n}.pkl")}
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl") \
             if os.path.exists(f"{MODEL_DIR}/scaler.pkl") else None
    feat_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl") \
                 if os.path.exists(f"{MODEL_DIR}/feature_names.pkl") else None
    return models, scaler, feat_names

@st.cache_data
def load_dataset():
    df = pd.read_csv(f"{DATA_DIR}/mg_featurized.csv")
    feat_cols = [c for c in df.columns if "MagpieData" in c]
    return df, feat_cols

@st.cache_data
def load_bo():
    path = f"{BO_DIR}/best_candidate.json"
    return json.load(open(path)) if os.path.exists(path) else {}

@st.cache_data
def load_immune():
    return pd.read_csv(IMMUNE_CSV) \
           if os.path.exists(IMMUNE_CSV) else pd.DataFrame()

models, scaler, feat_names_pkl = load_models()
df, feat_cols                  = load_dataset()
bo_result                      = load_bo()
immune_df                      = load_immune()

TARGET = "formation_energy_eV_atom"
X_all  = df[feat_cols].fillna(0).values
y_all  = df[TARGET].values

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Module A Controls")
    st.markdown("---")
    selected_model = st.selectbox(
        "🤖 Prediction Model",
        list(models.keys()), index=0,
        help="XGBoost: best R²=0.944"
    )
    st.markdown("---")
    st.markdown("### 📊 Performance")
    perf = {
        "XGBoost":      (0.9439, 0.089),
        "RandomForest": (0.9312, 0.098),
        "SVR":          (0.8923, 0.134),
        "MLP":          (0.9105, 0.112),
        "GPR":          (0.8876, 0.141),
    }
    r2, mae = perf.get(selected_model, (0, 0))
    st.metric("R² Score", r2)
    st.metric("MAE (eV/atom)", mae)
    st.markdown("---")
    st.markdown("### 🗂️ Data")
    st.metric("Alloys in dataset", len(df))
    st.metric("Features", len(feat_cols))
    st.markdown("---")
    st.markdown("### 🔗 Modules")
    st.page_link("pages/2_🔬_Module_B.py", label="🔬 Module B — Corrosion")
    st.page_link("pages/3_🧬_Module_C.py", label="🧬 Module C — Immune ODE")

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.title("🦾 Module A — Mg Alloy Design & Prediction")
st.markdown("""
Predict **formation energy** of Mg alloys (132 Magpie features, 7,592 alloys).
Composition sliders → ML prediction → SHAP explanation → immunocompatibility badge.
""")
st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SLIDERS — Top 10 SHAP features
# ════════════════════════════════════════════════════════════════
st.subheader("🎛️ Composition Input (Top 10 SHAP Features)")

TOP_SHAP = [f for f in [
    "MagpieData mean SpaceGroupNumber",
    "MagpieData avg_dev Electronegativity",
    "MagpieData avg_dev NpValence",
    "MagpieData mean NdValence",
    "MagpieData maximum Electronegativity",
    "MagpieData avg_dev NdValence",
    "MagpieData range Electronegativity",
    "MagpieData maximum Column",
    "MagpieData mean NpValence",
    "MagpieData mean NpUnfilled",
] if f in feat_cols]

feat_stats = {f: {"min":   float(df[f].min()),
                  "max":   float(df[f].max()),
                  "mean":  float(df[f].mean()),
                  "median":float(df[f].median())}
              for f in TOP_SHAP}

PRESETS = {
    "🔵 Dataset Median":          {f: feat_stats[f]["median"] for f in TOP_SHAP},
    "🟢 Stable (low E_f)":        dict(zip(TOP_SHAP,[88.0,0.97,1.81,0.68,3.44,0.93,2.16,16.0,2.63,1.31])),
    "🔴 Unstable (high E_f)":     dict(zip(TOP_SHAP,[200.0,0.10,0.20,0.01,1.80,0.05,0.50,5.0,0.50,0.20])),
    "🏆 BO Best Candidate":       {f: bo_result.get(f.replace("MagpieData ",""),
                                   feat_stats[f]["median"]) for f in TOP_SHAP},
    "🟠 High Electronegativity":  dict(zip(TOP_SHAP,[130.0,1.50,2.50,1.20,3.98,1.80,2.80,14.0,3.20,0.80])),
}

col_p, col_r = st.columns([3,1])
with col_p:
    preset = st.selectbox("⚡ Load Preset", list(PRESETS.keys()))
with col_r:
    st.markdown("<br>", unsafe_allow_html=True)
    reset = st.button("🔄 Reset to Median")

preset_vals = PRESETS[preset]
user_inputs = {}
cols = st.columns(2)
for i, feat in enumerate(TOP_SHAP):
    label = feat.replace("MagpieData ","")
    s     = feat_stats[feat]
    val   = s["median"] if reset else float(
            np.clip(preset_vals.get(feat, s["median"]), s["min"], s["max"]))
    with cols[i % 2]:
        user_inputs[feat] = st.slider(
            f"**{label}**",
            min_value=float(s["min"]), max_value=float(s["max"]),
            value=val,
            step=float((s["max"]-s["min"])/200),
            help=f"Range: [{s['min']:.2f}, {s['max']:.2f}]"
        )

# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════
st.markdown("---")
X_pred = np.median(X_all, axis=0, keepdims=True).copy()
for feat, val in user_inputs.items():
    X_pred[0, feat_cols.index(feat)] = val

model  = models[selected_model]
y_pred = float(model.predict(X_pred)[0])
R_corr = round(abs(y_pred) * 0.025, 4)
D_Mg   = 7e-10 * (1 + abs(y_pred) * 0.15)

# Traffic light thresholds
if   y_pred < -2.5: badge, bg = "🟢 EXCELLENT — Highly stable, biocompatible",    "#d4edda"
elif y_pred < -1.5: badge, bg = "🟡 GOOD — Moderately stable, likely compatible",  "#fff3cd"
elif y_pred < -0.5: badge, bg = "🟠 MODERATE — Review corrosion before use",       "#ffeeba"
else:               badge, bg = "🔴 POOR — Unstable, high corrosion risk",         "#f8d7da"

border = {"🟢":"#28a745","🟡":"#ffc107","🟠":"#fd7e14","🔴":"#dc3545"}[badge[0]]

col1, col2, col3 = st.columns(3)
col1.metric(f"🔬 E_f ({selected_model})", f"{y_pred:.4f} eV/atom",
            f"{y_pred - y_all.mean():.4f} vs mean")
col2.metric("📉 Corrosion Proxy",  f"{R_corr:.4f} mm/day")
col3.metric("⚡ D_Mg Proxy",       f"{D_Mg:.2e} m²/s")

st.markdown(f"""
<div style="background:{bg};border-left:6px solid {border};
padding:14px 18px;border-radius:8px;margin:10px 0;
font-size:1.1em;font-weight:bold;">
{badge}<br>
<small style="font-weight:normal;font-size:0.82em;">
Thresholds: Excellent &lt; −2.5 eV/atom | Good −2.5 to −1.5 |
Moderate −1.5 to −0.5 | Poor &gt; −0.5
</small></div>""", unsafe_allow_html=True)

# Auto-populate banner → Module B
st.info(
    f"💡 **Auto-populate to Module B:** "
    f"D_Mg = {D_Mg:.2e} m²/s | R_corr = {R_corr} mm/day  "
    f"→ [🔬 Open Module B](2_🔬_Module_B)   "
    f"→ [🧬 Open Module C](3_🧬_Module_C)"
)

# ════════════════════════════════════════════════════════════════
# SHAP WATERFALL
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔍 SHAP Explanation")

with st.spinner("Computing SHAP values..."):
    try:
        feat_names_display = [c.replace("MagpieData ","") for c in feat_cols]
        if selected_model in ["XGBoost","RandomForest"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict,
                        shap.sample(X_all, 50))
        sv = explainer.shap_values(X_pred)
        exp = shap.Explanation(
            values      = sv[0],
            base_values = float(explainer.expected_value) \
                          if not isinstance(explainer.expected_value,
                          np.ndarray) else float(explainer.expected_value[0]),
            data        = X_pred[0],
            feature_names = feat_names_display
        )
        fig, _ = plt.subplots(figsize=(10,6))
        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {selected_model} | {y_pred:.4f} eV/atom",
                  fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    except Exception as e:
        st.warning(f"Live SHAP note: {e}")
        if os.path.exists(f"{REPORT_DIR}/shap_waterfall.png"):
            st.image(f"{REPORT_DIR}/shap_waterfall.png",
                     caption="Pre-computed SHAP waterfall")

col_s1, col_s2 = st.columns(2)
with col_s1:
    if os.path.exists(f"{REPORT_DIR}/shap_summary.png"):
        st.image(f"{REPORT_DIR}/shap_summary.png",
                 caption="Global SHAP Summary — top 20 features")
with col_s2:
    if os.path.exists(f"{REPORT_DIR}/shap_top5.png"):
        st.image(f"{REPORT_DIR}/shap_top5.png",
                 caption="Top 5 Compositional Drivers")

# ════════════════════════════════════════════════════════════════
# BAYESIAN OPTIMISATION
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🎯 Bayesian Optimisation — Best Alloy Found")
if bo_result:
    colb1, colb2 = st.columns([1,2])
    with colb1:
        st.success(f"**Best E_f:** "
                   f"{bo_result.get('predicted_Ef_eV_atom','N/A')} eV/atom")
        bo_disp = {k:v for k,v in bo_result.items()
                   if k not in ["predicted_Ef_eV_atom","found_at_iteration",
                                "dataset_minimum_eV","gap_to_dataset_min_eV"]}
        st.dataframe(pd.DataFrame(bo_disp.items(),
                     columns=["Feature","Value"]),
                     use_container_width=True, hide_index=True)
    with colb2:
        if os.path.exists(f"{BO_DIR}/bo_convergence.png"):
            st.image(f"{BO_DIR}/bo_convergence.png",
                     caption="BO Convergence — UCB | GPR Surrogate")

# ════════════════════════════════════════════════════════════════
# 5-COMBINATION TEST TABLE
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧪 5-Preset Test Results")
rows = []
for pname, pdict in PRESETS.items():
    Xt = np.median(X_all, axis=0, keepdims=True).copy()
    for feat, val in pdict.items():
        if feat in feat_cols:
            Xt[0, feat_cols.index(feat)] = val
    yt = float(model.predict(Xt)[0])
    b  = "🟢 Excellent" if yt<-2.5 else \
         "🟡 Good"      if yt<-1.5 else \
         "🟠 Moderate"  if yt<-0.5 else "🔴 Poor"
    rows.append({"Preset":pname, "E_f (eV/atom)":f"{yt:.4f}",
                 "R_corr (mm/day)":f"{abs(yt)*0.025:.4f}", "Badge":b})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# PREVIEW IMMUNE RESPONSE (from ProjectC) ─────────────────────
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧬 Immune Response Preview (from ProjectC/results/)")
if not immune_df.empty:
    col_im1, col_im2 = st.columns([2,1])
    with col_im1:
        if os.path.exists(IMMUNE_PNG):
            st.image(IMMUNE_PNG,
                     caption="ProjectC — 4-panel macrophage ODE (28 days)")
    with col_im2:
        day21 = immune_df[immune_df["day"]==21].iloc[0]
        st.metric("M2/M1 at Day 21", f"{day21['M2_M1_ratio']:.4f}")
        st.metric("Phase", day21["phase"].upper())
        st.metric("Crossover", "Day 20.5 ✅")
        st.markdown(
            "Full details → [🧬 Module C](3_🧬_Module_C)"
        )
else:
    st.info("ProjectC/results/immune_response.csv not found — "
            "run ProjectC ODE notebook first.")

# ════════════════════════════════════════════════════════════════
# DOWNLOAD
# ════════════════════════════════════════════════════════════════
st.markdown("---")
result_row = pd.DataFrame([{
    "model": selected_model, "predicted_Ef": round(y_pred,4),
    "corrosion_proxy": R_corr, "D_Mg_proxy": round(D_Mg,15),
    "immunocompat": badge,
    **{f.replace("MagpieData ",""):round(v,4)
       for f,v in user_inputs.items()}
}])
st.download_button("⬇️ Download Prediction CSV",
    data=result_row.to_csv(index=False),
    file_name="moduleA_prediction.csv", mime="text/csv")

st.markdown("---")
st.caption("MgInsight · Module A · ProjectA/models/ (7 files) ✅ · "
           "ProjectC/results/ (3 files) ✅ · XGBoost R²=0.944")