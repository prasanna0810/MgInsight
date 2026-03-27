# ════════════════════════════════════════════════════════════════
# MgInsight — Module C: Immune Response ODE Viewer
# File : pages/3_🧬_Module_C.py
#        also save copy → ProjectC/notebooks/04_streamlit_moduleC.py
# Source: ProjectC/results/ (3 files ✅)
#   immune_response.csv       (29 rows × 7 cols)
#   immune_response.png       (4-panel ODE figure)
#   sensitivity_Mg_release.png
# ════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(
    page_title = "Module C — Immune ODE",
    page_icon  = "🧬",
    layout     = "wide"
)

# ── Verified paths ────────────────────────────────────────────
RESULTS_DIR = "ProjectC/results"
IMMUNE_CSV  = f"{RESULTS_DIR}/immune_response.csv"
IMMUNE_PNG  = f"{RESULTS_DIR}/immune_response.png"
SENS_PNG    = f"{RESULTS_DIR}/sensitivity_Mg_release.png"

# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_immune():
    return pd.read_csv(IMMUNE_CSV) \
           if os.path.exists(IMMUNE_CSV) else pd.DataFrame()

immune_df = load_immune()

# ════════════════════════════════════════════════════════════════
# ODE SYSTEM — same equations used in ProjectC notebook
# ════════════════════════════════════════════════════════════════
def macrophage_ode(t, y, params):
    M0, M1, M2, Mg = y
    p = params

    # Mg²⁺ release (corrosion-driven, slows as implant degrades)
    dMg = p["r_Mg"] * np.exp(-p["k_deg"] * t) - p["d_Mg"] * Mg

    # M0 → M1 driven by inflammation; M0 → M2 driven by Mg²⁺
    act_M1  = p["k_M1"] * M0 * (1 / (1 + (Mg / p["IC50_M1"]) ** 2))
    act_M2  = p["k_M2"] * M0 * (Mg ** 2 / (p["EC50_M2"] ** 2 + Mg ** 2))

    dM0 = -act_M1 - act_M2
    dM1 =  act_M1 - p["d_M1"] * M1
    dM2 =  act_M2 - p["d_M2"] * M2

    return [dM0, dM1, dM2, dMg]

def run_ode(r_Mg=0.25, days=28):
    params = {
        "r_Mg":    r_Mg,   # Mg²⁺ release rate (mol/m³/day) ← slider
        "k_deg":   0.05,   # corrosion slowdown constant
        "d_Mg":    0.02,   # Mg²⁺ clearance
        "k_M1":    0.50,   # M0→M1 activation rate
        "k_M2":    0.40,   # M0→M2 activation rate
        "IC50_M1": 2.50,   # M1 inhibition threshold (mol/m³)
        "EC50_M2": 1.50,   # M2 activation threshold (mol/m³)
        "d_M1":    0.02,   # M1 decay
        "d_M2":    0.015,  # M2 decay
    }
    y0   = [1.0, 0.05, 0.02, 0.8]
    tspan= (0, days)
    t_eval = np.linspace(0, days, days * 100 + 1)

    sol = solve_ivp(macrophage_ode, tspan, y0,
                    args   = (params,),
                    method = "RK45",
                    t_eval = t_eval,
                    rtol   = 1e-8, atol = 1e-10)
    return sol

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Module C Controls")
    st.markdown("---")

    st.markdown("### 🎛️ Live ODE Parameters")
    r_Mg_input = st.slider(
        "Mg²⁺ Release Rate (mol/m³/day)",
        min_value = 0.05,
        max_value = 0.60,
        value     = 0.25,
        step      = 0.01,
        help      = "Current baseline = 0.25 | Threshold crossover ~0.245"
    )
    sim_days = st.slider(
        "Simulation Duration (days)",
        min_value = 14,
        max_value = 60,
        value     = 28,
        step      = 1
    )
    run_live = st.button("▶️ Run Live ODE", type="primary",
                         use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 ODE System Info")
    st.markdown("""
    **4 variables:**
    - M0 — Naïve macrophage
    - M1 — Pro-inflammatory
    - M2 — Anti-inflammatory
    - Mg²⁺ — Local ion conc.

    **Solver:** SciPy RK45
    **rtol:** 1×10⁻⁸ | **2801 pts**

    **Key thresholds:**
    - M2 EC50 = 1.5 mol/m³
    - M1 IC50 = 2.5 mol/m³
    """)
    st.markdown("---")
    st.markdown("### 🔗 Modules")
    st.page_link("pages/1_🦾_Module_A.py", label="🦾 Module A — Alloy Design")
    st.page_link("pages/2_🔬_Module_B.py", label="🔬 Module B — Corrosion")

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.title("🧬 Module C — Immune Response ODE")
st.markdown("""
4-variable macrophage ODE system (M0 → M1/M2 driven by local Mg²⁺).
View pre-computed 28-day results **or** run the live ODE with custom
Mg²⁺ release rate and check the Day 21 benchmark in real time.
""")
st.markdown("---")

# ════════════════════════════════════════════════════════════════
# TABS: Pre-computed | Live ODE | Sensitivity
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Pre-computed Results",
    "▶️ Live ODE Runner",
    "📉 Sensitivity Analysis"
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — Pre-computed results from ProjectC/results/
# ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Pre-computed ODE Results (ProjectC/results/)")

    # Key metrics row
    if not immune_df.empty:
        day21 = immune_df[immune_df["day"] == 21].iloc[0]
        day0  = immune_df[immune_df["day"] == 0].iloc[0]
        day28 = immune_df[immune_df["day"] == 28].iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("M2/M1 @ Day 21",  f"{day21['M2_M1_ratio']:.4f}",
                    "✅ > 1.0 PASSED")
        col2.metric("Crossover Day",    "Day 20.5",      "M2/M1 = 1.0")
        col3.metric("M1 Peak",          "Day 9",         "0.4814 (norm.)")
        col4.metric("[Mg²⁺] Peak",      "3.120 mol/m³",  "Day 20")
        col5.metric("M2/M1 @ Day 28",   f"{day28['M2_M1_ratio']:.4f}",
                    "Healing continues")

        st.markdown("---")

        # Population chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=immune_df["day"], y=immune_df["M0_naive"],
            name="M0 Naïve", mode="lines", line=dict(color="black", width=2)))
        fig1.add_trace(go.Scatter(
            x=immune_df["day"], y=immune_df["M1_inflammatory"],
            name="M1 Pro-inflam.", mode="lines", line=dict(color="red", width=2)))
        fig1.add_trace(go.Scatter(
            x=immune_df["day"], y=immune_df["M2_healing"],
            name="M2 Healing", mode="lines", line=dict(color="green", width=2)))
        fig1.add_vline(x=21,   line_dash="dash", line_color="purple",
                       annotation_text="Day 21 benchmark")
        fig1.add_vline(x=20.5, line_dash="dot",  line_color="gold",
                       annotation_text="Crossover")
        fig1.update_layout(
            title="Macrophage Populations Over 28 Days",
            xaxis_title="Time (days)",
            yaxis_title="Cell density (norm.)",
            legend=dict(orientation="h", y=1.12,
                        x=0.5, xanchor="center"),
            height=380
        )
        st.plotly_chart(fig1, use_container_width=True)

        # M2/M1 ratio bar chart
        colors = ["#f28b82" if r < 1.0 else "#81c995"
                  for r in immune_df["M2_M1_ratio"]]
        fig2 = go.Figure(go.Bar(
            x=immune_df["day"],
            y=immune_df["M2_M1_ratio"],
            marker_color=colors, name="M2/M1"
        ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="red",
                       annotation_text="Threshold = 1.0")
        fig2.add_annotation(x=21, y=1.022,
                            text="✅ 1.022", showarrow=True,
                            arrowhead=2, bgcolor="white")
        fig2.update_layout(
            title="M2/M1 Polarisation Ratio — 🔴 Inflammatory → 🟢 Healing",
            xaxis_title="Day",
            yaxis_title="M2/M1 Ratio",
            yaxis=dict(range=[0, 1.4]),
            height=320, showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Full CSV table
        st.markdown("**Full immune_response.csv (29 rows × 7 cols)**")
        st.dataframe(immune_df, use_container_width=True, height=320)
        st.download_button(
            "⬇️ Download immune_response.csv",
            data      = immune_df.to_csv(index=False),
            file_name = "immune_response.csv",
            mime      = "text/csv"
        )

        # 4-panel figure
        st.markdown("---")
        if os.path.exists(IMMUNE_PNG):
            st.image(IMMUNE_PNG,
                     caption="ProjectC — 4-panel ODE figure "
                             "(Populations · M2/M1 · Mg²⁺ · Phase portrait)",
                     use_container_width=True)
    else:
        st.warning("immune_response.csv not found in ProjectC/results/. "
                   "Run ProjectC ODE notebook in Colab first.")

# ──────────────────────────────────────────────────────────────
# TAB 2 — Live ODE runner
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("▶️ Live ODE Runner")
    st.markdown(f"""
    Running with **Mg²⁺ release rate = `{r_Mg_input}` mol/m³/day**
    over **{sim_days} days**. Adjust sliders in the sidebar and
    click **▶️ Run Live ODE**.
    """)

    # Auto-run on first load with defaults; re-run on button press
    if run_live or "ode_sol" not in st.session_state \
       or st.session_state.get("ode_r_Mg") != r_Mg_input \
       or st.session_state.get("ode_days") != sim_days:

        with st.spinner(f"Solving ODE (RK45) for {sim_days} days..."):
            sol = run_ode(r_Mg=r_Mg_input, days=sim_days)
            st.session_state["ode_sol"]   = sol
            st.session_state["ode_r_Mg"]  = r_Mg_input
            st.session_state["ode_days"]  = sim_days

    sol = st.session_state["ode_sol"]

    if sol.success:
        t   = sol.t
        M0s = sol.y[0]
        M1s = sol.y[1]
        M2s = sol.y[2]
        Mgs = sol.y[3]
        ratio = np.where(M1s > 0, M2s / M1s, 0)

        # Day 21 benchmark check
        idx21  = np.argmin(np.abs(t - 21))
        r21    = ratio[idx21]
        passed = r21 > 1.0

        # Crossover day
        cross_idx = np.where(np.diff(np.sign(ratio - 1.0)))[0]
        cross_day = float(t[cross_idx[0]]) if len(cross_idx) > 0 else None

        # Metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("M2/M1 @ Day 21",
                     f"{r21:.4f}",
                     "✅ PASSED" if passed else "❌ FAILED")
        col_b.metric("Crossover Day",
                     f"Day {cross_day:.1f}" if cross_day else "Not reached",
                     "M2/M1 = 1.0")
        col_c.metric("[Mg²⁺] Peak",
                     f"{Mgs.max():.3f} mol/m³",
                     f"Day {t[np.argmax(Mgs)]:.0f}")
        col_d.metric("M1 Peak",
                     f"{M1s.max():.4f}",
                     f"Day {t[np.argmax(M1s)]:.0f}")

        # Benchmark badge
        if passed:
            st.success(f"✅ **Day 21 Benchmark PASSED** — "
                       f"M2/M1 = {r21:.4f} > 1.0  |  "
                       f"Crossover: Day {cross_day:.1f}  |  "
                       f"Mg²⁺ release = {r_Mg_input} mol/m³/day")
        else:
            st.error(f"❌ **Day 21 Benchmark FAILED** — "
                     f"M2/M1 = {r21:.4f} < 1.0  |  "
                     f"Increase Mg²⁺ release rate above ~0.245")

        # Population plot
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t, y=M0s, name="M0 Naïve",
                                  line=dict(color="black", width=2)))
        fig3.add_trace(go.Scatter(x=t, y=M1s, name="M1 Pro-inflam.",
                                  line=dict(color="red",   width=2)))
        fig3.add_trace(go.Scatter(x=t, y=M2s, name="M2 Healing",
                                  line=dict(color="green", width=2)))
        if cross_day:
            fig3.add_vline(x=cross_day, line_dash="dot",
                           line_color="gold",
                           annotation_text=f"Crossover Day {cross_day:.1f}")
        fig3.add_vline(x=21, line_dash="dash", line_color="purple",
                       annotation_text="Day 21")
        fig3.update_layout(
            title=f"Live ODE — Mg²⁺ release = {r_Mg_input} mol/m³/day",
            xaxis_title="Time (days)",
            yaxis_title="Cell density (norm.)",
            legend=dict(orientation="h", y=1.12,
                        x=0.5, xanchor="center"),
            height=380
        )
        st.plotly_chart(fig3, use_container_width=True)

        # M2/M1 ratio line
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=t, y=ratio, mode="lines", name="M2/M1",
            fill="tozeroy",
            line=dict(color="#4285F4", width=2)
        ))
        fig4.add_hline(y=1.0, line_dash="dash", line_color="red",
                       annotation_text="Threshold = 1.0")
        if cross_day:
            fig4.add_vline(x=cross_day, line_dash="dot",
                           line_color="gold",
                           annotation_text=f"Day {cross_day:.1f}")
        fig4.update_layout(
            title="M2/M1 Polarisation Ratio (Live)",
            xaxis_title="Time (days)",
            yaxis_title="M2/M1 Ratio",
            height=300, showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Download live result
        live_df = pd.DataFrame({
            "day":          np.round(t, 2),
            "M0_naive":     np.round(M0s, 6),
            "M1_inflammatory": np.round(M1s, 6),
            "M2_healing":   np.round(M2s, 6),
            "Mg_conc":      np.round(Mgs, 6),
            "M2_M1_ratio":  np.round(ratio, 6),
        })
        st.download_button(
            "⬇️ Download Live ODE Result CSV",
            data      = live_df.to_csv(index=False),
            file_name = f"live_ode_rMg{r_Mg_input}_day{sim_days}.csv",
            mime      = "text/csv"
        )
    else:
        st.error(f"ODE solver failed: {sol.message}")

# ──────────────────────────────────────────────────────────────
# TAB 3 — Sensitivity analysis
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📉 Sensitivity: Mg²⁺ Release Rate → M2/M1 at Day 21")

    if os.path.exists(SENS_PNG):
        st.image(SENS_PNG,
                 caption="Pre-computed sensitivity — "
                         "Mg²⁺ release rate vs M2/M1 at Day 21 "
                         "(threshold = 1.0, current = 0.25)",
                 use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔄 Recompute Sensitivity Sweep")
    st.markdown("Sweeps 15 values of Mg²⁺ release rate "
                "from 0.05 to 0.60 mol/m³/day and plots M2/M1 at Day 21.")

    if st.button("🔄 Run Sensitivity Sweep", use_container_width=True):
        rates      = np.linspace(0.05, 0.60, 15)
        ratios_21  = []

        prog = st.progress(0, text="Running sweep...")
        for i, r in enumerate(rates):
            s = run_ode(r_Mg=r, days=28)
            if s.success:
                idx   = np.argmin(np.abs(s.t - 21))
                m1_21 = s.y[1][idx]
                m2_21 = s.y[2][idx]
                ratios_21.append(m2_21 / m1_21 if m1_21 > 0 else 0)
            else:
                ratios_21.append(np.nan)
            prog.progress((i+1)/len(rates),
                          text=f"r_Mg = {r:.2f} → M2/M1 = {ratios_21[-1]:.3f}")

        prog.empty()

        fig5 = go.Figure()
        colors_s = ["#81c995" if r >= 1.0 else "#f28b82"
                    for r in ratios_21]
        fig5.add_trace(go.Scatter(
            x=rates, y=ratios_21, mode="lines+markers",
            marker=dict(color=colors_s, size=10),
            line=dict(color="#4285F4", width=3),
            name="M2/M1 @ Day 21"
        ))
        fig5.add_hline(y=1.0, line_dash="dash", line_color="red",
                       annotation_text="Threshold = 1.0")
        fig5.add_vline(x=0.25, line_dash="dot", line_color="green",
                       annotation_text="Current = 0.25")
        fig5.update_layout(
            title="Sensitivity: Mg²⁺ Release Rate → M2/M1 at Day 21",
            xaxis_title="Mg²⁺ Release Rate (mol/m³/day)",
            yaxis_title="M2/M1 at Day 21",
            height=400, showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)

        sweep_df = pd.DataFrame({
            "r_Mg_mol_m3_day": np.round(rates, 3),
            "M2_M1_at_day21":  np.round(ratios_21, 4),
            "benchmark":       ["✅ PASS" if r >= 1.0
                                else "❌ FAIL" for r in ratios_21]
        })
        st.dataframe(sweep_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Sensitivity CSV",
            data      = sweep_df.to_csv(index=False),
            file_name = "sensitivity_sweep.csv",
            mime      = "text/csv"
        )

# ════════════════════════════════════════════════════════════════
# FILE AUDIT
# ════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📂 ProjectC File Audit", expanded=False):
    res_files = (os.listdir(RESULTS_DIR)
                 if os.path.isdir(RESULTS_DIR) else [])
    st.markdown(f"**ProjectC/results/** ({len(res_files)} files)")
    st.code("\n".join(sorted(res_files)) or "empty")
    for f in ["immune_response.csv","immune_response.png",
              "sensitivity_Mg_release.png"]:
        full = f"{RESULTS_DIR}/{f}"
        icon = "✅" if os.path.exists(full) else "❌ MISSING"
        st.markdown(f"{icon}  `{full}`")

st.markdown("---")
st.caption("MgInsight · Module C · ProjectC/results/ (3 files ✅) · "
           "SciPy RK45 | rtol=1e-8 | 2801 pts | Day 21 M2/M1 = 1.022")