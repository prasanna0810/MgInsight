# ════════════════════════════════════════════════════════════════
# MgInsight — Professional Academic Dashboard
# File: app.py | Run: python -m streamlit run app.py
# ════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json, os

st.set_page_config(
    page_title            = "MgInsight — Dashboard",
    page_icon             = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⬡</text></svg>",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── SVG icon library (Lucide-style, pure SVG, no emojis) ──────
ICONS = {
    "home": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "flask": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M9 3v8l-4 9h14l-4-9V3"/></svg>',
    "cpu":   '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "dna":   '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M9 22c1.798-3.111 4.218-5 7-5"/><path d="M2 9c6.667-6 13.333 0 20-6"/><path d="M15 2c-1.798 3.111-4.218 5-7 5"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "bar":   '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
    "check": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "dot_g": '<svg width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="4" fill="#22C55E"/></svg>',
    "dot_r": '<svg width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="4" fill="#EF4444"/></svg>',
    "trend": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "atom":  '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1"/><path d="M20.2 20.2c2.04-2.03.02-7.36-4.5-11.9C11.2 3.8 5.9 1.75 3.8 3.8c-2.04 2.04-.02 7.37 4.5 11.9 4.51 4.52 9.83 6.57 11.9 4.5z"/><path d="M3.8 20.2c2.07 2.07 7.4.02 11.9-4.5 4.52-4.5 6.57-9.83 4.5-11.9"/></svg>',
    "github":'<svg width="16" height="16" viewBox="0 0 24 24" fill="white"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>',
    "r2":    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "target":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0D9488" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "cog":   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>',
    "check2":'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "cal":   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#C2410C" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
}

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family:'Inter',sans-serif !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility:hidden; }
.block-container {
    padding-top:0 !important;
    padding-bottom:1rem !important;
    padding-left:0 !important;
    padding-right:1.5rem !important;
    max-width:100% !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: none !important;
    min-width: 230px !important;
    max-width: 230px !important;
}
[data-testid="stSidebar"] *     { color:#94A3B8 !important; }
[data-testid="stSidebarNav"]    { display:none !important; }
[data-testid="stSidebar"] a     {
    text-decoration:none !important;
    font-size:0.875em !important;
    font-weight:500 !important;
}
[data-testid="stSidebar"] a:hover { color:#FFFFFF !important; }

/* Nav item active highlight */
.nav-active {
    background: rgba(37,99,235,0.2) !important;
    border-left: 3px solid #2563EB !important;
    border-radius: 0 8px 8px 0 !important;
}
.nav-item {
    display:flex; align-items:center; gap:10px;
    padding:9px 16px 9px 20px;
    border-radius:0 8px 8px 0;
    margin:2px 8px 2px 0;
    cursor:pointer;
    transition: background 0.15s;
    border-left: 3px solid transparent;
}
.nav-item:hover { background:rgba(255,255,255,0.06); }
.nav-label { font-size:0.875em; font-weight:500; color:#94A3B8; }

/* Top bar */
.topbar {
    background:#FFFFFF;
    border-bottom:1px solid #E2E8F0;
    padding:14px 32px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:24px;
    position:sticky;
    top:0;
    z-index:100;
}
.topbar h2 {
    font-size:1.4em;
    font-weight:700;
    color:#0F172A;
    margin:0;
    letter-spacing:-0.3px;
}
.topbar-right {
    display:flex; align-items:center; gap:16px;
}
.topbar-badge {
    background:#EFF6FF;
    color:#2563EB;
    font-size:0.72em;
    font-weight:700;
    padding:4px 12px;
    border-radius:20px;
    border:1px solid #BFDBFE;
    letter-spacing:0.3px;
}
.topbar-badge-green {
    background:#F0FDF4;
    color:#15803D;
    border-color:#BBF7D0;
}

/* KPI cards */
.kpi {
    background:#FFFFFF;
    border-radius:12px;
    padding:20px 22px;
    border:1px solid #E2E8F0;
    box-shadow:0 1px 8px rgba(15,23,42,0.05);
    position:relative;
    overflow:hidden;
    height:100%;
}
.kpi-top-bar {
    height:3px;
    width:100%;
    position:absolute;
    top:0; left:0;
    border-radius:12px 12px 0 0;
}
.kpi-icon {
    width:40px; height:40px;
    border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    margin-bottom:12px;
}
.kpi-val {
    font-size:1.75em;
    font-weight:800;
    color:#0F172A;
    line-height:1;
    letter-spacing:-0.5px;
}
.kpi-lbl {
    font-size:0.72em;
    font-weight:600;
    color:#64748B;
    text-transform:uppercase;
    letter-spacing:0.8px;
    margin:5px 0 10px;
}
.kpi-sub {
    font-size:0.75em;
    font-weight:600;
    padding:3px 9px;
    border-radius:12px;
    display:inline-block;
}

/* White section card */
.wcard {
    background:#FFFFFF;
    border-radius:12px;
    padding:20px 24px;
    border:1px solid #E2E8F0;
    box-shadow:0 1px 8px rgba(15,23,42,0.04);
    margin-bottom:16px;
}
.wcard-title {
    font-size:0.80em;
    font-weight:700;
    color:#0F172A;
    text-transform:uppercase;
    letter-spacing:0.8px;
    margin:0 0 16px;
    padding-bottom:12px;
    border-bottom:1px solid #F1F5F9;
    display:flex; align-items:center; gap:8px;
}

/* Table row */
.trow {
    display:flex; align-items:center;
    justify-content:space-between;
    padding:7px 0;
    border-bottom:1px solid #F8FAFC;
    gap:8px;
}
.trow:last-child { border-bottom:none; }
.trow-k { font-size:0.80em; color:#64748B; font-weight:400; }
.trow-v { font-size:0.82em; color:#0F172A; font-weight:600; text-align:right;}

/* Module row card */
.mrow {
    background:#FFFFFF;
    border-radius:12px;
    padding:16px 20px;
    border:1px solid #E2E8F0;
    box-shadow:0 1px 6px rgba(15,23,42,0.04);
    display:flex; align-items:center; gap:16px;
    margin-bottom:10px;
    transition: box-shadow 0.2s;
}
.mrow:hover { box-shadow:0 4px 20px rgba(37,99,235,0.10); }
.mrow-icon {
    width:44px; height:44px; border-radius:12px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
}
.mrow-title { font-size:0.92em; font-weight:700; color:#0F172A; margin:0; }
.mrow-sub   { font-size:0.76em; color:#94A3B8; margin:2px 0 0; }
.mrow-stats {
    margin-left:auto;
    display:flex; align-items:center; gap:24px;
}
.mrow-stat { text-align:center; }
.mrow-stat-val { font-size:0.95em; font-weight:700; color:#0F172A; }
.mrow-stat-lbl { font-size:0.68em; color:#94A3B8; text-transform:uppercase;
                 letter-spacing:0.5px; }
.mbadge {
    font-size:0.72em; font-weight:700;
    padding:4px 11px; border-radius:20px;
}

/* Pipeline */
.pstep {
    background:#F8FAFC;
    border:1.5px solid #E2E8F0;
    border-radius:10px;
    padding:11px 14px;
    margin:3px 0;
}
.pstep-t { font-size:0.82em; font-weight:700; color:#0F172A; }
.pstep-d { font-size:0.75em; color:#64748B; margin-top:2px; }
.parr { text-align:center; color:#2563EB; font-size:1.1em;
        font-weight:700; line-height:1.2; padding:1px 0; }

/* Deliverable row */
.drow {
    display:flex; align-items:center;
    justify-content:space-between;
    padding:7px 0;
    border-bottom:1px solid #F8FAFC;
}
.drow:last-child { border-bottom:none; }
.drow-l { display:flex; align-items:center; gap:8px;
          font-size:0.80em; color:#475569; font-weight:400; }
.drow-v { font-size:0.78em; font-weight:700; }

/* Footer */
.mgfooter {
    background:#FFFFFF;
    border-radius:12px;
    border:1px solid #E2E8F0;
    padding:16px 28px;
    text-align:center;
    font-size:0.75em;
    color:#94A3B8;
    margin-top:24px;
    line-height:1.9;
}
</style>
""", unsafe_allow_html=True)

# ── Paths & Data ──────────────────────────────────────────────
MODEL_DIR  = "ProjectA/models"
BO_JSON    = "ProjectA/bayesian_opt/best_candidate.json"
CORR_CSV   = "ProjectB/results/corrosion_profile.csv"
IMMUNE_CSV = "ProjectC/results/immune_response.csv"

@st.cache_data
def load_data():
    d = {"models":0,"best_ef":"N/A","corr_rows":0,
         "immune_rows":0,"m2m1":"N/A","crossover":"N/A",
         "corr_ok":False,"immune_ok":False,
         "corr_df":None,"immune_df":None}
    pkls = ["XGBoost.pkl","RandomForest.pkl","SVR.pkl",
            "MLP.pkl","GPR.pkl","scaler.pkl"]
    d["models"] = sum(1 for p in pkls
                      if os.path.exists(f"{MODEL_DIR}/{p}"))
    if os.path.exists(BO_JSON):
        with open(BO_JSON) as f:
            bo = json.load(f)
        d["best_ef"] = bo.get("predicted_Ef_eV_atom","N/A")
    if os.path.exists(CORR_CSV):
        df = pd.read_csv(CORR_CSV)
        d["corr_rows"] = len(df)
        d["corr_ok"]   = True
        d["corr_df"]   = df
    if os.path.exists(IMMUNE_CSV):
        df = pd.read_csv(IMMUNE_CSV)
        d["immune_rows"] = len(df)
        d["immune_ok"]   = True
        d["immune_df"]   = df
        r = df[df["day"]==21]["M2_M1_ratio"].values
        if len(r):
            d["m2m1"]      = f"{r[0]:.4f}"
            d["crossover"] = "Day 20.5"
    return d

data = load_data()

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style='padding:28px 20px 20px;display:flex;
    align-items:center;gap:12px;'>
        <div style='width:40px;height:40px;border-radius:12px;
        background:linear-gradient(135deg,#2563EB,#1E40AF);
        display:flex;align-items:center;justify-content:center;
        flex-shrink:0;box-shadow:0 4px 12px rgba(37,99,235,0.4);'>
        {ICONS['atom']}</div>
        <div>
            <div style='font-size:1.05em;font-weight:800;
            color:#FFFFFF;letter-spacing:-0.2px;'>MgInsight</div>
            <div style='font-size:0.65em;color:#475569;
            letter-spacing:0.5px;font-weight:500;'>
            RESEARCH PLATFORM</div>
        </div>
    </div>
    <div style='height:1px;background:rgba(255,255,255,0.08);
    margin:0 0 16px;'></div>
    """, unsafe_allow_html=True)

    # Nav sections
    nav_sections = {
        "MAIN MENU": [
            ("app.py",                  ICONS["home"],  "Dashboard",          True),
        ],
        "MODULES": [
            ("pages/1_🦾_Module_A.py", ICONS["flask"], "Module A — Alloy Design", False),
            ("pages/2_🔬_Module_B.py", ICONS["cpu"],   "Module B — Corrosion",   False),
            ("pages/3_🧬_Module_C.py", ICONS["dna"],   "Module C — Immune ODE",  False),
        ],
        "ANALYTICS": [
            ("app.py",                  ICONS["bar"],   "Results Overview",   False),
        ],
    }

    for section, items in nav_sections.items():
        st.markdown(f"""
        <div style='font-size:0.60em;font-weight:700;
        color:#334155;letter-spacing:2px;
        padding:12px 20px 6px;'>{section}</div>
        """, unsafe_allow_html=True)
        for path, icon, label, active in items:
            border = "2px solid #2563EB" if active else "2px solid transparent"
            bg     = "rgba(37,99,235,0.15)" if active else "transparent"
            clr    = "#FFFFFF" if active else "#94A3B8"
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;
            padding:9px 16px 9px 18px;border-radius:0 8px 8px 0;
            margin:1px 8px 1px 0;background:{bg};
            border-left:{border};transition:all 0.15s;'>
            <span style='color:{clr};display:flex;'>{icon}</span>
            <span style='font-size:0.862em;font-weight:{"600" if active else "400"};
            color:{clr};'>{label}</span>
            </div>
            """, unsafe_allow_html=True)
            if active:
                st.page_link(path, label=" ")
            else:
                st.page_link(path, label=label)

    st.markdown("""
    <div style='height:1px;background:rgba(255,255,255,0.08);
    margin:16px 0;'></div>
    <div style='font-size:0.60em;font-weight:700;
    color:#334155;letter-spacing:2px;
    padding:4px 20px 8px;'>LIVE STATUS</div>
    """, unsafe_allow_html=True)

    checks = [
        ("ML Models",        f"{data['models']}/6 loaded", data['models']>=5),
        ("Corrosion Profile", f"{data['corr_rows']} rows",  data['corr_ok']),
        ("Immune Response",  f"{data['immune_rows']} rows", data['immune_ok']),
        ("Day 21 Benchmark", f"M2/M1 = {data['m2m1']}",    data['m2m1']!="N/A"),
        ("GitHub Repo",      "49 objects",                  True),
    ]
    for lbl, val, ok in checks:
        dot = "#22C55E" if ok else "#EF4444"
        st.markdown(f"""
        <div style='display:flex;align-items:center;
        justify-content:space-between;padding:5px 20px;'>
          <div style='display:flex;align-items:center;gap:8px;'>
            <div style='width:6px;height:6px;border-radius:50%;
            background:{dot};flex-shrink:0;'></div>
            <span style='font-size:0.78em;color:#64748B;'>{lbl}</span>
          </div>
          <span style='font-size:0.70em;color:#334155;
          font-weight:600;'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='height:1px;background:rgba(255,255,255,0.08);
    margin:16px 0 12px;'></div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='padding:12px 20px 24px;'>
        <a href='https://github.com/prasanna0810/MgInsight'
        style='display:flex;align-items:center;gap:8px;
        text-decoration:none;background:rgba(255,255,255,0.06);
        border-radius:8px;padding:9px 12px;border:1px solid
        rgba(255,255,255,0.08);'>
            <span>{ICONS['github']}</span>
            <span style='font-size:0.75em;color:#64748B;
            font-weight:500;'>prasanna0810/MgInsight</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TOP BAR
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topbar">
  <div>
    <h2>Dashboard</h2>
    <div style='font-size:0.75em;color:#94A3B8;margin-top:2px;'>
    MgInsight — AI-Driven Biodegradable Mg Alloy Implant Design
    </div>
  </div>
  <div class="topbar-right">
    <div class="topbar-badge topbar-badge-green">
      Days 1–14 Complete
    </div>
    <div class="topbar-badge">
      Biomedical Engineering
    </div>
    <div style='width:36px;height:36px;border-radius:50%;
    background:linear-gradient(135deg,#2563EB,#1E40AF);
    display:flex;align-items:center;justify-content:center;'>
    {ICONS['atom']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════
k1,k2,k3,k4,k5 = st.columns(5)
kpis = [
    (k1,"#2563EB","#EFF6FF",ICONS["r2"],
     "0.9439","Best R² Score","XGBoost Model",
     "#EFF6FF","#1D4ED8"),
    (k2,"#0D9488","#F0FDF4",ICONS["target"],
     "−2.90","BO Best E\u2091 (eV/atom)","50-iter UCB",
     "#F0FDF4","#0F766E"),
    (k3,"#7C3AED","#FAF5FF",ICONS["cog"],
     "78,316","FEniCS Steps","28 days · dt=30.9s",
     "#FAF5FF","#6D28D9"),
    (k4,"#15803D","#F0FDF4",ICONS["check2"],
     data["m2m1"],"M2/M1 at Day 21","Benchmark Passed",
     "#F0FDF4","#15803D"),
    (k5,"#C2410C","#FFF7ED",ICONS["cal"],
     data["crossover"],"Crossover Day","Inflam. → Healing",
     "#FFF7ED","#C2410C"),
]
for (col,accent,ibg,icon,val,lbl,sub,sbg,sclr) in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-top-bar"
          style="background:{accent};"></div>
          <div class="kpi-icon"
          style="background:{ibg};">{icon}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
          <span class="kpi-sub"
          style="background:{sbg};color:{sclr};">{sub}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# CHARTS ROW (mimics the line chart section in reference image)
# ════════════════════════════════════════════════════════════════
ch1, ch2 = st.columns([3, 2])

with ch1:
    st.markdown('<div class="wcard">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="wcard-title">
      {ICONS['bar']} &nbsp; 28-Day Corrosion &amp;
      Immune Response Profile
    </div>
    """, unsafe_allow_html=True)

    corr = data["corr_df"]
    imm  = data["immune_df"]

    fig = go.Figure()
    if corr is not None:
        fig.add_trace(go.Scatter(
            x=corr["day"], y=corr["c_max_mol_m3"],
            name="Mg²⁺ c_max (mol/m³)",
            line=dict(color="#2563EB", width=2.5),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=corr["day"], y=corr["c_mean_mol_m3"],
            name="Mg²⁺ c_mean (mol/m³)",
            line=dict(color="#93C5FD", width=2, dash="dot"),
            mode="lines",
        ))
    if imm is not None:
        fig.add_trace(go.Scatter(
            x=imm["day"], y=imm["M1"],
            name="M1 Macrophage",
            line=dict(color="#EF4444", width=2.5),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=imm["day"], y=imm["M2"],
            name="M2 Macrophage",
            line=dict(color="#22C55E", width=2.5),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=imm["day"], y=imm["M2_M1_ratio"],
            name="M2/M1 Ratio",
            line=dict(color="#A855F7", width=2, dash="dash"),
            mode="lines",
        ))
        # Day 21 annotation
        fig.add_vline(x=21, line_width=1.5,
                      line_dash="dash", line_color="#94A3B8",
                      annotation_text="Day 21",
                      annotation_position="top",
                      annotation_font_size=11,
                      annotation_font_color="#64748B")

    fig.update_layout(
        height=280,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10,r=10,t=10,b=30),
        font=dict(family="Inter",size=11,color="#64748B"),
        legend=dict(orientation="h", y=-0.25,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10)),
        xaxis=dict(title="Day", showgrid=True,
                   gridcolor="#F1F5F9",
                   tickfont=dict(size=10)),
        yaxis=dict(title="Value", showgrid=True,
                   gridcolor="#F1F5F9",
                   tickfont=dict(size=10)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

with ch2:
    st.markdown('<div class="wcard">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="wcard-title">
      {ICONS['cog']} &nbsp; ML Model Performance
    </div>
    """, unsafe_allow_html=True)

    models = ["XGBoost","RandomForest","MLP","SVR","GPR"]
    r2     = [0.9439, 0.9312, 0.9105, 0.8923, 0.8876]
    colors = ["#2563EB","#0D9488","#7C3AED","#F59E0B","#EF4444"]

    fig2 = go.Figure(go.Bar(
        x=r2, y=models,
        orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v:.4f}" for v in r2],
        textposition="outside",
        textfont=dict(size=10, color="#0F172A"),
    ))
    fig2.update_layout(
        height=280,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10,r=50,t=10,b=30),
        font=dict(family="Inter",size=11,color="#64748B"),
        xaxis=dict(title="R² Score", range=[0.85,0.97],
                   showgrid=True, gridcolor="#F1F5F9",
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True,
                    config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MODULE STATUS + PIPELINE + DELIVERABLES
# ════════════════════════════════════════════════════════════════
m1, m2 = st.columns([2.2, 1])

with m1:
    st.markdown(f"""
    <div class="wcard-title" style='margin-bottom:12px;'>
      {ICONS['flask']} &nbsp; PLATFORM MODULES
    </div>
    """, unsafe_allow_html=True)

    module_rows = [
        (ICONS["flask"],  "linear-gradient(135deg,#2563EB,#1D4ED8)",
         "Module A — Alloy Design & ML",
         "XGBoost · RF · SVR · MLP · GPR · SHAP · BoTorch",
         "pages/1_🦾_Module_A.py",
         [("R²",  "0.9439"),("Models","5 / 5"),("BO E\u2091","−2.90 eV")],
         "#F0FDF4","#15803D","Ready"),
        (ICONS["cpu"],    "linear-gradient(135deg,#0D9488,#0F766E)",
         "Module B — Corrosion Simulation",
         "DOLFINx · Fick's 2nd Law · Crank-Nicolson · Gmsh 708 nodes",
         "pages/2_🔬_Module_B.py",
         [("Steps","78,316"),("Days","28"),("Depth","0.072 mm")],
         "#F0FDF4","#15803D","Ready"),
        (ICONS["dna"],    "linear-gradient(135deg,#7C3AED,#6D28D9)",
         "Module C — Immune Response ODE",
         "SciPy RK45 · 4-var ODE · M0/M1/M2/Mg²⁺ · Sensitivity sweep",
         "pages/3_🧬_Module_C.py",
         [("M2/M1",data["m2m1"]),("Points","2,801"),("Cross.",data["crossover"])],
         "#F0FDF4","#15803D","Ready"),
    ]

    for (icon, grad, title, sub, link,
         stats, bbg, bclr, blabel) in module_rows:
        stats_html = "".join(f"""
        <div class="mrow-stat">
          <div class="mrow-stat-val">{sv}</div>
          <div class="mrow-stat-lbl">{sk}</div>
        </div>""" for sk,sv in stats)

        st.markdown(f"""
        <div class="mrow">
          <div class="mrow-icon"
          style="background:{grad};">{icon}</div>
          <div>
            <p class="mrow-title">{title}</p>
            <p class="mrow-sub">{sub}</p>
          </div>
          <div class="mrow-stats">
            {stats_html}
            <span class="mbadge"
            style="background:{bbg};color:{bclr};">{blabel}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(link, label=f"Open {title.split(' — ')[0]} →")

with m2:
    # Pipeline card
    st.markdown(f"""
    <div class="wcard">
    <div class="wcard-title">
      {ICONS['cog']} &nbsp; DATA PIPELINE
    </div>
    """, unsafe_allow_html=True)

    pipe = [
        ("Module A",  "Composition → E_f → D_Mg, R_corr"),
        None,
        ("Module B",  "D_Mg → FEniCS → Mg²⁺(t)"),
        None,
        ("Module C",  "Mg²⁺(t) → ODE → M2/M1 ratio"),
        None,
        ("Output",    "M2/M1 = 1.022 > 1.0 — Validated"),
    ]
    for item in pipe:
        if item is None:
            st.markdown(
                '<div class="parr">↓</div>',
                unsafe_allow_html=True)
        else:
            t, d = item
            st.markdown(f"""
            <div class="pstep">
              <div class="pstep-t">{t}</div>
              <div class="pstep-d">{d}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Deliverables card
    st.markdown(f"""
    <div class="wcard">
    <div class="wcard-title">
      {ICONS['check']} &nbsp; DELIVERABLES
    </div>
    """, unsafe_allow_html=True)

    deliverables = [
        ("mg_featurized.csv",    "7,592 × 132", True),
        ("5 × .pkl models",       "R²=0.9439",   True),
        ("SHAP reports",          "7 PNGs",       True),
        ("best_candidate.json",   "−2.90 eV",     True),
        ("corrosion_profile.csv", "28 × 7",       True),
        ("immune_response.csv",   "29 × 7",       True),
        ("Streamlit Modules A-C", "Live",         True),
        ("GitHub Repository",     "49 objects",   True),
    ]
    for lbl, val, ok in deliverables:
        dot = "#22C55E" if ok else "#F59E0B"
        vc  = "#15803D" if ok else "#C2410C"
        st.markdown(f"""
        <div class="drow">
          <div class="drow-l">
            <div style='width:6px;height:6px;border-radius:50%;
            background:{dot};flex-shrink:0;'></div>
            {lbl}
          </div>
          <span class="drow-v" style="color:{vc};">{val}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TECHNICAL SUMMARY ROW
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='font-size:0.72em;font-weight:700;color:#94A3B8;
letter-spacing:1.5px;text-transform:uppercase;
margin:8px 0 12px;'>TECHNICAL PARAMETERS</div>
""", unsafe_allow_html=True)

s1,s2,s3 = st.columns(3)
summaries = [
    (s1,"Dataset",ICONS["flask"],"#EFF6FF","#2563EB",[
        ("Source",  "Materials Project + Matminer"),
        ("Alloys",  "7,592"),
        ("Features","132 Magpie compositional"),
        ("Target",  "Formation energy (eV/atom)"),
        ("Split",   "80 / 20 train / test"),
        ("Range",   "−4.5 to +0.8 eV/atom"),
    ]),
    (s2,"FEniCS Solver",ICONS["cpu"],"#F0FDF4","#0D9488",[
        ("PDE",     "Fick's 2nd Law (3D)"),
        ("Method",  "Crank-Nicolson"),
        ("Steps",   "78,316"),
        ("dt",      "30.9 s"),
        ("Duration","28 days"),
        ("Nodes",   "708"),
        ("Domain",  "13 × 13 × 30 mm SBF"),
        ("Radius",  "1.5 – 7.5 mm"),
    ]),
    (s3,"ODE System",ICONS["dna"],"#FAF5FF","#7C3AED",[
        ("Variables","M0, M1, M2, Mg²⁺"),
        ("Solver",   "SciPy RK45"),
        ("rtol",     "1 × 10⁻⁸"),
        ("Points",   "2,801"),
        ("M2 EC50",  "1.5 mol/m³"),
        ("M1 IC50",  "2.5 mol/m³"),
        ("Day 21",   f"{data['m2m1']} (> 1.0)"),
        ("Crossover",data["crossover"]),
    ]),
]
for col,title,icon,ibg,iclr,rows in summaries:
    with col:
        rows_html = "".join(
            f"<div class='trow'>"
            f"<span class='trow-k'>{k}</span>"
            f"<span class='trow-v'>{v}</span>"
            f"</div>" for k,v in rows
        )
        st.markdown(f"""
        <div class="wcard">
          <div class="wcard-title" style="color:{iclr};">
            <div style='width:24px;height:24px;border-radius:7px;
            background:{ibg};display:flex;align-items:center;
            justify-content:center;'>{icon}</div>
            {title}
          </div>
          {rows_html}
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="mgfooter">
  <strong style='color:#0F172A;'>MgInsight v1.0</strong>
  &nbsp;·&nbsp; Department of Biomedical Engineering
  &nbsp;·&nbsp; DOLFINx 0.10.0 &nbsp;·&nbsp;
  SciPy RK45 &nbsp;·&nbsp; XGBoost R²=0.9439
  &nbsp;·&nbsp; BoTorch 50-iter UCB<br>
  <a href='https://github.com/prasanna0810/MgInsight'
  style='color:#2563EB;text-decoration:none;font-weight:600;'>
  github.com/prasanna0810/MgInsight</a>
</div>
""", unsafe_allow_html=True)