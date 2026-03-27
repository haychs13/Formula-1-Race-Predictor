"""
F1 Race Predictor — Streamlit Web App
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────────────────────────────────────

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

DK = st.session_state.dark_mode

# Colour tokens
if DK:
    BG          = "#0c0c10"
    SURFACE     = "#15151e"
    SURFACE2    = "#1d1d28"
    BORDER      = "rgba(255,255,255,0.07)"
    BORDER_ACC  = "#e10600"
    TEXT        = "#f0f0f0"
    TEXT_SUB    = "#8a8a9a"
    TEXT_MUTED  = "#55556a"
    GOLD        = "#f5c518"
    SILVER      = "#c0c0c0"
    BRONZE      = "#cd7f32"
    RED         = "#e10600"
    PLOT_TEXT   = "#e0e0e0"
    PLOT_GRID   = "rgba(255,255,255,0.08)"
    BAR_NEUTRAL = "#3d6199"
    SIDEBAR_BG  = "#10101a"
else:
    BG          = "#f0f2f6"
    SURFACE     = "#ffffff"
    SURFACE2    = "#f7f8fc"
    BORDER      = "rgba(0,0,0,0.07)"
    BORDER_ACC  = "#e10600"
    TEXT        = "#1a1a2e"
    TEXT_SUB    = "#5a5a72"
    TEXT_MUTED  = "#9090a8"
    GOLD        = "#c8960c"
    SILVER      = "#7a7a8a"
    BRONZE      = "#a0622a"
    RED         = "#e10600"
    PLOT_TEXT   = "#1a1a2e"
    PLOT_GRID   = "rgba(0,0,0,0.08)"
    BAR_NEUTRAL = "#3060b0"
    SIDEBAR_BG  = "#e8eaf0"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, .stApp {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
.stApp header {{ background-color: {BG}; }}

/* ── Hide deploy button & hamburger ── */
#MainMenu {{ visibility: hidden; }}
header [data-testid="stToolbar"] {{ display: none; }}
footer {{ visibility: hidden; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {TEXT_SUB};
    font-size: 13px;
    line-height: 1.6;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {SURFACE};
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {TEXT_SUB};
    font-size: 14px;
    font-weight: 500;
    padding: 8px 18px;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background-color: {RED} !important;
    color: white !important;
    font-weight: 600;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    color: {TEXT};
    transition: border-color 0.2s;
}}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {{
    border-color: {RED};
}}
.stSelectbox label, .stNumberInput label, .stCheckbox label {{
    color: {TEXT_SUB};
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.3px;
}}

/* ── Expander ── */
div[data-testid="stExpander"] details {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
div[data-testid="stExpander"] summary {{
    color: {TEXT_SUB};
    font-weight: 500;
    font-size: 14px;
}}

/* ── Primary button ── */
.stButton > button[kind="primary"] {{
    background-color: {RED};
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.5px;
    padding: 10px 24px;
    transition: all 0.2s;
    box-shadow: 0 2px 12px rgba(225,6,0,0.3);
}}
.stButton > button[kind="primary"]:hover {{
    background-color: #ff1a15;
    box-shadow: 0 4px 20px rgba(225,6,0,0.45);
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"]:active {{
    transform: translateY(0);
}}

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    color: {TEXT_SUB};
    font-size: 13px;
    font-weight: 500;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {RED};
    color: {RED};
}}

/* ── Dataframe ── */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}
iframe {{ border-radius: 10px; }}

/* ── Divider ── */
hr {{ border-color: {BORDER}; }}

/* ── Metrics ── */
div[data-testid="stMetricValue"] {{
    color: {TEXT};
    font-size: 22px;
    font-weight: 700;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_SUB};
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Caption ── */
.stCaption {{ color: {TEXT_MUTED}; font-size: 12px; }}

/* ── Info / Warning / Error boxes ── */
div[data-testid="stInfo"] {{
    background-color: {'rgba(30,40,70,0.6)' if DK else 'rgba(220,235,255,0.8)'};
    border-left: 3px solid #4a90e2;
    border-radius: 8px;
    color: {TEXT};
}}
div[data-testid="stWarning"] {{
    background-color: {'rgba(60,45,10,0.6)' if DK else 'rgba(255,245,210,0.9)'};
    border-left: 3px solid #f5a623;
    border-radius: 8px;
    color: {TEXT};
}}
div[data-testid="stError"] {{
    background-color: {'rgba(60,10,10,0.6)' if DK else 'rgba(255,230,230,0.9)'};
    border-left: 3px solid {RED};
    border-radius: 8px;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_ACC}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — reusable HTML components
# ─────────────────────────────────────────────────────────────────────────────

def disclaimer():
    st.markdown(
        f"<div style='margin-top:40px;padding:14px 18px;border-top:1px solid {BORDER};"
        f"font-size:11px;color:{TEXT_MUTED};line-height:1.7;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:16px;'>"
        f"<div>"
        f"<b style='color:{TEXT_SUB};'>Disclaimer</b> &nbsp;·&nbsp; "
        f"This app uses F1 race data from the 2021–2025 seasons sourced from the Ergast F1 dataset. "
        f"Data has not been updated beyond this period and driver line-ups, team names, and circuits may have changed. "
        f"Predictions are generated by a machine learning model trained on historical patterns and are not guaranteed to reflect current or future performance. "
        f"This project is intended solely as a machine learning demonstration and must not be used for gambling, betting, or any commercial purposes. "
        f"For production use, this model could be extended with live data via the FastF1 or OpenF1 APIs and automated retraining after each race weekend. "
        f"<b>This is not an official Formula 1 product and is not affiliated with or endorsed by Formula 1, FIA, or any F1 team.</b>"
        f"</div>"
        f"<div style='white-space:nowrap;text-align:right;flex-shrink:0;'>"
        f"Created by<br><b style='color:{TEXT_SUB};'>Hayder Sayyid</b>"
        f"</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    sub_html = f'<div style="color:{TEXT_SUB};font-size:13px;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:3px;height:22px;background:{RED};border-radius:2px;flex-shrink:0;"></div>
            <div style="font-size:17px;font-weight:700;color:{TEXT};">{title}</div>
        </div>
        {sub_html}
    </div>
    """

def stat_card(label, value, icon=""):
    return f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:14px 16px;text-align:center;">
        <div style="font-size:20px;margin-bottom:2px;">{icon}</div>
        <div style="font-size:20px;font-weight:700;color:{TEXT};">{value}</div>
        <div style="font-size:11px;color:{TEXT_SUB};text-transform:uppercase;
                    letter-spacing:0.6px;margin-top:2px;">{label}</div>
    </div>
    """

def confidence_bar(pct, label="Model Confidence"):
    fill_color = RED if pct < 50 else ("#f5a623" if pct < 75 else "#22c55e")
    return f"""
    <div style="margin:16px 0 8px 0;">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:13px;color:{TEXT_SUB};font-weight:500;">{label}</span>
            <span style="font-size:14px;font-weight:700;color:{TEXT};">{pct}%</span>
        </div>
        <div style="background:{SURFACE2};border-radius:999px;height:8px;overflow:hidden;
                    border:1px solid {BORDER};">
            <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{fill_color}cc,{fill_color});
                        border-radius:999px;transition:width 0.4s ease;"></div>
        </div>
    </div>
    """

def info_card(icon, title, body):
    return f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:18px;height:100%;">
        <div style="font-size:26px;margin-bottom:8px;">{icon}</div>
        <div style="font-size:14px;font-weight:600;color:{TEXT};margin-bottom:6px;">{title}</div>
        <div style="font-size:13px;color:{TEXT_SUB};line-height:1.5;">{body}</div>
    </div>
    """

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    required = ["model.pkl", "scaler.pkl", "label_encoders.pkl",
                "feature_columns.pkl", "driver_list.pkl", "circuit_list.pkl"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, f"Missing files: {', '.join(missing)}"
    with open("model.pkl",          "rb") as f: model         = pickle.load(f)
    with open("scaler.pkl",         "rb") as f: scaler        = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f: label_encoders= pickle.load(f)
    with open("feature_columns.pkl","rb") as f: feature_cols  = pickle.load(f)
    with open("driver_list.pkl",    "rb") as f: driver_list   = pickle.load(f)
    with open("circuit_list.pkl",   "rb") as f: circuit_list  = pickle.load(f)
    driver_stats = pd.read_csv("driver_stats.csv", index_col=0) if os.path.exists("driver_stats.csv") else None
    circuit_map  = pd.read_csv("circuit_map.csv",  index_col=0) if os.path.exists("circuit_map.csv")  else None
    return {"model":model,"scaler":scaler,"label_encoders":label_encoders,
            "feature_cols":feature_cols,"driver_list":driver_list,
            "circuit_list":circuit_list,"driver_stats":driver_stats,"circuit_map":circuit_map}, None

artifacts, load_error = load_model_artifacts()

def read_model_report():
    if os.path.exists("model_report.txt"):
        with open("model_report.txt","r",encoding="utf-8") as f: return f.read()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 16px 0;border-bottom:1px solid {BORDER};margin-bottom:16px;">
        <div style="font-size:20px;font-weight:800;color:{TEXT};">🏎️ F1 Predictor</div>
        <div style="font-size:11px;color:{TEXT_MUTED};margin-top:2px;letter-spacing:0.5px;">
            2021 – 2025 SEASON DATA
        </div>
    </div>
    """, unsafe_allow_html=True)

    icon_label = "☀️  Switch to Light" if DK else "🌙  Switch to Dark"
    if st.button(icon_label, key="theme_sidebar", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(f"<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:12px;font-weight:600;color:{TEXT_MUTED};
                text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">
        About
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    Uses historical F1 data to estimate the probability of a driver finishing on the **podium** (top 3).

    The model weighs up grid position, recent form, qualifying pace, reliability, and championship standing.

    ---
    **Quick guide**
    1. Pick a driver and circuit
    2. Set the starting grid position
    3. Optionally expand Advanced Options
    4. Hit **Predict**
    """)

    st.markdown(f"""
    <div style="font-size:12px;font-weight:600;color:{TEXT_MUTED};
                text-transform:uppercase;letter-spacing:0.8px;margin:16px 0 8px 0;">
        Model
    </div>
    """, unsafe_allow_html=True)

    report = read_model_report()
    lines  = report.split("\n")
    comp   = []
    inside = False
    for ln in lines:
        if "MODEL COMPARISON" in ln: inside = True
        if inside: comp.append(ln)
        if inside and ln.strip() == "" and len(comp) > 3: break
    if comp:
        st.code("\n".join(comp[1:]), language=None)

    trained = (datetime.fromtimestamp(os.path.getmtime("model.pkl")).strftime("%d %b %Y")
               if os.path.exists("model.pkl") else "N/A")
    st.markdown(f"<div style='font-size:11px;color:{TEXT_MUTED};margin-top:4px;'>Last trained: {trained}</div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
            padding:24px 28px;margin-bottom:24px;
            display:flex;align-items:center;justify-content:space-between;">
    <div>
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="background:{RED};border-radius:8px;padding:8px 12px;
                        font-size:22px;line-height:1;">🏎️</div>
            <div>
                <div style="font-size:26px;font-weight:800;color:{TEXT};
                            letter-spacing:-0.5px;">F1 Race Predictor</div>
                <div style="font-size:13px;color:{TEXT_SUB};margin-top:2px;">
                    Machine learning · 2021–2025 seasons · 100 races · 2,000 entries
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Theme toggle top-right (rendered after header via columns trick)
_, toggle_col = st.columns([6, 1])
with toggle_col:
    icon = "☀️ Light" if DK else "🌙 Dark"
    if st.button(icon, key="theme_top"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

if load_error:
    st.error(f"**Model files not found.** Run `python train.py` first, then refresh.\n\n{load_error}")
    st.stop()

# Unpack
model        = artifacts["model"]
scaler       = artifacts["scaler"]
le           = artifacts["label_encoders"]
feature_cols = artifacts["feature_cols"]
driver_list  = artifacts["driver_list"]
circuit_list = artifacts["circuit_list"]
driver_stats = artifacts["driver_stats"]
circuit_map  = artifacts["circuit_map"]

circuit_names      = [c[1] if isinstance(c, list) else c for c in circuit_list]
circuit_ids        = [c[0] if isinstance(c, list) else None for c in circuit_list]
circuit_name_to_id = dict(zip(circuit_names, circuit_ids))

def encode_value(key, val):
    enc = le[key]
    return int(enc.transform([val])[0]) if val in enc.classes_ else int(enc.transform([enc.classes_[0]])[0])

def get_driver_stats(name):
    return driver_stats.loc[name] if driver_stats is not None and name in driver_stats.index else None

def make_prediction(driver_name, circuit_name):
    if driver_name not in le["driver_encoded"].classes_:
        return None, None, None, f"'{driver_name}' was not in the training data."
    stats          = get_driver_stats(driver_name)
    driver_enc     = encode_value("driver_encoded", driver_name)
    circuit_id_val = circuit_name_to_id.get(circuit_name, 0)
    circuit_enc    = encode_value("circuit_id", str(int(circuit_id_val))) if circuit_id_val else 0
    if stats is not None:
        rp5      = float(stats.get("rolling_points_5", 5))
        rf5      = float(stats.get("rolling_finish_5", 10))
        dnf      = float(stats.get("dnf_rate", 0.05))
        exp      = float(stats.get("driver_experience", 50))
        cpr      = float(stats.get("career_podium_rate", 0.0))
        cwr      = float(stats.get("career_win_rate", 0.0))
        qgap     = float(stats.get("quali_gap_to_pole", 1.0))
        champ    = float(stats.get("driver_championship_pos", 10))
        ctor_pos = float(stats.get("constructor_championship_pos", 5))
        pod5     = float(stats.get("podiums_last_5", 0))
        bfc      = float(stats.get("best_finish_at_circuit", 20))
    else:
        rp5=5.0; rf5=10.0; dnf=0.05; exp=50.0
        cpr=0.0; cwr=0.0; qgap=1.0; champ=10.0; ctor_pos=5.0; pod5=0.0; bfc=20.0
    vals = {
        "circuit_id":              float(circuit_enc),
        "driver_encoded":          float(driver_enc),
        "rolling_points_5":        rp5,
        "rolling_finish_5":        rf5,
        "dnf_rate":                dnf,
        "driver_experience":       exp,
        "quali_gap_to_pole":       qgap,
        "driver_championship_pos": champ,
        "constructor_championship_pos": ctor_pos,
        "podiums_last_5":          pod5,
        "best_finish_at_circuit":  bfc,
        "career_podium_rate":      cpr,
        "career_win_rate":         cwr,
    }
    X = np.array([[vals[c] for c in feature_cols]])
    prob = float(model.predict_proba(scaler.transform(X))[0][1])
    return int(prob >= 0.5), prob, vals, None

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🏁  Predict a Race",
    "📈  Model Accuracy",
    "❓  How It Works",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown(section_header("Driver & Race", "Select who and where"), unsafe_allow_html=True)

        selected_driver = st.selectbox("Driver", options=driver_list, index=0,
                                        label_visibility="collapsed", placeholder="Select a driver…")
        st.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};margin:-8px 0 10px 2px;'>Driver</div>",
                    unsafe_allow_html=True)

        selected_circuit = st.selectbox("Circuit", options=circuit_names, index=0,
                                         label_visibility="collapsed", placeholder="Select a circuit…")
        st.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};margin:-8px 0 4px 2px;'>Circuit</div>",
                    unsafe_allow_html=True)

        predict_clicked = st.button("🔮  Predict Podium", use_container_width=True, type="primary")

    # ── RIGHT: Result ──────────────────────────────────────────────────────
    with right_col:
        if not predict_clicked:
            st.markdown(f"""
            <div style="background:{SURFACE};border:1px dashed {BORDER};border-radius:12px;
                        padding:48px 24px;text-align:center;margin-top:0;">
                <div style="font-size:48px;margin-bottom:12px;">🏁</div>
                <div style="font-size:16px;font-weight:600;color:{TEXT_SUB};">
                    Your prediction will appear here
                </div>
                <div style="font-size:13px;color:{TEXT_MUTED};margin-top:6px;">
                    Select a driver and circuit, then click Predict
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                pred, prob, used_vals, warning = make_prediction(selected_driver, selected_circuit)
                driver_team = ""
                _ds = get_driver_stats(selected_driver)
                if _ds is not None and "team_name" in _ds:
                    driver_team = str(_ds["team_name"])

                if warning:
                    st.warning(f"⚠️  {warning}")
                elif pred is None:
                    st.error("Unable to make a prediction. Please check your inputs.")
                else:
                    confidence_pct = int(round(prob * 100))

                    if pred == 1:
                        st.markdown(f"""
                        <div style="background:linear-gradient(145deg,#1a0a00,#2d1200);
                                    border:1px solid {GOLD}40;border-radius:12px;
                                    padding:32px 24px;text-align:center;margin-bottom:16px;
                                    box-shadow:0 0 40px rgba(245,197,24,0.08);">
                            <div style="font-size:56px;margin-bottom:4px;">🏆</div>
                            <div style="font-size:11px;font-weight:700;color:{GOLD};
                                        letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">
                                Podium Finish Predicted
                            </div>
                            <div style="font-size:24px;font-weight:700;color:#fff;margin-bottom:4px;">
                                {selected_driver}
                            </div>
                            {"" if not driver_team else f"<div style='font-size:16px;color:{TEXT_SUB};margin-bottom:4px;'>{driver_team}</div>"}
                            <div style="font-size:13px;color:{TEXT_MUTED};">
                                at {selected_circuit}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:{SURFACE};border:1px solid {BORDER};
                                    border-radius:12px;padding:32px 24px;text-align:center;
                                    margin-bottom:16px;">
                            <div style="font-size:56px;margin-bottom:4px;">🚫</div>
                            <div style="font-size:11px;font-weight:700;color:{TEXT_SUB};
                                        letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">
                                Outside Podium
                            </div>
                            <div style="font-size:24px;font-weight:700;color:{TEXT};margin-bottom:4px;">
                                {selected_driver}
                            </div>
                            {"" if not driver_team else f"<div style='font-size:16px;color:{TEXT_SUB};margin-bottom:4px;'>{driver_team}</div>"}
                            <div style="font-size:13px;color:{TEXT_MUTED};">
                                at {selected_circuit}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(confidence_bar(confidence_pct), unsafe_allow_html=True)

                    # Stat cards from auto-looked-up driver data
                    cpr_pct  = int(round(used_vals.get("career_podium_rate", 0) * 100))
                    pod5_val = int(used_vals.get("podiums_last_5", 0))
                    champ_v  = int(used_vals.get("driver_championship_pos", 10))
                    last_pos = int(round(used_vals.get("rolling_finish_5", 10)))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(stat_card("Podium Rate", f"{cpr_pct}%", "🏆"), unsafe_allow_html=True)
                    c2.markdown(stat_card("Last 5 Podiums", str(pod5_val), "📈"), unsafe_allow_html=True)
                    c3.markdown(stat_card("Champ Pos", f"P{champ_v}", "🏅"), unsafe_allow_html=True)
                    c4.markdown(stat_card("Avg Finish Pos", f"P{last_pos}", "🏎️"), unsafe_allow_html=True)

                    # ── Why this prediction? ───────────────────────────────
                    reasons = []
                    cpr  = used_vals.get("career_podium_rate", 0)
                    cwr  = used_vals.get("career_win_rate", 0)
                    pod5 = used_vals.get("podiums_last_5", 0)
                    qgap = used_vals.get("quali_gap_to_pole", 1.0)
                    chmp = used_vals.get("driver_championship_pos", 10)
                    bfc  = used_vals.get("best_finish_at_circuit", 20)
                    rp5  = used_vals.get("rolling_points_5", 5)

                    if cpr >= 0.4:
                        reasons.append(f"historically finishes on the podium in <b>{int(cpr*100)}%</b> of races")
                    elif cpr >= 0.2:
                        reasons.append(f"career podium rate of <b>{int(cpr*100)}%</b> shows solid form")
                    else:
                        reasons.append(f"career podium rate of <b>{int(cpr*100)}%</b> reflects limited podium history")

                    if cwr >= 0.15:
                        reasons.append(f"strong winner — has won <b>{int(cwr*100)}%</b> of career races")

                    if pod5 >= 3:
                        reasons.append(f"in excellent recent form with <b>{int(pod5)}</b> podiums in the last 5 races")
                    elif pod5 == 0:
                        reasons.append("no podiums in the last 5 races")

                    if qgap < 0.3:
                        reasons.append("recent qualifying pace is very close to pole position")
                    elif qgap > 1.5:
                        reasons.append(f"average qualifying gap of <b>{qgap:.2f}s</b> to pole suggests a pace deficit")

                    if chmp <= 3:
                        reasons.append("currently at the sharp end of the championship")
                    elif chmp >= 15:
                        reasons.append("low championship standing reflects weaker overall pace this season")

                    if bfc <= 3:
                        reasons.append(f"has previously finished as high as <b>P{int(bfc)}</b> at this circuit")
                    elif bfc >= 15:
                        reasons.append("historically struggles at this circuit")

                    if rp5 >= 15:
                        reasons.append(f"averaging <b>{rp5:.0f} points</b> per race over the last 5 rounds")

                    if reasons:
                        bullets = "".join(f"<li style='margin-bottom:5px;'>{r}</li>" for r in reasons)
                        expl_html = (
                            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                            f"border-radius:10px;padding:16px 20px;margin-top:14px;'>"
                            f"<div style='font-size:12px;font-weight:600;color:{TEXT_SUB};"
                            f"text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>"
                            f"Why this prediction?</div>"
                            f"<ul style='margin:0;padding-left:18px;font-size:13px;color:{TEXT_SUB};line-height:1.8;'>"
                            f"{bullets}"
                            f"</ul></div>"
                        )
                        st.markdown(expl_html, unsafe_allow_html=True)

                    # ── Confidence key ────────────────────────────────────
                    conf_key_html = (
                        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                        f"border-radius:10px;padding:14px 18px;margin-top:12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT_SUB};"
                        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>"
                        f"Understanding the Confidence Score</div>"
                        f"<div style='display:flex;gap:8px;'>"
                        f"<div style='flex:1;background:{SURFACE2};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:2px;'>50% +</div>"
                        f"<div style='font-size:11px;color:{TEXT_SUB};'>Podium predicted</div>"
                        f"</div>"
                        f"<div style='flex:1;background:{SURFACE2};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:2px;'>Below 50%</div>"
                        f"<div style='font-size:11px;color:{TEXT_SUB};'>Outside podium</div>"
                        f"</div>"
                        f"<div style='flex:1;background:{SURFACE2};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:2px;'>Higher = more certain</div>"
                        f"<div style='font-size:11px;color:{TEXT_SUB};'>51% and 95% are both podium calls — very different certainty</div>"
                        f"</div>"
                        f"</div></div>"
                    )
                    st.markdown(conf_key_html, unsafe_allow_html=True)

                    # ── Stat card timespans ────────────────────────────────
                    spans_html = (
                        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                        f"border-radius:10px;padding:14px 18px;margin-top:8px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT_SUB};"
                        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>"
                        f"About the Stat Cards</div>"
                        f"<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
                        f"<div style='flex:1;min-width:120px;background:{SURFACE2};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:2px;'>Podium Rate</div>"
                        f"<div style='font-size:11px;color:{TEXT_SUB};'>Career average across all races in the dataset (2021–2025)</div>"
                        f"</div>"
                        f"<div style='flex:1;min-width:120px;background:{SURFACE2};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:2px;'>Last 5 Podiums</div>"
                        f"<div style='font-size:11px;color:{TEXT_SUB};'>Podium finishes from the driver's last 5 races only</div>"
                        f"</div>"
                        f"<div style='flex:1;min-width:120px;background:{SURFACE2};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:2px;'>Champ Pos</div>"
                        f"<div style='font-size:11px;color:{TEXT_SUB};'>Championship standing as of their last race in the dataset (mid-2025)</div>"
                        f"</div>"
                        f"<div style='flex:1;min-width:120px;background:{SURFACE2};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:2px;'>Avg Finish Pos</div>"
                        f"<div style='font-size:11px;color:{TEXT_SUB};'>Average finishing position across the driver's last 5 races</div>"
                        f"</div>"
                        f"</div></div>"
                    )
                    st.markdown(spans_html, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="text-align:center;margin-top:12px;font-size:11px;color:{TEXT_MUTED};">
                        Based on historical F1 data · 2021–2025
                    </div>
                    """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.prediction_history.insert(0, {
                        "driver":     selected_driver,
                        "circuit":    selected_circuit,
                        "team":       driver_team,
                        "result":     "Podium" if pred == 1 else "No Podium",
                        "confidence": confidence_pct,
                    })

            except Exception:
                st.error("Unable to predict — please check your inputs and try again.")

    # ── Previous Predictions ──────────────────────────────────────────────────
    if st.session_state.prediction_history:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(section_header("Previous Predictions"), unsafe_allow_html=True)

        for entry in st.session_state.prediction_history:
            is_podium = entry["result"] == "Podium"
            result_color = GOLD if is_podium else TEXT_SUB
            result_icon  = "🏆" if is_podium else "🚫"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:16px;padding:10px 16px;"
                f"background:{SURFACE};border:1px solid {BORDER};border-radius:8px;margin-bottom:6px;'>"
                f"<div style='font-size:20px;'>{result_icon}</div>"
                f"<div style='flex:1;'>"
                f"<div style='font-size:14px;font-weight:600;color:{TEXT};'>{entry['driver']}"
                f"<span style='font-size:12px;font-weight:400;color:{TEXT_MUTED};margin-left:8px;'>{entry['team']}</span></div>"
                f"<div style='font-size:12px;color:{TEXT_SUB};margin-top:2px;'>{entry['circuit']}</div>"
                f"</div>"
                f"<div style='text-align:right;'>"
                f"<div style='font-size:13px;font-weight:600;color:{result_color};'>{entry['result']}</div>"
                f"<div style='font-size:12px;color:{TEXT_MUTED};'>{entry['confidence']}% confidence</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if st.button("Clear history", key="clear_history"):
            st.session_state.prediction_history = []
            st.rerun()

    disclaimer()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — FULL GRID
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL ACCURACY
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(section_header("Model Accuracy",
        "How well does the model perform on races it has never seen?"), unsafe_allow_html=True)

    report_text  = read_model_report()
    report_lines = report_text.split("\n")

    def extract_section(lines, heading):
        out, inside, past_sep = [], False, False
        for ln in lines:
            if heading in ln: inside = True; continue
            if inside and not past_sep:
                if ln.startswith("==="): past_sep = True
                continue
            if inside and past_sep:
                if ln.startswith("==="): break
                out.append(ln)
        return out

    comp_raw = extract_section(report_lines, "MODEL COMPARISON")
    model_scores = {}; selected_model = ""
    for ln in comp_raw:
        ln = ln.strip()
        if not ln: continue
        if "<-- SELECTED" in ln: selected_model = ln.split(":")[0].strip()
        if ": F1 =" in ln:
            p = ln.replace("<-- SELECTED","").split(": F1 =")
            if len(p)==2:
                try: model_scores[p[0].strip()] = float(p[1].strip())
                except: pass

    # Parse precision & recall for Podium class from classification report
    podium_precision = podium_recall = None
    for ln in report_lines:
        if ln.strip().startswith("Podium"):
            parts = ln.split()
            try:
                podium_precision = float(parts[1])
                podium_recall    = float(parts[2])
            except: pass
            break

    # Score cards
    if model_scores:
        best_f1  = max(model_scores.values())
        n_models = len(model_scores)
        t_trained = (datetime.fromtimestamp(os.path.getmtime("model.pkl")).strftime("%d %b %Y")
                     if os.path.exists("model.pkl") else "N/A")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.markdown(stat_card("Selected Model", selected_model or "—", "🤖"), unsafe_allow_html=True)
        sc2.markdown(stat_card("F1 Score", f"{best_f1:.3f}", "🎯"), unsafe_allow_html=True)
        sc3.markdown(stat_card("Precision", f"{podium_precision:.2f}" if podium_precision else "—", "🔍"), unsafe_allow_html=True)
        sc4.markdown(stat_card("Recall", f"{podium_recall:.2f}" if podium_recall else "—", "📡"), unsafe_allow_html=True)

        st.markdown(
            f"<div style='font-size:13px;color:{TEXT_MUTED};margin-top:6px;margin-bottom:4px;'>"
            "F1 score is a machine learning evaluation metric — not a Formula 1 score."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # Bar chart
        st.markdown(section_header("Model Comparison"), unsafe_allow_html=True)
        score_df = pd.DataFrame(list(model_scores.items()), columns=["Model","F1"]).sort_values("F1", ascending=True)
        bar_c = [RED if m==selected_model else BAR_NEUTRAL for m in score_df["Model"]]

        fig_s = go.Figure(go.Bar(
            x=score_df["F1"], y=score_df["Model"], orientation="h",
            marker=dict(color=bar_c, line=dict(width=0)),
            text=[f"  {v:.3f}" for v in score_df["F1"]],
            textposition="outside", textfont=dict(color=PLOT_TEXT, size=12),
            hovertemplate="<b>%{y}</b><br>F1: %{x:.4f}<extra></extra>",
        ))
        fig_s.update_layout(
            xaxis=dict(title="F1 Score (0 = worst, 1 = perfect)", range=[0,1.18],
                       color=PLOT_TEXT, gridcolor=PLOT_GRID, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=13, color=PLOT_TEXT), showgrid=False),
            height=260, margin=dict(l=10,r=70,t=10,b=40),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PLOT_TEXT),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # Metric explanation cards
    st.markdown(section_header("What Do the Scores Mean?"), unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3)
    ex1.markdown(info_card("🎯","F1 Score",
        "A machine learning metric — not Formula 1. Balances precision and recall into a single number. 1.0 is perfect. Above 0.80 is considered strong for real-world prediction tasks."),
        unsafe_allow_html=True)
    ex2.markdown(info_card("🔍","Precision",
        "When the model says 'podium', how often is it right? High precision means it doesn't cry wolf on drivers who won't make it."),
        unsafe_allow_html=True)
    ex3.markdown(info_card("📡","Recall",
        "Of all the podiums that actually happened, how many did the model catch? High recall means it rarely misses a genuine podium."),
        unsafe_allow_html=True)

    # Training the Model
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(section_header("Training the Model"), unsafe_allow_html=True)
    tm1, tm2 = st.columns([1, 1], gap="large")
    with tm1:
        algos = [
            (1, "Logistic Regression", "Draws a mathematical boundary between podium and non-podium"),
            (2, "Random Forest",       "Builds 200 decision trees and combines their votes"),
            (3, "Gradient Boosting",   "Chains many small trees, each fixing the last one's errors"),
            (4, "XGBoost",             "An optimised gradient boosting variant — fast and accurate \u2014 \u2705 used by this predictor"),
        ]
        algo_rows = ""
        for num, aname, adesc in algos:
            algo_rows += (
                f"<div style='display:flex;align-items:flex-start;gap:10px;padding:8px 0;"
                f"border-bottom:1px solid {BORDER};'>"
                f"<div style='background:{RED};color:white;border-radius:4px;padding:2px 8px;"
                f"font-size:11px;font-weight:600;white-space:nowrap;margin-top:2px;'>{num}</div>"
                f"<div>"
                f"<div style='font-size:13px;font-weight:600;color:{TEXT};'>{aname}</div>"
                f"<div style='font-size:12px;color:{TEXT_SUB};'>{adesc}</div>"
                f"</div></div>"
            )
        algo_html = (
            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:10px;padding:20px;'>"
            f"<div style='font-size:14px;font-weight:600;color:{TEXT};margin-bottom:12px;'>"
            f"Four Algorithms Compared</div>"
            f"{algo_rows}"
            f"</div>"
        )
        st.markdown(algo_html, unsafe_allow_html=True)
    with tm2:
        rules_html = (
            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:10px;padding:20px;'>"
            f"<div style='font-size:14px;font-weight:600;color:{TEXT};margin-bottom:12px;'>The Rules</div>"
            f"<div style='font-size:13px;color:{TEXT_SUB};line-height:1.7;'>"
            f"<div style='margin-bottom:10px;'>"
            f"<span style='color:{RED};font-weight:600;'>No peeking at the future.</span> "
            f"Training always uses past races only. Tested using TimeSeriesSplit — "
            f"5 splits, each one training on earlier races and testing on later ones."
            f"</div>"
            f"<div style='margin-bottom:10px;'>"
            f"<span style='color:{RED};font-weight:600;'>Handling the imbalance.</span> "
            f"Only 3 of 20 drivers podium per race (15%). Models were given extra weight "
            f"on podium examples so they do not just predict no podium for everyone."
            f"</div>"
            f"<div>"
            f"<span style='color:{RED};font-weight:600;'>Winner chosen by F1 score</span> — "
            f"a balanced metric that rewards both catching podiums and avoiding false positives."
            f"</div>"
            f"</div></div>"
        )
        st.markdown(rules_html, unsafe_allow_html=True)

    # ── How the Scores Are Calculated ────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(section_header("How the Scores Are Calculated"), unsafe_allow_html=True)

    formulas = [
        (
            "F1 Score",
            "2 × (Precision × Recall) / (Precision + Recall)",
            "The harmonic mean of precision and recall — a single balanced score. "
            "Penalises models that are strong on one but weak on the other.",
        ),
        (
            "Precision",
            "TP / (TP + FP)",
            "Of every podium the model predicted, how many were correct. "
            "TP = true positives (correctly predicted podiums). FP = false positives (wrongly predicted podiums).",
        ),
        (
            "Recall",
            "TP / (TP + FN)",
            "Of every podium that actually happened, how many did the model catch. "
            "FN = false negatives (real podiums the model missed).",
        ),
    ]

    f1c, f2c, f3c = st.columns(3)
    for col, (name, formula, desc) in zip([f1c, f2c, f3c], formulas):
        col.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;height:100%;'>"
            f"<div style='font-size:13px;font-weight:600;color:{TEXT};margin-bottom:10px;'>{name}</div>"
            f"<div style='font-size:13px;font-family:monospace;color:{RED};background:{SURFACE2};"
            f"border-radius:6px;padding:8px 12px;margin-bottom:12px;'>{formula}</div>"
            f"<div style='font-size:12px;color:{TEXT_SUB};line-height:1.6;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(section_header("Confusion Matrix",
        "All 5 folds combined — real counts from the model's test predictions"), unsafe_allow_html=True)

    cm_section = extract_section(report_lines, "CONFUSION MATRIX")
    cm_tp = cm_fp = cm_fn = cm_tn = None
    for ln in cm_section:
        if "TP=" in ln:
            try:
                parts = dict(p.split("=") for p in ln.split() if "=" in p)
                cm_tp = int(parts["TP"]); cm_fp = int(parts["FP"])
                cm_fn = int(parts["FN"]); cm_tn = int(parts["TN"])
            except: pass

    if cm_tp is not None:
        cm_html = (
            f"<div style='display:flex;justify-content:center;margin-top:8px;'>"
            f"<div style='display:inline-block;border:1px solid {BORDER};border-radius:10px;overflow:hidden;'>"
            # Header row
            f"<div style='display:flex;'>"
            f"<div style='width:280px;background:{SURFACE};padding:16px 24px;font-size:13px;font-weight:600;color:{TEXT_MUTED};border-right:1px solid {BORDER};border-bottom:1px solid {BORDER};'></div>"
            f"<div style='width:240px;background:{SURFACE};padding:16px 24px;font-size:13px;font-weight:600;color:{TEXT_SUB};text-align:center;border-right:1px solid {BORDER};border-bottom:1px solid {BORDER};'>Predicted Podium</div>"
            f"<div style='width:280px;background:{SURFACE};padding:16px 24px;font-size:13px;font-weight:600;color:{TEXT_SUB};text-align:center;border-bottom:1px solid {BORDER};'>Predicted No Podium</div>"
            f"</div>"
            # Row 1 — Actually Podium
            f"<div style='display:flex;'>"
            f"<div style='width:280px;background:{SURFACE};padding:16px 24px;font-size:13px;font-weight:600;color:{TEXT_SUB};border-right:1px solid {BORDER};border-bottom:1px solid {BORDER};display:flex;align-items:center;'>Actually Podium</div>"
            f"<div style='width:240px;background:#0d2b1a;padding:20px 24px;text-align:center;border-right:1px solid {BORDER};border-bottom:1px solid {BORDER};'>"
            f"<div style='font-size:36px;font-weight:700;color:#4caf50;'>{cm_tp}</div>"
            f"<div style='font-size:12px;color:{TEXT_MUTED};margin-top:4px;'>TP — Correct podium calls</div>"
            f"</div>"
            f"<div style='width:280px;background:#2b0d0d;padding:20px 24px;text-align:center;border-bottom:1px solid {BORDER};'>"
            f"<div style='font-size:36px;font-weight:700;color:{RED};'>{cm_fn}</div>"
            f"<div style='font-size:12px;color:{TEXT_MUTED};margin-top:4px;'>FN — Missed podiums</div>"
            f"</div>"
            f"</div>"
            # Row 2 — Actually No Podium
            f"<div style='display:flex;'>"
            f"<div style='width:280px;background:{SURFACE};padding:16px 24px;font-size:13px;font-weight:600;color:{TEXT_SUB};border-right:1px solid {BORDER};display:flex;align-items:center;'>Actually No Podium</div>"
            f"<div style='width:240px;background:#2b0d0d;padding:20px 24px;text-align:center;border-right:1px solid {BORDER};'>"
            f"<div style='font-size:36px;font-weight:700;color:{RED};'>{cm_fp}</div>"
            f"<div style='font-size:12px;color:{TEXT_MUTED};margin-top:4px;'>FP — False podium calls</div>"
            f"</div>"
            f"<div style='width:280px;background:#0d2b1a;padding:20px 24px;text-align:center;'>"
            f"<div style='font-size:36px;font-weight:700;color:#4caf50;'>{cm_tn}</div>"
            f"<div style='font-size:12px;color:{TEXT_MUTED};margin-top:4px;'>TN — Correct non-podium calls</div>"
            f"</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )
        st.markdown(cm_html, unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;justify-content:center;gap:24px;margin-top:10px;flex-wrap:wrap;'>"
            f"<span style='font-size:12px;color:{TEXT_MUTED};'><b style='color:#4caf50;'>TP</b> — True Positive</span>"
            f"<span style='font-size:12px;color:{TEXT_MUTED};'><b style='color:{RED};'>FP</b> — False Positive</span>"
            f"<span style='font-size:12px;color:{TEXT_MUTED};'><b style='color:{RED};'>FN</b> — False Negative</span>"
            f"<span style='font-size:12px;color:{TEXT_MUTED};'><b style='color:#4caf50;'>TN</b> — True Negative</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    disclaimer()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    # ── Confidence Score ─────────────────────────────────────────────────────
    st.markdown(section_header("Understanding the Confidence Score"), unsafe_allow_html=True)
    conf_html = (
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;margin-bottom:8px;'>"
        f"<div style='font-size:13px;color:{TEXT_SUB};line-height:1.8;'>"
        f"When you click <b style='color:{TEXT};'>Predict</b>, the model outputs a number between 0% and 100% — that's the confidence. "
        f"For example, <b style='color:{TEXT};'>87%</b> means the model thinks there is an 87% chance the driver finishes on the podium, "
        f"while <b style='color:{TEXT};'>12%</b> means it thinks a podium is unlikely."
        f"</div>"
        f"<div style='display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;'>"
        f"<div style='flex:1;min-width:160px;background:{SURFACE2};border-radius:8px;padding:14px;'>"
        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:4px;'>50% or above</div>"
        f"<div style='font-size:12px;color:{TEXT_SUB};'>Podium predicted. The higher the number, the more certain the model is.</div>"
        f"</div>"
        f"<div style='flex:1;min-width:160px;background:{SURFACE2};border-radius:8px;padding:14px;'>"
        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:4px;'>Below 50%</div>"
        f"<div style='font-size:12px;color:{TEXT_SUB};'>Outside podium predicted. A 51% and a 95% are both podium calls — but very different levels of certainty.</div>"
        f"</div>"
        f"<div style='flex:1;min-width:160px;background:{SURFACE2};border-radius:8px;padding:14px;'>"
        f"<div style='font-size:12px;font-weight:600;color:{TEXT};margin-bottom:4px;'>In practice</div>"
        f"<div style='font-size:12px;color:{TEXT_SUB};'>Top drivers at strong circuits: 80–99%. Mid-field: 10–40%. Backmarkers: 1–5%.</div>"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(conf_html, unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Prediction steps ─────────────────────────────────────────────────────
    st.markdown(section_header("What Happens When You Click Predict"), unsafe_allow_html=True)

    steps = [
        ("Inputs collected",
         "Driver name and circuit — that's all. Everything else is looked up automatically."),
        ("Historical stats loaded",
         "The driver's recent form, career podium rate, DNF rate, championship position, and team reliability are looked up from stored training data."),
        ("Features scaled",
         "All 13 values are normalised to the same scale so the model treats them fairly."),
        ("Model scores the input",
         "XGBoost runs the features through an optimised sequence of decision trees, each one correcting the errors of the last."),
        ("Probability calculated",
         "The final tree ensemble outputs a podium probability between 0% and 100%."),
        ("Threshold applied",
         "50% or above = podium predicted. Below 50% = outside podium."),
    ]

    for i, (step_title, step_desc) in enumerate(steps):
        step_html = (
            f"<div style='display:flex;align-items:flex-start;gap:14px;margin-bottom:4px;'>"
            f"<div style='background:{RED};color:white;border-radius:50%;width:32px;height:32px;"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-weight:700;font-size:13px;flex-shrink:0;margin-top:2px;'>{i + 1}</div>"
            f"<div style='padding-bottom:4px;'>"
            f"<div style='font-size:14px;font-weight:600;color:{TEXT};'>{step_title}</div>"
            f"<div style='font-size:13px;color:{TEXT_SUB};margin-top:2px;'>{step_desc}</div>"
            f"</div></div>"
        )
        st.markdown(step_html, unsafe_allow_html=True)
        if i < len(steps) - 1:
            st.markdown(
                f"<div style='width:2px;height:16px;background:{BORDER};margin-left:15px;margin-bottom:4px;'></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.warning(
        "**What the model can't see:** weather, safety cars, first-lap incidents, pit strategy, "
        "tyre degradation, mechanical failures during the race, or any information that only "
        "emerges once the race has started. Predictions are probability estimates based on "
        "historical patterns — not guarantees. Formula 1 is unpredictable by design."
    )
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Data sources ─────────────────────────────────────────────────────────
    st.markdown(section_header("The Data"), unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:14px;color:{TEXT_SUB};margin-bottom:16px;'>"
        "Six official F1 data sources are merged — covering every race from 2021 to mid-2025."
        "</div>",
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)
    d4, d5, d6 = st.columns(3)
    for col, icon, title, desc in [
        (d1, "🏁", "Race Results",   "Finishing positions, points scored, and retirements for every driver in every race."),
        (d2, "⏱️", "Qualifying",     "Lap times from qualifying, used to calculate each driver's gap to pole position."),
        (d3, "👤", "Drivers",        "Driver identities and career histories spanning all 2021–2025 entries."),
        (d4, "🏎️", "Constructors",   "Team information — which car each driver raced for in every event."),
        (d5, "🗺️", "Circuits",       "Details on all 28 tracks used across the five seasons."),
        (d6, "📅", "Race Calendar",  "Dates and round numbers — critical for keeping training strictly chronological."),
    ]:
        col.markdown(info_card(icon, title, desc), unsafe_allow_html=True)

    # ── Features ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(section_header("The 13 Features"), unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px;color:{TEXT_SUB};margin:-8px 0 8px 0;'>Features are the individual, measurable properties or characteristics used as inputs for a model.</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:14px;color:{TEXT_SUB};margin-bottom:16px;'>"
        "Raw data is transformed into signals that actually matter. All historical features "
        "use only past race results — no future data ever leaks in. "
        "Grid position is intentionally excluded so the model predicts from driver ability, not just where they started."
        "</div>",
        unsafe_allow_html=True,
    )

    feat_groups = {
        "⏱️  Qualifying Pace": [
            ("Qualifying Gap to Pole", "Seconds behind the fastest qualifier — captures raw pace."),
        ],
        "📈  Recent Form": [
            ("Avg Points (Last 5 Races)",         "Current momentum in terms of championship points."),
            ("Avg Finishing Position (Last 5)",   "Where the driver has been finishing recently. Lower is better."),
            ("Podiums in Last 5 Races",           "Pure podium count from the last five races (0 to 5)."),
        ],
        "🏆  Career Record": [
            ("Career Podium Rate",   "Percentage of all career races the driver has finished in the top 3."),
            ("Career Win Rate",      "Percentage of all career races the driver has won outright."),
            ("Career Race Starts",   "Total Formula 1 starts — a measure of overall experience."),
        ],
        "🔧  Reliability": [
            ("Driver DNF Rate",    "How often the driver has retired in the last 10 races."),
        ],
        "🏅  Championship Context": [
            ("Driver Championship Position",  "Current standing in the Drivers' Championship."),
            ("Team Championship Position",    "Current standing in the Constructors' Championship — a proxy for car performance."),
        ],
        "🗺️  Circuit & Identity": [
            ("Best Past Finish at This Circuit",  "The driver's best-ever result at this track. 20 if never raced here."),
            ("Driver Identity",                   "Encoded driver identity — lets the model learn driver-specific patterns."),
            ("Circuit Identity",                  "Encoded circuit — some tracks suit certain drivers more than others."),
        ],
    }

    for group, features in feat_groups.items():
        with st.expander(group, expanded=False):
            for feat_name, feat_desc in features:
                html = (
                    f"<div style='padding:10px 0;border-bottom:1px solid {BORDER};'>"
                    f"<div style='font-size:14px;font-weight:600;color:{TEXT};margin-bottom:3px;'>{feat_name}</div>"
                    f"<div style='font-size:13px;color:{TEXT_SUB};'>{feat_desc}</div>"
                    f"</div>"
                )
                st.markdown(html, unsafe_allow_html=True)

    # ── Feature Importance ───────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(section_header("Feature Importance",
        "Higher = more influence on the final prediction"), unsafe_allow_html=True)
    FEAT_LABELS = {
        "quali_gap_to_pole":           "Qualifying Gap to Pole",
        "dnf_rate":                    "Driver DNF Rate",
        "rolling_points_5":            "Avg Points (Last 5 Races)",
        "driver_experience":           "Career Race Starts",
        "rolling_finish_5":            "Avg Finishing Position (Last 5)",
        "driver_encoded":              "Driver Identity",
        "podiums_last_5":              "Podiums in Last 5 Races",
        "career_win_rate":             "Career Win Rate",
        "career_podium_rate":          "Career Podium Rate",
        "circuit_id":                  "Circuit Identity",
        "best_finish_at_circuit":      "Best Past Finish at Circuit",
        "driver_championship_pos":     "Driver Championship Position",
        "constructor_championship_pos":"Team Championship Position",
    }

    imp_section = extract_section(report_lines, "FEATURE IMPORTANCES")
    imp_rows = []
    for ln in imp_section:
        parts = ln.split()
        if len(parts) >= 3:
            try:
                feat = parts[1]
                val  = float(parts[2])
                imp_rows.append((FEAT_LABELS.get(feat, feat), val))
            except ValueError:
                pass

    if imp_rows:
        imp_df = pd.DataFrame(imp_rows, columns=["Feature", "Importance"])
        imp_df = imp_df.sort_values("Importance", ascending=True)
        bar_colors = [RED if imp_df["Importance"].iloc[i] == imp_df["Importance"].max()
                      else BAR_NEUTRAL for i in range(len(imp_df))]

        fig_i = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"  {v*100:.1f}%" for v in imp_df["Importance"]],
            textposition="outside",
            textfont=dict(color=PLOT_TEXT, size=11),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_i.update_layout(
            xaxis=dict(title="Importance (share of total)", range=[0, max(imp_df["Importance"]) * 1.25],
                       color=PLOT_TEXT, gridcolor=PLOT_GRID, showticklabels=False),
            yaxis=dict(tickfont=dict(size=12, color=PLOT_TEXT), showgrid=False),
            height=420, margin=dict(l=10, r=80, t=10, b=40),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PLOT_TEXT),
        )
        st.plotly_chart(fig_i, use_container_width=True)

    st.info(
        "**Training approach:** All 4 models were trained using TimeSeriesSplit (5 folds) — "
        "older races train, newer races test. Class weights balance podium vs non-podium examples. "
        "The model with the highest F1 score was selected automatically."
    )

    feat_expl_html = (
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-radius:10px;padding:20px;margin-top:12px;'>"
        f"<div style='font-size:14px;font-weight:600;color:{TEXT};margin-bottom:12px;'>"
        f"Why does feature importance matter?</div>"
        f"<div style='font-size:13px;color:{TEXT_SUB};line-height:1.8;'>"
        f"<div style='margin-bottom:8px;'>"
        f"<span style='color:{RED};font-weight:600;'>Which signals actually drive predictions?</span> "
        f"Importance scores reveal which inputs the model relies on most. "
        f"A feature with near-zero importance adds noise, not signal."
        f"</div>"
        f"<div style='margin-bottom:8px;'>"
        f"<span style='color:{RED};font-weight:600;'>Does it match common sense?</span> "
        f"Qualifying gap to pole dominates because raw pace is the strongest predictor of a podium finish. "
        f"Championship position and circuit history contribute, but less so."
        f"</div>"
        f"<div>"
        f"<span style='color:{RED};font-weight:600;'>What was removed?</span> "
        f"Grid position was intentionally excluded so the model predicts from driver ability and form, "
        f"not just where they happened to start."
        f"</div>"
        f"</div></div>"
    )
    st.markdown(feat_expl_html, unsafe_allow_html=True)

    disclaimer()
