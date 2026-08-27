import os
import re
import tempfile
import textwrap

import streamlit as st

from src.analyzer import analyze_resume
from src.predictor import predict_top_roles
from src.resume_parser import extract_resume_text
from src.skill_extractor import load_skills, extract_skills


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "filename" not in st.session_state:
    st.session_state.filename = ""

if "dashboard_page" not in st.session_state:
    st.session_state.dashboard_page = "Overview"


# ============================================================
# HELPERS
# ============================================================

def html(value: str):
    """Render HTML without Streamlit treating indentation as a code block."""
    st.markdown(textwrap.dedent(value).strip(), unsafe_allow_html=True)


def safe_percent(value):
    try:
        return max(0.0, min(float(value), 100.0))
    except (TypeError, ValueError):
        return 0.0


def get_value(data, *keys, default=None):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def reset_analysis():
    st.session_state.analysis = None
    st.session_state.filename = ""
    st.session_state.dashboard_page = "Overview"


def role_value(role, *keys, default=None):
    if isinstance(role, dict):
        return get_value(role, *keys, default=default)
    return default


def render_skill_pills(skills, kind="skill", as_html=False):
    if not skills:
        if kind == "gap":
            content = '<div class="empty-text">🎉 No missing required skills!</div>'
        else:
            content = '<div class="empty-text">No known skills were detected.</div>'
        if as_html:
            return content
        html(content)
        return

    pills = []
    for skill in skills:
        cls = "skill-pill gap-pill" if kind == "gap" else "skill-pill"
        icon = "✕" if kind == "gap" else "✓"
        pills.append(f'<span class="{cls}">{icon} {skill}</span>')

    pill_html = "".join(pills)
    if as_html:
        return pill_html
    html('<div class="pill-wrap">' + pill_html + "</div>")


def render_topbar():
    page = st.session_state.dashboard_page
    filename = st.session_state.filename

    html(f"""
    <div class="topbar">
        <div class="brand-block">
            <div class="brand-icon">🤖</div>
            <div>
                <div class="brand-title">AI Resume Analyzer</div>
                <div class="brand-subtitle">Analyze&nbsp; + &nbsp;Predict&nbsp; + &nbsp;Grow</div>
            </div>
        </div>
        <div class="topbar-file">📄 {filename}</div>
    </div>
    """)

    pages = [
        ("📋", "Overview"),
        ("🎯", "Job Matches"),
        ("🛠", "Skills"),
        ("🎓", "Education"),
        ("💼", "Experience"),
        ("💰", "Job Information"),
    ]

    st.markdown('<div class="toolbar">', unsafe_allow_html=True)
    cols = st.columns([1.0, 1.0, 1.0, 1.0, 1.12, 1.0, 1.42])

    for index, ((icon, label), col) in enumerate(zip(pages, cols[:6])):
        with col:
            active = page == label
            if st.button(
                f"{icon}  {label}",
                key=f"toolbar_{label}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.dashboard_page = label
                st.rerun()

    with cols[6]:
        if st.button(
            "↻  Analyze Another Resume",
            key="toolbar_new_resume",
            use_container_width=True,
            type="secondary",
        ):
            reset_analysis()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_dashboard_header():
    filename = st.session_state.filename
    html(f"""
    <div class="dashboard-header">
        <div>
            <div class="eyebrow">TURN YOUR SKILLS INTO OPPORTUNITIES</div>
            <div class="dashboard-title">
                Resume Analysis <span>Dashboard</span>
            </div>
            <div class="dashboard-subtitle">
                Your personalized career insights, powered by AI
            </div>
        </div>
        <div class="analysis-file-card">
            <div class="file-label">Analyzed Resume</div>
            <div class="file-name">📄 {filename}</div>
            <div class="complete-badge">Analysis Complete ✓</div>
        </div>
    </div>
    """)


def render_footer():
    html("""
    <div class="footer">
        <div>🌱 <b>Analyze Today</b> &nbsp;•&nbsp; Build Tomorrow &nbsp;•&nbsp; You've Got This! 💜</div>
        <div>AI Resume Analyzer &nbsp;•&nbsp; NLP + TF-IDF + Calibrated Linear SVM</div>
    </div>
    """)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #172b68;
    --navy-dark: #102052;
    --blue: #3158c8;
    --purple: #6845e8;
    --pink: #df3d9b;
    --text: #16275b;
    --muted: #5e6b91;
    --line: rgba(35, 54, 112, 0.13);
    --white: #ffffff;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(243, 197, 255, .72), transparent 27%),
        radial-gradient(circle at 96% 4%, rgba(187, 207, 255, .78), transparent 31%),
        radial-gradient(circle at 72% 65%, rgba(229, 218, 255, .55), transparent 28%),
        linear-gradient(135deg, #fbf8ff 0%, #f2f5ff 50%, #ffffff 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding: 0 4.1rem 2rem;
}

/* ======================== TOP BAR ======================== */

.topbar {
    min-height: 88px;
    margin: 0 -4.1rem;
    padding: 0 4.1rem;
    display: flex;
    align-items: center;
    gap: 24px;
    background: linear-gradient(110deg, #0d1b4d 0%, #253f99 55%, #4b46a8 100%);
    box-shadow: 0 12px 30px rgba(23, 43, 104, .24);
}

.brand-block {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-icon {
    font-size: 39px;
}

.brand-title {
    color: #ffffff !important;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.15;
}

.brand-subtitle {
    color: #e4e9ff !important;
    font-size: 12px;
    font-weight: 600;
    margin-top: 5px;
}

.topbar-file {
    margin-left: auto;
    padding: 13px 18px;
    border-radius: 14px;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.22);
    color: #ffffff !important;
    font-size: 13px;
    font-weight: 800;
}

/* ======================== TOOLBAR ======================== */

.toolbar {
    margin: 17px -4.1rem 0;
    padding: 0 4.1rem;
}

.toolbar [data-testid="stHorizontalBlock"] {
    gap: 10px;
}

.toolbar .stButton {
    width: 100%;
}

.toolbar .stButton > button {
    min-height: 54px !important;
    border-radius: 14px !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    letter-spacing: .05px;
    transition: all .18s ease;
    white-space: nowrap;
}

/* Inactive navigation buttons */
.toolbar .stButton > button[kind="secondary"],
.toolbar .stButton > button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #20366f !important;
    border: 1px solid #cbd4ee !important;
    box-shadow: 0 5px 15px rgba(34, 48, 101, .08) !important;
}

.toolbar .stButton > button[kind="secondary"]:hover,
.toolbar .stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: #edf1ff !important;
    color: #4b35bd !important;
    border-color: #7966df !important;
    transform: translateY(-1px);
}

/* Active navigation button */
.toolbar .stButton > button[kind="primary"],
.toolbar .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6247e8 0%, #9b3fd0 100%) !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    box-shadow: 0 9px 20px rgba(98, 70, 224, .28) !important;
}

.toolbar .stButton > button[kind="primary"]:hover,
.toolbar .stButton > button[data-testid="stBaseButton-primary"]:hover {
    color: #ffffff !important;
    background: linear-gradient(135deg, #563bdd 0%, #8f31c5 100%) !important;
}

/* Make button text/icons inherit the intended contrast */
.toolbar .stButton > button p,
.toolbar .stButton > button span {
    color: inherit !important;
}

/* ======================== HEADERS ======================== */

.dashboard-header {
    padding: 48px 0 29px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 30px;
}

.eyebrow {
    color: #6657bd !important;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 800;
}

.dashboard-title {
    margin-top: 9px;
    color: #14265d !important;
    font-size: 46px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -1.7px;
}

.dashboard-title span {
    background: linear-gradient(90deg, #6244e8, #df3b9c);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.dashboard-subtitle {
    margin-top: 11px;
    color: #52658f !important;
    font-size: 17px;
    font-weight: 500;
}

.analysis-file-card {
    min-width: 350px;
    padding: 19px 22px;
    border-radius: 18px;
    background: rgba(255,255,255,.93);
    border: 1px solid rgba(68,82,145,.13);
    box-shadow: 0 13px 30px rgba(42,54,121,.11);
}

.file-label {
    color: #667397 !important;
    font-size: 12px;
    font-weight: 700;
}

.file-name {
    margin-top: 5px;
    color: #15275e !important;
    font-size: 18px;
    font-weight: 800;
}

.complete-badge {
    display: inline-block;
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 999px;
    background: #dff7ec;
    color: #087a58 !important;
    font-size: 11px;
    font-weight: 800;
}

.section-heading {
    color: #172b68 !important;
    font-size: 25px;
    font-weight: 800;
    margin-top: 9px;
}

.section-subheading {
    color: #5f6e94 !important;
    font-size: 13px;
    font-weight: 500;
    margin-top: 5px;
    margin-bottom: 17px;
}

.divider {
    height: 1px;
    background: rgba(35, 54, 112, 0.15);
    margin: 28px 0 32px;
}

/* ======================== KPI ======================== */

.kpi {
    min-height: 148px;
    padding: 21px 22px;
    border-radius: 19px;
    border: 1px solid rgba(62,75,139,.11);
    box-shadow: 0 12px 28px rgba(40,52,119,.09);
}

.kpi-pink { background: linear-gradient(135deg, #ffe0ec, #fff5f9); }
.kpi-mint { background: linear-gradient(135deg, #ddf8f0, #f5fffc); }
.kpi-purple { background: linear-gradient(135deg, #e8e1ff, #faf8ff); }
.kpi-yellow { background: linear-gradient(135deg, #ffefd0, #fffaf1); }

.kpi-icon {
    width: 51px;
    height: 51px;
    margin-right: 14px;
    float: left;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,.76);
    font-size: 23px;
}

.kpi-title {
    padding-top: 3px;
    color: #4d5d84 !important;
    font-size: 12px;
    font-weight: 800;
}

.kpi-value {
    color: #14275f !important;
    font-size: 34px;
    font-weight: 800;
    margin-top: 3px;
}

.kpi-note {
    color: #637195 !important;
    font-size: 11px;
    margin-top: 3px;
}

/* ======================== CARDS ======================== */

.card {
    background: rgba(255,255,255,.92);
    border: 1px solid rgba(65,79,142,.12);
    border-radius: 19px;
    box-shadow: 0 11px 27px rgba(40,50,115,.08);
}

.role-card {
    min-height: 180px;
    padding: 26px 30px;
    position: relative;
    overflow: hidden;
}

.role-card:after {
    content: '✨';
    position: absolute;
    right: 35px;
    bottom: 10px;
    font-size: 78px;
    opacity: .12;
}

.role-label {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: #eeeaff;
    color: #4f38c4 !important;
    font-size: 11px;
    letter-spacing: 1.2px;
    font-weight: 800;
}

.role-name {
    margin-top: 13px;
    color: #14275f !important;
    font-size: 32px;
    font-weight: 800;
}

.role-description {
    margin-top: 7px;
    color: #5d6e96 !important;
    font-size: 13px;
    font-weight: 500;
}

.req-card {
    min-height: 180px;
    padding: 24px;
}

.req-title {
    color: #182b66 !important;
    font-size: 15px;
    font-weight: 800;
}

.req-status {
    display: inline-block;
    margin-top: 16px;
    padding: 9px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
}

.good { background: #dcf7ec; color: #087a58 !important; }
.warning { background: #fff0d7; color: #a75d00 !important; }

.req-note {
    margin-top: 12px;
    color: #5e6e94 !important;
    font-size: 12px;
    line-height: 1.6;
}

/* ======================== MATCH CARDS ======================== */

.match-card {
    min-height: 157px;
    padding: 20px;
    background: rgba(255,255,255,.93);
    border: 1px solid rgba(65,79,142,.12);
    border-radius: 18px;
    box-shadow: 0 9px 24px rgba(40,50,115,.07);
}

.rank {
    width: 45px;
    height: 45px;
    margin-right: 12px;
    float: left;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #ffe8ae, #ffd56b);
    color: #17285f !important;
    font-size: 17px;
    font-weight: 800;
}

.match-role {
    padding-top: 4px;
    color: #172a64 !important;
    font-size: 16px;
    font-weight: 800;
}

.match-score {
    float: right;
    color: #172a64 !important;
    font-size: 17px;
    font-weight: 800;
}

.progress-bg {
    clear: both;
    height: 10px;
    margin-top: 17px;
    border-radius: 999px;
    overflow: hidden;
    background: #dfe3ee;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6048e8, #e344a4);
}

.match-note {
    color: #5f6f95 !important;
    font-size: 11px;
    margin-top: 9px;
    line-height: 1.55;
}

/* ======================== DETAIL CARDS ======================== */

.detail-card {
    min-height: 190px;
    padding: 23px;
    background: rgba(255,255,255,.93);
    border: 1px solid rgba(65,79,142,.12);
    border-radius: 19px;
    box-shadow: 0 9px 24px rgba(40,50,115,.07);
}

.detail-title {
    color: #182b66 !important;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 15px;
}

.pill-wrap { line-height: 2.6; }

.skill-pill {
    display: inline-block;
    margin: 4px 4px;
    padding: 8px 12px;
    border-radius: 999px;
    background: #e7f8f1;
    border: 1px solid #c5eadc;
    color: #126f55 !important;
    font-size: 12px;
    font-weight: 800;
}

.gap-pill {
    background: #fff0f5;
    border-color: #ffc9dc;
    color: #cf356c !important;
}

.empty-text {
    padding: 14px;
    border-radius: 12px;
    background: #eef3ff;
    color: #3d5f9e !important;
    font-size: 12px;
    font-weight: 700;
}

.info-box {
    margin-top: 11px;
    padding: 14px;
    border-radius: 12px;
    background: #eef4ff;
    color: #315a98 !important;
    font-size: 13px;
    font-weight: 700;
}

.big-percent {
    color: #172a64 !important;
    font-size: 34px;
    font-weight: 800;
    margin-top: 12px;
}

.stat-card {
    padding: 24px;
    min-height: 135px;
    border-radius: 18px;
    background: rgba(255,255,255,.93);
    border: 1px solid rgba(65,79,142,.12);
    box-shadow: 0 9px 23px rgba(40,50,115,.07);
}

.stat-label {
    color: #5c6c93 !important;
    font-size: 13px;
    font-weight: 700;
}

.stat-value {
    color: #172a64 !important;
    font-size: 30px;
    font-weight: 800;
    margin-top: 7px;
}

/* ======================== NATIVE STREAMLIT TEXT ======================== */

.stMarkdown, .stText, .stCaption, p, label {
    color: #26386f;
    font-size: 13px;
}

[data-testid="stMetricLabel"] {
    color: #536287 !important;
    font-size: 13px !important;
}

[data-testid="stMetricValue"] {
    color: #172a64 !important;
    font-size: 30px !important;
}

/* Make success/warning messages readable */
[data-testid="stAlert"] {
    font-size: 13px !important;
}

[data-testid="stAlert"] p {
    font-size: 13px !important;
}

/* Upload controls */
[data-testid="stFileUploader"] {
    border-radius: 15px;
}

.stButton > button {
    font-size: 14px !important;
}

/* ======================== FOOTER ======================== */

.footer {
    margin-top: 38px;
    padding: 18px 0 7px;
    border-top: 1px solid rgba(35,54,112,.15);
    display: flex;
    justify-content: space-between;
    gap: 20px;
    color: #617096 !important;
    font-size: 11px;
}

@media (max-width: 1100px) {
    .toolbar .stButton > button {
        font-size: 12px !important;
    }

    .dashboard-title, .upload-title {
        font-size: 38px;
    }
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    .topbar, .toolbar {
        margin-left: -1.2rem;
        margin-right: -1.2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    .dashboard-header {
        flex-direction: column;
        align-items: flex-start;
    }

    .analysis-file-card {
        width: 100%;
        min-width: 0;
    }

    .dashboard-title, .upload-title {
        font-size: 34px;
    }

    .footer {
        flex-direction: column;
    }
}


/* ======================== UPLOAD PAGE ======================== */

.upload-header {
    padding: 66px 0 30px;
}

.upload-title {
    color: #172b68;
    font-size: 43px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -1.5px;
}

.upload-title span {
    background: linear-gradient(90deg, #6244e8, #df3b9c);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.upload-subtitle {
    color: #5c6a91;
    font-size: 15px;
    margin-top: 10px;
}

.upload-card {
    margin-top: 20px;
    padding: 29px;
    border-radius: 20px;
    background: rgba(255,255,255,.84);
    border: 1px solid rgba(66,81,145,.11);
    box-shadow: 0 13px 31px rgba(40,50,115,.09);
}

.upload-card-title {
    color: #182b66;
    font-size: 21px;
    font-weight: 800;
}

.upload-card-text {
    color: #667394;
    font-size: 11px;
    margin-top: 6px;
}

[data-testid="stFileUploader"] {
    margin-top: 14px;
    border-radius: 15px;
    border: 1.5px dashed #9185dc;
    background: rgba(255,255,255,.74);
}

.stButton > button {
    border-radius: 13px;
    min-height: 47px;
    font-weight: 800;
}

/* ======================== FOOTER ======================== */

.footer {
    margin-top: 34px;
    padding: 17px 0 5px;
    border-top: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    gap: 20px;
    color: #697598;
    font-size: 10px;
}

/* ======================== NATIVE TEXT ======================== */

.stMarkdown, .stText, .stCaption, p, label {
    color: #26386f;
}

[data-testid="stMetricLabel"] {
    color: #536287 !important;
}

[data-testid="stMetricValue"] {
    color: #172a64 !important;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    .topbar, .toolbar {
        margin-left: -1.2rem;
        margin-right: -1.2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    .dashboard-header {
        flex-direction: column;
        align-items: flex-start;
    }
    .analysis-file-card {
        width: 100%;
        min-width: 0;
    }
    .dashboard-title, .upload-title {
        font-size: 34px;
    }
}

/* ======================== FINAL UI POLISH ======================== */

/* Strong readable primary/secondary text */
.stApp, .stApp p, .stApp label {
    color: #16285f !important;
}

.section-heading,
.detail-title,
.req-title,
.match-role,
.match-score,
.role-name,
.stat-value,
.kpi-value,
.dashboard-title,
.upload-title {
    color: #10245f !important;
}

.section-subheading,
.dashboard-subtitle,
.role-description,
.req-note,
.match-note,
.kpi-note,
.stat-label,
.upload-subtitle,
.upload-card-text,
.file-label {
    color: #465b8f !important;
}

/* Navigation: high contrast */
.toolbar .stButton > button {
    min-height: 58px !important;
    border-radius: 15px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    color: #172b68 !important;
    background: #ffffff !important;
    border: 1px solid #b9c6e8 !important;
    box-shadow: 0 5px 15px rgba(34, 48, 101, .08) !important;
}

.toolbar .stButton > button[kind="secondary"],
.toolbar .stButton > button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #172b68 !important;
}

.toolbar .stButton > button[kind="secondary"]:hover,
.toolbar .stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: #eef2ff !important;
    color: #4d32b9 !important;
    border-color: #735de0 !important;
}

.toolbar .stButton > button[kind="primary"],
.toolbar .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #5a3fe2 0%, #bd3fa5 100%) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 9px 20px rgba(90, 63, 226, .28) !important;
}

.toolbar .stButton > button[kind="primary"] p,
.toolbar .stButton > button[data-testid="stBaseButton-primary"] p,
.toolbar .stButton > button[kind="primary"] span,
.toolbar .stButton > button[data-testid="stBaseButton-primary"] span {
    color: #ffffff !important;
}

/* Larger readable dashboard text */
.section-heading { font-size: 29px !important; }
.section-subheading { font-size: 16px !important; }
.detail-title { font-size: 20px !important; }
.match-role { font-size: 18px !important; }
.match-score { font-size: 19px !important; }
.match-note { font-size: 13px !important; }
.skill-pill { font-size: 14px !important; padding: 9px 14px !important; }
.empty-text { font-size: 14px !important; }
.info-box { font-size: 15px !important; }
.req-title { font-size: 17px !important; }
.req-status { font-size: 14px !important; }
.req-note { font-size: 14px !important; }
.stat-label { font-size: 15px !important; }
.stat-value { font-size: 32px !important; }
.kpi-title { font-size: 14px !important; }
.kpi-value { font-size: 36px !important; }
.kpi-note { font-size: 13px !important; }
.file-label { font-size: 14px !important; }
.file-name { font-size: 20px !important; }
.complete-badge { font-size: 13px !important; }
.upload-card-title { font-size: 23px !important; }
.upload-card-text { font-size: 14px !important; }

/* Skills stay inside cards */
.skill-card {
    min-height: 0 !important;
    padding-bottom: 22px !important;
}

.skill-card .detail-title {
    margin-bottom: 14px !important;
}

.skill-card .pill-wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    margin: 0 !important;
    padding: 0 !important;
}

.skill-card .skill-pill {
    margin: 0 !important;
    font-size: 15px !important;
    padding: 10px 15px !important;
}

/* Education stays inside the white cards */
.education-items {
    margin-top: 12px;
}

.education-item {
    margin: 8px 0;
    padding: 12px 15px;
    border-radius: 12px;
    background: #eef4ff;
    color: #214b91 !important;
    font-size: 15px;
    font-weight: 700;
    line-height: 1.45;
    border: 1px solid #dce6fb;
}

.education-item.matched {
    background: #e9f8f2;
    color: #08785a !important;
    border-color: #ccecdf;
}

/* Upload controls */
[data-testid="stFileUploader"] button {
    color: #172b68 !important;
    font-size: 15px !important;
    font-weight: 800 !important;
}

.footer {
    color: #53658f !important;
    font-size: 13px !important;
}

@media (max-width: 1100px) {
    .toolbar .stButton > button {
        font-size: 14px !important;
    }
}

@media (max-width: 900px) {
    .dashboard-title, .upload-title {
        font-size: 36px !important;
    }
    .section-heading {
        font-size: 25px !important;
    }
}


/* Final card-content layout: content must remain inside the card */
.detail-card {
    overflow: hidden !important;
}

.detail-card .pill-wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    margin: 18px 0 0 0 !important;
    padding: 0 !important;
    line-height: 1.4 !important;
}

.detail-card .skill-pill {
    display: inline-flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 10px 15px !important;
    font-size: 15px !important;
    line-height: 1.2 !important;
}

.education-items {
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    margin: 18px 0 0 0 !important;
    padding: 0 !important;
}

.education-item {
    box-sizing: border-box !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 13px 16px !important;
    font-size: 15px !important;
    line-height: 1.4 !important;
}


/* ============================================================
   FINAL UI POLISH
   - High-contrast readable text
   - Larger typography
   - Clean navigation colours
   - Upload + Analyze button colours
   - Skills / education stay INSIDE their cards
   ============================================================ */

/* ---------- GLOBAL READABILITY ---------- */
.stApp {
    color: #13265c !important;
}

.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #334b80 !important;
    font-size: 15px !important;
    line-height: 1.55 !important;
}

/* ---------- NAVIGATION ---------- */
.toolbar .stButton > button {
    min-height: 60px !important;
    border-radius: 15px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: 0 !important;
    color: #17306f !important;
    background: #f8faff !important;
    border: 1.5px solid #b7c5e8 !important;
    box-shadow: 0 5px 14px rgba(31, 51, 110, .10) !important;
}

.toolbar .stButton > button:hover {
    color: #4525b8 !important;
    background: #eeeaff !important;
    border-color: #7058dd !important;
    box-shadow: 0 8px 18px rgba(79, 56, 190, .16) !important;
    transform: translateY(-1px);
}

.toolbar .stButton > button[kind="primary"],
.toolbar .stButton > button[data-testid="stBaseButton-primary"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #5639e3 0%, #a936c5 100%) !important;
    border: 1px solid transparent !important;
    box-shadow: 0 9px 20px rgba(86, 57, 227, .28) !important;
}

.toolbar .stButton > button[kind="primary"]:hover,
.toolbar .stButton > button[data-testid="stBaseButton-primary"]:hover {
    color: #ffffff !important;
    background: linear-gradient(135deg, #4529cf 0%, #9226b1 100%) !important;
}

.toolbar .stButton > button p,
.toolbar .stButton > button span {
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
}

/* ---------- MAIN HEADINGS ---------- */
.dashboard-title,
.upload-title {
    font-size: 48px !important;
    line-height: 1.1 !important;
}

.section-heading {
    font-size: 30px !important;
    color: #10255f !important;
    font-weight: 800 !important;
}

.section-subheading {
    font-size: 16px !important;
    color: #435b8f !important;
}

.dashboard-subtitle,
.upload-subtitle {
    font-size: 17px !important;
    color: #40598c !important;
}

/* ---------- CARDS / DETAIL HEADINGS ---------- */
.detail-title {
    font-size: 21px !important;
    color: #10265f !important;
    font-weight: 800 !important;
}

.role-name {
    font-size: 34px !important;
    color: #10265f !important;
}

.role-description {
    font-size: 15px !important;
    color: #4b6090 !important;
}

.req-title {
    font-size: 18px !important;
    color: #10265f !important;
}

.req-note {
    font-size: 14px !important;
    color: #435b8e !important;
}

/* ---------- SKILLS: KEEP PILLS INSIDE WHITE CARD ---------- */
.detail-card.skill-card {
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
    padding-bottom: 22px !important;
}

.detail-card.skill-card .pill-wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    margin: 14px 0 0 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}

.detail-card.skill-card .skill-pill {
    display: inline-flex !important;
    align-items: center !important;
    width: auto !important;
    margin: 0 !important;
    padding: 10px 15px !important;
    border-radius: 999px !important;
    font-size: 15px !important;
    line-height: 1.25 !important;
    font-weight: 800 !important;
}

.detail-card.skill-card .skill-pill,
.detail-card.skill-card .skill-pill * {
    color: #08735a !important;
}

.detail-card.skill-card .gap-pill,
.detail-card.skill-card .gap-pill * {
    color: #c52f68 !important;
}

.skill-pill {
    background: #e8f8f2 !important;
    border: 1px solid #bfe7d8 !important;
}

.gap-pill {
    background: #fff0f5 !important;
    border: 1px solid #ffc5d9 !important;
}

/* ---------- EDUCATION: KEEP ITEMS INSIDE WHITE CARDS ---------- */
.detail-card.education-card {
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
    padding-bottom: 20px !important;
}

.education-items {
    display: flex !important;
    flex-direction: column !important;
    gap: 9px !important;
    width: 100% !important;
    margin: 14px 0 0 0 !important;
    padding: 0 !important;
}

.education-item {
    display: block !important;
    box-sizing: border-box !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 12px 15px !important;
    border-radius: 11px !important;
    background: #edf4ff !important;
    border: 1px solid #d4e2fa !important;
    color: #214b91 !important;
    font-size: 15px !important;
    font-weight: 750 !important;
    line-height: 1.45 !important;
}

.education-item.matched {
    background: #e9f8f2 !important;
    border-color: #c5e9db !important;
    color: #08755a !important;
}

/* ---------- KPI / MATCH TEXT ---------- */
.kpi-title {
    font-size: 14px !important;
    color: #3e5688 !important;
}

.kpi-value {
    font-size: 37px !important;
    color: #10265f !important;
}

.kpi-note {
    font-size: 13px !important;
    color: #4f6593 !important;
}

.match-role {
    font-size: 18px !important;
    color: #10265f !important;
}

.match-score {
    font-size: 19px !important;
    color: #10265f !important;
}

.match-note {
    font-size: 13px !important;
    color: #4d6493 !important;
}

/* ---------- UPLOAD AREA ---------- */
.upload-card-title {
    font-size: 24px !important;
    color: #10265f !important;
}

.upload-card-text {
    font-size: 15px !important;
    color: #4a6090 !important;
}

/* File uploader drop-zone text */
[data-testid="stFileUploader"] {
    border-radius: 15px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f8faff !important;
    border: 1.5px dashed #7d6add !important;
    border-radius: 15px !important;
}

[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: #294477 !important;
    font-size: 15px !important;
    font-weight: 650 !important;
}

/* Browse / Upload button */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: #e8e4ff !important;
    color: #3d2aa5 !important;
    border: 1.5px solid #8975df !important;
    border-radius: 11px !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    min-height: 44px !important;
}

[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background: #dcd5ff !important;
    color: #2e1d91 !important;
    border-color: #6249d0 !important;
}

/* ---------- ANALYZE RESUME BUTTON ----------
   On the upload page there is no toolbar, so this targets
   the main Analyze button without changing nav buttons. */
.stApp:not(:has(.toolbar)) [data-testid="stButton"] > button {
    min-height: 54px !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #5639e3 0%, #d63d9e 100%) !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    box-shadow: 0 9px 20px rgba(86, 57, 227, .25) !important;
}

.stApp:not(:has(.toolbar)) [data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #4529cf 0%, #bd2d88 100%) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

.stApp:not(:has(.toolbar)) [data-testid="stButton"] > button p,
.stApp:not(:has(.toolbar)) [data-testid="stButton"] > button span {
    color: #ffffff !important;
    font-size: inherit !important;
    font-weight: inherit !important;
}

/* ---------- SUCCESS / WARNING / CAPTION ---------- */
[data-testid="stAlert"] {
    font-size: 15px !important;
}

[data-testid="stAlert"] p {
    font-size: 15px !important;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #526894 !important;
    font-size: 14px !important;
}

/* ---------- FOOTER ---------- */
.footer {
    color: #53678f !important;
    font-size: 13px !important;
}

.footer b {
    color: #334d85 !important;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 1100px) {
    .toolbar .stButton > button {
        font-size: 14px !important;
        min-height: 56px !important;
    }

    .dashboard-title,
    .upload-title {
        font-size: 40px !important;
    }
}

@media (max-width: 900px) {
    .dashboard-title,
    .upload-title {
        font-size: 36px !important;
    }

    .section-heading {
        font-size: 25px !important;
    }

    .detail-title {
        font-size: 19px !important;
    }
}


/* ============================================================
   FINAL TOOLBAR COLOR + READABILITY FIX
   ============================================================ */

/* Toolbar container */
.toolbar {
    margin-top: 18px !important;
    margin-bottom: 8px !important;
}

/* Every toolbar button */
.toolbar button {
    min-height: 60px !important;
    height: 60px !important;
    border-radius: 15px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: 0 !important;
    line-height: 1.2 !important;
    transition: all 0.18s ease !important;
}

/* Force ALL text inside toolbar buttons to use the intended color */
.toolbar button p,
.toolbar button span,
.toolbar button div,
.toolbar button [data-testid="stMarkdownContainer"],
.toolbar button [data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
}

/* INACTIVE buttons — dark navy text on a clean white/lilac surface */
.toolbar button[kind="secondary"],
.toolbar button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #172b68 !important;
    border: 1.5px solid #c5cdec !important;
    box-shadow: 0 5px 14px rgba(38, 55, 120, 0.10) !important;
}

.toolbar button[kind="secondary"] p,
.toolbar button[kind="secondary"] span,
.toolbar button[data-testid="stBaseButton-secondary"] p,
.toolbar button[data-testid="stBaseButton-secondary"] span {
    color: #172b68 !important;
}

/* INACTIVE hover */
.toolbar button[kind="secondary"]:hover,
.toolbar button[data-testid="stBaseButton-secondary"]:hover {
    background: #f0edff !important;
    color: #4b32b9 !important;
    border-color: #765ce0 !important;
    box-shadow: 0 8px 18px rgba(76, 52, 185, 0.16) !important;
    transform: translateY(-1px) !important;
}

.toolbar button[kind="secondary"]:hover p,
.toolbar button[kind="secondary"]:hover span,
.toolbar button[data-testid="stBaseButton-secondary"]:hover p,
.toolbar button[data-testid="stBaseButton-secondary"]:hover span {
    color: #4b32b9 !important;
}

/* ACTIVE button — strong purple/pink gradient + white text */
.toolbar button[kind="primary"],
.toolbar button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #5b3fe5 0%, #c43ca5 100%) !important;
    color: #ffffff !important;
    border: 1.5px solid transparent !important;
    box-shadow: 0 9px 22px rgba(91, 63, 229, 0.28) !important;
}

.toolbar button[kind="primary"] p,
.toolbar button[kind="primary"] span,
.toolbar button[kind="primary"] div,
.toolbar button[data-testid="stBaseButton-primary"] p,
.toolbar button[data-testid="stBaseButton-primary"] span,
.toolbar button[data-testid="stBaseButton-primary"] div {
    color: #ffffff !important;
}

/* ACTIVE hover */
.toolbar button[kind="primary"]:hover,
.toolbar button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #4d31d0 0%, #aa2f91 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 10px 24px rgba(77, 49, 208, 0.32) !important;
}

/* Analyze Another Resume — same readable style as inactive navigation */
.toolbar button[key="toolbar_new_resume"] {
    background: #ffffff !important;
    color: #172b68 !important;
    border: 1.5px solid #c5cdec !important;
}

.toolbar button[key="toolbar_new_resume"] p,
.toolbar button[key="toolbar_new_resume"] span {
    color: #172b68 !important;
}

.toolbar button[key="toolbar_new_resume"]:hover {
    background: #f0edff !important;
    color: #4b32b9 !important;
    border-color: #765ce0 !important;
}

.toolbar button[key="toolbar_new_resume"]:hover p,
.toolbar button[key="toolbar_new_resume"]:hover span {
    color: #4b32b9 !important;
}

/* Keep navigation readable on smaller screens too */
@media (max-width: 1100px) {
    .toolbar button,
    .toolbar button p,
    .toolbar button span {
        font-size: 14px !important;
    }
}

@media (max-width: 900px) {
    .toolbar button,
    .toolbar button p,
    .toolbar button span {
        font-size: 13px !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD PAGE
# ============================================================

if st.session_state.analysis is None:
    html("""
    <div class="upload-header">
        <div class="eyebrow">AI-POWERED CAREER ANALYSIS</div>
        <div class="upload-title">
            Resume Analysis <span>Dashboard</span>
        </div>
        <div class="upload-subtitle">
            Turn your resume into actionable career insights
        </div>
    </div>
    <div class="divider"></div>
    <div class="upload-card">
        <div class="upload-card-title">📄 Upload Your Resume</div>
        <div class="upload-card-text">
            Upload a PDF, DOCX or TXT resume to analyze skills,
            predict suitable job roles and identify skill gaps.
        </div>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose a resume",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        st.success(f"✓ {uploaded_file.name} uploaded successfully")
        st.caption(f"File size: {uploaded_file.size / 1024:.1f} KB")

        if st.button(
            "🔍  Analyze Resume",
            key="analyze_resume_button",
            type="primary",
            use_container_width=True,
        ):
            temp_path = None

            try:
                extension = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension,
                ) as temp:
                    temp.write(uploaded_file.getbuffer())
                    temp_path = temp.name

                with st.spinner("🔍 Analyzing your resume..."):
                    result = analyze_resume(temp_path)
                    resume_text = extract_resume_text(temp_path)
                    skill_list = load_skills()
                    detected_skills = extract_skills(resume_text, skill_list)
                    top_roles = predict_top_roles(
                        resume_text,
                        detected_skills,
                        top_n=3,
                    )

                # Keep the latest robust skill extractor result.
                if detected_skills:
                    result["detected_skills"] = detected_skills

                st.session_state.analysis = {
                    "result": result,
                    "top_roles": top_roles,
                }
                st.session_state.filename = uploaded_file.name
                st.session_state.dashboard_page = "Overview"
                st.rerun()

            except Exception as exc:
                st.error("Something went wrong while analyzing the resume.")
                st.exception(exc)

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    html("""
    <div class="footer">
        <div>AI Resume Analyzer</div>
        <div>NLP + TF-IDF + Calibrated Linear SVM</div>
    </div>
    """)


# ============================================================
# DASHBOARD
# ============================================================

else:
    result = st.session_state.analysis["result"]
    top_roles = st.session_state.analysis["top_roles"] or []

    # Defensive defaults so the UI does not break if a key is absent.
    detected_skills = get_value(result, "detected_skills", default=[]) or []
    matched_skills = get_value(result, "matched_skills", default=[]) or []
    missing_skills = get_value(result, "missing_skills", default=[]) or []
    required_skills = get_value(result, "required_skills", default=[]) or []

    resume_score = safe_percent(get_value(result, "resume_score", default=0))
    skill_match = safe_percent(
        get_value(result, "skill_match_percentage", default=0)
    )

    predicted_role = get_value(
        result,
        "predicted_role",
        "job_role",
        default="Unknown Role",
    )

    ml_score = safe_percent(
        role_value(top_roles[0], "confidence", "ml_score", "score", default=0)
        if top_roles
        else 0
    )

    render_topbar()
    render_dashboard_header()

    page = st.session_state.dashboard_page

    # ========================================================
    # OVERVIEW
    # ========================================================

    if page == "Overview":
        html("""
        <div class="section-heading">✨ Quick Insights</div>
        <div class="section-subheading">Your resume at a glance</div>
        """)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            html(f"""
            <div class="kpi kpi-pink">
                <div class="kpi-icon">⭐</div>
                <div class="kpi-title">Resume Match</div>
                <div class="kpi-value">{resume_score:.1f}%</div>
                <div class="kpi-note">Overall alignment with role</div>
            </div>
            """)

        with c2:
            html(f"""
            <div class="kpi kpi-mint">
                <div class="kpi-icon">📊</div>
                <div class="kpi-title">Skill Match</div>
                <div class="kpi-value">{skill_match:.1f}%</div>
                <div class="kpi-note">Required skills matched</div>
            </div>
            """)

        with c3:
            html(f"""
            <div class="kpi kpi-purple">
                <div class="kpi-icon">🤖</div>
                <div class="kpi-title">ML Confidence</div>
                <div class="kpi-value">{ml_score:.1f}%</div>
                <div class="kpi-note">Model prediction confidence</div>
            </div>
            """)

        with c4:
            html(f"""
            <div class="kpi kpi-yellow">
                <div class="kpi-icon">🛠</div>
                <div class="kpi-title">Skills Detected</div>
                <div class="kpi-value">{len(detected_skills)}</div>
                <div class="kpi-note">Relevant skills found</div>
            </div>
            """)

        html('<div class="divider"></div>')

        education_meets = bool(get_value(result, "education_meets", default=False))
        experience_meets = bool(get_value(result, "experience_meets", default=False))

        role_col, edu_col, exp_col = st.columns([1.65, 1, 1])

        with role_col:
            html(f"""
            <div class="card role-card">
                <div class="role-label">🎯 &nbsp; PREDICTED JOB ROLE</div>
                <div class="role-name">{predicted_role} 🚀</div>
                <div class="role-description">
                    A career direction based on your skills and ML prediction.
                </div>
            </div>
            """)

        with edu_col:
            if education_meets:
                status, css, note = (
                    "✓ Met",
                    "good",
                    "You meet the education requirement for this role.",
                )
            else:
                status, css, note = (
                    "⚠ Not detected",
                    "warning",
                    "The required education was not detected.",
                )

            html(f"""
            <div class="card req-card">
                <div class="req-title">🎓 &nbsp; Education Requirement</div>
                <div class="req-status {css}">{status}</div>
                <div class="req-note">{note}</div>
            </div>
            """)

        with exp_col:
            if experience_meets:
                status, css, note = (
                    "✓ Met",
                    "good",
                    "Your stated experience meets the role requirement.",
                )
            else:
                status, css, note = (
                    "⚠ Not met / Not stated",
                    "warning",
                    "Experience is below the requirement or was not explicitly stated.",
                )

            html(f"""
            <div class="card req-card">
                <div class="req-title">💼 &nbsp; Experience Requirement</div>
                <div class="req-status {css}">{status}</div>
                <div class="req-note">{note}</div>
            </div>
            """)

        html('<div class="divider"></div>')
        html("""
        <div class="section-heading">🏆 Top Job Role Matches</div>
        <div class="section-subheading">Other roles you're well-suited for</div>
        """)

        if top_roles:
            cols = st.columns(min(3, len(top_roles)))
            medals = ["🥇", "🥈", "🥉"]

            for i, role in enumerate(top_roles[:3]):
                role_name = role_value(role, "role", "job_title", "title", default="Unknown Role")
                overall = safe_percent(
                    role_value(role, "overall_match", "match", "match_score", "confidence", default=0)
                )
                confidence = safe_percent(
                    role_value(role, "confidence", "ml_score", "score", default=0)
                )
                role_skill = safe_percent(
                    role_value(role, "skill_match", "skill_match_percentage", default=0)
                )

                with cols[i]:
                    html(f"""
                    <div class="match-card">
                        <div class="rank">{i + 1}</div>
                        <div class="match-score">{overall:.1f}%</div>
                        <div class="match-role">{medals[i]} &nbsp; {role_name}</div>
                        <div class="progress-bg">
                            <div class="progress-fill" style="width:{overall:.1f}%;"></div>
                        </div>
                        <div class="match-note">
                            ML Confidence: <b>{confidence:.1f}%</b>
                            &nbsp;•&nbsp;
                            Skill Match: <b>{role_skill:.1f}%</b>
                        </div>
                    </div>
                    """)
        else:
            st.info("No job-role matches are available.")

        html('<div class="divider"></div>')
        html("""
        <div class="section-heading">🧩 Resume Snapshot</div>
        <div class="section-subheading">Skills and important requirements at a glance</div>
        """)

        s1, s2 = st.columns(2)
        with s1:
            skills_html = render_skill_pills(detected_skills, as_html=True)
            html(f'''
            <div class="detail-card skill-card">
                <div class="detail-title">🛠 Detected Skills</div>
                <div class="pill-wrap">{skills_html}</div>
            </div>
            ''')

        with s2:
            gaps_html = render_skill_pills(missing_skills, kind="gap", as_html=True)
            html(f'''
            <div class="detail-card skill-card">
                <div class="detail-title">❌ Skill Gaps</div>
                <div class="pill-wrap">{gaps_html}</div>
            </div>
            ''')

    # ========================================================
    # JOB MATCHES
    # ========================================================

    elif page == "Job Matches":
        html("""
        <div class="section-heading">🎯 Job Matches</div>
        <div class="section-subheading">
            Explore how your resume aligns with the predicted career paths.
        </div>
        """)

        if not top_roles:
            st.info("No job-role matches are available.")
        else:
            medals = ["🥇", "🥈", "🥉"]
            for i, role in enumerate(top_roles):
                role_name = role_value(role, "role", "job_title", "title", default="Unknown Role")
                overall = safe_percent(
                    role_value(role, "overall_match", "match", "match_score", "confidence", default=0)
                )
                confidence = safe_percent(
                    role_value(role, "confidence", "ml_score", "score", default=0)
                )
                role_skill = safe_percent(
                    role_value(role, "skill_match", "skill_match_percentage", default=0)
                )

                html(f"""
                <div class="detail-card" style="margin-bottom:16px; min-height:145px;">
                    <div class="rank">{i + 1}</div>
                    <div class="match-score">{overall:.1f}%</div>
                    <div class="match-role">{medals[i] if i < 3 else '🏅'} &nbsp; {role_name}</div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width:{overall:.1f}%;"></div>
                    </div>
                    <div class="match-note">
                        Overall Match: <b>{overall:.1f}%</b>
                        &nbsp;•&nbsp; ML Confidence: <b>{confidence:.1f}%</b>
                        &nbsp;•&nbsp; Skill Match: <b>{role_skill:.1f}%</b>
                    </div>
                </div>
                """)

        html('<div class="divider"></div>')
        html(f"""
        <div class="card role-card">
            <div class="role-label">🎯 &nbsp; PREDICTED JOB ROLE</div>
            <div class="role-name">{predicted_role} 🚀</div>
            <div class="role-description">
                This is the role selected by the calibrated ML prediction.
            </div>
        </div>
        """)

    # ========================================================
    # SKILLS
    # ========================================================

    elif page == "Skills":
        html("""
        <div class="section-heading">🛠 Skills Analysis</div>
        <div class="section-subheading">
            Skills detected from the uploaded resume and gaps against the predicted role.
        </div>
        """)

        a, b, c = st.columns(3)
        for col, label, value in [
            (a, "Detected Skills", len(detected_skills)),
            (b, "Matched Skills", len(matched_skills)),
            (c, "Skill Gaps", len(missing_skills)),
        ]:
            with col:
                html(f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """)

        html('<div class="divider"></div>')
        s1, s2 = st.columns(2)

        with s1:
            skills_html = render_skill_pills(detected_skills, as_html=True)
            html(f"""
            <div class="detail-card skill-card">
                <div class="detail-title">🛠 Detected Skills</div>
                <div class="pill-wrap">{skills_html}</div>
            </div>
            """)

        with s2:
            gaps_html = render_skill_pills(missing_skills, kind="gap", as_html=True)
            html(f"""
            <div class="detail-card skill-card">
                <div class="detail-title">❌ Skill Gaps</div>
                <div class="pill-wrap">{gaps_html}</div>
            </div>
            """)

        html('<div class="divider"></div>')
        html("""
        <div class="section-heading">📊 Skill Match Visualization</div>
        <div class="section-subheading">Required skills matched for the predicted role</div>
        """)
        html(f"""
        <div class="detail-card">
            <div class="stat-label">Required skills matched</div>
            <div class="progress-bg" style="height:13px; margin-top:14px;">
                <div class="progress-fill" style="width:{skill_match:.1f}%;"></div>
            </div>
            <div class="big-percent">{skill_match:.1f}%</div>
        </div>
        """)

        if matched_skills:
            matched_html = render_skill_pills(matched_skills, as_html=True)
            html(f'''
            <div style="height:18px"></div>
            <div class="detail-card skill-card">
                <div class="detail-title">✓ Matched Required Skills</div>
                <div class="pill-wrap">{matched_html}</div>
            </div>
            ''')

    # ========================================================
    # EDUCATION
    # ========================================================

    elif page == "Education":
        html("""
        <div class="section-heading">🎓 Education Analysis</div>
        <div class="section-subheading">Education requirements detected from your resume.</div>
        """)

        education_meets = bool(get_value(result, "education_meets", default=False))
        education_requirements = get_value(
            result,
            "education_requirements",
            "required_education",
            default="",
        )
        matched_education = get_value(
            result,
            "matched_education",
            default=[],
        ) or []

        status_text = "✓ Meets requirement" if education_meets else "⚠ Requirement not detected"
        status_class = "good" if education_meets else "warning"

        html(f"""
        <div class="detail-card">
            <div class="detail-title">🎓 Education Requirement</div>
            <div class="req-status {status_class}">{status_text}</div>
            <div class="req-note">Predicted role: <b>{predicted_role}</b></div>
        </div>
        """)

        e1, e2 = st.columns(2)

        if isinstance(education_requirements, (list, tuple)):
            education_items = education_requirements
        else:
            education_items = re.split(r"[|,]", str(education_requirements))

        clean_items = [str(x).strip() for x in education_items if str(x).strip()]

        with e1:
            if clean_items:
                required_html = "".join(
                    f'<span class="education-item">• {item}</span>'
                    for item in clean_items
                )
            else:
                required_html = '<div class="empty-text">No education requirement available.</div>'

            html(f'''
            <div class="detail-card education-card">
                <div class="detail-title">📚 Required Education</div>
                <div class="education-items">
                    {required_html}
                </div>
            </div>
            ''')

        with e2:
            if matched_education:
                matched_html = "".join(
                    f'<span class="education-item matched">✓ {item}</span>'
                    for item in matched_education
                )
            else:
                matched_html = '<div class="empty-text">No matching education requirement detected.</div>'

            html(f'''
            <div class="detail-card education-card">
                <div class="detail-title">✓ Matched Education</div>
                <div class="education-items">
                    {matched_html}
                </div>
            </div>
            ''')

    # ========================================================
    # EXPERIENCE
    # ========================================================

    elif page == "Experience":
        html("""
        <div class="section-heading">💼 Experience Analysis</div>
        <div class="section-subheading">Compare your stated experience with the role requirement.</div>
        """)

        candidate = get_value(result, "candidate_experience", "experience", default=None)
        required = get_value(result, "required_experience", default=0)
        experience_meets = bool(get_value(result, "experience_meets", default=False))

        candidate_display = "Not explicitly stated" if candidate is None else f"{candidate} years"
        required_display = f"{required} years" if required not in (None, "") else "Not specified"

        c1, c2 = st.columns(2)
        with c1:
            html(f"""
            <div class="stat-card">
                <div class="stat-label">Candidate Experience</div>
                <div class="stat-value">{candidate_display}</div>
            </div>
            """)
        with c2:
            html(f"""
            <div class="stat-card">
                <div class="stat-label">Required Experience</div>
                <div class="stat-value">{required_display}</div>
            </div>
            """)

        st.write("")
        if experience_meets:
            st.success("✓ Candidate meets the experience requirement.")
        elif candidate is None:
            st.warning("⚠ Professional experience was not explicitly stated in the resume.")
        else:
            st.warning("⚠ Candidate experience is below the requirement.")

    # ========================================================
    # JOB INFORMATION
    # ========================================================

    elif page == "Job Information":
        html("""
        <div class="section-heading">💰 Job Information</div>
        <div class="section-subheading">Career and role information associated with your analysis.</div>
        """)

        salary = get_value(result, "salary_range", "salary", default="Not available")
        category = get_value(result, "category", "job_category", default="Not available")

        c1, c2 = st.columns(2)
        with c1:
            html(f"""
            <div class="stat-card">
                <div class="stat-label">💰 Salary Range</div>
                <div class="stat-value">{salary}</div>
            </div>
            """)
        with c2:
            html(f"""
            <div class="stat-card">
                <div class="stat-label">🏷️ Job Category</div>
                <div class="stat-value">{category}</div>
            </div>
            """)

        st.write("")
        html(f"""
        <div class="card role-card">
            <div class="role-label">🎯 &nbsp; PREDICTED JOB ROLE</div>
            <div class="role-name">{predicted_role} 🚀</div>
            <div class="role-description">Career direction based on your resume and ML prediction.</div>
        </div>
        """)

        html('<div class="divider"></div>')
        html('<div class="section-heading">📋 Required Skills</div>')
        if required_skills:
            render_skill_pills(required_skills)
        else:
            html('<div class="empty-text">No required skills available.</div>')

        html('<div class="divider"></div>')
        html(f"""
        <div class="detail-card">
            <div class="detail-title">⭐ Overall Resume Match</div>
            <div class="progress-bg" style="height:13px; margin-top:14px;">
                <div class="progress-fill" style="width:{resume_score:.1f}%;"></div>
            </div>
            <div class="big-percent">{resume_score:.1f}/100</div>
        </div>
        """)

    render_footer()
