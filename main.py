import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="Institutional Multi-Timeframe Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    /* ════════════════════════════════════
       DESIGN TOKENS
       Contrast rules: labels >= #8AAAC8
       body >= #C0D4E8, values #E8F2FF
    ════════════════════════════════════ */
    :root {
        --bg:       #080C10;
        --surface:  #0E1520;
        --surface2: #131E2D;
        --border:   #1E2E42;
        --border2:  #253548;
        --txt-dim:  #607080;
        --txt-muted:#8AAAC8;
        --txt-body: #C0D4E8;
        --txt-bright:#E8F2FF;
        --accent:   #4A9EFF;
        --bull:     #00C853;
        --bear:     #FF3A5C;
        --bull-bg:  #06200E;
        --bear-bg:  #200608;
        --bull-bdr: #0A3818;
        --bear-bdr: #381018;
    }

    /* ── GLOBAL ── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: var(--bg);
        color: var(--txt-body);
    }
    .stApp { background-color: var(--bg); }
    .block-container {
        padding-top: 16px !important;
        padding-bottom: 40px !important;
        max-width: 1400px;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 260px !important;
        max-width: 300px !important;
    }
    /* Target text elements only — never SVG/icons or they render as broken text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--txt-body) !important;
    }
    [data-testid="stSidebar"] svg { fill: var(--txt-muted) !important; color: unset !important; }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--txt-muted) !important;
        font-size: 9px !important;
        font-weight: 600 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid var(--border) !important;
        padding-bottom: 8px !important;
        margin: 20px 0 12px 0 !important;
    }
    [data-testid="stSidebar"] label {
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stCaption {
        color: var(--txt-muted) !important;
        font-size: 11px !important;
    }
    [data-testid="stSidebar"] strong { color: var(--txt-bright) !important; }
    [data-testid="stSidebar"] input {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--txt-bright) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: #0A1E38 !important;
        border: 1px solid #1A4A80 !important;
        color: #90C8FF !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 2px !important;
        font-weight: 600 !important;
        padding: 10px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #1A3A60 !important;
        border-color: #4A9EFF !important;
        color: #C0E0FF !important;
    }

    /* ── MOBILE SIDEBAR ── */
    @media (max-width: 768px) {
        /* Sidebar: full-width drawer on mobile */
        [data-testid="stSidebar"] {
            min-width: 88vw !important;
            max-width: 88vw !important;
            transform: none !important;
        }
        /* Force sidebar visible — override Streamlit's mobile collapse */
        [data-testid="stSidebar"][aria-expanded="false"] {
            transform: translateX(-100%) !important;
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0) !important;
        }
        /* Hamburger button — large, obvious, always visible */
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            top: 8px !important;
            left: 8px !important;
            width: 44px !important;
            height: 44px !important;
            background: #1E88E5 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px !important;
            z-index: 99999 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
            cursor: pointer !important;
        }
        [data-testid="stSidebarCollapsedControl"] svg {
            fill: #FFFFFF !important;
            width: 24px !important;
            height: 24px !important;
        }
        /* Also style the open/close button inside sidebar */
        [data-testid="stSidebarCollapseButton"] {
            background: var(--surface2) !important;
            border-radius: 6px !important;
        }
        .block-container {
            padding-top: 64px !important;
            padding-left: 12px !important;
            padding-right: 12px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch !important;
            scrollbar-width: none !important;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
        .stTabs [data-baseweb="tab"] {
            padding: 12px 14px !important;
            font-size: 9px !important;
            letter-spacing: 1px !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
        }
        .term-header { flex-direction: column !important; gap: 4px !important; }
        .term-clock { display: none !important; }
        .term-title { font-size: 11px !important; letter-spacing: 2px !important; }
        .term-subtitle { font-size: 9px !important; letter-spacing: 1px !important; }
        .signal-banner { flex-direction: column !important; gap: 10px !important; }
        .signal-conf { text-align: left !important; font-size: 26px !important; }
        .signal-conf-label { text-align: left !important; }
        .risk-row { grid-template-columns: repeat(2, 1fr) !important; }
        .tf-matrix { grid-template-columns: repeat(2, 1fr) !important; }
        .bt-stat-grid { grid-template-columns: repeat(2, 1fr) !important; }
        .fib-grid { grid-template-columns: repeat(3, 1fr) !important; }
        .news-meta { flex-direction: column !important; align-items: flex-start !important; gap: 6px !important; }
        .sentiment-grid { grid-template-columns: 1fr !important; }
    }
    @media (max-width: 480px) {
        .risk-row { grid-template-columns: 1fr !important; }
        .signal-label { font-size: 16px !important; }
        .tf-conf { font-size: 18px !important; }
        .tf-matrix { grid-template-columns: repeat(2, 1fr) !important; }
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-bottom: 1px solid var(--border);
        gap: 0; padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--txt-muted);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 14px 22px;
        border: none;
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--txt-body); }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--txt-bright) !important;
        border-bottom: 2px solid var(--accent) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { background: var(--bg); padding-top: 20px; }

    /* ── METRICS ── */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 9px !important; font-weight: 600 !important;
        letter-spacing: 2px !important; text-transform: uppercase !important;
        color: var(--txt-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 18px !important; font-weight: 500 !important;
        color: var(--txt-bright) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
    }

    /* ── EXPANDER ── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important; font-weight: 600 !important;
        letter-spacing: 1px !important;
        color: var(--txt-body) !important;
        padding: 12px 16px !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--txt-bright) !important; }

    /* ── INPUTS ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 2px !important;
        color: var(--txt-body) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
    }
    hr { border-color: var(--border) !important; margin: 16px 0 !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
    [data-testid="stDataFrame"] * {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
    }

    /* ═══════════════════════════════
       CUSTOM COMPONENTS
    ═══════════════════════════════ */

    .term-header {
        display: flex; align-items: baseline;
        justify-content: space-between;
        padding: 0 0 16px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }
    .term-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px; font-weight: 600;
        letter-spacing: 4px; text-transform: uppercase;
        color: var(--txt-bright);
    }
    .term-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px; letter-spacing: 2px;
        color: var(--txt-muted); margin-top: 4px;
    }
    .term-clock {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; letter-spacing: 1px;
        color: var(--txt-muted);
    }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px; font-weight: 600;
        letter-spacing: 3px; text-transform: uppercase;
        color: var(--txt-muted);
        border-left: 2px solid var(--accent);
        padding: 2px 0 2px 10px;
        margin: 22px 0 14px 0;
    }

    /* Signal banner */
    .signal-banner {
        padding: 18px 22px;
        display: flex; align-items: center;
        justify-content: space-between;
        border-radius: 2px; margin-bottom: 4px; gap: 12px;
    }
    .signal-banner-bull { background: var(--bull-bg); border: 1px solid var(--bull-bdr); border-left: 3px solid var(--bull); }
    .signal-banner-bear { background: var(--bear-bg); border: 1px solid var(--bear-bdr); border-left: 3px solid var(--bear); }
    .signal-banner-neutral { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--border2); }

    .signal-label { font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600; letter-spacing: 3px; }
    .signal-label-bull   { color: var(--bull); }
    .signal-label-bear   { color: var(--bear); }
    .signal-label-neutral{ color: var(--txt-muted); }

    .signal-sub { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 2px; margin-top: 5px; }
    .signal-sub-bull   { color: #4ABA7A; }
    .signal-sub-bear   { color: #E06070; }
    .signal-sub-neutral{ color: var(--txt-muted); }

    .signal-conf { font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 300; letter-spacing: -1px; text-align: right; }
    .signal-conf-bull   { color: var(--bull); }
    .signal-conf-bear   { color: var(--bear); }
    .signal-conf-neutral{ color: var(--txt-muted); }

    .signal-conf-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 2px; text-align: right; text-transform: uppercase; }
    .signal-conf-label-bull   { color: #4ABA7A; }
    .signal-conf-label-bear   { color: #E06070; }
    .signal-conf-label-neutral{ color: var(--txt-dim); }

    /* TF matrix */
    .tf-matrix {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 1px; background: var(--border);
        border: 1px solid var(--border); border-radius: 2px;
        overflow: hidden; margin: 4px 0 16px 0;
    }
    .tf-card { padding: 14px 10px; text-align: center; border-top: 3px solid transparent; }
    .tf-card-bull    { background: #080F0B; border-top-color: var(--bull); }
    .tf-card-bear    { background: #0F0809; border-top-color: var(--bear); }
    .tf-card-neutral { background: var(--surface); border-top-color: var(--border2); }

    .tf-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 2px; color: var(--txt-muted); margin-bottom: 6px; }
    .tf-bias  { font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 1px; }
    .tf-bias-bull   { color: var(--bull); }
    .tf-bias-bear   { color: var(--bear); }
    .tf-bias-neutral{ color: var(--txt-muted); }
    .tf-conf { font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 300; letter-spacing: -1px; margin: 6px 0 4px 0; }
    .tf-conf-bull   { color: var(--bull); }
    .tf-conf-bear   { color: var(--bear); }
    .tf-conf-neutral{ color: var(--txt-dim); }
    .tf-votes  { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--txt-muted); letter-spacing: 1px; }
    .tf-source { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--txt-dim); letter-spacing: 1px; margin-top: 5px; text-transform: uppercase; }

    /* Risk table */
    .risk-row {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 1px; background: var(--border);
        border: 1px solid var(--border); border-radius: 2px;
        margin: 14px 0; overflow: hidden;
    }
    .risk-cell { background: var(--surface); padding: 12px 14px; text-align: center; }
    .risk-cell-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--txt-muted); margin-bottom: 6px; }
    .risk-cell-value { font-family: 'IBM Plex Mono', monospace; font-size: 15px; font-weight: 500; color: var(--txt-bright); word-break: break-all; }
    .risk-cell-delta { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--txt-muted); margin-top: 3px; }

    /* Rationale */
    .rationale-row { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--txt-body); padding: 6px 0; border-bottom: 1px solid var(--surface2); letter-spacing: 0.2px; line-height: 1.5; }
    .rationale-tf-header { font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 3px; color: var(--txt-muted); text-transform: uppercase; padding: 14px 0 6px 0; margin-top: 8px; border-top: 1px solid var(--border); }

    /* News card */
    .news-card { background: var(--surface); border: 1px solid var(--border); border-radius: 2px; padding: 16px 18px; margin: 8px 0; border-left: 3px solid var(--border2); }
    .news-card-bull    { border-left-color: var(--bull); }
    .news-card-bear    { border-left-color: var(--bear); }
    .news-card-neutral { border-left-color: var(--border2); }
    .news-title { font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; font-weight: 500; color: var(--txt-bright); margin-bottom: 6px; line-height: 1.45; }
    .news-desc  { font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: var(--txt-body); margin-bottom: 10px; line-height: 1.55; }
    .news-meta  { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; }
    .news-source { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--txt-muted); letter-spacing: 1px; text-transform: uppercase; }
    .news-score-bull    { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #80EEB0; background: var(--bull-bg); border: 1px solid var(--bull-bdr); padding: 2px 8px; border-radius: 1px; letter-spacing: 1px; }
    .news-score-bear    { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #FF9090; background: var(--bear-bg); border: 1px solid var(--bear-bdr); padding: 2px 8px; border-radius: 1px; letter-spacing: 1px; }
    .news-score-neutral { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--txt-muted); background: var(--surface2); border: 1px solid var(--border2); padding: 2px 8px; border-radius: 1px; letter-spacing: 1px; }
    .vader-bar-bg { height: 2px; background: var(--border); margin: 8px 0; border-radius: 1px; }

    /* Sentiment grid */
    .sentiment-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 20px; }
    .sentiment-cell { background: var(--surface); padding: 16px; text-align: center; }
    .sentiment-cell-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--txt-muted); margin-bottom: 8px; }
    .sentiment-cell-value { font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 300; }

    /* Backtest stat grid */
    .bt-stat-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 20px; }
    .bt-stat { background: var(--surface); padding: 14px 10px; text-align: center; }
    .bt-stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--txt-muted); margin-bottom: 8px; }
    .bt-stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 300; color: var(--txt-bright); }

    /* SMC badges */
    .smc-badge { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 1px; text-transform: uppercase; padding: 4px 8px; border-radius: 1px; margin: 2px; }
    .smc-bull { background: var(--bull-bg); color: #80EEB0; border: 1px solid var(--bull-bdr); }
    .smc-bear { background: var(--bear-bg); color: #FF9090; border: 1px solid var(--bear-bdr); }
    .smc-info { background: var(--surface2); color: #90C8FF; border: 1px solid var(--border2); }

    /* Data tag */
    .data-tag { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 1px; background: var(--surface2); color: var(--txt-muted); border: 1px solid var(--border2); padding: 3px 8px; border-radius: 1px; text-transform: uppercase; display: inline-block; margin: 2px 2px 8px 0; }

    /* Fibonacci */
    .fib-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; margin-top: 12px; }
    .fib-cell { background: var(--surface); padding: 10px 8px; text-align: center; }
    .fib-level { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--txt-muted); letter-spacing: 1px; margin-bottom: 4px; }
    .fib-price { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--txt-bright); font-weight: 500; }

    /* Mobile hint */
    .mobile-settings-hint { display: none; font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #FFFFFF; background: #1E88E5; border: none; border-radius: 8px; padding: 12px 16px; text-align: center; margin-bottom: 16px; letter-spacing: 0.5px; cursor: pointer; }
    @media (max-width: 768px) { .mobile-settings-hint { display: block !important; } }

    /* Chrome */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

# ── TERMINAL HEADER ──
now_utc = datetime.utcnow().strftime('%Y-%m-%d  %H:%M:%S  UTC')
st.markdown(f"""
<div class="term-header">
  <div>
    <div class="term-title">▐ INSTITUTIONAL TERMINAL</div>
    <div class="term-subtitle">SMART MONEY · MULTI-TIMEFRAME · REAL OHLCV · VADER SENTIMENT · BACKTEST</div>
  </div>
  <div class="term-clock">{now_utc}</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("### ── CONFIG ──")
    st.caption("Settings are in the **⚙ SETTINGS** tab.")
    st.divider()

    # Alpha Vantage key — stored directly in session_state via key=
    st.text_input(
        "Alpha Vantage Key (XAU/USD)",
        type="password",
        key="cfg_av_key",
        help="Required for XAU/USD. Free at alphavantage.co",
        placeholder="Paste key here…"
    )

    st.divider()
    run_btn = st.button("◈  REFRESH ANALYSIS", type="primary", use_container_width=True)
    if run_btn:
        st.cache_data.clear()
        st.rerun()
    st.caption(f"SYS · {datetime.utcnow().strftime('%H:%M:%S UTC')}")


# ==================================================
# DATA LAYER — REAL OHLC FROM FREE SOURCES
# ==================================================

class DataFetcher:
    """
    Fetches REAL OHLC data — all free, no API keys required:
    - Crypto: Kraken → Coinbase → CoinGecko
    - Forex: Kraken (EUR/USD, GBP/USD, USD/JPY) — free, no key
    """

    # Kraken interval in minutes
    KRAKEN_TF_MAP = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1day": 1440, "1week": 10080
    }

    # Coinbase granularity in seconds
    COINBASE_TF_MAP = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1day": 86400, "1week": 604800
    }

    # yfinance interval map for forex
    YF_TF_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "1h", "1day": "1d", "1week": "1wk"
    }
    # yfinance period map (how far back to fetch)
    YF_PERIOD_MAP = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "730d", "4h": "730d", "1day": "max", "1week": "max"
    }

    def __init__(self):
        pass

    def fetch(self, symbol: str, interval: str, limit: int = 500, av_key: str = "") -> Tuple[Optional[pd.DataFrame], str]:
        """Delegate to module-level cached fetch so cache is truly shared across reruns."""
        return _cached_fetch(symbol, interval, limit, av_key)

    def _fetch_kraken(self, symbol: str, interval: str, limit: int) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Kraken REST API — free, no key, generous rate limits, real OHLCV.
        Returns up to 720 candles per call.
        """
        try:
            pair_map = {
                # Crypto
                "BTC/USDT": "XBTUSD", "ETH/USDT": "ETHUSD",
                "BTC-USD":  "XBTUSD", "ETH-USD":  "ETHUSD",
                # Forex — Kraken trades these pairs natively
                "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY",
            }
            pair = pair_map.get(symbol)
            if pair is None:
                return None, f"Kraken: no mapping for {symbol}"
            tf_min = self.KRAKEN_TF_MAP.get(interval, 60)

            # since = earliest timestamp we want
            since = int((datetime.utcnow() - timedelta(minutes=tf_min * min(limit, 720))).timestamp())

            url = "https://api.kraken.com/0/public/OHLC"
            params = {"pair": pair, "interval": tf_min, "since": since}

            resp = requests.get(url, params=params, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()

            if data.get("error"):
                return None, f"Kraken: {data['error']}"

            result = data.get("result", {})
            # Kraken returns {pair_name: [[time,o,h,l,c,vwap,vol,count]], "last": ...}
            ohlc_key = [k for k in result if k != "last"]
            if not ohlc_key:
                return None, "Kraken: no OHLC key in response"

            raw = result[ohlc_key[0]]
            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df.sort_index()
            # Drop only if ALL of o/h/l/c are zero (truly empty)
            df = df[~((df["open"] == 0) & (df["close"] == 0))]

            display_map = {
                "XBTUSD": "BTC/USD", "ETHUSD": "ETH/USD",
                "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD",
                "USDJPY": "USD/JPY",
            }
            pair_display = display_map.get(pair, pair)
            return df, f"Kraken ({pair_display})"

        except Exception as e:
            return None, f"Kraken error: {e}"

    def _fetch_binance(self, symbol: str, interval: str, limit: int) -> Tuple[Optional[pd.DataFrame], str]:
        """Binance public API — free, no key, very reliable."""
        try:
            tf_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1day": "1d", "1week": "1w"
            }
            sym_map = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT",
                       "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
            binance_sym = sym_map.get(symbol)
            if not binance_sym:
                return None, f"Binance: no mapping for {symbol}"
            tf = tf_map.get(interval, "1h")
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": binance_sym, "interval": tf, "limit": min(limit, 1000)}
            resp = requests.get(url, params=params, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.json()
            df = pd.DataFrame(raw, columns=[
                "timestamp","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df[["open","high","low","close","volume"]].astype(float)
            df = df[df["volume"] > 0].sort_index()
            return df, f"Binance ({symbol})"
        except Exception as e:
            return None, f"Binance error: {e}"

    def _fetch_coinbase(self, symbol: str, interval: str, limit: int) -> Tuple[Optional[pd.DataFrame], str]:
        """Coinbase Exchange public API — free, no key."""
        try:
            sym_map = {
                "BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD",
                "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
            }
            product_id = sym_map.get(symbol, symbol)
            granularity = self.COINBASE_TF_MAP.get(interval, 3600)

            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - granularity * min(limit, 300)

            url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            params = {
                "granularity": granularity,
                "start": datetime.utcfromtimestamp(start_time).isoformat(),
                "end": datetime.utcfromtimestamp(end_time).isoformat()
            }

            resp = requests.get(url, params=params, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.json()

            if not raw or isinstance(raw, dict):
                return None, "Coinbase: unexpected response"

            # [timestamp, low, high, open, close, volume]
            df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df[df["volume"] > 0].sort_index()

            return df, f"Coinbase ({product_id})"

        except Exception as e:
            return None, f"Coinbase error: {e}"

    def _fetch_coingecko(self, symbol: str, interval: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        CoinGecko — last resort only. Real OHLC but rate-limited (30 calls/min free tier).
        Granularity is auto-selected by CoinGecko based on days requested.
        """
        try:
            coin_map = {
                "BTC/USDT": "bitcoin", "ETH/USDT": "ethereum",
                "BTC-USD": "bitcoin", "ETH-USD": "ethereum"
            }
            # Use minimal days to avoid rate limits and get finer granularity
            days_map = {
                "1m": 1, "5m": 2, "15m": 7, "30m": 14,
                "1h": 30, "4h": 30, "1day": 90, "1week": 90
            }
            coin_id = coin_map.get(symbol, "bitcoin")
            days = days_map.get(interval, 30)

            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            resp = requests.get(url, params={"vs_currency": "usd", "days": days},
                                timeout=15, headers={"User-Agent": "Mozilla/5.0"})

            if resp.status_code == 429:
                # Retry with backoff
                import time
                for wait in [2, 5]:
                    time.sleep(wait)
                    resp = requests.get(url, params={"vs_currency": "usd", "days": days},
                                        timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code != 429:
                        break
                if resp.status_code == 429:
                    return None, "CoinGecko rate-limited — data unavailable, retry in 60s"

            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                return None, "CoinGecko: no data"

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            df["volume"] = 0.0
            df = df.sort_index()

            return df, f"CoinGecko ({coin_id}, {days}d — volume unavailable)"

        except Exception as e:
            return None, f"CoinGecko error: {e}"


    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def macd(close: pd.Series, fast=12, slow=26, signal=9):
        exp_fast = close.ewm(span=fast, adjust=False).mean()
        exp_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        return volume.rolling(period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        atr_val = Indicators.atr(high, low, close, period)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(period).mean() / atr_val
        minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr_val
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean(), plus_di, minus_di

    @staticmethod
    def find_swing_highs(high: pd.Series, lookback: int = 3) -> pd.Series:
        """Vectorized swing high detection"""
        result = pd.Series(False, index=high.index)
        for i in range(lookback, len(high) - lookback):
            window = high.iloc[i-lookback:i+lookback+1]
            if high.iloc[i] == window.max():
                result.iloc[i] = True
        return result

    @staticmethod
    def find_swing_lows(low: pd.Series, lookback: int = 3) -> pd.Series:
        """Vectorized swing low detection"""
        result = pd.Series(False, index=low.index)
        for i in range(lookback, len(low) - lookback):
            window = low.iloc[i-lookback:i+lookback+1]
            if low.iloc[i] == window.min():
                result.iloc[i] = True
        return result



    def _fetch_alpha_vantage_gold(self, interval: str, api_key: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Alpha Vantage — XAU/USD gold OHLCV. Free tier: 25 calls/day."""
        try:
            if interval in ("1m", "5m", "15m", "30m", "1h"):
                av_int = {"1m": "1min", "5m": "5min", "15m": "15min",
                          "30m": "30min", "1h": "60min"}.get(interval, "60min")
                params = {
                    "function": "FX_INTRADAY", "from_symbol": "XAU",
                    "to_symbol": "USD", "interval": av_int,
                    "outputsize": "full", "apikey": api_key,
                }
                time_key = f"Time Series FX ({av_int})"
            else:
                params = {
                    "function": "FX_DAILY", "from_symbol": "XAU",
                    "to_symbol": "USD", "outputsize": "full", "apikey": api_key,
                }
                time_key = "Time Series FX (Daily)"

            resp = requests.get("https://www.alphavantage.co/query",
                                params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            if "Note" in data:
                return None, "Alpha Vantage rate limited (25 calls/day free tier)"
            if "Information" in data:
                return None, f"Alpha Vantage: {data['Information'][:80]}"
            if "Error Message" in data:
                return None, f"Alpha Vantage: {data['Error Message'][:80]}"
            if time_key not in data:
                return None, f"Alpha Vantage unexpected response: {list(data.keys())}"

            rows = []
            for dt_str, v in data[time_key].items():
                rows.append({
                    "timestamp": pd.to_datetime(dt_str, utc=True),
                    "open":  float(v["1. open"]),
                    "high":  float(v["2. high"]),
                    "low":   float(v["3. low"]),
                    "close": float(v["4. close"]),
                    "volume": 0.0,
                })

            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            df = df[df["close"] > 0].tail(500)
            if len(df) < 5:
                return None, "Alpha Vantage: insufficient data returned"
            return df, "Alpha Vantage (XAU/USD)"

        except Exception as e:
            return None, f"Alpha Vantage error: {e}"


@st.cache_data(ttl=600, show_spinner=False)
def _cached_fetch(symbol: str, interval: str, limit: int = 500, av_key: str = "") -> Tuple[Optional[pd.DataFrame], str]:
    """
    Module-level cached fetch — cache key is (symbol, interval, limit) only.
    This ensures desktop and mobile always get identical data for the same symbol+TF.
    Crypto: Kraken → Binance → Coinbase → CoinGecko
    Forex:  Kraken (EUR/USD, GBP/USD, USD/JPY)
    """
    _f = DataFetcher()
    is_crypto = symbol in ["BTC/USDT", "ETH/USDT", "BTC-USD", "ETH-USD"]
    if is_crypto:
        df, src = _f._fetch_kraken(symbol, interval, limit)
        if df is not None and len(df) >= 20:
            return df, src
        df, src = _f._fetch_binance(symbol, interval, limit)
        if df is not None and len(df) >= 20:
            return df, src
        df, src = _f._fetch_coinbase(symbol, interval, limit)
        if df is not None and len(df) >= 20:
            return df, src
        return _f._fetch_coingecko(symbol, interval)
    elif symbol == "XAU/USD":
        if av_key:
            df, src = _f._fetch_alpha_vantage_gold(interval, av_key)
            if df is not None and len(df) >= 20:
                return df, src
        return None, "XAU/USD requires an Alpha Vantage API key — add it in ⚙ Settings"
    else:
        df, src = _f._fetch_kraken(symbol, interval, limit)
        if df is not None and len(df) >= 20:
            return df, src
        return None, f"No data source available for {symbol}"


# ==================================================
# TECHNICAL INDICATORS — VECTORIZED
# ==================================================

class Indicators:
    """Stateless indicator calculations — all vectorized"""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def macd(close: pd.Series, fast=12, slow=26, signal=9):
        exp_fast = close.ewm(span=fast, adjust=False).mean()
        exp_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        return volume.rolling(period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Average Directional Index"""
        atr_val = Indicators.atr(high, low, close, period)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di  = 100 * pd.Series(plus_dm,  index=close.index).rolling(period).mean() / atr_val
        minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr_val
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean(), plus_di, minus_di

    @staticmethod
    def find_swing_highs(high: pd.Series, lookback: int = 3) -> pd.Series:
        result = pd.Series(False, index=high.index)
        for i in range(lookback, len(high) - lookback):
            window = high.iloc[i-lookback:i+lookback+1]
            if high.iloc[i] == window.max():
                result.iloc[i] = True
        return result

    @staticmethod
    def find_swing_lows(low: pd.Series, lookback: int = 3) -> pd.Series:
        result = pd.Series(False, index=low.index)
        for i in range(lookback, len(low) - lookback):
            window = low.iloc[i-lookback:i+lookback+1]
            if low.iloc[i] == window.min():
                result.iloc[i] = True
        return result


# ==================================================
# SMART MONEY CONCEPTS ENGINE — PROPER IMPLEMENTATION
# ==================================================

class SmartMoneyEngine:
    """
    Proper ICT / Smart Money pattern detection.
    All patterns use REAL OHLC — no synthetic data.
    """

    def __init__(self, df: pd.DataFrame, atr_series: pd.Series):
        self.df = df.copy()
        self.atr = atr_series
        self.df['atr'] = atr_series

    def detect_order_blocks(self) -> List[Dict]:
        """
        Bullish OB: Last bearish candle before a bullish impulsive move.
        Bearish OB: Last bullish candle before a bearish impulsive move.
        Proper definition: strong move that breaks prior structure.
        """
        df = self.df
        obs = []
        min_move_atr = 1.5  # minimum move size relative to ATR

        for i in range(2, len(df) - 2):
            candle = df.iloc[i]
            prev = df.iloc[i - 1]
            next_2 = df.iloc[i + 1]
            atr_val = self.atr.iloc[i]

            if pd.isna(atr_val) or atr_val == 0:
                continue

            move_up = next_2['close'] - candle['close']
            move_down = candle['close'] - next_2['close']

            # Bullish OB: bearish candle (i-1 or i) followed by strong upward move
            if (prev['close'] < prev['open'] and  # bearish candle
                    move_up > min_move_atr * atr_val and  # impulsive up move
                    next_2['close'] > prev['high']):  # breaks prior candle high (structure break)
                obs.append({
                    'type': 'BULLISH_OB',
                    'top': prev['high'],
                    'bottom': prev['low'],
                    'index': df.index[i - 1],
                    'atr': atr_val,
                    'strength': move_up / atr_val
                })

            # Bearish OB: bullish candle followed by strong downward move
            if (prev['close'] > prev['open'] and  # bullish candle
                    move_down > min_move_atr * atr_val and  # impulsive down move
                    next_2['close'] < prev['low']):  # breaks prior candle low
                obs.append({
                    'type': 'BEARISH_OB',
                    'top': prev['high'],
                    'bottom': prev['low'],
                    'index': df.index[i - 1],
                    'atr': atr_val,
                    'strength': move_down / atr_val
                })

        return obs[-10:]  # keep last 10

    def detect_fair_value_gaps(self) -> List[Dict]:
        """
        Proper 3-candle FVG:
        Bullish: candle[i-2].high < candle[i].low (gap between them)
        Bearish: candle[i-2].low > candle[i].high (gap between them)
        """
        df = self.df
        fvgs = []

        for i in range(2, len(df)):
            c0 = df.iloc[i - 2]
            c2 = df.iloc[i]

            # Bullish FVG
            if c0['high'] < c2['low']:
                fvgs.append({
                    'type': 'BULLISH_FVG',
                    'top': c2['low'],
                    'bottom': c0['high'],
                    'midpoint': (c0['high'] + c2['low']) / 2,
                    'index': df.index[i - 1],
                    'filled': False
                })

            # Bearish FVG
            if c0['low'] > c2['high']:
                fvgs.append({
                    'type': 'BEARISH_FVG',
                    'top': c0['low'],
                    'bottom': c2['high'],
                    'midpoint': (c0['low'] + c2['high']) / 2,
                    'index': df.index[i - 1],
                    'filled': False
                })

        return fvgs[-10:]

    def detect_liquidity_sweeps(self) -> List[Dict]:
        """
        Liquidity sweep: price takes out a recent swing high/low with a wick,
        then closes back inside — indicating stop hunt / liquidity grab.
        """
        df = self.df
        sweeps = []
        lookback = 20

        for i in range(lookback, len(df)):
            candle = df.iloc[i]
            window = df.iloc[i - lookback:i]
            atr_val = self.atr.iloc[i]

            if pd.isna(atr_val) or atr_val == 0:
                continue

            recent_high = window['high'].max()
            recent_low = window['low'].min()

            wick_up = candle['high'] - max(candle['open'], candle['close'])
            wick_down = min(candle['open'], candle['close']) - candle['low']

            # Bearish sweep: wick above recent high, closes below
            if (candle['high'] > recent_high and
                    candle['close'] < recent_high and
                    wick_up > 0.3 * atr_val):
                sweeps.append({
                    'type': 'SWEEP_HIGH',
                    'price': candle['high'],
                    'swept_level': recent_high,
                    'index': df.index[i]
                })

            # Bullish sweep: wick below recent low, closes above
            if (candle['low'] < recent_low and
                    candle['close'] > recent_low and
                    wick_down > 0.3 * atr_val):
                sweeps.append({
                    'type': 'SWEEP_LOW',
                    'price': candle['low'],
                    'swept_level': recent_low,
                    'index': df.index[i]
                })

        return sweeps[-5:]

    def detect_market_structure(self) -> Dict:
        """
        Proper market structure using swing highs/lows.
        HH+HL = uptrend, LH+LL = downtrend, otherwise ranging.
        Also detects Break of Structure (BOS) and Change of Character (CHOCH).
        """
        df = self.df
        if len(df) < 20:
            return {'structure': 'UNKNOWN', 'bos': False, 'choch': False}

        swing_highs = Indicators.find_swing_highs(df['high'], lookback=3)
        swing_lows = Indicators.find_swing_lows(df['low'], lookback=3)

        sh_prices = df['high'][swing_highs].values
        sl_prices = df['low'][swing_lows].values

        if len(sh_prices) < 2 or len(sl_prices) < 2:
            return {'structure': 'UNKNOWN', 'bos': False, 'choch': False}

        last_sh = sh_prices[-2:]
        last_sl = sl_prices[-2:]

        hh = last_sh[-1] > last_sh[-2]
        hl = last_sl[-1] > last_sl[-2]
        lh = last_sh[-1] < last_sh[-2]
        ll = last_sl[-1] < last_sl[-2]

        if hh and hl:
            structure = 'UPTREND'
        elif lh and ll:
            structure = 'DOWNTREND'
        elif hh and ll:
            structure = 'RANGING'
        else:
            structure = 'RANGING'

        # BOS: price breaks most recent swing high/low
        current = df['close'].iloc[-1]
        recent_sh = sh_prices[-1] if len(sh_prices) > 0 else None
        recent_sl = sl_prices[-1] if len(sl_prices) > 0 else None

        bos_bullish = current > recent_sh if recent_sh else False
        bos_bearish = current < recent_sl if recent_sl else False

        return {
            'structure': structure,
            'bos_bullish': bos_bullish,
            'bos_bearish': bos_bearish,
            'swing_highs': sh_prices.tolist(),
            'swing_lows': sl_prices.tolist(),
            'last_swing_high': float(recent_sh) if recent_sh else None,
            'last_swing_low': float(recent_sl) if recent_sl else None
        }

    def detect_rsi_divergence(self, rsi: pd.Series) -> List[str]:
        """Proper RSI divergence using swing points"""
        df = self.df
        divergences = []
        lookback = 30

        if len(df) < lookback:
            return divergences

        recent_df = df.tail(lookback)
        recent_rsi = rsi.tail(lookback)

        # Find local price lows and RSI values there
        lows_idx = []
        for i in range(2, len(recent_df) - 2):
            if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1]):
                lows_idx.append(i)

        highs_idx = []
        for i in range(2, len(recent_df) - 2):
            if (recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1]):
                highs_idx.append(i)

        # Bullish divergence: price lower low, RSI higher low
        if len(lows_idx) >= 2:
            i1, i2 = lows_idx[-2], lows_idx[-1]
            if (recent_df['low'].iloc[i2] < recent_df['low'].iloc[i1] and
                    recent_rsi.iloc[i2] > recent_rsi.iloc[i1]):
                divergences.append('BULLISH_DIVERGENCE')

        # Bearish divergence: price higher high, RSI lower high
        if len(highs_idx) >= 2:
            i1, i2 = highs_idx[-2], highs_idx[-1]
            if (recent_df['high'].iloc[i2] > recent_df['high'].iloc[i1] and
                    recent_rsi.iloc[i2] < recent_rsi.iloc[i1]):
                divergences.append('BEARISH_DIVERGENCE')

        return divergences


# ==================================================
# MULTI-TIMEFRAME ANALYZER
# ==================================================

class MultiTimeframeAnalyzer:

    # Timeframe hierarchy for HTF confirmation
    TF_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1day", "1week"]

    def __init__(self, data: Dict[str, Tuple[pd.DataFrame, str]], timeframes: List[str]):
        self.data = data  # {tf: (df, source_label)}
        self.timeframes = timeframes
        self.results = {}
        self._analyze_all()

    def _analyze_all(self):
        for tf in self.timeframes:
            if tf not in self.data:
                continue
            df, source = self.data[tf]
            if df is None or len(df) < 50:
                continue
            self.results[tf] = self._analyze_tf(df, tf, source)

    def _analyze_tf(self, df: pd.DataFrame, tf: str, source: str) -> Dict:
        """Full analysis of a single timeframe using real OHLCV"""
        ind = Indicators()

        # Core indicators — all vectorized, using real OHLC
        atr_s = ind.atr(df['high'], df['low'], df['close'])
        rsi_s = ind.rsi(df['close'])
        macd_line, signal_line, histogram = ind.macd(df['close'])
        bb_upper, bb_mid, bb_lower = ind.bollinger(df['close'])
        vol_sma = ind.volume_sma(df['volume'])
        adx_s, plus_di, minus_di = ind.adx(df['high'], df['low'], df['close'])

        # Current values (last bar)
        cur = df.iloc[-1]
        rsi_val = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
        atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
        macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        macd_sig = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
        hist_val = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0.0
        plus_di_val = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
        minus_di_val = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

        # Volume ratio (real volume vs SMA)
        vol_ratio = float(cur['volume'] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1.0

        # EMAs
        emas = {}
        for period in [9, 21, 50, 200]:
            ema_s = ind.ema(df['close'], period)
            emas[period] = float(ema_s.iloc[-1])

        # Smart money engine
        smc = SmartMoneyEngine(df, atr_s)
        order_blocks = smc.detect_order_blocks()
        fvgs = smc.detect_fair_value_gaps()
        sweeps = smc.detect_liquidity_sweeps()
        structure = smc.detect_market_structure()
        divergences = smc.detect_rsi_divergence(rsi_s)

        # Fibonacci levels from recent swing
        fib = self._fibonacci(df)

        # Nearest significant levels (swing-based)
        swing_h = structure.get('last_swing_high')
        swing_l = structure.get('last_swing_low')

        # Current price
        price = float(cur['close'])

        # Generate signals
        signals = self._generate_signals(
            price, rsi_val, macd_val, macd_sig, hist_val, adx_val,
            plus_di_val, minus_di_val, vol_ratio, emas, structure,
            order_blocks, fvgs, sweeps, divergences,
            bb_upper.iloc[-1], bb_mid.iloc[-1], bb_lower.iloc[-1]
        )

        confidence = self._calc_confidence(signals, adx_val, vol_ratio, tf)

        return {
            'tf': tf,
            'source': source,
            'price': price,
            'rsi': rsi_val,
            'macd': macd_val,
            'macd_signal': macd_sig,
            'macd_hist': hist_val,
            'adx': adx_val,
            'plus_di': plus_di_val,
            'minus_di': minus_di_val,
            'atr': atr_val,
            'vol_ratio': vol_ratio,
            'emas': emas,
            'bb_upper': float(bb_upper.iloc[-1]),
            'bb_mid': float(bb_mid.iloc[-1]),
            'bb_lower': float(bb_lower.iloc[-1]),
            'structure': structure,
            'order_blocks': order_blocks,
            'fvgs': fvgs,
            'sweeps': sweeps,
            'divergences': divergences,
            'fib': fib,
            'signals': signals,
            'bias': signals['bias'],
            'confidence': confidence,
            'swing_high': swing_h,
            'swing_low': swing_l,
            'df': df,  # keep reference for charting
            'atr_series': atr_s
        }

    def _generate_signals(self, price, rsi, macd, macd_sig, macd_hist, adx,
                          plus_di, minus_di, vol_ratio, emas, structure,
                          order_blocks, fvgs, sweeps, divergences,
                          bb_upper, bb_mid, bb_lower) -> Dict:
        """
        Rule-based signal generation with scoring.
        Each component votes +1 bullish, -1 bearish, 0 neutral.
        """
        votes = []
        reasons = []

        # 1. Trend structure
        s = structure['structure']
        if s == 'UPTREND':
            votes.append(1)
            reasons.append("✅ Uptrend (HH+HL)")
        elif s == 'DOWNTREND':
            votes.append(-1)
            reasons.append("🔴 Downtrend (LH+LL)")
        else:
            votes.append(0)
            reasons.append("⚪ Ranging market")

        # 2. BOS
        if structure.get('bos_bullish'):
            votes.append(1)
            reasons.append("✅ Bullish BOS (broke swing high)")
        elif structure.get('bos_bearish'):
            votes.append(-1)
            reasons.append("🔴 Bearish BOS (broke swing low)")

        # 3. EMA alignment (9 > 21 > 50 > 200)
        if all(k in emas for k in [9, 21, 50]):
            if emas[9] > emas[21] > emas[50]:
                votes.append(1)
                reasons.append("✅ EMA bullish stack (9>21>50)")
            elif emas[9] < emas[21] < emas[50]:
                votes.append(-1)
                reasons.append("🔴 EMA bearish stack (9<21<50)")
            else:
                votes.append(0)
                reasons.append("⚪ EMAs mixed")

        # 4. Price vs EMA 50
        if 50 in emas:
            if price > emas[50]:
                votes.append(1)
                reasons.append("✅ Price above EMA 50")
            else:
                votes.append(-1)
                reasons.append("🔴 Price below EMA 50")

        # 5. RSI (avoid extremes, use for confirmation)
        if 45 < rsi < 70:
            votes.append(1)
            reasons.append(f"✅ RSI bullish zone ({rsi:.1f})")
        elif 30 < rsi <= 45:
            votes.append(-1)
            reasons.append(f"🔴 RSI bearish zone ({rsi:.1f})")
        elif rsi >= 70:
            votes.append(-1)
            reasons.append(f"⚠️ RSI overbought ({rsi:.1f})")
        elif rsi <= 30:
            votes.append(1)
            reasons.append(f"⚠️ RSI oversold ({rsi:.1f})")

        # 6. MACD — use histogram for direction + cross
        if macd_hist > 0 and macd > macd_sig:
            votes.append(1)
            reasons.append("✅ MACD bullish cross + histogram positive")
        elif macd_hist < 0 and macd < macd_sig:
            votes.append(-1)
            reasons.append("🔴 MACD bearish cross + histogram negative")
        else:
            votes.append(0)

        # 7. ADX trend strength + DI direction
        if adx > 25:
            if plus_di > minus_di:
                votes.append(1)
                reasons.append(f"✅ ADX {adx:.1f} strong — +DI dominates")
            else:
                votes.append(-1)
                reasons.append(f"🔴 ADX {adx:.1f} strong — -DI dominates")
        else:
            votes.append(0)
            reasons.append(f"⚪ ADX {adx:.1f} weak trend")

        # 8. Volume confirmation (skip if no real volume data)
        if vol_ratio > 0 and vol_ratio != 1.0:
            if vol_ratio > 1.3:
                votes.append(1)
                reasons.append(f"✅ Above-average volume ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.7:
                votes.append(-1)
                reasons.append(f"⚠️ Low volume ({vol_ratio:.1f}x) — weak move")
            else:
                votes.append(0)
        else:
            reasons.append("⚪ Volume N/A (forex/commodity — no centralized volume)")

        # 9. Bollinger Bands position
        if price > bb_upper:
            votes.append(-1)
            reasons.append("⚠️ Price above BB upper (overextended)")
        elif price < bb_lower:
            votes.append(1)
            reasons.append("✅ Price below BB lower (oversold)")
        elif price > bb_mid:
            votes.append(1)
            reasons.append("✅ Price above BB midline")
        else:
            votes.append(-1)
            reasons.append("🔴 Price below BB midline")

        # 10. Smart Money patterns
        recent_bullish_ob = any(ob['type'] == 'BULLISH_OB' for ob in order_blocks[-3:])
        recent_bearish_ob = any(ob['type'] == 'BEARISH_OB' for ob in order_blocks[-3:])
        if recent_bullish_ob:
            votes.append(1)
            reasons.append("✅ Bullish Order Block detected")
        if recent_bearish_ob:
            votes.append(-1)
            reasons.append("🔴 Bearish Order Block detected")

        recent_bullish_fvg = any(f['type'] == 'BULLISH_FVG' for f in fvgs[-3:])
        recent_bearish_fvg = any(f['type'] == 'BEARISH_FVG' for f in fvgs[-3:])
        if recent_bullish_fvg:
            votes.append(1)
            reasons.append("✅ Bullish FVG present")
        if recent_bearish_fvg:
            votes.append(-1)
            reasons.append("🔴 Bearish FVG present")

        # 11. Liquidity sweeps
        for sw in sweeps[-2:]:
            if sw['type'] == 'SWEEP_LOW':
                votes.append(1)
                reasons.append("✅ Bullish liquidity sweep (stop hunt below)")
            elif sw['type'] == 'SWEEP_HIGH':
                votes.append(-1)
                reasons.append("🔴 Bearish liquidity sweep (stop hunt above)")

        # 12. Divergences
        for div in divergences:
            if 'BULLISH' in div:
                votes.append(1)
                reasons.append("✅ Bullish RSI divergence")
            elif 'BEARISH' in div:
                votes.append(-1)
                reasons.append("🔴 Bearish RSI divergence")

        # Tally
        bull_votes = votes.count(1)
        bear_votes = votes.count(-1)
        total = len(votes)

        if bull_votes > bear_votes and (bull_votes / total) >= 0.55:
            bias = 'BULLISH'
        elif bear_votes > bull_votes and (bear_votes / total) >= 0.55:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'

        return {
            'bias': bias,
            'bull_votes': bull_votes,
            'bear_votes': bear_votes,
            'total_votes': total,
            'reasons': reasons
        }

    def _calc_confidence(self, signals: Dict, adx: float, vol_ratio: float, tf: str) -> float:
        """
        Confidence = (dominant vote fraction) * modifiers.
        Not arbitrary — tied to actual signal agreement rate.
        """
        total = signals['total_votes']
        if total == 0:
            return 0.0

        bull = signals['bull_votes']
        bear = signals['bear_votes']
        dominant = max(bull, bear)
        base_confidence = (dominant / total) * 100

        # ADX modifier: stronger trend = more confidence
        adx_mod = min(1.0 + (adx - 20) / 100, 1.3) if adx > 20 else 0.8

        # Volume modifier
        vol_mod = min(1.0 + (vol_ratio - 1.0) * 0.2, 1.2) if vol_ratio > 1 else 0.9

        # Timeframe modifier — higher TF signals are more reliable
        tf_mod = {
            "1m": 0.7, "5m": 0.8, "15m": 0.9, "30m": 0.95,
            "1h": 1.0, "4h": 1.05, "1day": 1.1, "1week": 1.15
        }.get(tf, 1.0)

        conf = base_confidence * adx_mod * vol_mod * tf_mod
        return min(conf, 98.0)

    def _fibonacci(self, df: pd.DataFrame) -> Dict:
        """Fibonacci from last 50 bars swing"""
        recent = df.tail(50)
        high = float(recent['high'].max())
        low = float(recent['low'].min())
        diff = high - low
        return {
            '0.0': high, '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382, '0.5': high - diff * 0.5,
            '0.618': high - diff * 0.618, '0.786': high - diff * 0.786,
            '1.0': low
        }

    def get_consolidated_signal(self, min_conf: float = 60, require_htf: bool = True) -> Dict:
        """
        Consolidated signal across timeframes.
        Optionally requires HTF alignment for trade validation.
        """
        if not self.results:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'valid': False}

        tfs_sorted = sorted(self.results.keys(),
                            key=lambda x: self.TF_ORDER.index(x) if x in self.TF_ORDER else 99)

        bull_tfs, bear_tfs, neutral_tfs = [], [], []
        for tf in tfs_sorted:
            r = self.results[tf]
            if r['confidence'] >= min_conf:
                if r['bias'] == 'BULLISH':
                    bull_tfs.append(tf)
                elif r['bias'] == 'BEARISH':
                    bear_tfs.append(tf)
                else:
                    neutral_tfs.append(tf)
            else:
                neutral_tfs.append(tf)

        # Weighted confidence (HTF weighted more)
        weights = {"1m": 0.5, "5m": 0.6, "15m": 0.8, "30m": 0.9,
                   "1h": 1.0, "4h": 1.3, "1day": 1.6, "1week": 2.0}
        total_w = sum(weights.get(tf, 1.0) for tf in self.results)
        weighted_conf = sum(
            self.results[tf]['confidence'] * weights.get(tf, 1.0)
            for tf in self.results
        ) / total_w if total_w > 0 else 0

        # Determine signal
        if len(bull_tfs) > len(bear_tfs) and len(bull_tfs) >= 2:
            signal = 'BULLISH'
        elif len(bear_tfs) > len(bull_tfs) and len(bear_tfs) >= 2:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        # HTF confirmation check
        htf_confirmed = True
        if require_htf and len(tfs_sorted) > 1:
            highest_tf = tfs_sorted[-1]
            htf_bias = self.results[highest_tf]['bias']
            if signal != 'NEUTRAL' and htf_bias != signal:
                htf_confirmed = False

        # Position sizing hint (Kelly-lite)
        price = self.results[tfs_sorted[0]]['price'] if tfs_sorted else 0
        atr = self.results[tfs_sorted[0]]['atr'] if tfs_sorted else 0

        return {
            'signal': signal,
            'confidence': weighted_conf,
            'bull_tfs': bull_tfs,
            'bear_tfs': bear_tfs,
            'neutral_tfs': neutral_tfs,
            'htf_confirmed': htf_confirmed,
            'valid': htf_confirmed and weighted_conf >= min_conf and signal != 'NEUTRAL',
            'price': price,
            'atr': atr
        }


# ==================================================
# NEWS SENTIMENT — VADER (NOT KEYWORD COUNTING)
# ==================================================

class SentimentEngine:
    """
    Uses VADER for proper compound sentiment scoring on real text.
    Falls back to NewsData.io if API key provided.
    """

    # Fallback lexicon for when VADER is not installed
    POSITIVE = [
        'bullish', 'surge', 'rally', 'gain', 'rise', 'soar', 'jump', 'record',
        'high', 'strong', 'growth', 'inflow', 'adoption', 'optimism', 'support',
        'buy', 'accumulate', 'breakthrough', 'upside', 'recovery', 'positive',
        'beat', 'exceeded', 'outperform', 'upgrade', 'boom', 'momentum'
    ]
    NEGATIVE = [
        'bearish', 'crash', 'plunge', 'drop', 'fall', 'decline', 'dump', 'sell',
        'outflow', 'fear', 'risk', 'warn', 'weak', 'low', 'loss', 'concern',
        'negative', 'miss', 'underperform', 'downgrade', 'bust', 'collapse',
        'headwind', 'pressure', 'volatility', 'uncertain', 'halt', 'ban'
    ]
    NEGATORS = ['not', "n't", 'no', 'never', 'without', 'against']

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None

    def analyze_text(self, text: str) -> Dict:
        if self.vader is not None:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
        else:
            compound = self._lexicon_score(text)

        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        return {'label': label, 'compound': round(compound, 3)}

    def _lexicon_score(self, text: str) -> float:
        """Simple but negation-aware lexicon scoring (VADER fallback)"""
        words = text.lower().split()
        score = 0
        for i, word in enumerate(words):
            # Strip punctuation
            clean = word.strip('.,!?;:"\'')
            negated = any(n in words[max(0, i-3):i] for n in self.NEGATORS)
            if clean in self.POSITIVE:
                score += -1 if negated else 1
            elif clean in self.NEGATIVE:
                score += 1 if negated else -1
        # Normalize to [-1, 1]
        total_words = max(len(words), 1)
        return max(-1.0, min(1.0, score / (total_words ** 0.5)))

    def fetch_and_analyze(self, symbol: str) -> List[Dict]:
        articles = []
        if self.api_key:
            articles = self._fetch_newsdata(symbol)

        if not articles:
            articles = self._sample_articles(symbol)

        # Run VADER on each article
        for a in articles:
            sentiment = self.analyze_text(a.get('title', '') + ' ' + a.get('description', ''))
            a['sentiment'] = sentiment['label']
            a['compound'] = sentiment['compound']

        return articles

    def _fetch_newsdata(self, symbol: str) -> List[Dict]:
        query_map = {
            "BTC/USDT": "Bitcoin BTC crypto",
            "ETH/USDT": "Ethereum ETH crypto",
                        "EUR/USD": "Euro EUR USD forex",
            "GBP/USD": "Pound GBP sterling forex",
            "USD/JPY": "Yen JPY USD forex"
        }
        query = query_map.get(symbol, symbol)

        try:
            url = "https://newsdata.io/api/1/news"
            params = {"apikey": self.api_key, "q": query, "language": "en", "size": 8}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            results = []
            for item in data.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'description': item.get('description', '') or '',
                    'source': item.get('source_id', 'Unknown'),
                    'published': item.get('pubDate', '')
                })
            return results
        except:
            return []

    def _sample_articles(self, symbol: str) -> List[Dict]:
        samples = {
            "BTC/USDT": [
                {'title': 'Bitcoin ETF sees record institutional inflows amid market optimism',
                 'description': 'Major asset managers continue accumulating BTC through regulated vehicles.',
                 'source': 'CoinDesk', 'published': 'Sample'},
                {'title': 'Crypto market faces headwinds as Fed signals prolonged high rates',
                 'description': 'Risk assets under pressure as monetary tightening concerns resurface.',
                 'source': 'Reuters', 'published': 'Sample'},
                {'title': 'Bitcoin on-chain metrics show accumulation at current levels',
                 'description': 'Long-term holder supply reaches all-time high, suggesting strong conviction.',
                 'source': 'Glassnode', 'published': 'Sample'},
            ],
            "ETH/USDT": [
                {'title': 'Ethereum staking yields attract institutional capital',
                 'description': 'Post-merge ETH staking now offers competitive yields vs traditional bonds.',
                 'source': 'Decrypt', 'published': 'Sample'},
                {'title': 'Layer-2 activity surges as Ethereum gas fees remain elevated',
                 'description': 'Arbitrum and Optimism see record transaction volumes this month.',
                 'source': 'The Block', 'published': 'Sample'},
            ],
            "EUR/USD": [
                {'title': 'ECB holds rates steady amid slowing eurozone growth',
                 'description': 'Policymakers signal caution as inflation edges toward target.',
                 'source': 'FT', 'published': 'Sample'},
                {'title': 'Dollar strengthens on robust US jobs data',
                 'description': 'EUR/USD under pressure as Fed rate cut expectations get pushed back.',
                 'source': 'Reuters', 'published': 'Sample'},
            ],
            "GBP/USD": [
                {'title': 'Bank of England signals gradual easing cycle ahead',
                 'description': 'Sterling retreats as BoE hints at rate cuts later this year.',
                 'source': 'Bloomberg', 'published': 'Sample'},
            ],
            "USD/JPY": [
                {'title': 'BoJ surprises with rate hike, yen surges',
                 'description': 'Bank of Japan raises rates for first time in decades, triggering sharp yen rally.',
                 'source': 'Nikkei', 'published': 'Sample'},
            ],
        }
        return samples.get(symbol, [
            {'title': f'{symbol} trades within tight range amid low volatility',
             'description': 'Market participants await key economic data for directional cues.',
             'source': 'Reuters', 'published': 'Sample'}
        ])


# ==================================================
# BACKTESTER — ACTUAL HISTORICAL SIMULATION
# ==================================================

class Backtester:
    """
    Runs the signal logic on historical OHLCV data.
    Uses real candle OHLC — no synthetic prices.
    """

    def __init__(self, df: pd.DataFrame, atr_multiplier: float = 2.0, risk_pct: float = 1.0):
        self.df = df.copy()
        self.atr_mult = atr_multiplier
        self.risk_pct = risk_pct

    def run(self) -> Dict:
        df = self.df
        if len(df) < 100:
            return {'error': 'Insufficient data (need 100+ bars)'}

        ind = Indicators()
        atr_s = ind.atr(df['high'], df['low'], df['close'])
        rsi_s = ind.rsi(df['close'])
        macd_line, signal_line, _ = ind.macd(df['close'])
        ema9  = ind.ema(df['close'], 9)
        ema21 = ind.ema(df['close'], 21)
        ema50 = ind.ema(df['close'], 50)

        trades      = []
        equity      = 10000.0
        equity_curve = [equity]
        in_trade    = False
        trade_dir   = None
        entry_price = None
        stop_price  = None
        target_price = None
        entry_idx   = None

        for i in range(50, len(df) - 1):
            bar      = df.iloc[i]
            next_bar = df.iloc[i + 1]
            atr      = atr_s.iloc[i]

            if pd.isna(atr) or atr == 0:
                continue

            bullish = (ema9.iloc[i] > ema21.iloc[i] > ema50.iloc[i] and
                       rsi_s.iloc[i] > 45 and rsi_s.iloc[i] < 70 and
                       macd_line.iloc[i] > signal_line.iloc[i])
            bearish = (ema9.iloc[i] < ema21.iloc[i] < ema50.iloc[i] and
                       rsi_s.iloc[i] < 55 and rsi_s.iloc[i] > 30 and
                       macd_line.iloc[i] < signal_line.iloc[i])

            if not in_trade:
                if bullish:
                    in_trade     = True
                    trade_dir    = 'LONG'
                    entry_price  = next_bar['open']
                    stop_price   = entry_price - atr * self.atr_mult
                    target_price = entry_price + atr * self.atr_mult * 2
                    entry_idx    = i
                elif bearish:
                    in_trade     = True
                    trade_dir    = 'SHORT'
                    entry_price  = next_bar['open']
                    stop_price   = entry_price + atr * self.atr_mult
                    target_price = entry_price - atr * self.atr_mult * 2
                    entry_idx    = i
            else:
                hit_stop   = False
                hit_target = False
                if trade_dir == 'LONG':
                    hit_stop   = bar['low']  <= stop_price
                    hit_target = bar['high'] >= target_price
                else:
                    hit_stop   = bar['high'] >= stop_price
                    hit_target = bar['low']  <= target_price

                if hit_target or hit_stop or (i - entry_idx > 20):
                    exit_price = (target_price if hit_target
                                  else (stop_price if hit_stop else bar['close']))
                    pnl_pct = (exit_price - entry_price) / entry_price
                    if trade_dir == 'SHORT':
                        pnl_pct = -pnl_pct
                    stop_dist_pct = abs(entry_price - stop_price) / entry_price
                    position_size = ((equity * self.risk_pct / 100) / stop_dist_pct
                                     if stop_dist_pct > 0 else equity)
                    pnl_dollar = pnl_pct * position_size
                    equity += pnl_dollar
                    trades.append({
                        'direction':  trade_dir,
                        'entry':      entry_price,
                        'exit':       exit_price,
                        'pnl_pct':    pnl_pct * 100,
                        'pnl_dollar': pnl_dollar,
                        'won':        pnl_pct > 0,
                        'reason':     'target' if hit_target else ('stop' if hit_stop else 'timeout'),
                        'entry_date': df.index[entry_idx],
                        'exit_date':  df.index[i]
                    })
                    equity_curve.append(equity)
                    in_trade = False

        # Stats
        if not trades:
            return {'error': 'No trades generated', 'equity_curve': equity_curve}

        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['won']]
        losses = trades_df[~trades_df['won']]

        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins['pnl_dollar'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_dollar'].mean() if len(losses) > 0 else 0
        profit_factor = (wins['pnl_dollar'].sum() / abs(losses['pnl_dollar'].sum())
                         if losses['pnl_dollar'].sum() != 0 else float('inf'))

        # Max drawdown
        eq_series = pd.Series(equity_curve)
        rolling_max = eq_series.cummax()
        drawdown = (eq_series - rolling_max) / rolling_max * 100
        max_drawdown = float(drawdown.min())

        # Sharpe (simplified)
        returns = trades_df['pnl_pct'].values
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win_dollar': avg_win,
            'avg_loss_dollar': avg_loss,
            'total_pnl': equity - 10000,
            'final_equity': equity,
            'max_drawdown_pct': max_drawdown,
            'sharpe': sharpe,
            'equity_curve': equity_curve,
            'trades': trades_df
        }


# ==================================================
# RISK MANAGEMENT CALCULATOR
# ==================================================

def calculate_position_size(equity: float, risk_pct: float, entry: float,
                             stop: float) -> Dict:
    """Calculate position size based on fixed risk %"""
    if entry == 0 or stop == 0 or entry == stop:
        return {}
    risk_amount = equity * risk_pct / 100
    stop_distance = abs(entry - stop)
    stop_pct = stop_distance / entry * 100
    units = risk_amount / stop_distance
    position_value = units * entry
    return {
        'risk_amount': risk_amount,
        'stop_distance': stop_distance,
        'stop_pct': stop_pct,
        'units': units,
        'position_value': position_value
    }


# ==================================================
# CHARTING — INSTITUTIONAL-GRADE CHARTS
# ==================================================

def build_chart(df: pd.DataFrame, symbol: str, tf: str, result: Dict) -> go.Figure:
    """Build comprehensive candlestick chart with all levels"""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=[f"{symbol} {tf}", "Volume", "RSI", "MACD"]
    )

    # Candlesticks — bright fills + thick wicks for visibility
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Price", showlegend=False,
        increasing_line_color='#00E676',   # bright green wick
        increasing_fillcolor='#00C853',    # solid green body
        decreasing_line_color='#FF5252',   # bright red wick
        decreasing_fillcolor='#D50000',    # solid red body
        line=dict(width=1),
        whiskerwidth=1,
    ), row=1, col=1)

    # EMAs
    ind = Indicators()
    colors = {9: '#2196F3', 21: '#FF9800', 50: '#9C27B0', 200: '#F44336'}
    for period, color in colors.items():
        ema = ind.ema(df['close'], period)
        fig.add_trace(go.Scatter(
            x=df.index, y=ema,
            line=dict(color=color, width=0.8, dash='solid'), opacity=0.8,
            name=f"EMA {period}", showlegend=True
        ), row=1, col=1)

    # Bollinger Bands
    bb_u, bb_m, bb_l = ind.bollinger(df['close'])
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_u, line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
        name="BB Upper", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_l, line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
        name="BB Lower", fill='tonexty', fillcolor='rgba(100,100,100,0.05)',
        showlegend=False
    ), row=1, col=1)

    # Order blocks
    for ob in result.get('order_blocks', [])[-5:]:
        color = 'rgba(76,175,80,0.08)' if ob['type'] == 'BULLISH_OB' else 'rgba(244,67,54,0.08)'
        border = 'rgba(76,175,80,0.5)' if ob['type'] == 'BULLISH_OB' else 'rgba(244,67,54,0.5)'
        fig.add_hrect(y0=ob['bottom'], y1=ob['top'],
                      fillcolor=color, line_color=border, line_width=1,
                      annotation_text=ob['type'].replace('_OB', ' OB'),
                      annotation_position="top left",
                      annotation_font_size=9, row=1, col=1)

    # Swing highs/lows
    sw = result.get('structure', {})
    if sw.get('last_swing_high'):
        fig.add_hline(y=sw['last_swing_high'], line_dash="dash",
                      line_color="red", line_width=1,
                      annotation_text="Swing High", row=1, col=1)
    if sw.get('last_swing_low'):
        fig.add_hline(y=sw['last_swing_low'], line_dash="dash",
                      line_color="green", line_width=1,
                      annotation_text="Swing Low", row=1, col=1)

    # Volume with color coding
    colors_vol = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['close'], df['open'])]
    vol_sma = ind.volume_sma(df['volume'])
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'], name="Volume",
        marker_color=colors_vol, showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_sma, line=dict(color='orange', width=1),
        name="Vol MA", showlegend=False
    ), row=2, col=1)

    # RSI
    rsi = ind.rsi(df['close'])
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, line=dict(color='purple', width=1.5),
        name="RSI", showlegend=False
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=0.8, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.8, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=0.5, row=3, col=1)

    # MACD
    macd_l, sig_l, hist = ind.macd(df['close'])
    hist_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, name="MACD Hist",
        marker_color=hist_colors, showlegend=False
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_l, line=dict(color='blue', width=1.2),
        name="MACD", showlegend=False
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=sig_l, line=dict(color='orange', width=1.2),
        name="Signal", showlegend=False
    ), row=4, col=1)

    fig.update_layout(
        height=900, template="plotly_dark",
        title=dict(text=f"{symbol} — {tf} | Source: {result.get('source', 'N/A')}",
                   font=dict(size=14)),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=40, r=40, t=80, b=20)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    return fig


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():
    news_key = st.session_state.get("cfg_news_api_key", "")
    instruments    = st.session_state.get("cfg_instruments",    ["BTC/USDT", "ETH/USDT"])
    timeframes     = st.session_state.get("cfg_timeframes",     ["15m", "1h", "4h", "1day"])
    min_confidence = st.session_state.get("cfg_min_confidence", 60)
    require_htf    = st.session_state.get("cfg_require_htf",    True)
    max_risk       = st.session_state.get("cfg_max_risk",       1.0)
    atr_multiplier = st.session_state.get("cfg_atr_multiplier", 2.0)

    fetcher = DataFetcher()
    sentiment_engine = SentimentEngine(api_key=news_key)

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["  ⚙ SETTINGS  ", "  SIGNALS  ", "  CHARTS  ", "  NEWS  ", "  BACKTEST  "])

    # ========== TAB 0: SETTINGS (mobile) ==========
    with tab0:
        st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

        # Read current values from cfg_ keys (set by Apply button)
        cur_instruments  = st.session_state.get("cfg_instruments",    ["BTC/USDT", "ETH/USDT"])
        cur_timeframes   = st.session_state.get("cfg_timeframes",     ["15m", "1h", "4h", "1day"])
        cur_confidence   = st.session_state.get("cfg_min_confidence", 60)
        cur_htf          = st.session_state.get("cfg_require_htf",    True)
        cur_risk         = st.session_state.get("cfg_max_risk",       1.0)
        cur_atr          = st.session_state.get("cfg_atr_multiplier", 2.0)

        # Show current config as readable summary cards
        st.markdown(f"""
        <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="color:var(--txt-muted);font-size:10px;letter-spacing:2px;margin-bottom:8px;">── CURRENT SETTINGS ──</div>
            <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;">
                <tr><td style="color:var(--txt-muted);padding:4px 0;font-size:11px;">MARKETS</td>
                    <td style="color:var(--txt-bright);font-size:11px;text-align:right;">{" · ".join(cur_instruments) or "None"}</td></tr>
                <tr><td style="color:var(--txt-muted);padding:4px 0;font-size:11px;">TIMEFRAMES</td>
                    <td style="color:var(--txt-bright);font-size:11px;text-align:right;">{" · ".join(cur_timeframes) or "None"}</td></tr>
                <tr><td style="color:var(--txt-muted);padding:4px 0;font-size:11px;">MIN CONFIDENCE</td>
                    <td style="color:var(--txt-bright);font-size:11px;text-align:right;">{cur_confidence}%</td></tr>
                <tr><td style="color:var(--txt-muted);padding:4px 0;font-size:11px;">HTF CONFIRM</td>
                    <td style="color:var(--txt-bright);font-size:11px;text-align:right;">{"ON" if cur_htf else "OFF"}</td></tr>
                <tr><td style="color:var(--txt-muted);padding:4px 0;font-size:11px;">MAX RISK</td>
                    <td style="color:var(--txt-bright);font-size:11px;text-align:right;">{cur_risk}%</td></tr>
                <tr><td style="color:var(--txt-muted);padding:4px 0;font-size:11px;">ATR MULT</td>
                    <td style="color:var(--txt-bright);font-size:11px;text-align:right;">{cur_atr}x</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Mobile: provide direct controls here
        st.markdown("**Change Settings:**")

        new_instruments = st.multiselect(
            "Markets", ["BTC/USDT", "ETH/USDT", "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"],
            default=cur_instruments, key="instruments_mobile"
        )
        new_timeframes = st.multiselect(
            "Timeframes", ["15m", "1h", "4h", "1day"],
            default=cur_timeframes, key="timeframes_mobile"
        )
        new_confidence = st.slider("Min Confidence %", 0, 100, cur_confidence, key="confidence_mobile")
        new_htf = st.checkbox("Require HTF Confirmation", value=cur_htf, key="htf_mobile")
        new_risk = st.slider("Max Risk %", 0.1, 5.0, cur_risk, 0.1, key="risk_mobile")
        new_atr  = st.slider("ATR Multiplier", 1.0, 5.0, cur_atr, 0.1, key="atr_mobile")
        st.text_input(
            "Alpha Vantage Key (for XAU/USD)", type="password",
            key="av_mobile",
            value=st.session_state.get("cfg_av_key", ""),
            help="Free key at alphavantage.co — required for XAU/USD data",
            placeholder="Paste your free API key here…"
        )
        new_news = st.text_input("NewsData.io Key (optional)", type="password", key="news_mobile")

        st.divider()
        if st.button("◈  APPLY & RUN ANALYSIS", type="primary", use_container_width=True, key="apply_mobile"):
            # Write to unbound keys (cfg_ prefix) — not bound to any widget
            st.session_state["cfg_instruments"]    = new_instruments
            st.session_state["cfg_timeframes"]     = new_timeframes
            st.session_state["cfg_min_confidence"] = new_confidence
            st.session_state["cfg_require_htf"]    = new_htf
            st.session_state["cfg_max_risk"]       = new_risk
            st.session_state["cfg_atr_multiplier"] = new_atr
            _av_val = st.session_state.get("av_mobile", "")
            if _av_val:
                st.session_state["cfg_av_key"] = _av_val
            if new_news:
                st.session_state["cfg_news_api_key"] = new_news
            st.cache_data.clear()
            st.rerun()

    # ========== TAB 1: SIGNALS ==========
    with tab1:
        st.markdown('<div class="section-label">Live Multi-Timeframe Signal Matrix</div>', unsafe_allow_html=True)


        if not instruments:
            st.warning("Select instruments in the sidebar.")
            return

        for symbol in instruments:
            with st.expander(f"  {symbol}  ·  MARKET ANALYSIS", expanded=True):
                data = {}
                with st.spinner(f"Loading {symbol}..."):
                    for tf in timeframes:
                        df, source = fetcher.fetch(symbol, tf, limit=300, av_key=st.session_state.get("cfg_av_key", ""))
                        if df is not None and len(df) >= 50:
                            data[tf] = (df, source)
                        else:
                            st.caption(f"⚠ {symbol} {tf}: {source}")

                if not data:
                    st.error(f"No data available for {symbol}")
                    continue

                analyzer = MultiTimeframeAnalyzer(data, timeframes)
                consolidated = analyzer.get_consolidated_signal(
                    min_conf=min_confidence, require_htf=require_htf
                )

                sig = consolidated['signal']
                valid = consolidated['valid']
                conf = consolidated['confidence']
                price = consolidated['price']
                atr_val = consolidated['atr']

                # ── SIGNAL BANNER ──
                if sig == 'BULLISH':
                    banner_cls = 'signal-banner-bull'
                    label_cls = 'signal-label-bull'
                    sub_cls = 'signal-sub-bull'
                    conf_cls = 'signal-conf-bull'
                    label_txt = 'LONG  ▲'
                    sub_txt = 'CONFIRMED' if valid else 'UNCONFIRMED — HTF MISALIGN'
                elif sig == 'BEARISH':
                    banner_cls = 'signal-banner-bear'
                    label_cls = 'signal-label-bear'
                    sub_cls = 'signal-sub-bear'
                    conf_cls = 'signal-conf-bear'
                    label_txt = 'SHORT  ▼'
                    sub_txt = 'CONFIRMED' if valid else 'UNCONFIRMED — HTF MISALIGN'
                else:
                    banner_cls = 'signal-banner-neutral'
                    label_cls = 'signal-label-neutral'
                    sub_cls = 'signal-sub-neutral'
                    conf_cls = 'signal-conf-neutral'
                    label_txt = 'NO SIGNAL  ◆'
                    sub_txt = 'AWAIT CONFLUENCE'

                st.markdown(f"""
                <div class="signal-banner {banner_cls}">
                  <div>
                    <div class="signal-label {label_cls}">{label_txt}</div>
                    <div class="signal-sub {sub_cls}">{sub_txt}</div>
                  </div>
                  <div style="text-align:right">
                    <div class="signal-conf {conf_cls}">{conf:.1f}<span style="font-size:14px;letter-spacing:0">%</span></div>
                    <div class="signal-sub {sub_cls}">CONFIDENCE</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── KEY METRICS ROW — HTML grid stays horizontal on mobile ──
                htf_txt   = "YES ✅" if consolidated['htf_confirmed'] else "NO ❌"
                bull_tfs_n = len(consolidated['bull_tfs'])
                bear_tfs_n = len(consolidated['bear_tfs'])
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin:10px 0;grid-template-columns:repeat(auto-fit,minmax(60px,1fr));">
                  <div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px 6px;text-align:center;">
                    <div style="color:var(--txt-muted);font-size:9px;letter-spacing:1px;">PRICE</div>
                    <div style="color:var(--txt-bright);font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace;margin-top:3px;">{price:,.2f}</div>
                  </div>
                  <div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px 6px;text-align:center;">
                    <div style="color:var(--txt-muted);font-size:9px;letter-spacing:1px;">ATR</div>
                    <div style="color:var(--txt-bright);font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace;margin-top:3px;">{atr_val:.2f}</div>
                  </div>
                  <div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px 6px;text-align:center;">
                    <div style="color:var(--txt-muted);font-size:9px;letter-spacing:1px;">HTF</div>
                    <div style="font-size:11px;font-weight:600;margin-top:3px;">{"✅" if consolidated['htf_confirmed'] else "❌"}</div>
                  </div>
                  <div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px 6px;text-align:center;">
                    <div style="color:var(--txt-muted);font-size:9px;letter-spacing:1px;">BULL TF</div>
                    <div style="color:#4ABA7A;font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace;margin-top:3px;">{bull_tfs_n}</div>
                  </div>
                  <div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px 6px;text-align:center;">
                    <div style="color:var(--txt-muted);font-size:9px;letter-spacing:1px;">BEAR TF</div>
                    <div style="color:#E06070;font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace;margin-top:3px;">{bear_tfs_n}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── RISK LEVELS ──
                if sig != 'NEUTRAL' and price > 0 and atr_val > 0:
                    if sig == 'BULLISH':
                        stop = price - atr_val * atr_multiplier
                        target = price + atr_val * atr_multiplier * 2
                        stop_pct = f"−{abs((stop-price)/price*100):.2f}%"
                        tgt_pct  = f"+{abs((target-price)/price*100):.2f}%"
                    else:
                        stop = price + atr_val * atr_multiplier
                        target = price - atr_val * atr_multiplier * 2
                        stop_pct = f"+{abs((stop-price)/price*100):.2f}%"
                        tgt_pct  = f"−{abs((target-price)/price*100):.2f}%"

                    sizing = calculate_position_size(10000, max_risk, price, stop)
                    st.markdown(f"""
                    <div class="risk-row">
                      <div class="risk-cell">
                        <div class="risk-cell-label">Entry</div>
                        <div class="risk-cell-value">{price:.4f}</div>
                        <div class="risk-cell-delta">CURRENT</div>
                      </div>
                      <div class="risk-cell">
                        <div class="risk-cell-label">Stop Loss</div>
                        <div class="risk-cell-value" style="color:#FF4444">{stop:.4f}</div>
                        <div class="risk-cell-delta">{stop_pct} · ATR×{atr_multiplier}</div>
                      </div>
                      <div class="risk-cell">
                        <div class="risk-cell-label">Target  2:1</div>
                        <div class="risk-cell-value" style="color:#00C853">{target:.4f}</div>
                        <div class="risk-cell-delta">{tgt_pct} · ATR×{atr_multiplier*2}</div>
                      </div>
                      <div class="risk-cell">
                        <div class="risk-cell-label">Risk  $10K Acct</div>
                        <div class="risk-cell-value">${sizing.get('risk_amount', 0):.0f}</div>
                        <div class="risk-cell-delta">{max_risk}% of equity</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── TIMEFRAME BREAKDOWN ──
                st.markdown('<div class="section-label">Timeframe Matrix</div>', unsafe_allow_html=True)
                tf_cards_html = '<div class="tf-matrix">'
                for tf, res in analyzer.results.items():
                    bias = res['bias']
                    conf_tf = res['confidence']
                    sig_counts = res['signals']
                    source_label = res['source'].split('(')[0].strip()
                    if bias == 'BULLISH':
                        card_cls, bias_cls, conf_cls = 'tf-card-bull', 'tf-bias-bull', 'tf-conf-bull'
                    elif bias == 'BEARISH':
                        card_cls, bias_cls, conf_cls = 'tf-card-bear', 'tf-bias-bear', 'tf-conf-bear'
                    else:
                        card_cls, bias_cls, conf_cls = 'tf-card-neutral', 'tf-bias-neutral', 'tf-conf-neutral'
                    bv = sig_counts['bull_votes']
                    bev = sig_counts['bear_votes']
                    tv = sig_counts['total_votes']
                    tf_cards_html += (
                        f'<div class="tf-card {card_cls}">'
                        f'<div class="tf-label">{tf}</div>'
                        f'<div class="tf-bias {bias_cls}">{bias}</div>'
                        f'<div class="tf-conf {conf_cls}">{conf_tf:.0f}'
                        f'<span style="font-size:11px">%</span></div>'
                        f'<div class="tf-votes">&#9650;{bv} &#9660;{bev} of {tv}</div>'
                        f'<div class="tf-source">{source_label}</div>'
                        f'</div>'
                    )
                tf_cards_html += '</div>'
                st.markdown(tf_cards_html, unsafe_allow_html=True)

                # ── SIGNAL RATIONALE ──
                with st.expander("SIGNAL RATIONALE  ·  SMART MONEY PATTERNS"):
                    for tf, res in analyzer.results.items():
                        st.markdown(f'<div class="rationale-tf-header">{tf}  ·  {res["bias"]}  ·  {res["confidence"]:.1f}%</div>', unsafe_allow_html=True)
                        for reason in res['signals']['reasons']:
                            st.markdown(f'<div class="rationale-row">{reason}</div>', unsafe_allow_html=True)

                        # SMC badges
                        smc_html = ""
                        for ob in res['order_blocks'][-3:]:
                            cls = 'smc-bull' if 'BULLISH' in ob['type'] else 'smc-bear'
                            smc_html += f'<span class="smc-badge {cls}">{ob["type"].replace("_"," ")} @ {ob["bottom"]:.4f}–{ob["top"]:.4f}</span>'
                        for fvg in res['fvgs'][-3:]:
                            cls = 'smc-bull' if 'BULLISH' in fvg['type'] else 'smc-bear'
                            smc_html += f'<span class="smc-badge {cls}">{fvg["type"].replace("_"," ")} @ {fvg["bottom"]:.4f}–{fvg["top"]:.4f}</span>'
                        for sw in res['sweeps'][-2:]:
                            smc_html += f'<span class="smc-badge smc-info">{sw["type"].replace("_"," ")} @ {sw["price"]:.4f}</span>'
                        if smc_html:
                            st.markdown(f'<div style="margin:10px 0 4px 0">{smc_html}</div>', unsafe_allow_html=True)

    # ========== TAB 2: CHARTS ==========
    with tab2:
        st.markdown('<div class="section-label">Price Chart · Institutional Levels</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        chart_symbol = col1.selectbox("Instrument", instruments, key="chart_symbol")
        chart_tf = col2.selectbox("Timeframe", timeframes, key="chart_tf")

        if chart_symbol and chart_tf:
            with st.spinner("Building chart..."):
                df, source = fetcher.fetch(chart_symbol, chart_tf, limit=300, av_key=st.session_state.get("cfg_av_key", ""))

                if df is not None and len(df) > 50:
                    st.markdown(
                        f'<span class="data-tag">SOURCE · {source.upper()}</span>&nbsp;'
                        f'<span class="data-tag">BARS · {len(df)}</span>&nbsp;'
                        f'<span class="data-tag">LAST · {df.index[-1].strftime("%Y-%m-%d %H:%M UTC")}</span>',
                        unsafe_allow_html=True
                    )

                    ind = Indicators()
                    atr_s = ind.atr(df['high'], df['low'], df['close'])
                    smc = SmartMoneyEngine(df, atr_s)
                    quick_result = {
                        'order_blocks': smc.detect_order_blocks(),
                        'fvgs': smc.detect_fair_value_gaps(),
                        'sweeps': smc.detect_liquidity_sweeps(),
                        'structure': smc.detect_market_structure(),
                        'source': source
                    }

                    fig = build_chart(df, chart_symbol, chart_tf, quick_result)
                    st.plotly_chart(fig, use_container_width=True)

                    # Fibonacci levels
                    high50 = float(df['high'].tail(50).max())
                    low50  = float(df['low'].tail(50).min())
                    diff   = high50 - low50
                    fibs   = {
                        '0.000': high50,
                        '0.236': high50 - diff * 0.236,
                        '0.382': high50 - diff * 0.382,
                        '0.500': high50 - diff * 0.500,
                        '0.618': high50 - diff * 0.618,
                        '1.000': low50,
                    }
                    st.markdown('<div class="section-label">Fibonacci Retracement · Last 50 Bars</div>', unsafe_allow_html=True)
                    fib_html = '<div class="fib-grid">'
                    for level, fprice in fibs.items():
                        fib_html += f'<div class="fib-cell"><div class="fib-level">FIB {level}</div><div class="fib-price">{fprice:.4f}</div></div>'
                    fib_html += '</div>'
                    st.markdown(fib_html, unsafe_allow_html=True)
                else:
                    st.error(f"Chart data unavailable: {source}")

    # ========== TAB 3: NEWS ==========
    with tab3:
        st.markdown('<div class="section-label">News Sentiment · VADER NLP Scoring</div>', unsafe_allow_html=True)

        news_symbol = st.selectbox("Instrument", instruments, key="news_sym")

        with st.spinner("Analyzing..."):
            articles = sentiment_engine.fetch_and_analyze(news_symbol)

        if articles:
            compounds = [a['compound'] for a in articles]
            avg_compound = np.mean(compounds)

            if avg_compound > 0.05:
                overall_txt = "BULLISH BIAS"
                overall_col = "#00C853"
            elif avg_compound < -0.05:
                overall_txt = "BEARISH BIAS"
                overall_col = "#FF1744"
            else:
                overall_txt = "NEUTRAL"
                overall_col = "#2A5080"

            # Aggregate row
            st.markdown(f"""
            <div class="sentiment-grid">
              <div class="sentiment-cell">
                <div class="sentiment-cell-label">Avg VADER Score</div>
                <div class="sentiment-cell-value" style="color:{overall_col}">{avg_compound:+.3f}</div>
              </div>
              <div class="sentiment-cell">
                <div class="sentiment-cell-label">Sentiment Bias</div>
                <div class="sentiment-cell-value" style="font-size:18px; font-weight:600; letter-spacing:3px; color:{overall_col}">{overall_txt}</div>
              </div>
              <div class="sentiment-cell">
                <div class="sentiment-cell-label" style="font-size:9px; letter-spacing:2px;
                            color:#2A4060; margin-bottom:8px; text-transform:uppercase">Articles Analyzed</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:28px; font-weight:300;
                            color:#4A9EFF">{len(articles)}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            for a in articles:
                compound = a.get('compound', 0)
                bar_pct = int(abs(compound) * 100)
                if compound > 0.05:
                    card_cls = 'news-card-bull'
                    score_cls = 'news-score-bull'
                    bar_color = '#00C853'
                elif compound < -0.05:
                    card_cls = 'news-card-bear'
                    score_cls = 'news-score-bear'
                    bar_color = '#FF1744'
                else:
                    card_cls = 'news-card-neutral'
                    score_cls = 'news-score-neutral'
                    bar_color = '#2A4060'

                st.markdown(f"""
                <div class="news-card {card_cls}">
                  <div class="news-title">{a['title']}</div>
                  <div class="news-desc">{a['description']}</div>
                  <div class="vader-bar-bg">
                    <div style="height:2px; width:{bar_pct}%; background:{bar_color}; border-radius:1px;
                                transition:width 0.3s ease;"></div>
                  </div>
                  <div class="news-meta">
                    <span class="news-source">{a['source']}  ·  {a['published']}</span>
                    <span class="{score_cls}">VADER {compound:+.3f}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ========== TAB 4: BACKTEST ==========
    with tab4:
        st.markdown('<div class="section-label">Historical Backtest · EMA Stack + RSI + MACD · ATR Stops · 2:1 RR</div>', unsafe_allow_html=True)

        bt_col1, bt_col2, bt_col3 = st.columns(3)
        bt_symbol  = bt_col1.selectbox("Symbol", instruments, key="bt_sym")
        bt_tf      = bt_col2.selectbox("Timeframe", timeframes, key="bt_tf")
        bt_account = bt_col3.number_input("Account Size ($)", value=10000, step=1000, key="bt_acct")

        if st.button("  ▶  RUN BACKTEST  ", type="primary"):
            with st.spinner("Running simulation..."):
                df, source = fetcher.fetch(bt_symbol, bt_tf, limit=500, av_key=st.session_state.get("cfg_av_key", ""))

                if df is None or len(df) < 100:
                    st.error(f"Insufficient data: {source}")
                else:
                    bt = Backtester(df, atr_multiplier=atr_multiplier, risk_pct=max_risk)
                    results = bt.run()

                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        pnl = results['total_pnl']
                        pnl_color = "#00C853" if pnl >= 0 else "#FF1744"
                        pf = results['profit_factor']
                        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"

                        # Stat grid
                        st.markdown(f"""
                        <div class="bt-stat-grid">
                          <div class="bt-stat">
                            <div class="bt-stat-label">Total Trades</div>
                            <div class="bt-stat-value">{results['total_trades']}</div>
                          </div>
                          <div class="bt-stat">
                            <div class="bt-stat-label">Win Rate</div>
                            <div class="bt-stat-value">{results['win_rate']:.1f}<span style="font-size:14px">%</span></div>
                          </div>
                          <div class="bt-stat">
                            <div class="bt-stat-label">Profit Factor</div>
                            <div class="bt-stat-value">{pf_str}</div>
                          </div>
                          <div class="bt-stat">
                            <div class="bt-stat-label">Net P&L</div>
                            <div class="bt-stat-value" style="color:{pnl_color}">${pnl:+.0f}</div>
                          </div>
                          <div class="bt-stat">
                            <div class="bt-stat-label">Max Drawdown</div>
                            <div class="bt-stat-value" style="color:#FF4444">{results['max_drawdown_pct']:.1f}<span style="font-size:14px">%</span></div>
                          </div>
                          <div class="bt-stat">
                            <div class="bt-stat-label">Sharpe Ratio</div>
                            <div class="bt-stat-value">{results['sharpe']:.2f}</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Equity curve
                        eq_color = '#00C853' if pnl >= 0 else '#FF1744'
                        eq_fill  = 'rgba(0,200,83,0.06)' if pnl >= 0 else 'rgba(255,23,68,0.06)'
                        eq_fig = go.Figure()
                        eq_fig.add_trace(go.Scatter(
                            y=results['equity_curve'],
                            mode='lines', name='Equity',
                            line=dict(color=eq_color, width=1.5),
                            fill='tozeroy', fillcolor=eq_fill
                        ))
                        eq_fig.update_layout(
                            height=300,
                            template='plotly_dark',
                            paper_bgcolor='#080C10',
                            plot_bgcolor='#0C1117',
                            font=dict(family='IBM Plex Mono', size=10, color='#4A6080'),
                            title=dict(
                                text=f"EQUITY CURVE  ·  {bt_symbol}  ·  {bt_tf}",
                                font=dict(family='IBM Plex Mono', size=10, color='#2A5080'),
                                x=0.01
                            ),
                            xaxis=dict(gridcolor='#0F1A28', showgrid=True, title="TRADE #",
                                       title_font=dict(size=9, color='#2A4060')),
                            yaxis=dict(gridcolor='#0F1A28', showgrid=True, title="EQUITY ($)",
                                       title_font=dict(size=9, color='#2A4060')),
                            margin=dict(l=50, r=20, t=40, b=40)
                        )
                        st.plotly_chart(eq_fig, use_container_width=True)

                        # Trade log
                        with st.expander("TRADE LOG"):
                            trades_df = results['trades'].copy()
                            trades_df['entry_date'] = trades_df['entry_date'].dt.strftime('%Y-%m-%d %H:%M')
                            trades_df['exit_date']  = trades_df['exit_date'].dt.strftime('%Y-%m-%d %H:%M')
                            trades_df['pnl_pct']    = trades_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
                            trades_df['pnl_dollar']  = trades_df['pnl_dollar'].apply(lambda x: f"${x:+.0f}")
                            st.dataframe(
                                trades_df[['entry_date', 'exit_date', 'direction', 'entry',
                                           'exit', 'pnl_pct', 'pnl_dollar', 'won', 'reason']],
                                use_container_width=True
                            )

                        st.markdown(
                            f'<span class="data-tag">SOURCE · {source.upper()}</span>&nbsp;'
                            f'<span class="data-tag">RISK · {max_risk}%</span>&nbsp;'
                            f'<span class="data-tag">ATR MULT · {atr_multiplier}×</span>&nbsp;'
                            f'<span class="data-tag">BARS · {len(df)}</span>',
                            unsafe_allow_html=True
                        )


if __name__ == "__main__":
    main()
