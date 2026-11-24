"""
frontend/theme.py
Two-theme design system for QubRoute: Dark (High Contrast) and Light.
Pure Plus Jakarta Sans typography, prestiguous styling, no emojis, no gradient title text.
"""

from __future__ import annotations

import streamlit as st
from typing import Dict, Any

THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "name": "Dark (High Contrast)",
        "bg": "#040914",
        "panel": "#0b1a38",
        "panel_border": "#1c325c",
        "text": "#ffffff",
        "text_secondary": "#bccbe6",
        "muted": "#bccbe6",
        "rails": "#93b4ea",
        "gold": "#ffd762",
        "gold_border": "#ffd762",
        "red": "#ff6a70",
        "red_bg": "rgba(255, 106, 112, 0.12)",
        "green": "#6fe9b5",
        "green_bg": "rgba(111, 233, 181, 0.12)",
        "sidebar_bg": "#071226",
        "sidebar_border": "#142647",
        "card_bg": "#0d2044",
        "card_border": "#1e3a6a",
        "active_tab_bg": "#ffd762",
        "active_tab_text": "#040914",
        "tab_border": "#2b4b80",
        "badge_border": "#2b4b80",
        "table_header_bg": "#10254c",
        "table_row_even": "#0b1a38",
        "table_row_odd": "#08142c",
    },
    "light": {
        "name": "Light",
        "bg": "#eceae3",
        "panel": "#f8f7f3",
        "panel_border": "#dcd8ce",
        "text": "#14213d",
        "text_secondary": "#5a6580",
        "muted": "#5a6580",
        "rails": "#7d8aa3",
        "gold": "#7f6220",
        "gold_border": "#b8923b",
        "red": "#c23a43",
        "red_bg": "rgba(194, 58, 67, 0.08)",
        "green": "#0b7f5a",
        "green_bg": "rgba(11, 127, 90, 0.08)",
        "sidebar_bg": "#e4e1d8",
        "sidebar_border": "#d2cdbf",
        "card_bg": "#ffffff",
        "card_border": "#ded9cd",
        "active_tab_bg": "#14213d",
        "active_tab_text": "#ffffff",
        "tab_border": "#cbd2de",
        "badge_border": "#c5cbd6",
        "table_header_bg": "#ebe7dc",
        "table_row_even": "#f8f7f3",
        "table_row_odd": "#f0ede5",
    },
}


def get_current_theme() -> Dict[str, str]:
    if "qubroute_theme" not in st.session_state:
        # Default to dark (high contrast)
        st.session_state["qubroute_theme"] = "dark"
    return THEMES[st.session_state["qubroute_theme"]]


def inject_theme_css() -> None:
    theme = get_current_theme()
    is_dark = st.session_state["qubroute_theme"] == "dark"

    css = f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

    <style>
    /* Base reset & typography */
    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {theme["bg"]} !important;
        color: {theme["text"]} !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 2.2rem !important;
    }}

    .block-container {{
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 96% !important;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        color: {theme["text"]} !important;
        letter-spacing: -0.025em !important;
    }}

    p, span, label, div, td, th, [data-testid="stMarkdownContainer"] p {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: {theme["text"]} !important;
    }}

    .text-muted, .stCaption, [data-testid="stCaptionContainer"] {{
        color: {theme["muted"]} !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
    }}

    code, pre, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {theme["sidebar_bg"]} !important;
        border-right: 1.5px solid {theme["sidebar_border"]} !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: {theme["text"]} !important;
        font-weight: 600 !important;
    }}

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {theme["gold"] if is_dark else theme["text"]} !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        border: 1px solid {theme["card_border"]} !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
        background: {theme["panel"]} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
        font-weight: 700 !important;
        color: {theme["text"]} !important;
    }}

    /* Brand Header */
    .brand-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1.5px solid {theme["panel_border"]};
        padding-bottom: 1.2rem;
        margin-bottom: 1.5rem;
    }}

    .brand-title {{
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        color: {theme["text"]} !important;
        letter-spacing: -0.04em !important;
        line-height: 1.05 !important;
        margin: 0 !important;
    }}

    .brand-sub {{
        font-size: 0.95rem !important;
        color: {theme["muted"]} !important;
        font-weight: 500 !important;
        margin-top: 4px !important;
        letter-spacing: -0.01em !important;
    }}

    .theme-pill {{
        background: {theme["card_bg"]};
        border: 1.5px solid {theme["card_border"]};
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        color: {theme["text"]};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
        border-bottom: 2px solid {theme["panel_border"]} !important;
        padding-bottom: 6px !important;
        margin-bottom: 20px !important;
        background: transparent !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: {theme["muted"]} !important;
        padding: 8px 16px !important;
        background-color: {theme["panel"]} !important;
        border: 1.5px solid {theme["card_border"]} !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {theme["text"]} !important;
        border-color: {theme["rails"]} !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {theme["active_tab_text"]} !important;
        background-color: {theme["active_tab_bg"]} !important;
        border: 1.5px solid {theme["active_tab_bg"]} !important;
        font-weight: 700 !important;
        box-shadow: { '0 2px 10px rgba(255, 215, 98, 0.3)' if is_dark else '0 2px 8px rgba(20, 33, 61, 0.2)' } !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    /* Native Bordered Container Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {theme["panel"]} !important;
        border: 1.5px solid {theme["card_border"]} !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: {'0 4px 16px rgba(0,0,0,0.4)' if is_dark else '0 2px 8px rgba(0,0,0,0.04)'} !important;
    }}

    /* QUBO Accent Card */
    .qubo-card {{
        border: 1.5px solid {theme["gold"]} !important;
        background: {theme["card_bg"]} !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
        box-shadow: {'0 0 16px rgba(255, 215, 98, 0.18)' if is_dark else '0 2px 10px rgba(127, 98, 32, 0.12)'} !important;
        margin-bottom: 14px !important;
    }}

    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: {theme["panel"]} !important;
        border: 1.5px solid {theme["card_border"]} !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
    }}

    div[data-testid="stMetricLabel"] span,
    div[data-testid="stMetricLabel"] p {{
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        color: {theme["muted"]} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }}

    div[data-testid="stMetricValue"] div {{
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: {theme["text"]} !important;
        letter-spacing: -0.03em !important;
    }}

    div[data-testid="stMetricDelta"] {{
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }}

    /* DataFrames / Tables */
    div[data-testid="stDataFrame"] {{
        border: 1.5px solid {theme["card_border"]} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background-color: {theme["panel"]} !important;
    }}

    /* Action Buttons */
    div.stButton > button {{
        background: {theme["panel"]} !important;
        color: {theme["text"]} !important;
        border: 1.5px solid {theme["card_border"]} !important;
        border-radius: 6px !important;
        padding: 8px 20px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
    }}

    div.stButton > button:hover {{
        border-color: {theme["gold"]} !important;
        color: {theme["gold"]} !important;
        box-shadow: 0 0 10px {'rgba(255, 215, 98, 0.25)' if is_dark else 'rgba(127, 98, 32, 0.15)'} !important;
    }}

    div.stButton > button[kind="primary"] {{
        background: {theme["gold"] if is_dark else theme["text"]} !important;
        color: {theme["bg"] if is_dark else "#ffffff"} !important;
        border: 1.5px solid {theme["gold"] if is_dark else theme["text"]} !important;
    }}

    /* Badges */
    .badge-ok {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        background: {theme["green_bg"]};
        color: {theme["green"]} !important;
        font-weight: 700;
        font-size: 0.88rem;
        border: 1.5px solid {theme["green"]};
        margin: 6px 0;
    }}

    .badge-warn {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        background: {theme["red_bg"]};
        color: {theme["red"]} !important;
        font-weight: 700;
        font-size: 0.88rem;
        border: 1.5px solid {theme["red"]};
        margin: 6px 0;
    }}

    .badge-neutral {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        background: {theme["panel"]};
        color: {theme["text_secondary"]} !important;
        font-weight: 700;
        font-size: 0.88rem;
        border: 1.5px solid {theme["card_border"]};
        margin: 6px 0;
    }}

    .badge-gold {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        background: {'rgba(255, 215, 98, 0.12)' if is_dark else 'rgba(127, 98, 32, 0.10)'};
        color: {theme["gold"]} !important;
        font-weight: 700;
        font-size: 0.88rem;
        border: 1.5px solid {theme["gold"]};
        margin: 6px 0;
    }}

    /* Pipeline Step Box */
    .pipeline-step {{
        background: {theme["panel"]};
        border: 1.5px solid {theme["card_border"]};
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease;
    }}
    .pipeline-step:hover {{
        border-color: {theme["rails"]};
    }}
    .pipeline-num {{
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 12px;
        background: {theme["gold"] if is_dark else theme["text"]};
        color: {theme["bg"] if is_dark else "#ffffff"};
        font-weight: 800;
        font-size: 0.8rem;
        text-align: center;
        line-height: 24px;
        margin-right: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
