"""
Dashboard Theme Configuration
==============================
Centralized color scheme and CSS styling for the Streamlit dashboard.
"""

import streamlit as st


# ============================================================================
# COLOR PALETTE
# ============================================================================

class Colors:
    """Centralized color definitions for the dark theme."""

    # Primary Colors
    PRIMARY = "#00d4ff"
    SECONDARY = "#00b8d4"

    # Background Colors
    BG_DARK = "#1e1e1e"
    BG_DARKER = "#121212"
    BG_CARD = "#2d2d30"
    BG_CARD_HOVER = "#3a3a3e"

    # Text Colors
    TEXT_PRIMARY = "#00d4ff"
    TEXT_SECONDARY = "#a0a0a0"
    TEXT_MUTED = "#6c757d"

    # Status Colors
    SUCCESS = "#4ade80"
    SUCCESS_BG = "#1a3d2e"
    SUCCESS_BG_ALT = "#0f2820"

    ERROR = "#f87171"
    ERROR_BG = "#3d1a1a"
    ERROR_BG_ALT = "#281010"

    WARNING = "#fbbf24"
    WARNING_BG = "#3d3a1a"
    WARNING_BG_ALT = "#282510"

    # Border Colors
    BORDER_PRIMARY = "#00d4ff33"
    BORDER_ACCENT = "#00d4ff"


# ============================================================================
# CSS THEME
# ============================================================================

def get_theme_css():
    """
    Returns the complete CSS theme for the dashboard.

    Returns:
        str: CSS styles as a string
    """
    return f"""
    <style>
    /* ========================================================================
       GLOBAL STYLES
       ======================================================================== */

    /* Page title styling */
    h1 {{
        background: linear-gradient(135deg, {Colors.PRIMARY} 0%, {Colors.SECONDARY} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* Section headers */
    h2, h3 {{
        color: {Colors.PRIMARY};
    }}

    /* Dark dividers */
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {Colors.PRIMARY}, transparent);
        margin: 1.5rem 0;
        opacity: 0.3;
    }}

    /* ========================================================================
       METRIC CARDS
       ======================================================================== */

    div[data-testid="metric-container"] {{
        background: linear-gradient(135deg, {Colors.BG_CARD} 0%, {Colors.BG_DARK} 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.15);
        border-left: 3px solid {Colors.BORDER_ACCENT};
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 1.6rem;
        font-weight: 600;
        color: {Colors.TEXT_PRIMARY} !important;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {Colors.TEXT_SECONDARY} !important;
    }}

    div[data-testid="stMetricDelta"] {{
        color: {Colors.SUCCESS} !important;
    }}

    /* ========================================================================
       BUTTONS
       ======================================================================== */

    .stButton > button {{
        background-color: {Colors.BG_CARD};
        border: 1px solid {Colors.BORDER_ACCENT};
        color: {Colors.PRIMARY};
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, {Colors.PRIMARY} 0%, {Colors.SECONDARY} 100%);
        color: #000;
        border-color: {Colors.BORDER_ACCENT};
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {Colors.PRIMARY} 0%, {Colors.SECONDARY} 100%);
        border: none;
        color: #000;
    }}

    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 6px 24px rgba(0, 212, 255, 0.5);
    }}

    /* ========================================================================
       SIDEBAR
       ======================================================================== */

    section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, {Colors.BG_DARK} 0%, {Colors.BG_DARKER} 100%);
    }}

    /* ========================================================================
       DATAFRAMES
       ======================================================================== */

    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.1);
        border: 1px solid {Colors.BORDER_PRIMARY};
    }}

    /* ========================================================================
       EXPANDERS
       ======================================================================== */

    .streamlit-expanderHeader {{
        background-color: {Colors.BG_CARD};
        border-radius: 8px;
        border: 1px solid {Colors.BORDER_PRIMARY};
    }}

    /* ========================================================================
       TABS
       ======================================================================== */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background-color: {Colors.BG_CARD};
        color: {Colors.TEXT_SECONDARY};
        border: 1px solid {Colors.BORDER_PRIMARY};
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {Colors.PRIMARY} 0%, {Colors.SECONDARY} 100%);
        color: #000;
        border-color: {Colors.BORDER_ACCENT};
    }}

    /* ========================================================================
       FILE UPLOADER
       ======================================================================== */

    div[data-testid="stFileUploader"] {{
        background: linear-gradient(135deg, {Colors.BG_CARD} 0%, {Colors.BG_DARK} 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed {Colors.BORDER_ACCENT};
    }}

    div[data-testid="stFileUploader"]:hover {{
        border-color: {Colors.SECONDARY};
        background: linear-gradient(135deg, {Colors.BG_CARD_HOVER} 0%, {Colors.BG_CARD} 100%);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.2);
    }}

    /* ========================================================================
       ALERTS
       ======================================================================== */

    div[data-testid="stAlert"] {{
        border-radius: 10px;
        border-left: 4px solid;
    }}

    /* ========================================================================
       DOWNLOAD BUTTON
       ======================================================================== */

    .stDownloadButton > button {{
        background: linear-gradient(135deg, {Colors.PRIMARY} 0%, {Colors.SECONDARY} 100%);
        color: #000;
        border: none;
    }}

    .stDownloadButton > button:hover {{
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }}

    /* ========================================================================
       CUSTOM COMPONENTS
       ======================================================================== */

    /* Status boxes */
    .status-box {{
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }}

    .status-success {{
        background: linear-gradient(135deg, {Colors.SUCCESS_BG} 0%, {Colors.SUCCESS_BG_ALT} 100%);
        box-shadow: 0 2px 12px rgba(74, 222, 128, 0.2);
        border: none;
        border-left: 4px solid {Colors.SUCCESS};
        color: {Colors.SUCCESS};
    }}

    .status-error {{
        background: linear-gradient(135deg, {Colors.ERROR_BG} 0%, {Colors.ERROR_BG_ALT} 100%);
        box-shadow: 0 2px 12px rgba(248, 113, 113, 0.2);
        border: none;
        border-left: 4px solid {Colors.ERROR};
        color: {Colors.ERROR};
    }}

    /* Pulse animation for online status */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}

    .status-indicator {{
        display: inline-block;
        width: 10px;
        height: 10px;
        background: {Colors.SUCCESS};
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 10px rgba(74, 222, 128, 0.6);
    }}

    /* Status badges */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    .status-completed {{
        background: linear-gradient(135deg, {Colors.SUCCESS_BG} 0%, {Colors.SUCCESS_BG_ALT} 100%);
        color: {Colors.SUCCESS};
        border: 1px solid {Colors.SUCCESS};
    }}

    .status-failed {{
        background: linear-gradient(135deg, {Colors.ERROR_BG} 0%, {Colors.ERROR_BG_ALT} 100%);
        color: {Colors.ERROR};
        border: 1px solid {Colors.ERROR};
    }}

    .status-processing {{
        background: linear-gradient(135deg, {Colors.WARNING_BG} 0%, {Colors.WARNING_BG_ALT} 100%);
        color: {Colors.WARNING};
        border: 1px solid {Colors.WARNING};
    }}

    /* Card containers */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {{
        background: linear-gradient(135deg, {Colors.BG_CARD} 0%, {Colors.BG_DARK} 100%);
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.1);
        border: 1px solid {Colors.BORDER_PRIMARY};
        transition: all 0.2s ease;
    }}

    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover {{
        box-shadow: 0 6px 24px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
        border-color: {Colors.BORDER_ACCENT};
    }}

    /* Main header (for dashboard.py) */
    .main-header {{
        background: linear-gradient(135deg, {Colors.PRIMARY} 0%, {Colors.SECONDARY} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}

    /* Subheader with accent */
    .stSubheader {{
        border-left: 4px solid {Colors.BORDER_ACCENT};
        padding-left: 0.75rem;
    }}
    </style>
    """


def apply_theme():
    """
    Apply the centralized theme to the current Streamlit page.

    Usage:
        from theme import apply_theme
        apply_theme()
    """
    st.markdown(get_theme_css(), unsafe_allow_html=True)
