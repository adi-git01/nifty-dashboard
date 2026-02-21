
import streamlit as st

# ─── Unified Color Palette ───
# Single source of truth for ALL colors used across the dashboard
COLORS = {
    # Core semantic colors
    'positive': '#00C853',       # Green — profits, uptrends, success
    'negative': '#FF5252',       # Red — losses, downtrends, errors
    'warning': '#FF9800',        # Orange — caution, attention
    'info': '#448AFF',           # Blue — neutral info, links
    'accent': '#764ba2',         # Purple — primary brand accent
    'accent_light': '#667eea',   # Light purple — gradients
    
    # Surfaces
    'bg_primary': '#0E1117',     # App background
    'bg_card': 'rgba(255, 255, 255, 0.05)',
    'bg_hover': 'rgba(255, 255, 255, 0.08)',
    'border': 'rgba(255, 255, 255, 0.1)',
    'border_hover': 'rgba(118, 75, 162, 0.5)',
    
    # Text
    'text_primary': '#FFFFFF',
    'text_secondary': '#a0a0a0',
    'text_muted': '#666',
    
    # Verdict
    'verdict_buy': '#00C853',
    'verdict_hold': '#FFCC00',
    'verdict_avoid': '#FF5252',
}

# Single gradient used for ALL page headers
PAGE_HEADER_GRADIENT = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'

def page_header(title, subtitle=""):
    """Unified page header with consistent gradient across all pages."""
    sub_html = f'<p style="color: rgba(255,255,255,0.8); margin-top: 8px; font-size: 1.05em;">{subtitle}</p>' if subtitle else ''
    return f"""
    <div style="background: {PAGE_HEADER_GRADIENT};
                padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);">
        <h1 style="color: white; margin: 0; font-size: 2em; font-weight: 700;
                   font-family: 'Outfit', sans-serif;">{title}</h1>
        {sub_html}
    </div>
    """

def css_styles():
    c = COLORS
    return f"""
    <style>
        /* ── Global Font & Colors ── */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {{
            --color-positive: {c['positive']};
            --color-negative: {c['negative']};
            --color-warning: {c['warning']};
            --color-info: {c['info']};
            --color-accent: {c['accent']};
            --color-accent-light: {c['accent_light']};
            --color-bg: {c['bg_primary']};
            --color-text: {c['text_primary']};
            --color-text-secondary: {c['text_secondary']};
            --color-text-muted: {c['text_muted']};
        }}
        
        .stApp {{
            background-color: var(--color-bg);
            font-family: 'Outfit', sans-serif;
        }}
        
        /* ── Glassmorphism Cards ── */
        .glass-card {{
            background: {c['bg_card']};
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid {c['border']};
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            border-color: {c['border_hover']};
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
        }}

        /* ── Verdict Badges ── */
        .verdict-badge {{
            display: inline-block;
            padding: 6px 18px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }}
        .verdict-buy {{
            background: rgba(0, 200, 83, 0.15);
            color: var(--color-positive);
            border: 1px solid var(--color-positive);
        }}
        .verdict-hold {{
            background: rgba(255, 204, 0, 0.15);
            color: {c['verdict_hold']};
            border: 1px solid {c['verdict_hold']};
        }}
        .verdict-avoid {{
            background: rgba(255, 82, 82, 0.15);
            color: var(--color-negative);
            border: 1px solid var(--color-negative);
        }}
        
        /* ── Metric Styling ── */
        .metric-label {{
            color: var(--color-text-secondary);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            color: var(--color-text);
            font-size: 1.8em;
            font-weight: 600;
        }}
        .metric-delta-pos {{ color: var(--color-positive); font-size: 0.9em; }}
        .metric-delta-neg {{ color: var(--color-negative); font-size: 0.9em; }}

        /* ── Standardized Headers ── */
        h1, h2, h3 {{
            color: var(--color-text);
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }}
        h1 {{ font-size: 2em; }}
        h2 {{ font-size: 1.5em; }}
        h3 {{ font-size: 1.2em; }}
        
        /* ── Data Table Enhancements ── */
        /* Alternating row backgrounds */
        [data-testid="stDataFrame"] tbody tr:nth-child(even) td {{
            background: rgba(255, 255, 255, 0.02) !important;
        }}
        [data-testid="stDataFrame"] tbody tr:hover td {{
            background: rgba(102, 126, 234, 0.08) !important;
        }}
        /* Sticky header */
        [data-testid="stDataFrame"] thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: #1a1a2e !important;
        }}
        
        /* ── Sidebar Visual Hierarchy ── */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0E1117 0%, #131722 100%);
        }}
        [data-testid="stSidebar"] .stMarkdown h3 {{
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--color-text-secondary);
            margin-bottom: 4px;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255, 255, 255, 0.06);
            margin: 12px 0;
        }}
        [data-testid="stSidebar"] .stCaption {{
            font-size: 0.78em;
        }}
        
        /* ── Smooth Scrollbars ── */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}
        
        /* ── Alert Cards (reusable) ── */
        .alert-card {{
            background: rgba(255,255,255,0.03);
            padding: 14px 18px;
            border-radius: 10px;
            margin-bottom: 8px;
            transition: background 0.2s ease;
        }}
        .alert-card:hover {{
            background: rgba(255,255,255,0.06);
        }}
        .alert-card-danger {{
            border-left: 4px solid var(--color-negative);
        }}
        .alert-card-success {{
            border-left: 4px solid var(--color-positive);
        }}
        .alert-card-warning {{
            border-left: 4px solid var(--color-warning);
        }}
        .alert-card-info {{
            border-left: 4px solid var(--color-info);
        }}
        
        /* ── Responsive helpers ── */
        @media (max-width: 768px) {{
            .glass-card {{
                padding: 16px;
                border-radius: 12px;
            }}
            .metric-value {{
                font-size: 1.4em;
            }}
        }}
        
        /* ── Streamlit metric override for consistent look ── */
        [data-testid="stMetric"] {{
            background: {c['bg_card']};
            border: 1px solid {c['border']};
            border-radius: 12px;
            padding: 12px 16px;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: {c['border_hover']};
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--color-text-secondary) !important;
            font-size: 0.85em !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        [data-testid="stMetricValue"] {{
            font-weight: 600 !important;
        }}
        [data-testid="stMetricDelta"] svg {{
            display: none;
        }}
        
        /* ── Tab styling ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-weight: 500;
        }}
        
        /* ── Button polish ── */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        /* ── Loading progress ── */
        .load-step {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 0;
            font-size: 0.95em;
        }}
        .load-step .step-dot {{
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--color-accent);
            animation: pulse 1.2s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 0.4; transform: scale(0.8); }}
            50% {{ opacity: 1; transform: scale(1.2); }}
        }}
        
    </style>
    """

def card_metric(label, value, delta=None):
    delta_html = ""
    if delta:
        is_positive = "+" in str(delta) or (isinstance(delta, str) and not delta.startswith("-"))
        color = COLORS['positive'] if is_positive else COLORS['negative']
        delta_html = f"<div style='color: {color}; font-size: 0.9em; margin-top: 4px;'>{delta}</div>"
        
    return f"""
    <div class="glass-card" style="padding: 16px; text-align: center;">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def card_verdict(verdict, score):
    v_class = "verdict-buy" if verdict == "BUY" else "verdict-hold" if verdict == "HOLD" else "verdict-avoid"
    return f"""
    <div class="glass-card" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="metric-label">Algorithmic Verdict</div>
            <div style="font-size: 1.5em; font-weight: bold; margin-top: 5px;">{score}/10</div>
        </div>
        <div class="{v_class} verdict-badge">{verdict}</div>
    </div>
    """

