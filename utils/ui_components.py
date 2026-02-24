
import streamlit as st

# ─── Unified Color Palette ───
# Single source of truth for ALL colors used across the dashboard
COLORS = {
    # Core semantic colors
    'positive': '#00A389',       # Stripe Green
    'negative': '#D92D20',       # Destructive Red
    'warning': '#DC6803',        # Orange
    'info': '#0054D2',           # Blue
    'accent': '#635BFF',         # Stripe Blurple
    'accent_light': '#7F77F1',   # Hover Blurple
    
    # Surfaces
    'bg_primary': '#F7F9FC',     # App background (very light slate)
    'bg_card': '#FFFFFF',        # Clean white cards
    'bg_hover': '#F3F4F6',       # Light hover state
    'border': '#E3E8EE',         # Crisp light borders
    'border_hover': '#635BFF',   # Accent borders on hover
    
    # Text
    'text_primary': '#1A1F36',   # Deep slate (header text)
    'text_secondary': '#4F566B', # Mid slate (body text)
    'text_muted': '#8792A2',     # Light slate
    
    # Verdict
    'verdict_buy': '#00A389',
    'verdict_hold': '#FDB022',
    'verdict_avoid': '#D92D20',
}

# Single gradient used for ALL page headers
PAGE_HEADER_GRADIENT = 'linear-gradient(135deg, #635BFF 0%, #1A1F36 100%)'

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
        
        /* ── Glassmorphism Cards (Now Solid White Cards) ── */
        .glass-card {{
            background: {c['bg_card']};
            border-radius: 12px;
            border: 1px solid {c['border']};
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            border-color: {c['border']};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04);
        }}

        /* ── Verdict Badges ── */
        .verdict-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .verdict-buy {{
            background: #E3FBEF;
            color: var(--color-positive);
        }}
        .verdict-hold {{
            background: #FFF9E6;
            color: {c['warning']};
        }}
        .verdict-avoid {{
            background: #FEE4E2;
            color: var(--color-negative);
        }}
        
        /* ── Metric Styling ── */
        .metric-label {{
            color: var(--color-text-secondary);
            font-size: 0.85em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            color: var(--color-text);
            font-size: 1.8em;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}
        .metric-delta-pos {{ color: var(--color-positive); font-size: 0.9em; font-weight: 500; }}
        .metric-delta-neg {{ color: var(--color-negative); font-size: 0.9em; font-weight: 500; }}

        /* ── Standardized Headers ── */
        h1, h2, h3, h4, h5 {{
            color: var(--color-text);
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}
        h1 {{ font-size: 2em; }}
        h2 {{ font-size: 1.5em; }}
        h3 {{ font-size: 1.2em; }}
        
        /* ── Data Table Enhancements ── */
        [data-testid="stDataFrame"] {{
            border: 1px solid {c['border']};
            border-radius: 8px;
            overflow: hidden;
            background: white;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        [data-testid="stDataFrame"] tbody tr:nth-child(even) td {{
            background: #F9FAFB !important;
        }}
        [data-testid="stDataFrame"] tbody tr:hover td {{
            background: #F3F4F6 !important;
        }}
        [data-testid="stDataFrame"] thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: white !important;
            color: var(--color-text-secondary) !important;
            text-transform: uppercase;
            font-size: 0.8em;
            letter-spacing: 0.5px;
            border-bottom: 2px solid {c['border']} !important;
        }}
        
        /* ── Sidebar Visual Hierarchy ── */
        [data-testid="stSidebar"] {{
            background: white;
            border-right: 1px solid {c['border']};
        }}
        [data-testid="stSidebar"] .stMarkdown h3 {{
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--color-text-secondary);
            margin-bottom: 8px;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: {c['border']};
            margin: 16px 0;
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
            background: rgba(0, 0, 0, 0.15);
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(0, 0, 0, 0.25);
        }}
        
        /* ── Alert Cards (reusable) ── */
        .alert-card {{
            background: #FFFFFF;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid {c['border']};
            margin-bottom: 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .alert-card:hover {{
            background: #F9FAFB;
        }}
        .alert-card-danger {{ border-left: 4px solid var(--color-negative); }}
        .alert-card-success {{ border-left: 4px solid var(--color-positive); }}
        .alert-card-warning {{ border-left: 4px solid var(--color-warning); }}
        .alert-card-info {{ border-left: 4px solid var(--color-info); }}
        
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
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }}
        [data-testid="stMetric"]:hover {{
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--color-text-secondary) !important;
            font-size: 0.85em !important;
            font-weight: 500 !important;
            text-transform: none;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--color-text) !important;
            font-weight: 600 !important;
            letter-spacing: -0.5px !important;
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
        
        /* ── Hero P&L Card (Stripe-style) ── */
        .hero-pnl {{
            background: #FFFFFF;
            border: 1px solid {c['border']};
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            position: relative;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            overflow: hidden;
        }}
        .hero-pnl::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: var(--color-accent);
        }}
        .hero-pnl .hero-value {{
            font-size: 3.2em;
            font-weight: 600;
            letter-spacing: -1.5px;
            color: var(--color-text);
            line-height: 1.1;
        }}
        .hero-pnl .hero-return {{
            font-size: 1.2em;
            font-weight: 500;
            margin-top: 8px;
            color: var(--color-positive);
        }}
        .hero-pnl .hero-label {{
            color: var(--color-text-secondary);
            font-size: 0.85em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }}
        .hero-pnl .hero-meta {{
            display: flex;
            gap: 32px;
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid {c['border']};
        }}
        .hero-pnl .hero-meta-item {{
            font-size: 0.85em;
        }}
        .hero-pnl .hero-meta-item .meta-label {{
            color: var(--color-text-secondary);
            font-size: 0.85em;
            font-weight: 500;
        }}
        .hero-pnl .hero-meta-item .meta-value {{
            color: var(--color-text);
            font-weight: 600;
            font-size: 1.1em;
        }}
        
        /* ── Ticker Tape Animation ── */
        .ticker-tape {{
            overflow: hidden;
            white-space: nowrap;
            background: #FFFFFF;
            border: 1px solid {c['border']};
            border-radius: 8px;
            padding: 8px 0;
            margin: 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }}
        .ticker-tape-inner {{
            display: inline-block;
            animation: scroll-left 30s linear infinite;
        }}
        @keyframes scroll-left {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        
        /* ── Compact Grid Cards ── */
        .grid-card {{
            background: #FFFFFF;
            border: 1px solid {c['border']};
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }}
        .grid-card h4 {{
            margin: 0 0 8px 0;
            font-size: 0.9em;
            color: var(--color-text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* ── Persistent Alert Panel ── */
        .alert-panel {{
            max-height: 200px;
            overflow-y: auto;
            padding: 4px;
        }}
        .alert-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 6px;
            margin-bottom: 4px;
            font-size: 0.82em;
            background: white;
            border: 1px solid {c['border']};
            color: var(--color-text-secondary);
            transition: background 0.15s;
        }}
        .alert-item:hover {{
            background: #F9FAFB;
        }}
        .alert-dot {{
            width: 6px; height: 6px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .alert-dot--red {{ background: var(--color-negative); }}
        .alert-dot--green {{ background: var(--color-positive); }}
        .alert-dot--yellow {{ background: var(--color-warning); }}
        
        /* ── Live Badge ── */
        .live-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #E3FBEF;
            border: 1px solid #A6F4C5;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.75em;
            font-weight: 600;
            color: var(--color-positive);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .live-badge::before {{
            content: '';
            width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--color-positive);
            animation: pulse 1.5s ease-in-out infinite;
        }}
        
        /* ── Enhanced Mobile Responsiveness ── */
        @media (max-width: 768px) {{
            .hero-pnl {{
                padding: 18px 20px;
                border-radius: 14px;
            }}
            .hero-pnl .hero-value {{
                font-size: 2em;
            }}
            .hero-pnl .hero-meta {{
                flex-wrap: wrap;
                gap: 12px;
            }}
            [data-testid="stMetric"] {{
                padding: 8px 10px !important;
            }}
            [data-testid="stMetricValue"] {{
                font-size: 1.2em !important;
            }}
            /* Stack columns on mobile */
            [data-testid="column"] {{
                min-width: 100% !important;
            }}
        }}
        
        @media (max-width: 480px) {{
            .hero-pnl .hero-value {{
                font-size: 1.6em;
            }}
            .hero-pnl .hero-return {{
                font-size: 1.1em;
            }}
            .glass-card {{
                padding: 12px;
                border-radius: 10px;
            }}
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

def hero_pnl_card(portfolio_value, return_pct, holdings_count, strategy_name="OptComp-V21", since="Feb '26"):
    """Robinhood-style hero card with big P&L number."""
    ret_color = COLORS['positive'] if return_pct >= 0 else COLORS['negative']
    arrow = "▲" if return_pct >= 0 else "▼"
    
    return f"""
    <div class="hero-pnl">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="hero-label">{strategy_name} Model Portfolio</div>
                <div class="hero-value" style="color: white;">₹{portfolio_value:,.0f}</div>
                <div class="hero-return" style="color: {ret_color};">{arrow} {return_pct:+.1f}%
                    <span style="font-size: 0.6em; color: rgba(255,255,255,0.4); margin-left: 8px;">since {since}</span>
                </div>
            </div>
            <div class="live-badge">Live</div>
        </div>
        <div class="hero-meta">
            <div class="hero-meta-item">
                <div class="meta-label">Holdings</div>
                <div class="meta-value">{holdings_count} stocks</div>
            </div>
            <div class="hero-meta-item">
                <div class="meta-label">Strategy</div>
                <div class="meta-value">Price > MA50 + RS > 0</div>
            </div>
            <div class="hero-meta-item">
                <div class="meta-label">Rebalance</div>
                <div class="meta-value">Every 13 days</div>
            </div>
        </div>
    </div>
    """

def sidebar_alert_panel(alerts):
    """Persistent alert panel for sidebar — shows unread position alerts."""
    if not alerts:
        return "<div style='color: rgba(255,255,255,0.4); font-size: 0.8em; padding: 8px;'>No active alerts</div>"
    
    items_html = ""
    for a in alerts[:10]:  # Max 10 in sidebar
        ticker = a.get('ticker', '').replace('.NS', '').replace('.BO', '')
        alert_type = a.get('alert_type', 'ALERT')
        current = a.get('current_price', 0)
        
        if alert_type == 'STOP_LOSS':
            dot_class = 'alert-dot--red'
            icon = '🔴'
            label = 'SL HIT'
        elif alert_type == 'TARGET':
            dot_class = 'alert-dot--green'
            icon = '🎯'
            label = 'TARGET'
        else:
            dot_class = 'alert-dot--yellow'
            icon = '⚠️'
            label = 'ALERT'
        items_html += f"""
<div class="alert-item">
    <span class="alert-dot {dot_class}"></span>
    <span>{icon} <b>{ticker}</b> {label} ₹{current:.0f}</span>
</div>
"""
    
    return f"""
<div class="alert-panel">
{items_html}
</div>
"""
