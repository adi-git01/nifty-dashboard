
import streamlit as st

def css_styles():
    return """
    <style>
        /* Global Font & Colors */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        .stApp {
            background-color: #0E1117;
            font-family: 'Outfit', sans-serif;
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(0, 240, 255, 0.3);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }

        /* Verdict Badges */
        .verdict-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .verdict-buy {
            background: rgba(52, 199, 89, 0.2);
            color: #34C759;
            border: 1px solid #34C759;
        }
        .verdict-hold {
            background: rgba(255, 204, 0, 0.2);
            color: #FFCC00;
            border: 1px solid #FFCC00;
        }
        .verdict-avoid {
            background: rgba(255, 59, 48, 0.2);
            color: #FF3B30;
            border: 1px solid #FF3B30;
        }
        
        /* Metric Styling */
        .metric-label {
            color: #888;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            color: #FFF;
            font-size: 1.8em;
            font-weight: 600;
        }
        .metric-delta-pos { color: #34C759; font-size: 0.9em; }
        .metric-delta-neg { color: #FF3B30; font-size: 0.9em; }

        /* Headers */
        h1, h2, h3 {
            color: #FFFFFF;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }
        
    </style>
    """

def card_metric(label, value, delta=None):
    delta_html = ""
    if delta:
        color_class = "metric-delta-pos" if "+" in delta or not delta.startswith("-") else "metric-delta-neg"
        delta_html = f"<div class='{color_class}'>{delta}</div>"
        
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
