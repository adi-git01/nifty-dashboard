"""
ULTIMATE SUB-INDUSTRY ALPHA PLAYBOOK GENERATOR
==============================================
Maps the 58 specific NSE Sub-Industries requested by the user to 
our historical 15-year databases:
1. Cyclicity (Long Cycle Wide Stops vs Short Cycle Tight Stops)
2. Seasonality (Golden Months vs Traps)
3. PEAD (Post Earnings Announcement Drift classification)

Outputs a comprehensive Markdown Playbook.
"""

import pandas as pd
import json
import os

OUTPUT_DIR = "analysis_2026/encyclopedia"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 58 NSE Sub-Industries supplied by the user
SUB_INDUSTRIES = [
    "Aerospace & Defense", "Agricultural Food & other Products", "Agricultural, Commercial & Construction Vehicles", 
    "Auto Components", "Automobiles", "Banks", "Beverages", "Capital Markets", "Cement & Cement Products", 
    "Chemicals & Petrochemicals", "Cigarettes & Tobacco Products", "Commercial Services & Supplies", 
    "Construction", "Consumable Fuels", "Consumer Durables", "Diversified", "Diversified FMCG", 
    "Diversified Metals", "Electrical Equipment", "Engineering Services", "Entertainment", "Ferrous Metals", 
    "Fertilizers & Agrochemicals", "Finance", "Financial Technology (Fintech)", "Food Products", "Gas", 
    "Healthcare Equipment & Supplies", "Healthcare Services", "Household Products", "Industrial Manufacturing", 
    "Industrial Products", "Insurance", "IT - Hardware", "IT - Services", "IT - Software", "Leisure Services", 
    "Media", "Metals & Minerals Trading", "Minerals & Mining", "Non - Ferrous Metals", "Oil", 
    "Other Construction Materials", "Other Consumer Services", "Other Utilities", "Paper, Forest & Jute Products", 
    "Personal Products", "Petroleum Products", "Pharmaceuticals & Biotechnology", "Power", "Printing & Publication", 
    "Realty", "Retailing", "Telecom - Equipment & Accessories", "Telecom - Services", "Textiles & Apparels", 
    "Transport Infrastructure", "Transport Services"
]

# ---------------------------------------------------------
# 1. CYCLICITY MAPPING (From sub_portfolio_cycles.py findings)
# ---------------------------------------------------------
LONG_CYCLE_KWS = ['auto', 'farm', 'heavy', 'metal', 'medical', 'conglomerates', 'machinery', 'agricultural', 'electric', 'software', 'aluminum', 'coal', 'defense', 'aerospace', 'cement', 'power', 'infrastructure', 'equipment', 'engineering', 'construction', 'minerals']
SHORT_CYCLE_KWS = ['insurance', 'capital market', 'chemical', 'tobacco', 'power producer', 'gas', 'tools', 'real estate', 'realty', 'lodging', 'diagnostic', 'broker', 'fmcg', 'consumer', 'retail', 'textile', 'apparel', 'food', 'beverage', 'leisure', 'media', 'entertainment', 'finance', 'banks']

def get_cyclicity(ind_str):
    s = str(ind_str).lower()
    for kw in LONG_CYCLE_KWS:
        if kw in s: return "🕰️ **LONG CYCLE** (Generational Wealth: Secular 3-4 year trends. Use Wide -20% trailing stops. Do not take targets.)"
    for kw in SHORT_CYCLE_KWS:
        if kw in s: return "⚡ **SHORT CYCLE** (Fast Flips: Volatile 6-month momentum. Use Tight -8% trailing stops and +20% hard targets.)"
    return "⚖️ **MID CYCLE** (Standard Momentum. Use Baseline -12% V3.1 trailing stops.)"


# ---------------------------------------------------------
# 2. SEASONALITY MAPPING (From utils/live_desk.py odds)
# ---------------------------------------------------------
SEASONAL_RULES = {
    'Financial': {'TRAP': [1, 2, 5], 'EDGE': [10, 11, 12]},
    'Industrial': {'TRAP': [1], 'EDGE': [3, 4, 5]},
    'IT': {'TRAP': [2, 3], 'EDGE': [6, 7]},
    'Auto': {'TRAP': [2], 'EDGE': [4, 5]},
    'Real Estate': {'TRAP': [2], 'EDGE': []},
    'Consumer': {'TRAP': [], 'EDGE': [9, 10]},
    'Healthcare': {'TRAP': [], 'EDGE': [7, 8]},
}

def get_seasonality(ind_str):
    s = str(ind_str).upper()
    cat = 'Other'
    if any(x in s for x in ['BANK', 'FINANC', 'INSUR', 'BROKER', 'CAPITAL']): cat = 'Financial'
    elif any(x in s for x in ['INDUST', 'CAPITAL GOODS', 'ENGINEERING', 'METAL', 'CONSTRUCTION', 'CEMENT', 'POWER', 'AEROSPACE', 'DEFENSE', 'EQUIPMENT']): cat = 'Industrial'
    elif any(x in s for x in ['IT', 'SOFTWARE', 'TECH', 'HARDWARE']): cat = 'IT'
    elif any(x in s for x in ['AUTO', 'VEHICLE']): cat = 'Auto'
    elif any(x in s for x in ['CONSUMER', 'FMCG', 'RETAIL', 'FOOD', 'BEVERAGE', 'CIGARETTE', 'PERSONAL']): cat = 'Consumer'
    elif any(x in s for x in ['HEALTH', 'PHARMA', 'HOSPITAL', 'BIOTECH']): cat = 'Healthcare'
    elif any(x in s for x in ['REALTY', 'REAL ESTATE', 'BUILDING']): cat = 'Real Estate'
    
    if cat in SEASONAL_RULES:
        months = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}
        traps = [months[m] for m in SEASONAL_RULES[cat].get('TRAP', [])]
        edges = [months[m] for m in SEASONAL_RULES[cat].get('EDGE', [])]
        
        t_str = ", ".join(traps) if traps else "None"
        e_str = ", ".join(edges) if edges else "None"
        return f"*   **🔥 Golden Edge Months:** {e_str}\n*   **⚠️ Seasonal Traps (Veto):** {t_str}"
    return "*   Neutral Seasonality (No distinct historical edge or trap. Trade pure momentum.)"


# ---------------------------------------------------------
# 3. PEAD (POST EARNINGS DRIFT) MAPPING 
# ---------------------------------------------------------
def load_pead_data():
    path = "analysis_2026/earnings_shocks/industry_drift_analysis.csv"
    if not os.path.exists(path): return None
    return pd.read_csv(path)

pead_df = load_pead_data()

def get_pead_profile(ind_str):
    if pead_df is None or pead_df.empty:
        return "Not enough historical shock events."
        
    s = str(ind_str).lower()
    
    # Heuristic mapping between NSE strings and Yahoo Finance strings in our DataFrame
    matches = pead_df[pead_df['Industry'].str.lower().str.contains(s.split()[0][:4], case=False, na=False)] 
    
    # Specific manual overrides for better mapping
    if "bank" in s: matches = pead_df[pead_df['Industry'].str.contains("Bank", case=False, na=False)]
    elif "auto" in s: matches = pead_df[pead_df['Industry'].str.contains("Auto", case=False, na=False)]
    elif "pharma" in s or "biotech" in s: matches = pead_df[pead_df['Industry'].str.contains("Drug|Biotech", case=False, na=False)]
    elif "it -" in s: matches = pead_df[pead_df['Industry'].str.contains("Software|Information Tech", case=False, na=False)]
    elif "fmcg" in s or "consumer durables" in s: matches = pead_df[pead_df['Industry'].str.contains("Household|Packaged Food", case=False, na=False)]
    elif "chemical" in s: matches = pead_df[pead_df['Industry'].str.contains("Chemical", case=False, na=False)]
    elif "telecom" in s: matches = pead_df[pead_df['Industry'].str.contains("Telecom", case=False, na=False)]
    elif "construction" in s or "engineering" in s: matches = pead_df[pead_df['Industry'].str.contains("Infrastructure|Engineering", case=False, na=False)]
    elif "metal" in s: matches = pead_df[pead_df['Industry'].str.contains("Steel|Aluminum|Metal", case=False, na=False)]
    elif "healthcare" in s: matches = pead_df[pead_df['Industry'].str.contains("Medical|Diagnostics", case=False, na=False)]
    
    if not matches.empty:
        match = matches.iloc[0] # Take best match
        c = match['Classification']
        drift = match['Drift_Ret_Tpos60']
        lead = match['Lead_Ret_T20']
        shock = match['Shock_Ret_T0']
        
        behavior_type = c.split('(')[0].strip()
        behavior_desc = c.split('(')[1].replace(')', '')
        
        # Color coding logic for markdown
        if "FRONT" in behavior_type: 
            icon = "🔴"
            action = "**FADE IT:** The move is usually over by the time earnings print."
        elif "DRIFT" in behavior_type: 
            icon = "🟢"
            action = "**BUY IT:** Massive continuation drift. Institutional sizing takes 2 months."
        else: 
            icon = "🟡"
            action = "**TRADE WITH CAUTION:** Standard continuation or priced perfectly on day 1."
            
        return f"{icon} **{behavior_type}** ({behavior_desc})\n   * {action}\n   * *Math Avg:* Pre-Shock Lead: {lead:+.1f}% | Day 1 Shock: {shock:+.1f}% | 60-Day PEAD: {drift:+.1f}%"
    
    return "🟡 **INSUFFICIENT DATA** (This specific niche hasn't produced enough massive single-day shocks in the last 5 years to build a statistically valid PEAD profile)."


# ---------------------------------------------------------
# GENERATE MASTER PLAYBOOK
# ---------------------------------------------------------
def generate_playbook():
    print("Generating Master Sub-Industry Alpha Playbook...")
    
    report = """# 📚 THE ULTIMATE SUB-INDUSTRY ALPHA PLAYBOOK
*(Generated February 2026 based on 15-Year Historical Averages)*

> **How to Use This Playbook**
> Keep this document open when you are executing discretionary trades or reviewing the scanner output. Before you buy into a specific sector (like Chemicals or Hospitals), check its playbook entry below. 
> 
> 1. Identify which **Risk Strategy (Cyclicity)** to use (Wide stops vs Tight stops).
> 2. Ensure you aren't buying into a **Seasonal Trap** month.
> 3. Know precisely how to trade its **Earnings Shocks (PEAD)**.

---
"""
    
    for sub in sorted(SUB_INDUSTRIES):
        report += f"## {sub}\n"
        report += f"**⏳ Business Cycle & Risk Allocation**\n"
        report += f"{get_cyclicity(sub)}\n\n"
        
        report += f"**📅 Seasonality Odds Matrix**\n"
        report += f"{get_seasonality(sub)}\n\n"
        
        report += f"**⚡ Earnings Shock & PEAD Behavior**\n"
        report += f"{get_pead_profile(sub)}\n\n"
        
        report += f"---\n\n"
        
    out_file = f"{OUTPUT_DIR}/Sub_Industry_Alpha_Playbook.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"✅ Success! Playbook written to: {out_file}")

if __name__ == "__main__":
    generate_playbook()
