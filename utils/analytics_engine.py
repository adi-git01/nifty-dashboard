
import pandas as pd
import numpy as np
import yfinance as yf

def safe_get(data, key, default=0):
    val = data.get(key)
    return val if val is not None else default


def calculate_cycle_position(ticker: str) -> dict:
    """
    Calculates where a stock is in its margin cycle.
    
    Returns:
        {
            "phase": "EARLY_RECOVERY" | "MID_CYCLE" | "LATE_CYCLE" | "DOWNTURN",
            "phase_num": 1-4 (for visualization),
            "current_margin": float (current OPM %),
            "avg_margin": float (historical avg OPM %),
            "margin_vs_avg": float (ratio: current/avg),
            "trend": "rising" | "falling" | "stable",
            "volatility": float (std dev of margins),
            "is_cyclical": bool (high volatility = cyclical)
        }
    """
    try:
        stock = yf.Ticker(ticker)
        income = stock.quarterly_income_stmt
        
        if income is None or income.empty:
            return {"phase": "UNKNOWN", "phase_num": 0, "error": "No income statement data"}
        
        # Check for required rows
        if 'Operating Income' not in income.index or 'Total Revenue' not in income.index:
            # Try alternative row names
            op_income_row = None
            revenue_row = None
            
            for row in income.index:
                if 'operating' in row.lower() and 'income' in row.lower():
                    op_income_row = row
                if 'revenue' in row.lower() and 'total' in row.lower():
                    revenue_row = row
            
            if not op_income_row or not revenue_row:
                return {"phase": "UNKNOWN", "phase_num": 0, "error": "Missing operating income or revenue"}
            
            op_income = income.loc[op_income_row]
            revenue = income.loc[revenue_row]
        else:
            op_income = income.loc['Operating Income']
            revenue = income.loc['Total Revenue']
        
        # Calculate OPM history
        opm_history = (op_income / revenue * 100).dropna()
        
        if len(opm_history) < 3:
            return {"phase": "UNKNOWN", "phase_num": 0, "error": "Insufficient history"}
        
        # Current margin (most recent quarter)
        current_margin = opm_history.iloc[0]
        
        # Average margin (all available periods)
        avg_margin = opm_history.mean()
        
        # Margin trend (compare recent 2 quarters vs previous 2)
        if len(opm_history) >= 4:
            recent_avg = opm_history.iloc[:2].mean()
            previous_avg = opm_history.iloc[2:4].mean()
            trend_change = recent_avg - previous_avg
            
            if trend_change > 1.0:  # Rising by more than 1%
                trend = "rising"
            elif trend_change < -1.0:  # Falling by more than 1%
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Calculate volatility (cyclicality indicator)
        volatility = opm_history.std()
        is_cyclical = volatility > 3.0  # High margin volatility = cyclical stock
        
        # Margin vs average ratio
        margin_vs_avg = current_margin / avg_margin if avg_margin != 0 else 1.0
        
        # Determine cycle phase
        # 4 quadrants based on: margin position + trend direction
        above_avg = margin_vs_avg > 1.05  # 5% above average
        near_avg = 0.90 <= margin_vs_avg <= 1.10  # Within 10% of average
        below_avg = margin_vs_avg < 0.95  # 5% below average
        
        if below_avg and trend == "rising":
            phase = "EARLY_RECOVERY"
            phase_num = 1
        elif (above_avg or near_avg) and trend in ["rising", "stable"]:
            # Stable at or above average = mid-cycle
            phase = "MID_CYCLE"
            phase_num = 2
        elif above_avg and trend == "falling":
            phase = "LATE_CYCLE"
            phase_num = 3
        elif below_avg and trend in ["falling", "stable"]:
            # Only DOWNTURN if clearly below avg and not improving
            phase = "DOWNTURN"
            phase_num = 4
        else:
            # Default to mid-cycle for edge cases
            phase = "MID_CYCLE"
            phase_num = 2
        
        return {
            "phase": phase,
            "phase_num": phase_num,
            "current_margin": round(current_margin, 1),
            "avg_margin": round(avg_margin, 1),
            "margin_vs_avg": round(margin_vs_avg, 2),
            "trend": trend,
            "volatility": round(volatility, 1),
            "is_cyclical": is_cyclical
        }
        
    except Exception as e:
        return {"phase": "UNKNOWN", "phase_num": 0, "error": str(e)}


def calculate_sector_cycle(sector_tickers: list) -> dict:
    """
    Aggregates cycle positions for all stocks in a sector.
    
    Returns:
        {
            "phase": Dominant phase in sector,
            "phase_distribution": {"EARLY_RECOVERY": n, "MID_CYCLE": n, ...},
            "avg_margin_vs_avg": float,
            "pct_at_peak": float (% of stocks in Late Cycle)
        }
    """
    phase_counts = {"EARLY_RECOVERY": 0, "MID_CYCLE": 0, "LATE_CYCLE": 0, "DOWNTURN": 0, "UNKNOWN": 0}
    margin_ratios = []
    
    for ticker in sector_tickers:
        cycle = calculate_cycle_position(ticker)
        phase = cycle.get("phase", "UNKNOWN")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        if "margin_vs_avg" in cycle:
            margin_ratios.append(cycle["margin_vs_avg"])
    
    # Determine dominant phase (excluding UNKNOWN)
    valid_phases = {k: v for k, v in phase_counts.items() if k != "UNKNOWN"}
    dominant_phase = max(valid_phases, key=valid_phases.get) if valid_phases else "UNKNOWN"
    
    # Calculate sector averages
    avg_margin_vs_avg = sum(margin_ratios) / len(margin_ratios) if margin_ratios else 1.0
    
    # Percentage at peak (Late Cycle or Mid-Cycle with high margins)
    total_valid = sum(valid_phases.values())
    pct_at_peak = (phase_counts["LATE_CYCLE"] / total_valid * 100) if total_valid > 0 else 0
    
    return {
        "phase": dominant_phase,
        "phase_distribution": phase_counts,
        "avg_margin_vs_avg": round(avg_margin_vs_avg, 2),
        "pct_at_peak": round(pct_at_peak, 1),
        "total_stocks": len(sector_tickers),
        "analyzed": total_valid
    }


def analyze_stock_health(info, scores):
    """
    Analyzes stock data and returns a dictionary of qualitative insights (rationales, scenarios, verdict).
    Updated for 4-pillar scoring system.
    """
    
    # --- 1. 4-Pillar Rationales ---
    rationales = {}
    
    # Quality Pillar
    roe = safe_get(info, 'roe', 0)
    roe_pct = roe * 100 if roe < 1 else roe
    npm = safe_get(info, 'profitMargins', 0) * 100
    
    if roe_pct > 20:
        rationales['quality'] = f"Exceptional capital efficiency (ROE: {roe_pct:.1f}%) with strong profitability."
    elif roe_pct < 8:
        rationales['quality'] = f"Capital efficiency is low (ROE: {roe_pct:.1f}%), suggesting tough industry economics."
    else:
        rationales['quality'] = "Solid business quality with stable margins and reasonable returns."

    # Value Pillar
    pe = safe_get(info, 'pe', 25)
    peg = safe_get(info, 'pegRatio', 1.5)
    if peg < 1.0 and pe < 20:
        rationales['value'] = "Stock appears undervalued relative to growth (PEG < 1)."
    elif pe > 50:
        rationales['value'] = "Trading at premium valuation - pricing in significant future growth."
    else:
        rationales['value'] = "Fairly valued relative to sector peers and growth outlook."

    # Growth Pillar
    rev_growth = safe_get(info, 'revenueGrowth', 0) * 100
    earn_growth = safe_get(info, 'earningsGrowth', 0) * 100
    if rev_growth > 15 and earn_growth > 20:
        rationales['growth'] = f"Strong growth trajectory (Revenue: {rev_growth:.1f}%, Earnings: {earn_growth:.1f}%)."
    elif rev_growth < 5 or earn_growth < 0:
        rationales['growth'] = "Growth is stagnating - needs re-acceleration of top/bottom line."
    else:
        rationales['growth'] = "Moderate growth with steady revenue and earnings progression."

    # Momentum Pillar
    change = safe_get(info, '52WeekChange', 0) * 100
    if change > 40:
        rationales['momentum'] = f"Strong momentum ({change:.1f}% 1Y) - market confidence is high."
    elif change < -10:
        rationales['momentum'] = f"Negative momentum ({change:.1f}% 1Y) - market skepticism or sector headwinds."
    else:
        rationales['momentum'] = "Stock is consolidating with no clear directional trend."

    # --- 2. Three-Scenario Framework ---
    de = safe_get(info, 'debtToEquity', 0)
    
    # BULL CASE
    bull_case = []
    if rev_growth > 15:
        bull_case.append("Sustained revenue growth (>15%) drives operating leverage.")
    if safe_get(info, 'grossMargins', 0) > 0.40:
        bull_case.append("Premium product mix expansion further boosts margins.")
    bull_case.append("Market share gains from peers accelerate growth.")
    
    # BASE CASE
    base_case = []
    base_case.append("Steady mid-teen earnings growth inline with sector averages.")
    base_case.append("Margins remain stable as input cost pressures are passed on.")
    
    # BEAR CASE
    bear_case = []
    if de > 100:
        bear_case.append("Rising interest rates squeeze net profit margins due to high debt.")
    if pe > 50:
        bear_case.append("Valuation de-rating risk if growth misses estimates.")
    bear_case.append("Intensifying competition erodes pricing power.")

    # --- 3. Key Metrics Status ---
    metrics_status = {
        "pe": "Attractive" if pe < 25 else "Premium",
        "quality": "Strong" if roe_pct > 15 else "Moderate" if roe_pct > 8 else "Weak",
        "de": "Safe" if de < 100 else "Elevated",
        "growth": "High" if rev_growth > 15 else "Moderate" if rev_growth > 5 else "Low"
    }
    
    return {
        "rationales": rationales,
        "bull_case": bull_case,
        "base_case": base_case,
        "bear_case": bear_case,
        "metrics_status": metrics_status
    }


def analyze_sectors(df):
    """
    Aggregates metrics by sector to find hot/cold industries.
    Returns a DataFrame indexed by sector with standardized column names.
    """
    if df.empty or 'sector' not in df.columns:
        return pd.DataFrame()
    
    # Ensure numeric columns
    numeric_cols = ['pe', 'overall', 'momentum', 'quality', 'value', 'growth', 'trend_score']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Build agg dict dynamically based on available columns
    agg_dict = {'ticker': 'count'}
    
    if 'overall' in df.columns:
        agg_dict['overall'] = 'mean'
    if 'trend_score' in df.columns:
        agg_dict['trend_score'] = 'mean'
    if 'pe' in df.columns:
        agg_dict['pe'] = 'median'
    if 'momentum' in df.columns:
        agg_dict['momentum'] = 'mean'
    if 'quality' in df.columns:
        agg_dict['quality'] = 'mean'

    sector_stats = df.groupby('sector').agg(agg_dict)
    
    # Rename columns to expected format
    rename_map = {
        'ticker': 'count',
        'overall': 'avg_overall',
        'trend_score': 'avg_trend_score',
        'pe': 'median_pe',
        'momentum': 'avg_momentum',
        'quality': 'avg_quality'
    }
    sector_stats = sector_stats.rename(columns=rename_map)
    
    # Round values
    if 'avg_overall' in sector_stats.columns:
        sector_stats['avg_overall'] = sector_stats['avg_overall'].round(1)
    if 'avg_trend_score' in sector_stats.columns:
        sector_stats['avg_trend_score'] = sector_stats['avg_trend_score'].round(0)
    if 'avg_momentum' in sector_stats.columns:
        sector_stats['avg_momentum'] = sector_stats['avg_momentum'].round(1)
    
    # Sort by average overall score
    if 'avg_overall' in sector_stats.columns:
        sector_stats = sector_stats.sort_values(by='avg_overall', ascending=False)
    
    return sector_stats


def get_monthly_alpha_calendar():
    """
    Parses `granular_industry_analysis.csv` and `evolution_report.txt` to build 
    a 12-month calendar of which industries to Accumulate/Avoid, including 
    price-earnings lead/lag times.
    """
    import os
    
    cycles_path = "analysis_2026/industry_cycles/granular_industry_analysis.csv"
    lags_path = "analysis_2026/industry_lags/evolution_report.txt"
    
    if not os.path.exists(cycles_path):
        return pd.DataFrame()
        
    df_cycles = pd.read_csv(cycles_path)
    
    # 1. Parse Post-COVID lags from the text report
    lag_map = {}
    if os.path.exists(lags_path):
        with open(lags_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Basic parsing: look for lines with ' -> ' which indicate the shift
                if '->' in line and not line.startswith(' ' * 10): # skip header rows
                    parts = line.split()
                    try:
                        # Extract industry name (can be multiple words)
                        # The last 3 tokens are usually ['Leads', 'by', 'Xmo'] or similar shift text
                        # Before that is Post-COVID_Lag (int), Pre-COVID_Lag (int)
                        # Everything before that is Industry
                        
                        # Let's use a simpler heuristic: the line contains the industry name
                        # We can just match the industry names from df_cycles
                        pass # too risky to split blindly
                    except:
                        pass
        
        # Safer parsing: regex or fixed width. Let's just do a simpler pass over the lines.
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith('STRUCTURAL') or line_str.startswith('Industry') or line_str.startswith('STRATEGY') or line_str.startswith('Horizon'):
                continue
                
            # Attempt to split. The structure is: [Industry Name] [Pre_Lag] [Post_Lag] [Shift Text]
            # Since Industry Name can have spaces, we can regex or split from right.
            # Example: "Aerospace & Defense             -1               0   Leads by 1mo -> Concurrent"
            # It's fixed width or space delimited. Let's find the numbers.
            import re
            match = re.search(r'(-\d+|\d+)\s+(-\d+|\d+)\s+([A-Za-z0-9 >-]+)$', line_str)
            if match:
                post_lag = int(match.group(2))
                industry = line_str[:match.start()].strip()
                lag_map[industry] = post_lag
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    calendar_data = []
    
    for month in months:
        # Best industries
        best_df = df_cycles[df_cycles['Best_Month'] == month]
        best_list = []
        for _, row in best_df.iterrows():
            ind = row['Industry']
            lag = lag_map.get(ind, 0)
            if lag < 0:
                best_list.append(f"{ind} (Lead: {abs(lag)}mo)")
            else:
                best_list.append(ind)
                
        # Worst industries
        worst_df = df_cycles[df_cycles['Worst_Month'] == month]
        worst_list = worst_df['Industry'].tolist()
        
        calendar_data.append({
            '📅 Month': month,
            '🟢 Historical Best (Accumulate)': " • ".join(best_list) if best_list else "-",
            '🔴 Historical Worst (Avoid/Lighten)': " • ".join(worst_list) if worst_list else "-"
        })
        
    return pd.DataFrame(calendar_data)
