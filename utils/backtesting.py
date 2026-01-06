
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def calculate_historical_scores(ticker, lookback_quarters=4):
    """
    Reconstructs approximate 4-pillar scores for historical quarters.
    
    Returns:
        DataFrame with columns: [date, quality, value, growth, momentum, overall]
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly financials
        income = stock.quarterly_income_stmt
        balance = stock.quarterly_balance_sheet
        hist = stock.history(period="2y")
        
        if income is None or income.empty or hist.empty:
            return pd.DataFrame()
        
        results = []
        
        # Get available quarters (columns are dates)
        quarters = list(income.columns)[:lookback_quarters]
        
        for q_date in quarters:
            try:
                # === QUALITY METRICS ===
                # ROE = Net Income / Equity
                ni = income.loc['Net Income', q_date] if 'Net Income' in income.index else 0
                equity = balance.loc['Stockholders Equity', q_date] if 'Stockholders Equity' in balance.index else 1
                roe = (ni / equity) if equity != 0 else 0
                
                # NPM = Net Income / Revenue  
                revenue = income.loc['Total Revenue', q_date] if 'Total Revenue' in income.index else 1
                npm = (ni / revenue) if revenue != 0 else 0
                
                # Quality score (simplified)
                roe_score = min(10, max(0, (roe * 100) / 2))  # 20% ROE = 10
                npm_score = min(10, max(0, (npm * 100) / 1.5))  # 15% NPM = 10
                quality = (roe_score + npm_score) / 2
                
                # === VALUE METRICS ===
                # Get price at quarter end
                q_date_dt = pd.Timestamp(q_date)
                price_at_q = hist[hist.index <= q_date_dt]['Close'].iloc[-1] if not hist[hist.index <= q_date_dt].empty else 0
                
                # Trailing EPS (annualize quarter)
                eps = (ni * 4) / stock.info.get('sharesOutstanding', 1) if ni else 0
                pe_at_q = (price_at_q / eps) if eps > 0 else 50
                
                # Value score (lower PE = higher score)
                value = max(0, min(10, 10 - (pe_at_q - 10) / 4))  # PE 10 = 10, PE 50 = 0
                
                # === GROWTH ===
                # Compare to previous quarter
                q_idx = quarters.index(q_date)
                if q_idx < len(quarters) - 1:
                    prev_q = quarters[q_idx + 1]
                    prev_ni = income.loc['Net Income', prev_q] if 'Net Income' in income.index else 0
                    growth_rate = ((ni - prev_ni) / abs(prev_ni)) if prev_ni != 0 else 0
                    growth = min(10, max(0, 5 + growth_rate * 10))  # 50% growth = 10
                else:
                    growth = 5  # Neutral
                
                # === MOMENTUM ===
                # 3-month return ending at quarter
                start_date = q_date_dt - timedelta(days=90)
                period_hist = hist[(hist.index >= start_date) & (hist.index <= q_date_dt)]
                if len(period_hist) > 5:
                    ret_3m = (period_hist['Close'].iloc[-1] / period_hist['Close'].iloc[0] - 1) * 100
                    momentum = min(10, max(0, 5 + ret_3m / 4))  # 20% return = 10
                else:
                    momentum = 5
                
                # === OVERALL ===
                overall = (quality * 0.3 + value * 0.25 + growth * 0.25 + momentum * 0.2)
                
                results.append({
                    'date': q_date_dt,
                    'quality': round(quality, 1),
                    'value': round(value, 1),
                    'growth': round(growth, 1),
                    'momentum': round(momentum, 1),
                    'overall': round(overall, 1)
                })
                
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        df = df.sort_values('date').reset_index(drop=True)
        return df
        
    except Exception as e:
        return pd.DataFrame()


def get_score_trend_insight(hist_scores_df, current_scores):
    """
    Generates a text insight comparing historical and current scores.
    """
    if hist_scores_df.empty:
        return "Insufficient historical data for comparison."
    
    oldest = hist_scores_df.iloc[0]
    current = current_scores.get('overall', 5)
    oldest_score = oldest['overall']
    
    diff = current - oldest_score
    
    if diff > 1.5:
        return f"📈 **Strong Improvement**: Score rose from {oldest_score} to {current} (+{diff:.1f}) over the past year."
    elif diff > 0.5:
        return f"📊 **Gradual Improvement**: Score increased from {oldest_score} to {current} (+{diff:.1f})."
    elif diff < -1.5:
        return f"📉 **Deteriorating**: Score fell from {oldest_score} to {current} ({diff:.1f}). Investigate fundamentals."
    elif diff < -0.5:
        return f"⚠️ **Slight Decline**: Score dropped from {oldest_score} to {current} ({diff:.1f})."
    else:
        return f"➡️ **Stable**: Score remained around {current:.1f} over the past year."
