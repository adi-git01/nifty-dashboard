
import pandas as pd
import numpy as np
import yfinance as yf
from utils.score_history import calculate_historical_scores

def scan_volume_changes(tickers, lookback_days=5):
    """
    Scans list of tickers for Volume Score changes >= 2 or <= -2.
    Returns list of alert objects.
    """
    print(f"Scanning {len(tickers)} stocks for Volume Alerts...")
    
    # Batch download (efficient)
    # We need enough history to calculate scores (approx 200 days for safety, though volume score uses less)
    # Using '1y' to be safe for moving averages
    try:
        data = yf.download(tickers, period="1y", group_by='ticker', progress=False, auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return []
        
    alerts = []
    
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                if ticker not in data.columns.get_level_values(0): continue
                df = data[ticker].dropna()
            else:
                df = data.dropna()
                
            if len(df) < 50: continue
            
            # Calculate Scores
            scores = calculate_historical_scores(df)
            
            if 'volume_score_hist' not in scores.columns: continue
            
            # Get last two valid scores
            valid_scores = scores['volume_score_hist'].dropna()
            if len(valid_scores) < 2: continue
            
            current_score = valid_scores.iloc[-1]
            prev_score = valid_scores.iloc[-2]
            
            delta = current_score - prev_score
            
            if abs(delta) >= 2:
                # Trigger Alert
                alert_type = "JUMP" if delta > 0 else "DROP"
                
                # Determine 'Smart' vs 'Panic' based on Magnitude and Trend
                # (Optional optimization: Check trend score too)
                trend_score = scores['trend_score_hist'].iloc[-1] if 'trend_score_hist' in scores.columns else 50
                
                signal_type = "Unknown"
                if delta > 0:
                    if delta <= 3 and trend_score < 30:
                        signal_type = "SMART ACCUMULATION" # The Elite Setup
                    elif delta > 3:
                        signal_type = "PANIC SPIKE"
                    else:
                        signal_type = "VOLUME SURGE"
                else:
                    signal_type = "VOLUME DRY-UP"
                
                alerts.append({
                    'ticker': ticker,
                    'change': delta,
                    'current': current_score,
                    'previous': prev_score,
                    'signal': signal_type,
                    'price': df['Close'].iloc[-1]
                })
                
        except Exception as e:
            continue
            
    return alerts
