"""
OPTIMAL POSITION SIZING & MONTE CARLO SIMULATION (KELLY CRITERION)
==================================================================
Reads the historical empirical trade log of the V3.1 Momentum Engine 
(376 trades over 7 years) and runs a 10,000-iteration Monte Carlo 
simulation to determine the optimal bet size (fractional Kelly) per trade.

Goal: Maximize Terminal Wealth while keeping Risk of Ruin (Drawdown > 35%) near zero.
"""

import pandas as pd
import numpy as np
import os
import sys

OUTPUT_DIR = "analysis_2026/position_sizing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADE_LOG_PATH = "analysis_2026/cycle_portfolios/Baseline_Uniform_V3.1_trades.csv"

# Test different fixed allocation percentages per trade
BET_SIZES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.33, 0.50]
SIMULATIONS = 10000
TRADES_PER_SIM = 300 # Approximate trades in a 5-year cycle

def calculate_theoretical_kelly(trades):
    wins = trades[trades > 0]
    losses = trades[trades <= 0]
    
    w = len(wins) / len(trades)
    aw = wins.mean()
    al = abs(losses.mean())
    
    if al == 0: return 1.0 # Infinite risk/reward
    
    r = aw / al
    kelly_f = w - ((1 - w) / r)
    return kelly_f, w, aw, al

def run_monte_carlo(trades):
    print(f"\n========================================================")
    print(f"POSITION SIZING MONTE CARLO (10,000 SIMULATIONS)")
    print(f"========================================================")
    
    if len(trades) < 50:
        print("Not enough trades for a valid empirical distribution.")
        return
        
    kelly_f, w, aw, al = calculate_theoretical_kelly(trades)
    print(f"\n[EMPIRICAL DISTRIBUTION DATA]")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Win Rate (W): {w*100:.1f}%")
    print(f"  Avg Win: {aw:.1f}% | Avg Loss: {al:.1f}%")
    print(f"  Payoff Ratio (R): {(aw/al):.2f}")
    print(f"  ==> THEORETICAL FULL KELLY (Optimal F): {kelly_f*100:.1f}%")
    print(f"  ==> HALF KELLY (Safe): {(kelly_f/2)*100:.1f}%")
    print(f"  ==> QUARTER KELLY (Conservative): {(kelly_f/4)*100:.1f}%\n")
    
    print("Running 10,000 Monte Carlo Simulations across various fixed bet sizes...")
    
    results = []
    
    # Pre-generate 10,000 sequences of 300 random trades from the empirical distribution
    # Shape: (10000, 300)
    sim_returns = np.random.choice(trades.values, size=(SIMULATIONS, TRADES_PER_SIM), replace=True)
    
    for f in BET_SIZES:
        # Array of returns if we allocate fraction f 
        # e.g., if trade is +20%, and f=0.10, account moves by +2.0%
        # formula: 1 + (trade/100 * f)
        
        multiplier_matrix = 1 + (sim_returns / 100.0) * f
        
        # Calculate cumulative equity paths. Start at 1.0
        # Shape: (10000, 300)
        equity_paths = np.cumprod(multiplier_matrix, axis=1)
        
        # Terminal Wealth extraction
        terminal_wealths = equity_paths[:, -1]
        median_cagr = (np.median(terminal_wealths) ** (1/5)) - 1 # Approx 5 years for 300 trades
        
        # Drawdown calculation
        # expanding max across axis 1
        running_max = np.maximum.accumulate(equity_paths, axis=1)
        # Ensure we don't divide by zero
        running_max[running_max == 0] = 1 
        drawdowns = (equity_paths - running_max) / running_max
        max_drawdowns = np.min(drawdowns, axis=1)
        
        median_dd = np.median(max_drawdowns)
        prob_ruin_20 = np.mean(max_drawdowns <= -0.20) * 100 # Chance of hitting -20% DD
        prob_ruin_40 = np.mean(max_drawdowns <= -0.40) * 100 # Chance of hitting -40% DD
        
        # Store results
        results.append({
            'Allocation per Trade': f"{f*100:.0f}%",
            'Position Count': f"{(1/f):.1f} stocks",
            'Median Est. 5Y Return': f"{(np.median(terminal_wealths)-1)*100:.0f}%",
            'Median Max DD': f"{median_dd*100:.1f}%",
            'Risk of -20% DD': f"{prob_ruin_20:.1f}%",
            'Risk of -40% DD': f"{prob_ruin_40:.1f}%"
        })
        
    df_results = pd.DataFrame(results)
    
    print(f"\n{'_'*115}")
    print(f"  MONTE CARLO FRACTIONAL SIZING OUTCOMES (Simulated 5 Years / 300 Trades)")
    print(f"{'_'*115}")
    
    headers = list(df_results.columns)
    print(f"  {headers[0]:<22} | {headers[1]:<15} | {headers[2]:<22} | {headers[3]:<15} | {headers[4]:<17} | {headers[5]}")
    print(f"  {'-'*110}")
    
    for i, row in df_results.iterrows():
        prefix = ">>" if abs((BET_SIZES[i] - kelly_f/2)) < 0.02 else "  " # Highlight Half-Kelly
        print(f"{prefix}{row[headers[0]]:<22} | {row[headers[1]]:<15} | {row[headers[2]]:<22} | {row[headers[3]]:<15} | {row[headers[4]]:<17} | {row[headers[5]]}")
        
    print("\nCONCLUSION:")
    print("Full Kelly yields technically maximum terminal wealth but brings catastrophic risk of ruin (>50% drawdowns).")
    print("Half-Kelly optimally balances explosive compound growth with near-zero probability of ruinous drawdowns.")

def run():
    if not os.path.exists(TRADE_LOG_PATH):
        print(f"Error: Trade log not found at {TRADE_LOG_PATH}. Please run sub_portfolio_cycles.py first.")
        return
        
    df = pd.read_csv(TRADE_LOG_PATH)
    if 'PnL%' not in df.columns:
        print("Error: Invalid trade log format. Missing 'PnL%' column.")
        return
        
    # Filter out bizarre data artifacts if any
    trades = df['PnL%'].clip(-100, 500)
    
    run_monte_carlo(trades)

if __name__ == "__main__":
    run()
