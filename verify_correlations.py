import pandas as pd

df = pd.read_csv('multi_index_timing_results.csv')

print("=== VERIFICATION: Bank & IT Correlations ===\n")

print("NIFTY BANK Correlations with avg_trend_score:")
print(f"  30D: {df[['avg_trend_score', 'Nifty Bank_30d']].dropna().corr().iloc[0,1]:+.3f}")
print(f"  60D: {df[['avg_trend_score', 'Nifty Bank_60d']].dropna().corr().iloc[0,1]:+.3f}")
print(f"  90D: {df[['avg_trend_score', 'Nifty Bank_90d']].dropna().corr().iloc[0,1]:+.3f}")

print("\nNIFTY IT Correlations with avg_trend_score:")
print(f"  30D: {df[['avg_trend_score', 'Nifty IT_30d']].dropna().corr().iloc[0,1]:+.3f}")
print(f"  60D: {df[['avg_trend_score', 'Nifty IT_60d']].dropna().corr().iloc[0,1]:+.3f}")
print(f"  90D: {df[['avg_trend_score', 'Nifty IT_90d']].dropna().corr().iloc[0,1]:+.3f}")

# Returns by score bin
print("\n\n=== RETURNS BY SCORE BIN ===\n")

# Low score
low = df[df['avg_trend_score'] < 40]
print(f"When Score < 40 (Bearish) - {len(low)} samples:")
print(f"  Bank 90D avg return: {low['Nifty Bank_90d'].mean():+.2f}%")
print(f"  IT 90D avg return: {low['Nifty IT_90d'].mean():+.2f}%")

# High score
high = df[df['avg_trend_score'] > 65]
print(f"\nWhen Score > 65 (Strong Bull) - {len(high)} samples:")
print(f"  Bank 90D avg return: {high['Nifty Bank_90d'].mean():+.2f}%")
print(f"  IT 90D avg return: {high['Nifty IT_90d'].mean():+.2f}%")

# Statistical significance check
print("\n\n=== SAMPLE DATA CHECK ===")
print("\nLow Score dates with Bank/IT returns:")
low_with_data = low[['date', 'avg_trend_score', 'Nifty Bank_90d', 'Nifty IT_90d']].dropna()
print(low_with_data.head(10).to_string(index=False))
