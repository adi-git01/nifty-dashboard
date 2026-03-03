import pandas as pd

# Load the historical data
df = pd.read_csv('data/market_mood_history.csv')

# The universe was precisely 500 stocks until today (Feb 24, 2026) when it became 1000
# We will iterate through and convert the absolute counts to percentages
for idx, row in df.iterrows():
    # If it's today's record, it was based on Nifty 1000. 
    # If it's a past record, it was based on Nifty 500.
    date_str = row['date']
    if date_str == '2026-02-24':
        universe_size = 980 # approximate yielding from live Nifty 1000 due to NaN cleaning
    else:
        universe_size = 500 

    df.at[idx, 'strong_momentum'] = round((row['strong_momentum'] / universe_size) * 100, 1)
    df.at[idx, 'total_uptrends'] = round((row['total_uptrends'] / universe_size) * 100, 1)
    df.at[idx, 'breakout_alerts'] = round((row['breakout_alerts'] / universe_size) * 100, 1)
    
# Save the converted percentages back
df.to_csv('data/market_mood_history.csv', index=False)
print("Historical Market Mood CSV successfully converted to percentages.")
