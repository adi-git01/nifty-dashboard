
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def chart_market_heatmap(df):
    """
    Creates a Market Map (Treemap) where:
    - Size = Market Cap
    - Color = Overall Score (Red=Bad, Green=Good)
    - Hierarchy = Sector -> Ticker
    """
    if df.empty:
        return None
        
    # Ensure market cap is numeric
    df['marketCap'] = pd.to_numeric(df['marketCap'], errors='coerce').fillna(0)
    
    fig = px.treemap(
        df,
        path=[px.Constant("Nifty 500"), 'sector', 'ticker'],
        values='marketCap',
        color='overall',
        color_continuous_scale='RdYlGn',
        range_color=[0, 10],
        custom_data=['name', 'price', 'overall', 'recommendation'],
        title="Market Quality Map (Size=M.Cap, Color=Score)"
    )
    
    fig.update_traces(
        textposition='middle center',
        texttemplate="%{label}<br>%{customdata[2]:.1f}",
        hovertemplate="<b>%{label}</b><br>Price: ₹%{customdata[1]:.2f}<br>Score: %{customdata[2]:.1f}<br>Verdict: %{customdata[3]}"
    )
    
    fig.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig


def chart_score_radar(scores):
    """
    Creates a Radar chart for the 4 investment pillars.
    """
    categories = ['Quality', 'Value', 'Growth', 'Momentum']
    values = [
        scores.get('quality', 5), 
        scores.get('value', 5), 
        scores.get('growth', 5), 
        scores.get('momentum', 5)
    ]
    
    # Close the loop
    categories = [*categories, categories[0]]
    values = [*values, values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Score',
        line_color='#00F0FF',
        fillcolor='rgba(0, 240, 255, 0.2)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=False,
                gridcolor='#333'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        font=dict(color='white', family='Inter')
    )
    return fig

def chart_price_history(hist_df):
    """
    Standard Price History Chart.
    """
    if hist_df is None or hist_df.empty:
        return None
        
    fig = go.Figure()

    # Price Candlestick
    fig.add_trace(go.Candlestick(x=hist_df.index,
                open=hist_df['Open'],
                high=hist_df['High'],
                low=hist_df['Low'],
                close=hist_df['Close'],
                name='Price',
                increasing_line_color='#00C853',
                decreasing_line_color='#FF5252'
    ))

    # Add Smart Volume if available
    if 'Volume' in hist_df.columns:
        # Calculate colors
        colors = []
        avg_vol = hist_df['Volume'].rolling(20).mean()
        
        for i in range(len(hist_df)):
            close = hist_df['Close'].iloc[i]
            open_p = hist_df['Open'].iloc[i]
            vol = hist_df['Volume'].iloc[i]
            ma = avg_vol.iloc[i] if i >= 19 else vol
            
            is_up = close >= open_p
            is_surge = vol > (2 * ma)
            is_panic = vol > (4 * ma)
            
            if is_panic:
                colors.append('#D50000' if not is_up else '#00E676') # Brightest
            elif is_surge:
                colors.append('#FF1744' if not is_up else '#69F0AE') # Bright
            else:
                colors.append('rgba(255, 82, 82, 0.3)' if not is_up else 'rgba(0, 200, 83, 0.3)') # Dim

        fig.add_trace(go.Bar(
            x=hist_df.index,
            y=hist_df['Volume'],
            name='Smart Volume',
            marker_color=colors,
            yaxis='y2',
            opacity=0.8
        ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", overlaying='y', side='left', showgrid=False, range=[0, hist_df['Volume'].max() * 4]),
        showlegend=False
    )
    return fig

def chart_gauge(score):
    """
    Creates a semi-circle gauge for the overall score.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00F0FF"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 4], 'color': 'rgba(255, 59, 48, 0.3)'},
                {'range': [4, 7], 'color': 'rgba(255, 204, 0, 0.3)'},
                {'range': [7, 10], 'color': 'rgba(52, 199, 89, 0.3)'}],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        font={'color': "white", 'family': "Inter"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=200
    )
    return fig

# === NEW CHARTS FOR TIME TRENDS ===

def chart_sector_rotation(sector_history):
    """
    Line chart showing Relative Strength of Sectors vs Nifty 500 (Base 100).
    """
    if sector_history.empty:
        return None
    
    # Calculate Relative Strength (Sector / Nifty500 * 100)
    rel_strength = sector_history.copy()
    benchmark = rel_strength['Nifty 500 (Eq Wt)']
    
    for col in rel_strength.columns:
        if col != 'Nifty 500 (Eq Wt)':
            rel_strength[col] = (rel_strength[col] / benchmark) * 100
            
    # Visualize top performing sectors
    fig = go.Figure()
    
    # Add traces for sectors
    for col in rel_strength.columns:
        if col == 'Nifty 500 (Eq Wt)': continue
        
        # Highlight if it ends > 100 (Outperforming)
        last_val = rel_strength[col].iloc[-1]
        width = 3 if last_val > 110 else 1
        opacity = 1.0 if last_val > 105 else 0.4
        
        fig.add_trace(go.Scatter(
            x=rel_strength.index,
            y=rel_strength[col],
            mode='lines',
            name=col,
            line=dict(width=width),
            opacity=opacity
        ))
        
    # Add Baseline (100)
    fig.add_hline(y=100, line_dash="dash", line_color="white", annotation_text="Market Perform")
    
    fig.update_layout(
        title="Sector Relative Strength (vs Nifty 500)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Rel Strength (100 = Market)",
        height=500
    )
    return fig

def chart_stock_cycle(trend_df):
    """
    Trend Score Evolution Chart with optional quarterly aggregation.
    Main focus: Trend Score over time with color-coded zones.
    """
    if trend_df.empty:
        return None
        
    # Create subplots - Trend Score (top), Price (mid), Volume (bottom)
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.50, 0.30, 0.20],
                        subplot_titles=("📊 Trend Score Evolution", "📈 Price Action", "b📊 Smart Volume"))

    # 1. MAIN CHART: Trend Score Area with gradient coloring
    # Color the area based on trend score zones
    fig.add_trace(go.Scatter(
        x=trend_df.index,
        y=trend_df['trend_score'],
        fill='tozeroy',
        mode='lines',
        line=dict(color='#00F0FF', width=2),
        fillcolor='rgba(0, 240, 255, 0.3)',
        name='Trend Score',
        hovertemplate='Date: %{x}<br>Trend Score: %{y:.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Add zone backgrounds using shapes
    # Strong Uptrend Zone (>75)
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(0, 200, 83, 0.15)", 
                  line_width=0, row=1, col=1)
    # Neutral Zone (25-75)
    fig.add_hrect(y0=25, y1=75, fillcolor="rgba(255, 214, 0, 0.08)", 
                  line_width=0, row=1, col=1)
    # Downtrend Zone (<25)
    fig.add_hrect(y0=0, y1=25, fillcolor="rgba(213, 0, 0, 0.15)", 
                  line_width=0, row=1, col=1)
    
    # Add threshold lines with labels
    fig.add_hline(y=75, line_dash="dash", line_color="#00C853", opacity=0.7, row=1, col=1,
                  annotation_text="Strong Uptrend (75)", annotation_position="right")
    fig.add_hline(y=50, line_dash="dot", line_color="#888", opacity=0.5, row=1, col=1,
                  annotation_text="Neutral (50)", annotation_position="right")
    fig.add_hline(y=25, line_dash="dash", line_color="#D50000", opacity=0.7, row=1, col=1,
                  annotation_text="Downtrend (25)", annotation_position="right")
    
    # Add signal color markers at key points (optional: monthly markers)
    if 'signal_color' in trend_df.columns:
        # Sample monthly for cleaner visualization
        monthly_df = trend_df.resample('ME').last().dropna()
        if not monthly_df.empty:
            fig.add_trace(go.Scatter(
                x=monthly_df.index,
                y=monthly_df['trend_score'],
                mode='markers',
                marker=dict(
                    color=monthly_df['signal_color'],
                    size=8,
                    line=dict(color='white', width=1)
                ),
                name='Monthly Signal',
                hovertemplate='%{x}<br>Score: %{y:.0f}<extra></extra>'
            ), row=1, col=1)
    

    # 2. SECONDARY CHART: Price Reference
    if 'Close' in trend_df.columns:
        fig.add_trace(go.Scatter(
            x=trend_df.index,
            y=trend_df['Close'],
            mode='lines',
            line=dict(color='#888', width=1),
            name='Price',
            hovertemplate='Price: ₹%{y:.2f}<extra></extra>'
        ), row=2, col=1)
        
        # Add MAs if available
        if 'MA50' in trend_df.columns:
            fig.add_trace(go.Scatter(
                x=trend_df.index, y=trend_df['MA50'], 
                line=dict(color='orange', width=1, dash='dot'), 
                name='50 DMA'
            ), row=2, col=1)
        if 'MA200' in trend_df.columns:
            fig.add_trace(go.Scatter(
                x=trend_df.index, y=trend_df['MA200'], 
                line=dict(color='white', width=1, dash='dot'), 
                name='200 DMA'
            ), row=2, col=1)

    # 3. TERTIARY CHART: Smart Volume
    if 'Volume' in trend_df.columns:
        # Calculate Volume Colors (same logic as price chart)
        vol_colors = []
        avg_vol = trend_df['Volume'].rolling(20).mean()
        
        for i in range(len(trend_df)):
            close = trend_df['Close'].iloc[i] if 'Close' in trend_df.columns else 0
            open_p = trend_df['Open'].iloc[i] if 'Open' in trend_df.columns else close
            vol = trend_df['Volume'].iloc[i]
            ma = avg_vol.iloc[i] if i >= 19 else vol
            
            is_up = close >= open_p
            is_surge = vol > (2 * ma)
            is_panic = vol > (4 * ma)
            
            if is_panic:
                vol_colors.append('#D50000' if not is_up else '#00E676') # Brightest
            elif is_surge:
                vol_colors.append('#FF1744' if not is_up else '#69F0AE') # Bright
            else:
                vol_colors.append('rgba(255, 82, 82, 0.3)' if not is_up else 'rgba(0, 200, 83, 0.3)') # Dim

        fig.add_trace(go.Bar(
            x=trend_df.index,
            y=trend_df['Volume'],
            name='Smart Volume',
            marker_color=vol_colors,
            hovertemplate='Volume: %{y:.2s}<extra></extra>'
        ), row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=60, t=40, b=0),
        height=700,
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        yaxis=dict(range=[0, 100], title="Trend Score"),
        yaxis2=dict(title="Price (₹)"),
        yaxis3=dict(title="Volume", showgrid=False)
    )
    
    return fig

def chart_relative_performance(stock_history_dict):
    """
    Line chart comparing normalized performance of multiple stocks (Base 0%).
    stock_history_dict: {ticker: series_of_prices}
    """
    if not stock_history_dict:
        return None
        
    fig = go.Figure()
    
    for ticker, series in stock_history_dict.items():
        if series.empty: continue
        
        # Normalize to percentage return
        start_price = series.iloc[0]
        if start_price == 0: continue
        
        normalized = ((series - start_price) / start_price) * 100
        
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized,
            mode='lines',
            name=ticker,
            hovertemplate='%{y:.1f}%'
        ))
        
    fig.update_layout(
        title="Relative Performance (Normalized %)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Return (%)",
        xaxis_title=None,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def chart_score_history(hist_df, current_scores=None):
    """
    Line chart showing historical evolution of 4-pillar scores.
    hist_df: DataFrame with [date, quality, value, growth, momentum, overall]
    """
    if hist_df is None or hist_df.empty:
        return None
    
    fig = go.Figure()
    
    # Add lines for each pillar
    colors = {'quality': '#00F0FF', 'value': '#34C759', 'growth': '#FF9500', 'momentum': '#AF52DE', 'overall': '#FFFFFF'}
    
    for col in ['quality', 'value', 'growth', 'momentum', 'overall']:
        if col in hist_df.columns:
            width = 3 if col == 'overall' else 1.5
            fig.add_trace(go.Scatter(
                x=hist_df['date'],
                y=hist_df[col],
                mode='lines+markers',
                name=col.title(),
                line=dict(color=colors.get(col, '#888'), width=width),
                hovertemplate=f'{col.title()}: %{{y:.1f}}<extra></extra>'
            ))
    
    # Add current score as final point if provided
    if current_scores:
        today = pd.Timestamp.now()
        for col in ['quality', 'value', 'growth', 'momentum', 'overall']:
            if col in current_scores:
                fig.add_trace(go.Scatter(
                    x=[today],
                    y=[current_scores[col]],
                    mode='markers',
                    name=f'{col.title()} (Now)',
                    marker=dict(color=colors.get(col, '#888'), size=12, symbol='star'),
                    showlegend=False
                ))
    
    fig.update_layout(
        title="Score Evolution Over Time",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(range=[0, 10], title="Score"),
        xaxis_title=None,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def chart_volume_analysis(trend_df):
    """
    Dedicated Multi-Factor Volume Analysis Chart.
    Shows:
    1. Volume Bars (Color coded by 'Smart' logic)
    2. Price Line overlay
    3. Volume MA
    """
    if trend_df is None or trend_df.empty or 'Volume' not in trend_df.columns:
        return None
        
    fig = go.Figure()
    
    # Calculate Colors
    vol_colors = []
    avg_vol = trend_df['Volume'].rolling(20).mean()
    
    for i in range(len(trend_df)):
        close = trend_df['Close'].iloc[i] if 'Close' in trend_df.columns else 0
        open_p = trend_df['Open'].iloc[i] if 'Open' in trend_df.columns else close
        vol = trend_df['Volume'].iloc[i]
        ma = avg_vol.iloc[i] if i >= 19 else vol
        
        is_up = close >= open_p
        is_surge = vol > (2 * ma)
        is_panic = vol > (4 * ma)
        
        if is_panic:
            vol_colors.append('#D50000' if not is_up else '#00E676') # Brightest (Panic/Smart Buy)
        elif is_surge:
            vol_colors.append('#FF1744' if not is_up else '#69F0AE') # Bright
        else:
            vol_colors.append('rgba(255, 255, 255, 0.1)' if not is_up else 'rgba(255, 255, 255, 0.1)') # Dim/Neutral

    # 1. Volume Bars
    fig.add_trace(go.Bar(
        x=trend_df.index,
        y=trend_df['Volume'],
        name='Volume',
        marker_color=vol_colors,
        yaxis='y'
    ))
    
    # 2. Volume MA
    fig.add_trace(go.Scatter(
        x=trend_df.index,
        y=avg_vol,
        mode='lines',
        name='20 DMA',
        line=dict(color='yellow', width=1, dash='dot')
    ))
    
    # 3. Price Overlay (Secondary Y)
    if 'Close' in trend_df.columns:
        fig.add_trace(go.Scatter(
            x=trend_df.index,
            y=trend_df['Close'],
            name='Price',
            mode='lines',
            line=dict(color='white', width=2),
            yaxis='y2'
        ))

    fig.update_layout(
        title="📊 Multi-Factor Volume Analysis (Smart Money Flow)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        showlegend=True,
        yaxis=dict(title="Volume", showgrid=False),
        yaxis2=dict(title="Price", overlaying='y', side='right', showgrid=False, gridcolor='#333'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
