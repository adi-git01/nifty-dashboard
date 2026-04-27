"""
Theme Engine — Reusable Multi-Theme Momentum Scanner
=====================================================
Configurable RS-based momentum scoring for any stock universe.
Each theme defines its own tickers, RS weights, benchmark, and layer mapping.

Usage:
    from utils.theme_engine import ThemeEngine, AI_CAPEX_THEME
    engine = ThemeEngine(AI_CAPEX_THEME)
    results_df = engine.scan()
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================
# THEME DEFINITIONS
# ============================================================

AI_CAPEX_THEME = {
    'name': 'AI Datacenter Capex',
    'benchmark': 'SMH',           # VanEck Semiconductor ETF
    'rs_weights': [(5, 0.25), (21, 0.50), (63, 0.25)],
    'ma_period': 30,
    'lookback_days': 400,
    'min_volume_usd': 2_000_000,
    'tickers': [
        # Tier 1: Core AI Compute
        'NVDA', 'AMD', 'AVGO', 'MRVL', 'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU',
        # Tier 2: Power & Infrastructure
        'GEV', 'ETN', 'VRT', 'PWR', 'MTZ', 'CEG', 'CCJ', 'BE', 'NVT', 'EQIX', 'DLR',
        # Tier 3: Networking & Optics
        'ANET', 'COHR', 'LITE', 'MTSI', 'FN', 'APH', 'SMTC',
        # Tier 4: Design & Software Moat
        'SNPS', 'CDNS', 'ARM', 'INTC', 'GFS', 'ON', 'MPWR',
        # Tier 5: Cooling, Water, Construction
        'ECL', 'XYL', 'DOV', 'FIX', 'EME', 'OKLO', 'SMR',
        # Tier 6: Hyperscalers
        'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL',
        # Tier 7: Niche / Specialist
        'AXTI', 'PLAB', 'ACLS', 'ONTO', 'WOLF', 'FSLR',
    ],
    'layer_map': {
        'NVDA': 'L6: Chip Design', 'AMD': 'L6: Chip Design', 'AVGO': 'L6: Custom ASIC',
        'MRVL': 'L6: Custom ASIC', 'TSM': 'L4: Foundry', 'ASML': 'L3: Equipment',
        'AMAT': 'L3: Equipment', 'LRCX': 'L3: Equipment', 'KLAC': 'L3: Equipment',
        'MU': 'L5: HBM Memory', 'GEV': 'L9A: Power Gen', 'ETN': 'L9: Power Dist',
        'VRT': 'L9: Power Dist', 'PWR': 'L11: Construction', 'MTZ': 'L11: Construction',
        'CEG': 'L9A: Nuclear', 'CCJ': 'L9A: Nuclear', 'BE': 'L9A: Fuel Cell',
        'NVT': 'L10: Cooling', 'EQIX': 'L11: DC REIT', 'DLR': 'L11: DC REIT',
        'ANET': 'L8: Networking', 'COHR': 'L8: Optics', 'LITE': 'L8: Optics',
        'MTSI': 'L8: Optics', 'FN': 'L8: Optics', 'APH': 'L8: Connectors',
        'SMTC': 'L8: Optics', 'SNPS': 'L0: EDA Software', 'CDNS': 'L0: EDA Software',
        'ARM': 'L0: IP Cores', 'INTC': 'L4: Foundry', 'GFS': 'L4: Foundry',
        'ON': 'L6: Power Semi', 'MPWR': 'L6: VRM', 'ECL': 'L10: Cooling',
        'XYL': 'L10A: Water', 'DOV': 'L10: Cooling', 'FIX': 'L11: Construction',
        'EME': 'L11: Construction', 'OKLO': 'L9A: Nuclear', 'SMR': 'L9A: Nuclear',
        'MSFT': 'L12: Hyperscaler', 'GOOGL': 'L12: Hyperscaler', 'AMZN': 'L12: Hyperscaler',
        'META': 'L12: Hyperscaler', 'ORCL': 'L12: Hyperscaler',
        'AXTI': 'L1: InP Substrate', 'PLAB': 'L3: Photomasks', 'ACLS': 'L3: Ion Implant',
        'ONTO': 'L3: Metrology', 'WOLF': 'L1: SiC/GaN', 'FSLR': 'L9A: Solar',
    },
    # Bottleneck severity for visual coding
    'bottleneck_map': {
        'ASML': 'CRITICAL', 'TSM': 'CRITICAL', 'MU': 'CRITICAL',
        'COHR': 'CRITICAL', 'LITE': 'CRITICAL', 'AXTI': 'CRITICAL',
        'SNPS': 'CRITICAL', 'CDNS': 'CRITICAL', 'GEV': 'CRITICAL',
        'AVGO': 'TIGHT', 'MRVL': 'TIGHT', 'ETN': 'TIGHT', 'VRT': 'TIGHT',
        'MPWR': 'TIGHT', 'ARM': 'TIGHT', 'CCJ': 'TIGHT', 'LRCX': 'TIGHT',
    }
}

# Template for adding more themes
INDIA_DEFENSE_THEME = {
    'name': 'India Defense',
    'benchmark': '^NSEI',
    'rs_weights': [(5, 0.15), (21, 0.50), (63, 0.35)],
    'ma_period': 50,
    'lookback_days': 400,
    'min_volume_usd': 1_000_000,
    'tickers': [],  # To be populated
    'layer_map': {},
    'bottleneck_map': {},
}


class ThemeEngine:
    """
    Generic theme-based momentum scanner.
    Configurable via theme dict (tickers, RS weights, benchmark).
    """

    def __init__(self, theme_config: dict):
        self.config = theme_config
        self.name = theme_config['name']
        self.tickers = theme_config['tickers']
        self.benchmark = theme_config['benchmark']
        self.rs_weights = theme_config['rs_weights']
        self.ma_period = theme_config.get('ma_period', 50)
        self.min_vol = theme_config.get('min_volume_usd', 2_000_000)
        self.layer_map = theme_config.get('layer_map', {})
        self.bottleneck_map = theme_config.get('bottleneck_map', {})
        self.data_cache = {}
        self.benchmark_data = None

    def fetch_data(self, period_days=400):
        """Bulk-download all theme tickers + benchmark."""
        start = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')

        # Benchmark
        try:
            bench = yf.download(self.benchmark, start=start, progress=False, auto_adjust=True)
            if isinstance(bench.columns, pd.MultiIndex):
                bench.columns = bench.columns.get_level_values(0)
            if bench.index.tz is not None:
                bench.index = bench.index.tz_localize(None)
            self.benchmark_data = bench
        except Exception as e:
            print(f"  [ThemeEngine] Benchmark {self.benchmark} fetch failed: {e}")
            return False

        # Bulk download tickers
        try:
            bulk = yf.download(self.tickers, start=start, group_by='ticker',
                               threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  [ThemeEngine] Bulk download failed: {e}")
            return False

        loaded = 0
        if bulk is not None and not bulk.empty:
            for t in self.tickers:
                try:
                    if len(self.tickers) == 1:
                        df = bulk.copy()
                    elif t in bulk.columns.get_level_values(0):
                        df = bulk[t].dropna(how='all')
                    else:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    if not df.empty and len(df) > 60:
                        self.data_cache[t] = df
                        loaded += 1
                except Exception:
                    pass

        print(f"  [ThemeEngine:{self.name}] Loaded {loaded}/{len(self.tickers)} stocks + {self.benchmark}")
        return loaded > 0

    def composite_rs(self, ticker):
        """Calculate composite RS vs benchmark using theme-specific weights."""
        if ticker not in self.data_cache or self.benchmark_data is None:
            return None

        df = self.data_cache[ticker]
        bench = self.benchmark_data

        if len(df) < 64 or len(bench) < 64:
            return None

        price = float(df['Close'].iloc[-1])
        bench_price = float(bench['Close'].iloc[-1])

        rs_total = 0.0
        for period, weight in self.rs_weights:
            if len(df) < period + 1 or len(bench) < period + 1:
                return None
            stock_ret = price / float(df['Close'].iloc[-period]) - 1
            bench_ret = bench_price / float(bench['Close'].iloc[-period]) - 1
            rs_total += (stock_ret - bench_ret) * 100 * weight

        return rs_total

    def scan(self):
        """Run full scan and return sorted DataFrame."""
        if not self.data_cache:
            if not self.fetch_data(self.config.get('lookback_days', 400)):
                return pd.DataFrame()

        results = []
        for t in self.tickers:
            if t not in self.data_cache:
                continue

            df = self.data_cache[t]
            try:
                price = float(df['Close'].iloc[-1])
                ma = float(df['Close'].rolling(self.ma_period).mean().iloc[-1])
                ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
                ma200 = float(df['Close'].rolling(200).mean().iloc[-1]) if len(df) > 200 else ma50

                high_52w = float(df['High'].rolling(252).max().iloc[-1]) if len(df) > 252 else float(df['High'].max())
                dist_52w = (price - high_52w) / high_52w * 100

                # Volume
                vol_20d = float(df['Volume'].rolling(20).mean().iloc[-1]) * price
                vol_ratio = float(df['Volume'].iloc[-1]) / float(df['Volume'].rolling(60).mean().iloc[-1]) if len(df) > 60 else 1.0

                # RS
                rs = self.composite_rs(t)
                if rs is None:
                    continue

                # Individual RS periods for analysis
                rs_5d = (price / float(df['Close'].iloc[-5]) - 1) * 100 if len(df) > 5 else 0
                rs_21d = (price / float(df['Close'].iloc[-21]) - 1) * 100 if len(df) > 21 else 0
                rs_63d = (price / float(df['Close'].iloc[-63]) - 1) * 100 if len(df) > 63 else 0

                # Signal
                above_ma = price > ma
                trend = 'STRONG UP' if (rs > 10 and above_ma) else \
                        'UPTREND' if (rs > 0 and above_ma) else \
                        'NEUTRAL' if above_ma else \
                        'DOWNTREND' if rs > -10 else 'STRONG DOWN'

                results.append({
                    'Ticker': t,
                    'Layer': self.layer_map.get(t, '?'),
                    'Bottleneck': self.bottleneck_map.get(t, ''),
                    'Price': round(price, 2),
                    'CompRS': round(rs, 2),
                    'RS_5D': round(rs_5d, 2),
                    'RS_21D': round(rs_21d, 2),
                    'RS_63D': round(rs_63d, 2),
                    'Dist_52W': round(dist_52w, 1),
                    f'vs_MA{self.ma_period}': round((price / ma - 1) * 100, 1),
                    'vs_MA200': round((price / ma200 - 1) * 100, 1) if len(df) > 200 else None,
                    'Vol_20D_USD': round(vol_20d, 0),
                    'Vol_Ratio': round(vol_ratio, 2),
                    'Signal': trend,
                    'Above_MA': above_ma,
                })
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results).sort_values('CompRS', ascending=False)
        return result_df

    def get_layer_rotation(self):
        """Show which supply chain layers have best/worst RS right now."""
        scan = self.scan()
        if scan.empty:
            return pd.DataFrame()

        rotation = scan.groupby('Layer').agg(
            Avg_RS=('CompRS', 'mean'),
            Best_Stock=('CompRS', 'idxmax'),
            Count=('Ticker', 'count'),
            Pct_Above_MA=('Above_MA', 'mean'),
        ).round(2)

        # Get best ticker name for each layer
        rotation['Best_Stock'] = rotation['Best_Stock'].apply(
            lambda idx: scan.loc[idx, 'Ticker'] if idx in scan.index else '?'
        )
        rotation['Pct_Above_MA'] = (rotation['Pct_Above_MA'] * 100).round(0)

        return rotation.sort_values('Avg_RS', ascending=False)
