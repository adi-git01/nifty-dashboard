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
    'benchmark': 'SMH',           # VanEck Semiconductor ETF — better alpha baseline than SPY
    'rs_weights': [(5, 0.30), (21, 0.50), (63, 0.20)],   # Config D (backtest-optimal)
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
    'bottleneck_map': {
        'ASML': 'CRITICAL', 'TSM': 'CRITICAL', 'MU': 'CRITICAL',
        'COHR': 'CRITICAL', 'LITE': 'CRITICAL', 'AXTI': 'CRITICAL',
        'SNPS': 'CRITICAL', 'CDNS': 'CRITICAL', 'GEV': 'CRITICAL',
        'AVGO': 'TIGHT', 'MRVL': 'TIGHT', 'ETN': 'TIGHT', 'VRT': 'TIGHT',
        'MPWR': 'TIGHT', 'ARM': 'TIGHT', 'CCJ': 'TIGHT', 'LRCX': 'TIGHT',
    },
}

# Template for adding more themes
INDIA_DEFENSE_THEME = {
    'name': 'India Defense',
    'benchmark': '^NSEI',
    'rs_weights': [(5, 0.15), (21, 0.50), (63, 0.35)],
    'ma_period': 50,
    'lookback_days': 400,
    'min_volume_usd': 1_000_000,
    'tickers': [],
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

        # Benchmark — single ticker download
        try:
            bench = yf.download(
                self.benchmark, start=start,
                threads=False, progress=False, auto_adjust=True
            )
            if isinstance(bench.columns, pd.MultiIndex):
                bench.columns = bench.columns.get_level_values(0)
            if bench.index.tz is not None:
                bench.index = bench.index.tz_localize(None)
            self.benchmark_data = bench
        except Exception as e:
            print(f"  [ThemeEngine] Benchmark {self.benchmark} fetch failed: {e}")
            return False

        # Bulk download tickers
        # threads=False prevents 401 cascade on GitHub Actions CI
        try:
            bulk = yf.download(
                self.tickers, start=start, group_by='ticker',
                threads=False, progress=False, auto_adjust=True
            )
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

    def _comp_rs_slice(self, close_series, bench_series):
        """Compute composite RS from two pre-sliced Close series.
        Uses iloc[-(period+1)] so RS5 = exactly 5-interval return."""
        if len(close_series) < 64 or len(bench_series) < 64:
            return None
        price = float(close_series.iloc[-1])
        b_price = float(bench_series.iloc[-1])
        rs_total = 0.0
        for period, weight in self.rs_weights:
            if len(close_series) < period + 1 or len(bench_series) < period + 1:
                return None
            s_ret = price / float(close_series.iloc[-(period + 1)]) - 1
            b_ret = b_price / float(bench_series.iloc[-(period + 1)]) - 1
            rs_total += (s_ret - b_ret) * 100 * weight
        return rs_total

    def composite_rs(self, ticker):
        """Calculate composite RS vs benchmark using theme-specific weights."""
        if ticker not in self.data_cache or self.benchmark_data is None:
            return None
        return self._comp_rs_slice(
            self.data_cache[ticker]['Close'],
            self.benchmark_data['Close'],
        )

    def composite_rs_offset(self, ticker, offset_days=5):
        """Composite RS computed as of `offset_days` trading days ago (for velocity)."""
        if ticker not in self.data_cache or self.benchmark_data is None:
            return None
        df_close = self.data_cache[ticker]['Close']
        bench_close = self.benchmark_data['Close']
        if offset_days >= len(df_close) or offset_days >= len(bench_close):
            return None
        return self._comp_rs_slice(
            df_close.iloc[:-offset_days],
            bench_close.iloc[:-offset_days],
        )

    def scan(self):
        """
        Run full scan and return sorted DataFrame.

        Key columns:
          CompRS   — composite RS vs benchmark (theme-weighted)
          RS_Vel   — RS velocity: how much CompRS changed vs 5 days ago
          RS5vs    — 5-day RS vs benchmark  (true relative strength, not absolute return)
          RS21vs   — 21-day RS vs benchmark
          RS63vs   — 63-day RS vs benchmark
          Ret5D/21D/63D — stock's own absolute return (for reference)
          Signal   — STRONG UP / UPTREND / NEUTRAL / DOWNTREND / STRONG DOWN
        """
        if not self.data_cache:
            if not self.fetch_data(self.config.get('lookback_days', 400)):
                return pd.DataFrame()

        bench = self.benchmark_data
        if bench is None:
            return pd.DataFrame()

        # Pre-compute benchmark returns once (reused across all tickers)
        b_close = bench['Close']
        b_last = float(b_close.iloc[-1])
        b_ret5  = (b_last / float(b_close.iloc[-5])  - 1) if len(b_close) >= 6  else 0.0
        b_ret21 = (b_last / float(b_close.iloc[-21]) - 1) if len(b_close) >= 22 else 0.0
        b_ret63 = (b_last / float(b_close.iloc[-63]) - 1) if len(b_close) >= 64 else 0.0

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

                lb = min(len(df), 252)
                h_col = 'High' if 'High' in df.columns else 'Close'
                high_52w = float(df[h_col].iloc[-lb:].max())
                dist_52w = (price - high_52w) / high_52w * 100

                # Volume
                vol_20d_usd = float(df['Volume'].rolling(20).mean().iloc[-1]) * price
                vol_60d_avg = float(df['Volume'].rolling(60).mean().iloc[-1]) if len(df) > 60 else float(df['Volume'].mean())
                vol_ratio = float(df['Volume'].iloc[-1]) / vol_60d_avg if vol_60d_avg > 0 else 1.0

                # Composite RS (vs benchmark, using theme weights)
                rs = self.composite_rs(t)
                if rs is None:
                    continue

                # RS velocity: change in CompRS over last 5 trading days
                rs_5d_ago = self.composite_rs_offset(t, offset_days=5)
                rs_vel = round(rs - rs_5d_ago, 2) if rs_5d_ago is not None else None

                # Stock absolute returns per period
                ret5d  = (price / float(df['Close'].iloc[-5])  - 1) * 100 if len(df) >= 6  else 0.0
                ret21d = (price / float(df['Close'].iloc[-21]) - 1) * 100 if len(df) >= 22 else 0.0
                ret63d = (price / float(df['Close'].iloc[-63]) - 1) * 100 if len(df) >= 64 else 0.0

                # Per-period RS vs benchmark (stock return - benchmark return for same period)
                rs5vs  = round((ret5d  / 100 - b_ret5)  * 100, 2)
                rs21vs = round((ret21d / 100 - b_ret21) * 100, 2)
                rs63vs = round((ret63d / 100 - b_ret63) * 100, 2)

                above_ma = price > ma
                signal = (
                    'STRONG UP'   if (rs > 10 and above_ma) else
                    'UPTREND'     if (rs > 0  and above_ma) else
                    'NEUTRAL'     if above_ma else
                    'DOWNTREND'   if rs > -10 else
                    'STRONG DOWN'
                )

                results.append({
                    'Ticker':      t,
                    'Layer':       self.layer_map.get(t, 'Unknown'),
                    'Bottleneck':  self.bottleneck_map.get(t, ''),
                    'Price':       round(price, 2),
                    'CompRS':      round(rs, 2),
                    'RS_Vel':      rs_vel,
                    'RS5vs':       rs5vs,
                    'RS21vs':      rs21vs,
                    'RS63vs':      rs63vs,
                    'Ret5D':       round(ret5d, 2),
                    'Ret21D':      round(ret21d, 2),
                    'Ret63D':      round(ret63d, 2),
                    'Dist_52W':    round(dist_52w, 1),
                    f'vs_MA{self.ma_period}': round((price / ma - 1) * 100, 1),
                    'vs_MA200':    round((price / ma200 - 1) * 100, 1) if len(df) > 200 else None,
                    'Vol_20D_USD': round(vol_20d_usd, 0),
                    'Vol_Ratio':   round(vol_ratio, 2),
                    'Signal':      signal,
                    'Above_MA':    above_ma,
                })
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).sort_values('CompRS', ascending=False).reset_index(drop=True)

    def get_layer_rotation(self):
        """
        Aggregate CompRS, RS_Vel, and signal counts by supply-chain layer.
        Returns a DataFrame sorted by CompRS descending.
        """
        scan = self.scan()
        if scan.empty:
            return pd.DataFrame()

        rotation = scan.groupby('Layer').agg(
            CompRS=('CompRS', 'mean'),
            RS_Vel=('RS_Vel', 'mean'),
            RS5vs=('RS5vs', 'mean'),
            RS21vs=('RS21vs', 'mean'),
            RS63vs=('RS63vs', 'mean'),
            Count=('Ticker', 'count'),
            Pct_Above_MA=('Above_MA', 'mean'),
        ).round(2)

        rotation['Pct_Above_MA'] = (rotation['Pct_Above_MA'] * 100).round(0)

        # Best stock per layer: safest approach using idxmax on grouped series
        best = (
            scan.loc[scan.groupby('Layer')['CompRS'].idxmax(), ['Layer', 'Ticker']]
                .set_index('Layer')['Ticker']
        )
        rotation['Best_Stock'] = best

        return rotation.sort_values('CompRS', ascending=False)
